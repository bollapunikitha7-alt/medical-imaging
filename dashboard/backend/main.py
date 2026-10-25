"""FastAPI front for JEPA-Med-OOD. Read-only + inference; no retraining."""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

# CPU-only laptop: cap torch threads so the asyncio loop stays responsive
# during long forward passes (prevents listener starvation / ECONNRESET).
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
import config  # noqa: E402
import inference as I  # noqa: E402
import alerts as A  # noqa: E402

DISCLAIMER = ("Research prototype for OOD detection. "
              "Not a medical device and not for diagnosis.")

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true",
                help="Serve clearly fake sample data for screenshots.")
_demo = ap.parse_known_args()[0].demo

app = FastAPI(title="JEPA-Med-OOD dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/results", StaticFiles(directory=str(BACKEND.parent.parent / "experiments")),
          name="results")


def demo_guard(payload):
    if _demo:
        payload["demo"] = True
        payload["banner"] = "DEMO DATA - NOT REAL RESULTS"
    return payload


@app.get("/api/models")
def models():
    st = I.checkpoint_status()
    return demo_guard({
        "models": [
            {"id": m, "label": I.PRETTY[m] + (" (Self-supervised)" if m != "supervised"
                                              else " (Baseline)"),
             "encoder": st[m]["available"],
             "head": st.get(f"{m}_probe", {}).get("available", False),
             "detail": "Checkpoint not found" if not st[m]["available"]
                       else ("Head missing: Energy unavailable"
                             if not st.get(f"{m}_probe", {}).get("available", False)
                             else "ok")}
            for m in I.MODELS],
        "disclaimer": DISCLAIMER})


def _score(name, method, data, filename):
    img = I.validate_upload(data, filename)
    tensor, _ = I.preprocess(img)
    try:
        return I.score_tensor(name, method, tensor)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), model: str = "jepa",
                  method: str = Query("mahalanobis")):
    if _demo:
        return demo_guard({"model": model, "method": method, "raw": 0.82,
                           "normalized": 0.82, "percentile": 82.0,
                           "threshold": 0.5, "decision": "OOD"})
    if model not in I.MODELS or method not in ("energy", "mahalanobis"):
        raise HTTPException(400, "Bad model/method.")
    data = await image.read()
    fname = image.filename or "upload.png"
    try:
        out = await run_in_threadpool(_score, model, method, data, fname)
    except HTTPException as e:
        if e.status_code == 404:
            A.from_exception("predict", FileNotFoundError(str(e.detail)),
                             model=model, method=method, data=data)
        else:
            A.from_exception("predict", ValueError(str(e.detail)),
                             model=model, method=method, data=data)
        raise
    except Exception as e:
        A.from_exception("predict", e, model=model, method=method, data=data)
        raise
    A.from_result(model, method, data, out)
    out.update({"model": I.PRETTY[model], "method": method})
    return demo_guard(out)


def _compare_rows(data, filename, method):
    rows = []
    for m in I.MODELS:
        try:
            img = I.validate_upload(data, filename)
            tensor, _ = I.preprocess(img)
            r = I.score_tensor(m, method, tensor)
            r["model"] = I.PRETTY[m]
            rows.append(r)
            A.from_result(m, method, data, r)
        except FileNotFoundError as e:
            A.from_exception("compare", e, model=m, method=method, data=data)
            rows.append({"model": I.PRETTY[m], "error": str(e)})
        except Exception as e:
            A.from_exception("compare", e, model=m, method=method, data=data)
            rows.append({"model": I.PRETTY[m], "error": str(e)})
    return rows


@app.post("/api/compare")
async def compare(image: UploadFile = File(...), method: str = Query("mahalanobis")):
    data = await image.read()
    rows = await run_in_threadpool(_compare_rows, data, image.filename or "upload.png", method)
    return demo_guard({"method": method, "rows": rows})


@app.post("/api/batch")
async def batch(images: list[UploadFile] = File(...), model: str = "jepa",
                method: str = Query("mahalanobis")):
    out = []
    for im in images:
        data = await im.read()
        fname = im.filename or "upload.png"
        try:
            r = await run_in_threadpool(_score, model, method, data, fname)
            A.from_result(model, method, data, r)
            out.append({"file": im.filename, "score": r["raw"],
                        "normalized": r["normalized"], "decision": r["decision"]})
        except (ValueError, FileNotFoundError) as e:
            A.from_exception("batch", e, model=model, method=method, data=data)
            out.append({"file": im.filename, "error": str(e)})
    return demo_guard({"model": model, "method": method, "rows": out})


def _attention_maps(data, filename, model, method="rollout", target="energy"):
    """Returns (overlay_png, heat_png, preview_png, method_label).

    method="rollout": Abnarain attention rollout (no target, no gradients).
    method="gradcam": Grad-CAM of `target` ("energy" = the served OOD score,
      "logit" = predicted-class logit). Mahalanobis has no gradients and is
    rejected with 400, never substituted."""
    if method not in ("rollout", "gradcam"):
        raise HTTPException(400, "method must be 'rollout' or 'gradcam'.")
    img = I.validate_upload(data, filename)
    tensor, preview = I.preprocess(img)
    short = "attention rollout"
    try:
        if method == "gradcam":
            take, label = I.gradcam_map(model, tensor, target=target)
            short = "grad-cam"
        else:
            roll = I.attention_rollout(model, tensor)  # [196,196]
            take = roll.mean(axis=0).reshape(14, 14)
            take = (take - take.min()) / (take.max() - take.min() + 1e-9)
            label = "attention-rollout (not Grad-CAM)"
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Smooth 14x14 -> 224x224 with bilinear (avoids blocky nearest-neighbor look)
    from PIL import Image as _PIL
    base = preview.resize((224, 224), _PIL.BILINEAR).convert("RGB")
    heat = _PIL.fromarray((take * 255).astype("uint8")).resize((224, 224), _PIL.BILINEAR)
    heat_arr = np.asarray(heat).astype(float) / 255.0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fig_png(draw):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        draw(ax)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _draw_base(ax):
        ax.imshow(base)

    def _draw_heat(ax):
        ax.imshow(heat_arr, cmap="jet", interpolation="bilinear")
        ax.set_title(short, fontsize=9)

    def _draw_overlay(ax):
        ax.imshow(base)
        ax.imshow(heat_arr, cmap="jet", alpha=0.5, extent=(0, 224, 224, 0),
                  interpolation="bilinear")
        ax.set_title("overlay", fontsize=9)

    return _fig_png(_draw_overlay), _fig_png(_draw_heat), _fig_png(_draw_base), label


def _attention_png(data, filename, model):
    overlay, _, _, _ = _attention_maps(data, filename, model)
    return overlay


@app.post("/api/attention")
async def attention(image: UploadFile = File(...), model: str = "jepa",
                    view: str = Query("overlay"),
                    method: str = Query("rollout"),
                    target: str = Query("energy")):
    if model not in I.MODELS:
        raise HTTPException(400, "Bad model.")
    data = await image.read()
    fname = image.filename or "upload.png"
    if view == "split":
        import base64

        def _b64(png: bytes):
            return "data:image/png;base64," + base64.b64encode(png).decode()
        try:
            overlay, heat, base, label = await run_in_threadpool(
                _attention_maps, data, fname, model, method, target)
        except HTTPException as e:
            A.from_exception("attention",
                             FileNotFoundError(str(e.detail))
                             if e.status_code == 404 else ValueError(str(e.detail)),
                             model=model, method=method, data=data)
            raise
        except Exception as e:
            A.from_exception("attention", e, model=model, method=method,
                             data=data)
            raise
        kind = "gradcam" if method == "gradcam" else "attention_rollout"
        return demo_guard({
            "model": I.PRETTY[model], "method": label,
            "method_kind": kind,
            "is_gradcam": kind == "gradcam",
            "target": target if kind == "gradcam" else None,
            "original": _b64(base), "heatmap": _b64(heat), "overlay": _b64(overlay),
        })
    try:
        png = await run_in_threadpool(_attention_png, data, fname, model)
    except HTTPException as e:
        A.from_exception("attention",
                         FileNotFoundError(str(e.detail))
                         if e.status_code == 404 else ValueError(str(e.detail)),
                         model=model, method=method, data=data)
        raise
    except Exception as e:
        A.from_exception("attention", e, model=model, method=method, data=data)
        raise
    return Response(png, media_type="image/png",
                    headers={"X-Method": "attention-rollout (not Grad-CAM)"})


@app.get("/api/alerts")
def alerts_list(include_acked: bool = False):
    if _demo:
        return demo_guard({"alerts": []})
    return demo_guard({"alerts": A.list_alerts(include_acked=include_acked)})


@app.post("/api/alerts/{alert_id}/ack")
def alerts_ack(alert_id: str):
    a = A.ack_alert(alert_id)
    if a is None:
        raise HTTPException(404, f"Unknown alert {alert_id}.")
    return demo_guard(a)


@app.get("/api/score-distribution")
def score_distribution(model: str = "jepa", method: str = "mahalanobis"):
    try:
        dist = I.get_id_scores(model, method).tolist()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return demo_guard({"model": model, "method": method, "id_scores": dist,
                       "threshold": I.get_threshold(model, method),
                       "note": "ID validation scores only; no OOD used."})


@app.get("/api/metrics")
def metrics():
    return demo_guard(I.read_metrics())


@app.get("/api/dataset-info")
def dataset_info():
    return demo_guard({"datasets": I.dataset_info()})
