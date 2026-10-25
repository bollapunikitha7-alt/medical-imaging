"""Reusable inference. Imports repo scoring code; copies nothing.

Sign convention (cxr_ood/evaluation/ood_detection.py): higher = more OOD
for both Energy (-T*logsumexp logits) and Mahalanobis (min-class squared
distance, EmpiricalCovariance fit on ID TRAIN only).

0-1 OOD score (documented formula): empirical percentile of the raw score
within cached ID-validation scores, i.e. P(ID_score <= raw). Higher = more
OOD. Primary value stays the raw score.

Threshold (ID-only): 95th percentile of ID calibration scores
(TPR_TARGET). OOD data never touches fitting or thresholds.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND.parent.parent))
sys.path.insert(0, str(BACKEND.parent.parent / "IJEPA_Meta"))

from cxr_ood.analysis.baseline_comparison import (  # noqa: E402
    JEPAEncoder, MAEEncoder, SupervisedEncoder, LinearProbe)
from cxr_ood.evaluation.ood_detection import (  # noqa: E402
    compute_energy_score, MahalanobisDetector)

import config  # noqa: E402

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
MODELS = ("jepa", "mae", "supervised")
PRETTY = {"jepa": "JEPA", "mae": "MAE", "supervised": "Supervised"}

_cache = {}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_upload(data: bytes, filename: str):
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds {config.MAX_UPLOAD_MB} MB.")
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("Only PNG/JPG/JPEG uploads.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception:
        raise ValueError("File is not a decodable image.")
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")  # same as repo loaders
    return img


def preprocess(img: Image.Image):
    preview = img.resize((224, 224), Image.BILINEAR)
    return PREPROCESS(img).unsqueeze(0), preview


def checkpoint_status():
    out = {}
    for m in MODELS:
        p = config.CHECKPOINTS[m]
        out[m] = {"available": p.exists(), "path": str(p)}
    for m in MODELS:
        hp = config.CHECKPOINTS.get(f"{m}_probe", config.CHECKPOINTS["jepa_probe"])
        out[f"{m}_probe"] = {"available": hp.exists(), "path": str(hp)}
    return out


def get_encoder(name):
    if name in _cache.get("enc", {}):
        return _cache["enc"][name]
    p = config.CHECKPOINTS[name]
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    dev = str(device())
    enc = {"jepa": lambda: JEPAEncoder(str(p), model_name="vit_small", device=dev),
           "mae": lambda: MAEEncoder(checkpoint_path=str(p), device=dev),
           "supervised": lambda: SupervisedEncoder(checkpoint_path=str(p), device=dev)}[name]()
    _cache.setdefault("enc", {})[name] = enc.to(device())
    return _cache["enc"][name]


def get_probe(name):
    if name not in MODELS:
        raise FileNotFoundError(f"Unknown model {name}.")
    hp = config.CHECKPOINTS.get(f"{name}_probe", config.CHECKPOINTS["jepa_probe"])
    if not hp.exists():
        raise FileNotFoundError(f"Probe head not found: {hp}")
    dev = device()
    enc = get_encoder(name)
    model = LinearProbe(enc, enc.embed_dim, num_classes=2).to(dev)
    ckpt = torch.load(hp, map_location=dev, weights_only=False)
    model.head.load_state_dict(ckpt["head_state_dict"])
    return model.eval()


def _id_train_index():
    from cxr_ood.evaluation.ood_detection import TBClassificationDataset
    ds = TBClassificationDataset(
        str(config.DATA_ROOT / "TB_Chest_Radiography_Database"), transform=PREPROCESS)
    if config.SPLIT_FILE.exists():
        return ds, json.loads(config.SPLIT_FILE.read_text())["train_indices"], True
    return ds, list(range(len(ds))), False  # all ID = training data for encoder


def get_mahal(name):
    key = ("mahal", name)
    if key in _cache:
        return _cache[key]
    dev = device()
    enc = get_encoder(name)
    ds, idx, _ = _id_train_index()
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, idx), batch_size=32, shuffle=False, num_workers=0)
    embs, labels = [], []
    enc.eval()
    with torch.no_grad():
        for imgs, lbs in loader:
            embs.append(enc(imgs.to(dev)).cpu().numpy())
            labels.append(np.asarray(lbs))
    det = MahalanobisDetector()
    det.fit(np.vstack(embs), np.concatenate(labels))
    _cache[key] = (det, len(labels))
    return _cache[key]


def get_id_scores(name, method):
    key = ("idscores", name, method)
    if key in _cache:
        return _cache[key]
    dev = device()
    ds, idx, _ = _id_train_index()
    idx = idx[:config.ID_SCORE_CAP]
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, idx), batch_size=32, shuffle=False, num_workers=0)
    vals = []
    with torch.no_grad():
        if method == "energy":
            model = get_probe(name)
            for imgs, _ in loader:
                vals.extend(compute_energy_score(model(imgs.to(dev))).cpu().numpy().tolist())
        else:
            enc = get_encoder(name)
            det, _ = get_mahal(name)
            enc.eval()
            for imgs, _ in loader:
                vals.extend(det.score(enc(imgs.to(dev)).cpu().numpy()).tolist())
    _cache[key] = np.array(vals)
    return _cache[key]


def get_threshold(name, method):
    dist = get_id_scores(name, method)
    return float(np.quantile(dist, config.TPR_TARGET))


@torch.no_grad()
def score_tensor(name, method, tensor):
    dev = device()
    if method == "energy":
        v = float(compute_energy_score(get_probe(name)(tensor.to(dev))).cpu().numpy()[0])
    else:
        det, _ = get_mahal(name)
        enc = get_encoder(name)
        enc.eval()
        v = float(det.score(enc(tensor.to(dev)).cpu().numpy())[0])
    dist = get_id_scores(name, method)
    return {"raw": v, "normalized": float((dist <= v).mean()),
            "percentile": float((dist <= v).mean() * 100),
            "threshold": get_threshold(name, method),
            "decision": "OOD" if v >= get_threshold(name, method) else "ID",
            "tpr_target": config.TPR_TARGET,
            "formula": "normalized = P(ID_validation_score <= raw); "
                       "threshold = ID 95th percentile (ID-only)"}


@torch.no_grad()
def attention_rollout(name, tensor):
    """Genuine attention rollout via forward hooks on Block.attn modules
    (Abnarain-style: A=0.5I+0.5*mean_heads, rollout=product). 14x14 map.
    Labelled rollout everywhere; never called Grad-CAM."""
    dev = device()
    enc = get_encoder(name)
    vit = enc.encoder
    saved = []

    def hook(_m, _i, o):
        # Block.attn returns (x, attn_weights); keep the weights [B, heads, N, N]
        w = o[1] if isinstance(o, tuple) else o
        saved.append(w.detach().cpu())

    if not hasattr(vit, "blocks"):
        raise RuntimeError("Encoder has no transformer blocks for rollout.")
    handles = [b.attn.register_forward_hook(hook) for b in vit.blocks]
    try:
        vit.eval()
        _ = vit(tensor.to(dev))
    finally:
        for h in handles:
            h.remove()
    if not saved:
        raise RuntimeError("No attention captured.")
    rollout = None
    for a in saved:  # [1, heads, 196, 196]
        a = a[0].mean(dim=0) + torch.eye(a.shape[-1])
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else rollout @ a
    return rollout.cpu().numpy()  # [196, 196]


def gradcam_map(name, tensor, target="energy"):
    """Grad-CAM for ViT: gradient of a differentiable scalar w.r.t. the last
    transformer block's patch tokens. Returns (14x14 float map in [0,1], label).

    Supported targets (verified differentiable through the real computation):
      - "energy": Energy OOD score = -logsumexp(probe logits), i.e. the actual
        OOD score served by /api/predict.
      - "logit": predicted-class logit from the saved linear probe head.
    "mahalanobis" is rejected: it is computed in numpy/sklearn on detached
    embeddings and has no gradients — never silently substituted.

    No-CLS ViT assumed (196 tokens -> 14x14); a 197-token output drops index 0.
    Cached encoders are left exactly as found (eval mode, frozen flags)."""
    if target == "mahalanobis":
        raise ValueError(
            "Mahalanobis distance is computed in numpy/sklearn on detached "
            "embeddings and has no gradients; Grad-CAM supports target="
            "'energy' or 'logit'.")
    if target not in ("energy", "logit"):
        raise ValueError(f"Unknown Grad-CAM target {target!r}; use 'energy' or 'logit'.")
    if name not in MODELS:
        raise FileNotFoundError(f"Unknown model {name!r}.")
    if tensor.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single-image tensor [1, 3, 224, 224].")
    dev = device()
    enc = get_encoder(name)  # FileNotFoundError if checkpoint missing
    vit = enc.encoder
    if not hasattr(vit, "blocks"):
        raise RuntimeError("Encoder has no transformer blocks for Grad-CAM.")
    probe = get_probe(name)  # FileNotFoundError if probe head missing
    head = probe.head

    feats = {}

    def fhook(_m, _i, o):
        feats["t"] = o

    params = list(vit.parameters()) + list(head.parameters())
    prev_flags = [p.requires_grad for p in params]
    handle = vit.blocks[-1].register_forward_hook(fhook)
    try:
        for p in params:
            p.requires_grad_(True)
        vit.eval()
        head.eval()
        with torch.enable_grad():
            tokens = vit(tensor.to(dev))  # [1, N, D]
            if tokens.shape[1] == 197:  # defensive: drop CLS if ever present
                tokens = tokens[:, 1:, :]
            if tokens.shape[1] != 196:
                raise RuntimeError(
                    f"Expected 196 patch tokens, got {tokens.shape[1]}.")
            pooled = tokens.mean(dim=1)
            logits = head(pooled)  # [1, 2]
            if target == "energy":
                from cxr_ood.evaluation.ood_detection import compute_energy_score
                scalar = compute_energy_score(logits)[0]
                label = "grad-cam (energy OOD score)"
            else:
                cls = int(logits.argmax(dim=1).item())
                scalar = logits[0, cls]
                label = f"grad-cam (predicted logit, class {cls})"
            feat = feats["t"]
            feat.retain_grad()
            vit.zero_grad()
            head.zero_grad()
            scalar.backward()
            grad = feat.grad  # [1, 196, D]
            if grad is None:
                raise RuntimeError("No gradients reached the target layer.")
            w = grad.mean(dim=1, keepdim=True)  # [1, 1, D] channel weights
            cam = torch.relu((feat.detach() * w).sum(dim=-1))  # [1, 196]
            cam = cam.reshape(14, 14)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
            out = cam.detach().cpu().numpy()
            if not np.isfinite(out).all():
                raise RuntimeError("Non-finite Grad-CAM values.")
            return out, label
    finally:
        handle.remove()
        for p, r in zip(params, prev_flags):
            p.requires_grad_(r)
        vit.zero_grad()
        head.zero_grad()


def read_metrics():
    out = {}
    for key, p in config.RESULT_FILES.items():
        if p.exists():
            try:
                out[key] = {"found": True, "data": json.loads(p.read_text()), "path": str(p)}
            except Exception as e:
                out[key] = {"found": False, "path": str(p), "error": str(e)}
        else:
            out[key] = {"found": False, "path": str(p)}
    return out


def dataset_info():
    def count(d):
        p = config.DATA_ROOT / d
        return sum(1 for f in p.iterdir() if f.is_file() and f.suffix in EXTS) if p.is_dir() else 0

    n = count("TB_Chest_Radiography_Database/Normal")
    t = count("TB_Chest_Radiography_Database/Tuberculosis")
    return [
        {"name": "TB Chest Radiography Database", "role": "ID train/test only",
         "normal": n, "tuberculosis": t, "total": n + t},
        {"name": "Montgomery County", "role": "OOD test only",
         "images": count("Montgomery TB CXR/images")},
        {"name": "Shenzhen Hospital", "role": "OOD test only",
         "images": count("Shenzhen TB CXR/images/images")},
    ]
