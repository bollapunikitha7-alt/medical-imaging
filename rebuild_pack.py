"""Rebuild kaggle pack with forward-slash paths (Kaggle-safe)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kaggle_pack.zip"
INCLUDE = ["run_pipeline.py", "subset_select.py", "requirements.txt",
           "README.md", "app", "cxr_ood", "IJEPA_Meta",
           "Datasets/CXR_subset_5pct/subset_manifest.csv",
           "experiments/cxr_5pct/checkpoint_ep50.pth",
           "experiments/cxr_5pct/config.yaml",
           "experiments/cxr_5pct/baseline_comparison/mae_pretrain/encoder_final.pth",
           "experiments/cxr_5pct/baseline_comparison/mae_pretrain/best_model.pth",
           "experiments/cxr_5pct/baseline_comparison/mae_pretrain/training_history.json"]
SKIP = {"__pycache__", ".cache", "checkpoint_latest.pth"}


def members():
    for rel in INCLUDE:
        p = ROOT / rel
        if p.is_file():
            yield p
        else:
            for f in sorted(p.rglob("*")):
                if f.is_file() and not any(s in f.parts for s in SKIP) \
                        and f.name not in SKIP:
                    yield f


tmp = OUT.with_suffix(".new.zip")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in members():
        z.write(f, f.relative_to(ROOT).as_posix())
tmp.replace(OUT)
print("rebuilt:", OUT, round(OUT.stat().st_size / 1e6, 1), "MB")
