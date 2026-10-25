"""Paths from env, never machine-specific. Defaults are repo-relative."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _p(env, default):
    v = os.environ.get(env, default)
    p = Path(v)
    return p if p.is_absolute() else REPO_ROOT / p


DATA_ROOT = _p("DATA_ROOT", "Datasets/CXR")
CHECKPOINT_DIR = _p("CHECKPOINT_DIR", "experiments/cxr_5pct")
RESULTS_DIR = _p("RESULTS_DIR", "experiments/cxr_5pct")
CACHE_DIR = REPO_ROOT / "dashboard" / "backend" / ".cache"

CHECKPOINTS = {
    "jepa": CHECKPOINT_DIR / "checkpoint_latest.pth",
    "jepa_probe": CHECKPOINT_DIR / "linear_probe" / "best_linear_probe.pth",
    "mae": CHECKPOINT_DIR / "baseline_comparison" / "mae_pretrain" / "encoder_final.pth",
    "supervised": CHECKPOINT_DIR / "baseline_comparison" / "supervised_pretrain" / "best_model.pth",
    "mae_probe": CHECKPOINT_DIR / "baseline_comparison" / "mae" / "best_linear_probe.pth",
    "supervised_probe": CHECKPOINT_DIR / "baseline_comparison" / "supervised" / "best_linear_probe.pth",
}
RESULT_FILES = {
    "baseline": RESULTS_DIR / "baseline_comparison" / "baseline_comparison.json",
    "ood_detection": RESULTS_DIR / "ood_detection" / "ood_detection_results.json",
    "robustness": RESULTS_DIR / "robustness" / "robustness_ablation_results.json",
    "threshold": RESULTS_DIR / "threshold_eval" / "threshold_metrics.json",
    "calibration": RESULTS_DIR / "calibration_uncertainty" / "calibration_uncertainty.json",
}
SPLIT_FILE = RESULTS_DIR / "linear_probe" / "dataset_split.json"

MAX_UPLOAD_MB = 10
ID_SCORE_CAP = 200
TPR_TARGET = 0.95
