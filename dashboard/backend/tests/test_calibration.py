"""Tests for calibration/uncertainty protocol. No training, no checkpoints
except one missing-file probe test (fails fast, no weights loaded)."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
from cxr_ood.analysis.calibration_uncertainty import (  # noqa: E402
    resolve_splits, ensemble_stats, summarize, build_probe, reliability_data,
    fit_temperature)
from cxr_ood.analysis.calibration_analysis import (  # noqa: E402
    compute_ece, compute_nll)

SPLIT = REPO / "experiments/cxr_5pct/linear_probe/dataset_split.json"
TSPLIT = REPO / "experiments/cxr_5pct/threshold_eval/threshold_split.json"


def test_splits_disjoint_and_documented():
    s = resolve_splits(str(SPLIT), str(TSPLIT))
    assert len(s["fit"]) == 21 and len(s["eval"]) == 21, s
    assert set(s["fit"]).isdisjoint(s["eval"])
    full = json.load(open(SPLIT))
    assert set(s["fit"]).isdisjoint(full["train_indices"])
    assert set(s["eval"]).isdisjoint(full["train_indices"])
    assert set(s["fit"]) | set(s["eval"]) <= set(full["val_indices"])


def test_splits_reject_overlap():
    base = json.load(open(SPLIT))
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "s.json")
        p2 = os.path.join(d, "t.json")
        json.dump(base, open(p1, "w"))
        json.dump({"calib_indices": [0, 1], "heldout_indices": [1, 2]},
                  open(p2, "w"))
        try:
            resolve_splits(p1, p2)
        except ValueError as e:
            assert "overlap" in str(e), e
        else:
            raise AssertionError("overlapping fit/eval accepted")


def test_splits_missing_file():
    try:
        resolve_splits(str(SPLIT), str(REPO / "nope.json"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing split accepted")


def test_ece_perfect_is_zero():
    probs = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    labels = np.array([0] * 5 + [1] * 5)
    assert compute_ece(probs, labels) < 1e-9


def test_ece_confident_wrong_is_high():
    probs = np.array([[0.05, 0.95]] * 10)
    labels = np.zeros(10, dtype=int)
    assert compute_ece(probs, labels) > 0.5


def test_nll_sanity():
    import torch
    logits = torch.tensor([[2.0, -2.0]])
    labels = torch.tensor([0])
    assert 0 < compute_nll(logits, labels) < 0.5


def test_ensemble_identical_members_zero_disagreement():
    p = np.array([[[0.2, 0.8]]] * 3)  # [3,1,2]
    st = ensemble_stats(p)
    assert abs(st["disagreement"][0]) < 1e-9
    assert abs(st["entropy"][0] - (-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))) < 1e-9
    assert np.allclose(st["mean_probs"][0], [0.2, 0.8])


def test_ensemble_entropy_bounded():
    rng = np.random.default_rng(0)
    p = rng.random((3, 50, 2))
    p = p / p.sum(axis=2, keepdims=True)
    st = ensemble_stats(p)
    assert ((st["entropy"] >= 0) & (st["entropy"] <= np.log(2) + 1e-9)).all()
    assert ((st["disagreement"] >= 0) & (st["disagreement"] <= 0.5 + 1e-9)).all()


def test_ensemble_rejects_bad_shape():
    try:
        ensemble_stats(np.zeros((10, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("bad ensemble shape accepted")


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == 2.5 and s["p50"] == 2.5


def test_reliability_empty_last_bin():
    # All confidences in bin 0; last bin empty must not crash (upstream bug).
    probs = np.array([[0.9, 0.1]] * 4)
    labels = np.array([0] * 4)
    r = reliability_data(probs, labels, n_bins=5)
    assert len(r["bin_accuracies"]) == 5 and len(r["bin_counts"]) == 5
    assert r["bin_counts"][-1] == 4 and r["bin_counts"][0] == 0
    assert r["bin_accuracies"][-1] == 1.0
    assert abs(r["bin_confidences"][-1] - 0.9) < 1e-9  # mean, not mis-centered
    assert abs(r["bin_confidences"][0] - 0.1) < 1e-9  # true empty-bin center


def test_build_probe_missing_checkpoint():
    import torch
    try:
        build_probe("jepa", str(REPO / "nope.pth"), str(REPO / "nope2.pth"),
                    torch.device("cpu"))
    except (FileNotFoundError, OSError, RuntimeError):
        pass
    else:
        raise AssertionError("missing probe checkpoint accepted")


def test_temperature_guard_rejects_nonpositive():
    import torch

    class BadScaler:
        def __init__(self):
            self.temperature = torch.nn.Parameter(torch.ones(1))

        def fit(self, logits, labels):
            return -11.0

    t, deg = fit_temperature(BadScaler(), torch.zeros(2, 2),
                             torch.zeros(2, dtype=torch.long),
                             torch.device("cpu"), "test")
    assert t == 1.0 and deg is True


def test_temperature_guard_accepts_positive():
    import torch

    class OkScaler:
        def __init__(self):
            self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

        def fit(self, logits, labels):
            return 1.44

    t, deg = fit_temperature(OkScaler(), torch.zeros(2, 2),
                             torch.zeros(2, dtype=torch.long),
                             torch.device("cpu"), "test")
    assert t == 1.44 and deg is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")
