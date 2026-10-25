"""Tests for Grad-CAM explainability. No training. Heavy tests reuse one cached encoder."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import config  # noqa: E402
import inference as I  # noqa: E402


def _tensor():
    img = Image.new("RGB", (300, 300), (120, 130, 140))
    t, _ = I.preprocess(img)
    return t


def test_reject_mahalanobis_target():
    try:
        I.gradcam_map("jepa", _tensor(), target="mahalanobis")
    except ValueError as e:
        assert "no gradients" in str(e), e
    else:
        raise AssertionError("mahalanobis target not rejected")


def test_reject_unknown_target():
    try:
        I.gradcam_map("jepa", _tensor(), target="logits")
    except ValueError as e:
        assert "Unknown Grad-CAM target" in str(e), e
    else:
        raise AssertionError("bogus target not rejected")


def test_reject_unknown_model():
    try:
        I.gradcam_map("vit_giant", _tensor(), target="energy")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("unknown model not rejected")


def test_reject_bad_tensor_shape():
    bad = torch.zeros(2, 3, 224, 224)
    try:
        I.gradcam_map("jepa", bad, target="energy")
    except ValueError as e:
        assert "single-image" in str(e), e
    else:
        raise AssertionError("bad batch shape not rejected")


def test_reject_missing_checkpoint():
    real = config.CHECKPOINTS["jepa"]
    config.CHECKPOINTS["jepa"] = BACKEND / "does_not_exist.pth"
    I._cache.pop("enc", None)
    try:
        try:
            I.gradcam_map("jepa", _tensor(), target="energy")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing checkpoint not rejected")
    finally:
        config.CHECKPOINTS["jepa"] = real
        I._cache.pop("enc", None)


def test_energy_map_shape_range():
    m, label = I.gradcam_map("jepa", _tensor(), target="energy")
    assert isinstance(m, np.ndarray) and m.shape == (14, 14), m.shape
    assert np.isfinite(m).all()
    assert m.min() >= 0.0 and m.max() <= 1.0 + 1e-6, (m.min(), m.max())
    assert "energy" in label, label


def test_logit_map_shape_range():
    m, label = I.gradcam_map("jepa", _tensor(), target="logit")
    assert m.shape == (14, 14) and np.isfinite(m).all()
    assert "logit" in label, label


def test_cache_state_restored():
    I.gradcam_map("jepa", _tensor(), target="energy")
    enc = I.get_encoder("jepa")
    assert all(p.requires_grad is False for p in enc.encoder.parameters())
    assert not enc.encoder.training


def test_rollout_untouched():
    r = I.attention_rollout("jepa", _tensor())
    assert isinstance(r, np.ndarray) and r.shape == (196, 196), r.shape
    assert np.isfinite(r).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")
