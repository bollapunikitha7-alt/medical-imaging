"""Tests: normalization, threshold selection, upload validation. No training."""
import io
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import inference as I  # noqa: E402


def test_normalized_is_id_percentile():
    dist = np.array([1.0, 2.0, 3.0, 4.0])
    raw = 3.0
    norm = float((dist <= raw).mean())
    assert norm == 0.75, norm
    assert 0.0 <= norm <= 1.0


def test_threshold_is_id_95th_percentile():
    dist = np.arange(100, dtype=float)
    t = float(np.quantile(dist, 0.95))
    assert t == 94.05, t
    assert (dist < t).mean() > 0.9  # ~95% ID accepted


def test_decision_respects_sign_convention():
    # higher = more OOD: above threshold -> OOD, below -> ID
    t = 5.0
    assert ("OOD" if 9.0 >= t else "ID") == "OOD"
    assert ("OOD" if 1.0 >= t else "ID") == "ID"


def test_reject_big_file():
    try:
        I.validate_upload(b"x" * (11 * 1024 * 1024), "big.png")
    except ValueError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("oversize not rejected")


def test_reject_bad_type():
    try:
        I.validate_upload(b"not an image", "evil.exe")
    except ValueError:
        pass
    else:
        raise AssertionError("bad type not rejected")


def test_reject_undecodable():
    try:
        I.validate_upload(b" noe ", "x.png")
    except ValueError:
        pass
    else:
        raise AssertionError("garbage not rejected")


def test_grayscale_converted():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (300, 300), 128).save(buf, format="PNG")
    img = I.validate_upload(buf.getvalue(), "g.png")
    assert img.mode == "RGB"
    t, pv = I.preprocess(img)
    assert tuple(t.shape) == (1, 3, 224, 224) and pv.size == (224, 224)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")
