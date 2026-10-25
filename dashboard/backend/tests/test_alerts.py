"""Tests for functional automatic alerts. No training, no checkpoints."""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import alerts as A  # noqa: E402


def _ood_out(raw=30.0, thr=24.0):
    return {"raw": raw, "normalized": 0.97, "percentile": 97.0,
            "threshold": thr, "decision": "OOD"}


def _id_out():
    return {"raw": 3.0, "normalized": 0.2, "percentile": 20.0,
            "threshold": 24.0, "decision": "ID"}


def test_ood_decision_creates_warning():
    A.clear()
    a = A.from_result("jepa", "mahalanobis", b"img-1", _ood_out())
    assert a is not None and a["kind"] == "ood_threshold_crossed"
    assert a["severity"] == "warning"
    assert a["score"]["raw"] == 30.0 and a["score"]["threshold"] == 24.0
    assert a["created_at"] and a["results"]
    low = a["message"].lower()
    assert "not a diagnosis" in low
    for w in ("tuberculosis", "pneumonia", "disease", "lesion", "patient"):
        assert w not in low, w


def test_id_decision_creates_nothing():
    A.clear()
    assert A.from_result("jepa", "mahalanobis", b"img-1", _id_out()) is None
    assert A.list_alerts() == []


def test_duplicate_requests_do_not_duplicate():
    A.clear()
    A.from_result("jepa", "mahalanobis", b"same-bytes", _ood_out())
    A.from_result("jepa", "mahalanobis", b"same-bytes", _ood_out())
    items = A.list_alerts()
    assert len(items) == 1, items
    assert items[0]["repeat_count"] == 2


def test_different_images_alert_separately():
    A.clear()
    A.from_result("jepa", "mahalanobis", b"img-a", _ood_out())
    A.from_result("jepa", "mahalanobis", b"img-b", _ood_out())
    assert len(A.list_alerts()) == 2


def test_missing_checkpoint_maps_to_error():
    A.clear()
    a = A.from_exception("predict", FileNotFoundError("Checkpoint not found: x"),
                         model="mae", method="energy", data=b"img-1")
    assert a["kind"] == "checkpoint_unavailable" and a["severity"] == "error"


def test_attention_missing_file_maps_to_explanation_failed():
    A.clear()
    a = A.from_exception("attention", FileNotFoundError("gone"), model="jepa",
                         data=b"img-1")
    assert a["kind"] == "explanation_failed" and a["severity"] == "warning"


def test_unexpected_failure_maps_to_inference_failed():
    A.clear()
    a = A.from_exception("predict", RuntimeError("boom"), model="jepa",
                         method="energy", data=b"img-1")
    assert a["kind"] == "inference_failed" and a["severity"] == "error"


def test_ack_and_list():
    A.clear()
    a = A.from_result("jepa", "energy", b"img-1", _ood_out())
    assert len(A.list_alerts()) == 1
    got = A.ack_alert(a["id"])
    assert got and got["acked"] is True
    assert A.list_alerts() == []
    assert len(A.list_alerts(include_acked=True)) == 1
    assert A.ack_alert("nope") is None


def test_cap_evicts_oldest_acked_first():
    A.clear()
    old = A.MAX_ALERTS
    A.MAX_ALERTS = 5
    try:
        ids = []
        for i in range(7):
            a = A.from_result("jepa", "energy", f"img-{i}".encode(), _ood_out())
            ids.append(a["id"])
        A.ack_alert(ids[5])
        A.ack_alert(ids[6])
        A.from_result("jepa", "energy", b"img-new", _ood_out())
        items = A.list_alerts(include_acked=True)
        assert len(items) <= 6, len(items)
        remaining = {x["id"] for x in items}
        assert ids[5] not in remaining  # oldest acked evicted first
    finally:
        A.MAX_ALERTS = old
        A.clear()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")
