"""Functional automatic alerts, separated from UI rendering.

Alert generation reuses the exact outputs of the existing scoring/threshold
logic (score_tensor dicts); it never recomputes scores or thresholds with
different rules.

Kinds:
  - ood_threshold_crossed (warning): a real OOD decision from score_tensor.
  - inference_failed (error): unexpected exception during scoring.
  - checkpoint_unavailable (error): FileNotFoundError for model/head files.
  - explanation_failed (warning): attention/gradcam could not be generated.

Messages are research flags only — never disease diagnoses.
No database, no external notifications. In-memory store with dedup:
repeated identical requests update repeat_count/last_seen instead of
creating duplicates.
"""
import hashlib
import threading
from datetime import datetime, timezone

MAX_ALERTS = 200

_kinds = {
    "ood_threshold_crossed": "warning",
    "inference_failed": "error",
    "checkpoint_unavailable": "error",
    "explanation_failed": "warning",
}

_lock = threading.Lock()
_store = {}  # id -> alert dict, insertion ordered
_order = []  # ids in creation order for eviction


def _now():
    return datetime.now(timezone.utc).isoformat()


def sha_of(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def results_id():
    try:
        import config  # noqa: E402
        return config.CHECKPOINT_DIR.name
    except Exception:
        return "unknown"


def _check_kind(kind):
    if kind not in _kinds:
        raise ValueError(f"Unknown alert kind {kind!r}.")


def new_alert(kind, *, model=None, method=None, data=b"", score=None, error=None):
    """Create (or deduplicate) an alert. score is a score_tensor-style dict
    (raw/normalized/threshold/decision); error is an exception/string."""
    _check_kind(kind)
    image = sha_of(data) if data else "none"
    aid = f"{kind}:{image}:{model or '-'}:{method or '-'}"
    with _lock:
        if aid in _store:
            a = _store[aid]
            a["repeat_count"] += 1
            a["last_seen"] = _now()
            return a
        if kind == "ood_threshold_crossed":
            s = score or {}
            msg = (f"Image {image} flagged OUT-OF-DISTRIBUTION by "
                   f"{model}/{method}: raw {s.get('raw')} crossed ID threshold "
                   f"{s.get('threshold')} (ID 95th percentile). "
                   f"Research flag only - not a diagnosis.")
        elif kind == "checkpoint_unavailable":
            msg = (f"Required checkpoint unavailable for {model}"
                   + (f"/{method}" if method else "")
                   + f": {error}. Scoring skipped for this request.")
        elif kind == "explanation_failed":
            msg = (f"Explanation could not be generated for {model} "
                   f"(image {image}): {error}")
        else:  # inference_failed
            msg = (f"Inference failed for {model}/{method} "
                   f"(image {image}): {error}")
        a = {
            "id": aid,
            "kind": kind,
            "severity": _kinds[kind],
            "message": msg,
            "model": model,
            "method": method,
            "image": image,
            "score": ({k: (score or {}).get(k) for k in
                       ("raw", "normalized", "threshold", "decision")}
                      if kind == "ood_threshold_crossed" else None),
            "results": results_id(),
            "created_at": _now(),
            "last_seen": _now(),
            "repeat_count": 1,
            "acked": False,
        }
        _store[aid] = a
        _order.append(aid)
        while len(_order) > MAX_ALERTS:
            # Evict oldest acked first; only then oldest overall.
            victim = next((c for c in _order if _store.get(c, {}).get("acked")),
                          _order[0])
            _order.remove(victim)
            _store.pop(victim, None)
        return a


def from_result(model, method, data, out):
    """Alert on a real OOD decision, reusing out's own threshold/decision."""
    if isinstance(out, dict) and out.get("decision") == "OOD":
        return new_alert("ood_threshold_crossed", model=model, method=method,
                         data=data, score=out)
    return None


def from_exception(where, exc, *, model=None, method=None, data=b""):
    """Map a real failure to an alert kind. Returns the alert dict."""
    if isinstance(exc, FileNotFoundError):
        kind = ("explanation_failed" if where == "attention"
                else "checkpoint_unavailable")
    elif isinstance(exc, ValueError):
        kind = ("explanation_failed" if where == "attention"
                else "inference_failed")
    else:
        kind = "inference_failed"
    return new_alert(kind, model=model, method=method, data=data,
                     error=str(exc))


def list_alerts(include_acked=False):
    with _lock:
        out = [dict(a) for a in _store.values()
               if include_acked or not a["acked"]]
    return sorted(out, key=lambda a: a["last_seen"], reverse=True)


def ack_alert(aid):
    with _lock:
        if aid not in _store:
            return None
        _store[aid]["acked"] = True
        return dict(_store[aid])


def clear():
    with _lock:
        _store.clear()
        _order.clear()
