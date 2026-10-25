#!/usr/bin/env python3
"""
Calibration + uncertainty analysis with a correct fit/eval protocol.

Protocol (documented, ID-only fitting):
  - Temperature T is fitted on the 21-image threshold-CALIBRATION split.
  - All calibration metrics are reported on the disjoint 21-image
    threshold-HELDOUT split (never used for any fitting).
  - Both splits partition the 42-image linear-probe validation set and are
    disjoint from the 168 training images (asserted at runtime).
  - Montgomery/Shenzhen OOD images are REPORTING-ONLY (distributions of
    uncertainty measures). They never touch temperature fitting, thresholds,
    or calibration metrics.

Uncertainty method: heterogeneous ensemble of the three EXISTING probe heads
(JEPA/MAE/Supervised). No extra training. Reported: mean probability,
predictive entropy, inter-member disagreement (std of P(TB)).
NOT claimed calibrated; classification uncertainty is NOT an OOD score
(Energy/Mahalanobis are computed by the separate OOD pipeline).

Fixes protocol flaw in calibration_analysis.py, which fitted T and evaluated
on the SAME validation logits (optimistic ECE-after). That script is reused
for primitives only (imported, never modified).

Outputs go to a NEW directory (default experiments/cxr_5pct/
calibration_uncertainty/). Existing results are never overwritten.
"""

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
IJEPA_PATH = os.path.join(REPO_ROOT, "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)

from cxr_ood.analysis.baseline_comparison import (  # noqa: E402
    JEPAEncoder, MAEEncoder, SupervisedEncoder, LinearProbe)
from cxr_ood.analysis.calibration_analysis import (  # noqa: E402
    TemperatureScaling, compute_ece, compute_nll, compute_brier_score,
    plot_reliability_diagram, TBClassificationDataset)
from cxr_ood.evaluation.ood_detection import (  # noqa: E402
    MontgomeryDataset, ShenzhenDataset)

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================================
# Splits with enforced data separation
# ============================================================================

def resolve_splits(split_file, threshold_split_file):
    """Return {fit, eval} index lists with asserted separation.

    fit = threshold calib indices (temperature fitting only).
    eval = threshold heldout indices (reporting only).
    Raises ValueError on any overlap or out-of-range index.
    """
    with open(split_file) as f:
        split = json.load(f)
    with open(threshold_split_file) as f:
        tsplit = json.load(f)
    n_total = split["n_total"]
    train = set(split["train_indices"])
    val = set(split.get("val_indices", []))
    fit = list(tsplit["calib_indices"])
    evl = list(tsplit["heldout_indices"])
    for name, idx in (("fit", fit), ("eval", evl)):
        bad = [i for i in idx if not 0 <= i < n_total]
        if bad:
            raise ValueError(f"{name} indices out of range: {bad[:5]}")
    if set(fit) & set(evl):
        raise ValueError("fit/eval overlap: OOD-style leakage in protocol")
    if set(fit) & train or set(evl) & train:
        raise ValueError("fit/eval overlap train indices")
    if val and (not set(fit) <= val or not set(evl) <= val):
        raise ValueError("fit/eval must partition the validation set")
    return {"fit": fit, "eval": evl, "n_total": n_total,
            "n_train": len(train)}


# ============================================================================
# Numpy-only ensemble helpers (unit-testable without checkpoints)
# ============================================================================

def ensemble_stats(member_probs):
    """member_probs: [M, N, C] numpy. Returns dict with mean_probs [N, C],
    predictive entropy (nats) [N], disagreement = std of P(class=1) [N]."""
    p = np.asarray(member_probs, dtype=float)
    if p.ndim != 3 or p.shape[2] < 2:
        raise ValueError(f"expected [M, N, C>=2], got {p.shape}")
    mean = p.mean(axis=0)
    ent = -(mean * np.log(np.clip(mean, 1e-12, 1.0))).sum(axis=1)
    dis = p[:, :, 1].std(axis=0)
    return {"mean_probs": mean, "entropy": ent, "disagreement": dis}


def summarize(values):
    v = np.asarray(values, dtype=float)
    return {"n": int(len(v)), "mean": float(v.mean()),
            "std": float(v.std()),
            "p50": float(np.quantile(v, 0.50)),
            "p95": float(np.quantile(v, 0.95))}


def reliability_data(probs, labels, n_bins=5):
    """Local reliability-binning (same schema as calibration_analysis.
    get_reliability_data, which is NOT reused: it indexes
    bin_centers[len(bin_accuracies)] AFTER appending, crashing when the last
    bin is empty and mis-centering earlier empty bins. Upstream left intact.)"""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    bounds = np.linspace(0, 1, n_bins + 1)
    accs, confs, counts = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            accs.append(float(acc[m].mean()))
            confs.append(float(conf[m].mean()))
            counts.append(int(m.sum()))
        else:
            accs.append(0.0)
            confs.append(float((lo + hi) / 2))
            counts.append(0)
    return {"bin_centers": (bounds[:-1] + bounds[1:]) / 2,
            "bin_accuracies": np.array(accs),
            "bin_confidences": np.array(confs),
            "bin_counts": np.array(counts)}


# ============================================================================
# Temperature fitting with positivity guard
# ============================================================================

def fit_temperature(scaler, logits, labels, device, tag):
    """Fit T on the given logits; temperature is defined positive, so a
    non-positive LBFGS solution (possible on tiny n) is rejected: keep T=1.0
    (no scaling) and flag degenerate. Returns (T, degenerate)."""
    t = scaler.fit(logits.to(device), labels.to(device))
    if not np.isfinite(t) or t <= 0:
        logger.warning(f"{tag}: degenerate temperature {t}; "
                       f"keeping T=1.0 (no scaling)")
        scaler.temperature.data.fill_(1.0)
        return 1.0, True
    return float(t), False


# ============================================================================
# Logit collection
# ============================================================================

def collect_logits(model, loader, device):
    model.eval()
    outs, labels, paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, lbs, ph = batch
            else:
                imgs, lbs = batch
                ph = [""] * len(lbs)
            outs.append(model(imgs.to(device)).cpu())
            labels.append(torch.as_tensor(lbs))
            paths.extend(list(ph))
    return torch.cat(outs), torch.cat(labels), paths


def build_probe(kind, enc_ckpt, probe_ckpt, device):
    if kind == "jepa":
        enc = JEPAEncoder(enc_ckpt, model_name="vit_small", device=str(device))
    elif kind == "mae":
        enc = MAEEncoder(checkpoint_path=enc_ckpt, device=str(device))
    elif kind == "supervised":
        enc = SupervisedEncoder(checkpoint_path=enc_ckpt, device=str(device))
    else:
        raise ValueError(f"unknown model {kind}")
    ckpt = torch.load(probe_ckpt, map_location=device, weights_only=False)
    if "head_state_dict" not in ckpt:
        raise KeyError(f"{probe_ckpt} has no head_state_dict")
    model = LinearProbe(enc, enc.embed_dim, num_classes=2).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    return model.eval()


class _OODWrapper(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.paths = list(getattr(base, "image_paths", []))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        return img, -1, self.paths[i] if i < len(self.paths) else ""


def main():
    ap = argparse.ArgumentParser(description="Calibration + uncertainty (held-out protocol)")
    ap.add_argument("--data-root", required=True,
                    help="Parent CXR dir (also locates Montgomery/Shenzhen)")
    ap.add_argument("--tb-dir", required=True,
                    help="TB_Chest_Radiography_Database dir")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--threshold-split", required=True)
    ap.add_argument("--jepa-checkpoint", required=True)
    ap.add_argument("--jepa-probe", required=True)
    ap.add_argument("--mae-checkpoint", required=True)
    ap.add_argument("--mae-probe", required=True)
    ap.add_argument("--supervised-checkpoint", required=True)
    ap.add_argument("--supervised-probe", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-bins", type=int, default=5,
                    help="Few bins: eval n is small (default 21 images)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    for p in [args.split_file, args.threshold_split, args.tb_dir,
              args.jepa_checkpoint, args.jepa_probe, args.mae_checkpoint,
              args.mae_probe, args.supervised_checkpoint, args.supervised_probe]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input: {p}")

    splits = resolve_splits(args.split_file, args.threshold_split)
    logger.info(f"fit={len(splits['fit'])} eval={len(splits['eval'])} "
                f"train={splits['n_train']} (disjoint, asserted)")

    full = TBClassificationDataset(args.tb_dir, transform=TRANSFORM)
    fit_loader = DataLoader(Subset(full, splits["fit"]),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    eval_loader = DataLoader(Subset(full, splits["eval"]),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    probes = {
        "jepa": build_probe("jepa", args.jepa_checkpoint, args.jepa_probe, device),
        "mae": build_probe("mae", args.mae_checkpoint, args.mae_probe, device),
        "supervised": build_probe("supervised", args.supervised_checkpoint,
                                  args.supervised_probe, device),
    }

    # Per-model logits on fit + eval splits.
    per_model = {}
    for name, model in probes.items():
        fit_logits, fit_labels, _ = collect_logits(model, fit_loader, device)
        eval_logits, eval_labels, eval_paths = collect_logits(
            model, eval_loader, device)
        per_model[name] = {"fit_logits": fit_logits, "fit_labels": fit_labels,
                           "eval_logits": eval_logits,
                           "eval_labels": eval_labels,
                           "eval_paths": [full.image_paths[i]
                                          for i in splits["eval"]]}

    # Temperature fitted on FIT only; every metric reported on EVAL only.
    report = {"models": {}, "protocol": {
        "fit": "threshold calib split (temperature fitting only)",
        "eval": "threshold heldout split (reporting only)",
        "n_fit": len(splits["fit"]), "n_eval": len(splits["eval"]),
        "n_train_excluded": splits["n_train"], "n_bins": args.n_bins,
        "seed": args.seed,
        "ood_role": "reporting-only (entropy/disagreement distributions); "
                    "never fitted, no labels used"}}
    for name, d in per_model.items():
        scaler = TemperatureScaling().to(device)
        t, degenerate = fit_temperature(scaler, d["fit_logits"],
                                        d["fit_labels"], device, name)
        pb = F.softmax(d["eval_logits"], dim=1)
        with torch.no_grad():
            pa = F.softmax(scaler(d["eval_logits"].to(device)).cpu(), dim=1)
        m = {
            "temperature": float(t),
            "degenerate_fit": degenerate,
            "n_eval": len(d["eval_labels"]),
            "accuracy": float((pa.argmax(1).numpy()
                               == d["eval_labels"].numpy()).mean()),
            "before": {"ece": float(compute_ece(pb, d["eval_labels"],
                                                n_bins=args.n_bins)),
                       "nll": float(compute_nll(d["eval_logits"],
                                                d["eval_labels"])),
                       "brier": float(compute_brier_score(
                           pb, d["eval_labels"]))},
            "after": {"ece": float(compute_ece(pa, d["eval_labels"],
                                               n_bins=args.n_bins)),
                      "nll": float(compute_nll(
                          d["eval_logits"] / t, d["eval_labels"])),
                      "brier": float(compute_brier_score(
                          pa, d["eval_labels"]))},
        }
        report["models"][name] = m
        plot_reliability_diagram(
            reliability_data(pb, d["eval_labels"], n_bins=args.n_bins),
            reliability_data(pa, d["eval_labels"], n_bins=args.n_bins),
            os.path.join(args.output_dir, f"reliability_{name}.png"),
            m["before"]["ece"], m["after"]["ece"])
        logger.info(f"{name}: T={t:.3f} ECE {m['before']['ece']:.4f} -> "
                    f"{m['after']['ece']:.4f} (eval n={m['n_eval']})")

    # Heterogeneous ensemble on mean logits; T_ens fitted on FIT only.
    stack_fit = np.stack([per_model[m]["fit_logits"].numpy()
                          for m in ("jepa", "mae", "supervised")])  # [3,Nf,2]
    stack_eval = np.stack([per_model[m]["eval_logits"].numpy()
                           for m in ("jepa", "mae", "supervised")])
    ens_scaler = TemperatureScaling().to(device)
    t_ens, ens_degenerate = fit_temperature(
        ens_scaler, torch.from_numpy(stack_fit.mean(axis=0)),
        per_model["jepa"]["fit_labels"], device, "ensemble")
    ens_eval_logits = torch.from_numpy(stack_eval.mean(axis=0))
    with torch.no_grad():
        ens_eval_scaled = ens_scaler(ens_eval_logits.to(device)).cpu()
    ens_probs = F.softmax(ens_eval_scaled, dim=1).numpy()
    ens_labels = per_model["jepa"]["eval_labels"].numpy()
    stats = ensemble_stats([F.softmax(torch.from_numpy(stack_eval[m]), dim=1).numpy()
                            for m in range(3)])
    # Disagreement/entropy use UNCALIBRATED member probs (honest: ensemble
    # disagreement is a heuristic, not a calibrated uncertainty).
    report["ensemble"] = {
        "members": ["jepa", "mae", "supervised"],
        "temperature_fit_on_fit_split": float(t_ens),
        "degenerate_fit": ens_degenerate,
        "accuracy": float((ens_probs.argmax(1) == ens_labels).mean()),
        "ece": float(compute_ece(ens_probs, ens_labels, n_bins=args.n_bins)),
        "nll": float(compute_nll(ens_eval_scaled, torch.from_numpy(ens_labels))),
        "brier": float(compute_brier_score(ens_probs, ens_labels)),
        "mean_entropy_nats": float(stats["entropy"].mean()),
        "mean_disagreement": float(stats["disagreement"].mean()),
    }
    plot_reliability_diagram(
        reliability_data(F.softmax(ens_eval_logits, dim=1).numpy(),
                             ens_labels, n_bins=args.n_bins),
        reliability_data(ens_probs, ens_labels, n_bins=args.n_bins),
        os.path.join(args.output_dir, "reliability_ensemble.png"),
        float(compute_ece(F.softmax(ens_eval_logits, dim=1).numpy(),
                          ens_labels, n_bins=args.n_bins)),
        report["ensemble"]["ece"])

    # OOD reporting-only: uncertainty distributions, thresholds from ID eval.
    eval_dis = stats["disagreement"]
    eval_ent = stats["entropy"]
    dis_thr = float(np.quantile(eval_dis, 0.95))
    ent_thr = float(np.quantile(eval_ent, 0.95))
    ood_rows = []
    report["ood_reporting_only"] = {}
    for ds_name, ds_cls in (("montgomery", MontgomeryDataset),
                            ("shenzhen", ShenzhenDataset)):
        ds = ds_cls(args.data_root, transform=TRANSFORM)
        loader = DataLoader(_OODWrapper(ds), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        outs = []
        for name in ("jepa", "mae", "supervised"):
            lg, _, ph = collect_logits(probes[name], loader, device)
            outs.append(F.softmax(lg, dim=1).numpy())
        st = ensemble_stats(outs)
        files = [ds.image_paths[i] for i in range(len(ds))]
        for f, e, dd, pj, pm, ps in zip(files, st["entropy"],
                                        st["disagreement"], outs[0][:, 1],
                                        outs[1][:, 1], outs[2][:, 1]):
            ood_rows.append({"split": ds_name, "file": f, "label": "",
                             "p_jepa": float(pj), "p_mae": float(pm),
                             "p_sup": float(ps),
                             "p_mean": float((pj + pm + ps) / 3),
                             "entropy": float(e), "disagreement": float(dd)})
        report["ood_reporting_only"][ds_name] = {
            "n": len(ds),
            "entropy": summarize(st["entropy"]),
            "disagreement": summarize(st["disagreement"]),
            "frac_disagreement_above_id95": float((st["disagreement"]
                                                   >= dis_thr).mean()),
            "frac_entropy_above_id95": float((st["entropy"] >= ent_thr).mean()),
        }
    # ID eval rows for the same CSV (labels known here).
    for f, lb, pj, pm, ps, e, dd in zip(
            per_model["jepa"]["eval_paths"], ens_labels,
            F.softmax(torch.from_numpy(stack_eval[0]), dim=1).numpy()[:, 1],
            F.softmax(torch.from_numpy(stack_eval[1]), dim=1).numpy()[:, 1],
            F.softmax(torch.from_numpy(stack_eval[2]), dim=1).numpy()[:, 1],
            stats["entropy"], stats["disagreement"]):
        ood_rows.append({"split": "id_eval", "file": f, "label": int(lb),
                         "p_jepa": float(pj), "p_mae": float(pm),
                         "p_sup": float(ps),
                         "p_mean": float((pj + pm + ps) / 3),
                         "entropy": float(e), "disagreement": float(dd)})
    with open(os.path.join(args.output_dir, "per_image_uncertainty.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "file", "label", "p_jepa",
                                          "p_mae", "p_sup", "p_mean",
                                          "entropy", "disagreement"])
        w.writeheader()
        w.writerows(ood_rows)

    report["assumptions_limits"] = {
        "temperature": "single scalar; assumes miscalibration is global "
                       "over-confidence/under-confidence",
        "eval_n": "heldout n=21: ECE with 5 bins is noisy; treat ECE-after "
                  "as indicative, not definitive",
        "ensemble": "heterogeneous members (different encoders), not i.i.d.; "
                    "disagreement is a heuristic, NOT calibrated epistemic "
                    "uncertainty; no claim that low disagreement means correct",
        "ood_vs_uncertainty": "classification entropy/disagreement and OOD "
                              "scores (Energy/Mahalanobis) measure different "
                              "things; reported side by side, never fused",
        "compute": "3 probe forwards/image (~0.3-0.5s CPU); full run "
                   "~5 min CPU for 884 images",
    }
    with open(os.path.join(args.output_dir,
                           "calibration_uncertainty.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved to {args.output_dir}")
    for name, m in report["models"].items():
        logger.info(f"{name}: ECE {m['before']['ece']:.4f} -> "
                    f"{m['after']['ece']:.4f} | NLL {m['before']['nll']:.4f} "
                    f"-> {m['after']['nll']:.4f}")
    logger.info(f"ensemble: acc {report['ensemble']['accuracy']:.4f} "
                f"ECE {report['ensemble']['ece']:.4f} "
                f"mean_dis {report['ensemble']['mean_disagreement']:.4f}")
    return report


if __name__ == "__main__":
    main()

