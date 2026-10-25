#!/usr/bin/env python3
"""
OOD threshold evaluation (NEW stage, does not modify existing scripts).

Turns existing OOD scores into ID/OOD decisions with a threshold chosen
ONLY on in-distribution validation data.

Sign convention (from ood_detection.py / baseline_comparison.py):
  higher score = more OOD, for both Energy and Mahalanobis.
  Decision: predict OOD if score >= threshold, else ID.
  Threshold = quantile(calibration ID scores, tpr_target), default 0.95
  so ~95% of ID calibration images are accepted as ID.

Positive class for precision/recall/F1: OOD (1), ID = 0.
OOD data (Montgomery/Shenzhen) is NEVER used for fitting or thresholding.

Reuses scoring code by import (no copies):
  from cxr_ood.evaluation.ood_detection import (
    compute_energy_score, MahalanobisDetector, compute_ood_auroc,
    compute_fpr_at_tpr, extract_features,
    TBClassificationDataset, MontgomeryDataset, ShenzhenDataset)
  from cxr_ood.analysis.baseline_comparison import (
    MAEEncoder, SupervisedEncoder, JEPAEncoder,
    LinearProbe as BCProbe)
Probe training for MAE/Supervised mirrors baseline_comparison.py train
settings (AdamW lr=0.001, CosineAnnealing, 50 epochs) but trains in-memory
on TRAIN split only, writing no extra checkpoints.
"""

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Make repo-root imports work when run as script from project root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
IJEPA_PATH = REPO_ROOT / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))

from cxr_ood.evaluation.ood_detection import (  # noqa: E402
    compute_energy_score,
    MahalanobisDetector,
    compute_ood_auroc,
    compute_fpr_at_tpr,
    extract_features,
    TBClassificationDataset,
    MontgomeryDataset,
    ShenzhenDataset,
)
from cxr_ood.analysis.baseline_comparison import (  # noqa: E402
    MAEEncoder,
    SupervisedEncoder,
    JEPAEncoder,
    LinearProbe as BCProbe,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

POSITIVE_CLASS = "OOD (1); ID = 0 (negative)"
PROBE_EPOCHS = 50  # mirrors baseline_comparison.py default --epochs
PROBE_LR = 0.001


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def train_probe_head(model, train_loader, device, epochs=PROBE_EPOCHS, lr=PROBE_LR):
    """Train only model.head on TRAIN data (mirrors baseline settings)."""
    model.train()
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.head.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def threshold_metrics(heldout_scores, ood_scores, thresh):
    """Threshold metrics with positive = OOD."""
    heldout_scores = np.asarray(heldout_scores)
    ood_scores = np.asarray(ood_scores)
    id_accept = float((heldout_scores < thresh).mean()) if len(heldout_scores) else 0.0
    ood_rate = float((ood_scores >= thresh).mean()) if len(ood_scores) else 0.0
    tp = int((ood_scores >= thresh).sum())
    fn = len(ood_scores) - tp
    tn = int((heldout_scores < thresh).sum())
    fp = len(heldout_scores) - tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "id_acceptance_heldout": id_accept,
        "ood_detection_rate": ood_rate,
        "recall_ood": recall,
        "precision_ood": precision,
        "f1_ood": f1,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def build_models(args, device):
    """Build (encoder+probe) per model with checkpoint; skip missing."""
    models = {}
    # JEPA: encoder from jepa ckpt + head from probe ckpt (both required for JEPA).
    if args.jepa_checkpoint and os.path.exists(args.jepa_checkpoint):
        enc = JEPAEncoder(args.jepa_checkpoint, model_name='vit_small', device=str(device))
        probe_model = BCProbe(enc, enc.embed_dim, num_classes=2).to(device)
        pckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=False)
        probe_model.head.load_state_dict(pckpt['head_state_dict'])
        probe_model.eval()
        models['JEPA'] = probe_model
        logger.info("JEPA ready (encoder %s + probe %s)", args.jepa_checkpoint, args.probe_checkpoint)
    else:
        logger.warning("JEPA skipped: checkpoint missing (%s)", args.jepa_checkpoint)
    # MAE / Supervised: encoder from ckpt, head trained later on TRAIN only.
    if args.mae_checkpoint:
        if os.path.exists(args.mae_checkpoint):
            enc = MAEEncoder(checkpoint_path=args.mae_checkpoint, device=str(device))
            models['MAE'] = BCProbe(enc, enc.embed_dim, num_classes=2).to(device)
            logger.info("MAE encoder ready (%s), probe head will be trained on TRAIN",
                        args.mae_checkpoint)
        else:
            logger.warning("MAE skipped: checkpoint missing (%s)", args.mae_checkpoint)
    if args.supervised_checkpoint:
        if os.path.exists(args.supervised_checkpoint):
            enc = SupervisedEncoder(checkpoint_path=args.supervised_checkpoint, device=str(device))
            models['Supervised'] = BCProbe(enc, enc.embed_dim, num_classes=2).to(device)
            logger.info("Supervised encoder ready (%s), probe head will be trained on TRAIN",
                        args.supervised_checkpoint)
        else:
            logger.warning("Supervised skipped: checkpoint missing (%s)", args.supervised_checkpoint)
    return models


def main():
    ap = argparse.ArgumentParser(description='OOD threshold evaluation (ID-calibrated)')
    ap.add_argument('--jepa-checkpoint', type=str, required=True)
    ap.add_argument('--mae-checkpoint', type=str, default=None)
    ap.add_argument('--supervised-checkpoint', type=str, default=None)
    ap.add_argument('--probe-checkpoint', type=str, required=True,
                    help='JEPA linear-probe head (head_state_dict)')
    ap.add_argument('--split-file', type=str, required=True,
                    help='dataset_split.json from linear_probe.py')
    ap.add_argument('--data-root', type=str, required=True,
                    help='Parent Datasets/CXR')
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tpr-target', type=float, default=0.95)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tb_path = os.path.join(args.data_root, 'TB_Chest_Radiography_Database')
    full_id = TBClassificationDataset(tb_path, transform=transform)
    montgomery = MontgomeryDataset(args.data_root, transform=transform)
    shenzhen = ShenzhenDataset(args.data_root, transform=transform)

    with open(args.split_file) as f:
        split = json.load(f)
    train_idx = list(split['train_indices'])
    val_idx = list(split['val_indices'])

    # Disjoint calib / held-out split of VAL only, fixed seed. Never touches OOD.
    rng = np.random.RandomState(args.seed)
    perm = np.array(val_idx)
    rng.shuffle(perm)
    half = len(perm) // 2
    calib_idx, heldout_idx = perm[:half].tolist(), perm[half:].tolist()
    with open(out / 'threshold_split.json', 'w') as f:
        json.dump({"parent_split": args.split_file, "seed": args.seed,
                   "tpr_target": args.tpr_target, "n_train": len(train_idx),
                   "calib_indices": calib_idx, "heldout_indices": heldout_idx}, f)

    def mk_loader(ds, idx, shuffle):
        return DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=4, pin_memory=True)

    train_loader = mk_loader(full_id, train_idx, True)
    calib_loader = mk_loader(full_id, calib_idx, False)
    heldout_loader = mk_loader(full_id, heldout_idx, False)
    mont_loader = DataLoader(montgomery, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    shen_loader = DataLoader(shenzhen, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    counts = {"train": len(train_idx), "calib": len(calib_idx),
              "heldout": len(heldout_idx), "Montgomery": len(montgomery),
              "Shenzhen": len(shenzhen)}
    print("Dataset counts:", counts)
    for k, v in counts.items():
        if v == 0:
            raise ValueError(f"Dataset '{k}' has 0 images - aborting (check paths/split).")

    models = build_models(args, device)
    if not models:
        raise FileNotFoundError("No model checkpoints found; nothing to evaluate.")
    # Train MAE/Supervised heads on TRAIN only.
    for name in ('MAE', 'Supervised'):
        if name in models:
            logger.info("Training %s probe head on TRAIN only (%d epochs)...", name, PROBE_EPOCHS)
            train_probe_head(models[name], train_loader, device)

    ood_sets = {"Montgomery": mont_loader, "Shenzhen": shen_loader}
    all_rows = []
    results = {}

    for name, model in models.items():
        model.eval()
        feats = {k: extract_features(model, dl, device) for k, dl in
                 [("train", train_loader), ("calib", calib_loader),
                  ("heldout", heldout_loader), ("Montgomery", mont_loader),
                  ("Shenzhen", shen_loader)]}
        det = MahalanobisDetector()
        det.fit(feats["train"]['embeddings'], feats["train"]['labels'])
        scores = {}
        for k, ft in feats.items():
            scores[k] = {"energy": compute_energy_score(ft['logits']).numpy(),
                         "mahal": det.score(ft['embeddings'])}
        # Thresholds from CALIBRATION ID only.
        thresh = {m: float(np.quantile(scores["calib"][m], args.tpr_target))
                  for m in ("energy", "mahal")}
        results[name] = {"threshold": thresh, "tpr_target": args.tpr_target, "ood": {}}
        for ood_name in ood_sets:
            results[name]["ood"][ood_name] = {}
            for m in ("energy", "mahal"):
                tm = threshold_metrics(scores["heldout"][m], scores[ood_name][m], thresh[m])
                tm["auroc_ref"] = float(compute_ood_auroc(scores["train"][m], scores[ood_name][m]))
                tm["fpr95_ref"] = float(compute_fpr_at_tpr(scores["train"][m], scores[ood_name][m], 0.95))
                tm["threshold"] = thresh[m]
                tm["n_train"] = counts["train"]
                tm["n_heldout"] = counts["heldout"]
                tm["n_ood"] = counts[ood_name]
                results[name]["ood"][ood_name][m] = tm
        # CSV rows (per-image scores + decisions).
        for split_name in ("train", "calib", "heldout", "Montgomery", "Shenzhen"):
            n = len(scores[split_name]["energy"])
            true = "ID" if split_name in ("train", "calib", "heldout") else "OOD"
            for i in range(n):
                all_rows.append({
                    "model": name, "split": split_name, "true": true, "index": i,
                    "energy": float(scores[split_name]["energy"][i]),
                    "mahal": float(scores[split_name]["mahal"][i]),
                    "energy_pred": "OOD" if scores[split_name]["energy"][i] >= thresh["energy"] else "ID",
                    "mahal_pred": "OOD" if scores[split_name]["mahal"][i] >= thresh["mahal"] else "ID",
                })

    with open(out / 'per_image_scores.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "true", "index",
                                          "energy", "mahal", "energy_pred", "mahal_pred"])
        w.writeheader()
        w.writerows(all_rows)

    # Plot: rows=models, cols=methods, with threshold line.
    names = list(results.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        # Recompute lightweight score cache for plotting from CSV rows.
        for c, m in enumerate(("energy", "mahal")):
            ax = axes[r][c]
            # Plot heldout ID vs OOD histograms using stored rows.
            h = np.array([row[m] for row in all_rows
                          if row["model"] == name and row["split"] == "heldout"])
            for ood_name in ("Montgomery", "Shenzhen"):
                o = np.array([row[m] for row in all_rows
                              if row["model"] == name and row["split"] == ood_name])
                ax.hist(o, bins=50, alpha=0.5, label=ood_name, density=True)
            ax.hist(h, bins=50, alpha=0.5, label='ID heldout', density=True)
            ax.axvline(results[name]["threshold"][m], color='k', linestyle='--',
                       label=f'thresh={results[name]["threshold"][m]:.3f}')
            ax.set_title(f"{name} {m} (higher=more OOD)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    run_info = {"args": vars(args), "seed": args.seed, "tpr_target": args.tpr_target,
                "positive_class": POSITIVE_CLASS,
                "sign_convention": "higher score = more OOD (Energy, Mahalanobis)",
                "timestamp": datetime.now().isoformat(), "git_hash": git_hash(),
                "python": sys.version, "torch": torch.__version__,
                "counts": counts, "probe_train": f"MAE/Supervised heads trained on TRAIN only "
                f"({PROBE_EPOCHS} epochs, lr={PROBE_LR}); JEPA head from probe-checkpoint"}
    with open(out / 'threshold_metrics.json', 'w') as f:
        json.dump({"run_info": run_info, "results": results}, f, indent=2)

    print("\n" + "=" * 100)
    print("THRESHOLD OOD RESULTS (positive = OOD; threshold from ID calib @ "
          f"{args.tpr_target:.2f})")
    print("=" * 100)
    for name in names:
        for ood_name in ("Montgomery", "Shenzhen"):
            for m in ("energy", "mahal"):
                t = results[name]["ood"][ood_name][m]
                print(f"{name:<10} {ood_name:<10} {m:<7} thresh={t['threshold']:.4f} "
                      f"IDacc={t['id_acceptance_heldout']:.3f} OODrate={t['ood_detection_rate']:.3f} "
                      f"P={t['precision_ood']:.3f} R={t['recall_ood']:.3f} F1={t['f1_ood']:.3f} "
                      f"AUROCref={t['auroc_ref']:.4f} FPR95ref={t['fpr95_ref']:.4f}")
    print("=" * 100)
    print(f"Saved: {out/'per_image_scores.csv'}, {out/'threshold_metrics.json'}, "
          f"{out/'score_distributions.png'}, {out/'threshold_split.json'}")


if __name__ == '__main__':
    main()
