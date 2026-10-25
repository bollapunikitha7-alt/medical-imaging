#!/usr/bin/env python3
"""
Thin orchestrator for the existing manual cxr_ood staged workflow.

Wraps (never edits) the existing stage scripts in canonical order:
  jepa_pretrain -> mae_pretrain -> supervised_pretrain -> linear_probe
  -> ood_detection -> baseline_comparison -> robustness_ablation

Each stage entry defines: name, command builder, required inputs,
expected outputs - in ONE list (STAGES) so adding a stage = one edit.

OOD data (Montgomery/Shenzhen/any --ood-datasets entry) is only ever
used as a pre-flight existence check before evaluation stages and is
only passed to evaluation-stage scripts via --data-root (parent dir).
It is NEVER passed to any training-stage script.

Assumptions (from approved STEP-1 table):
  A1. JEPA output dir is controlled by the YAML config's logging.folder
      (default experiments/cxr_jepa_pilot/, matching default --output-dir).
      The orchestrator does NOT rewrite the config; it expects
      <output-dir>/checkpoint_latest.pth (+config.yaml) to appear.
      If your config points elsewhere, input checks fail with a clear msg.
  A2. JEPA ckpt downstream = <output-dir>/checkpoint_latest.pth.
  A3. MAE ckpt downstream = baseline_comparison/mae_pretrain/encoder_final.pth;
      supervised ckpt = baseline_comparison/supervised_pretrain/best_model.pth
      (per README examples).
  A4. Probe ckpt for ood_detection = linear_probe/best_linear_probe.pth
      (what linear_probe.py actually writes; README's best_probe.pth does not exist).
  A5. --seed is passed only to scripts that accept it (all except jepa_pretrain).
  A6. --ood-datasets is check-only (scripts hardcode folder names); never a CLI passthrough.
  A7. Hyperparameters untouched: only path/seed/batch-size wiring, all else defaults.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

STAGE_ORDER = [
    "jepa_pretrain",
    "mae_pretrain",
    "supervised_pretrain",
    "linear_probe",
    "ood_detection",
    "baseline_comparison",
    "robustness_ablation",
]

JEPA_CONFIG = REPO_ROOT / "cxr_ood" / "configs" / "cxr_vit_small.yaml"
TB_DIRNAME = "TB_Chest_Radiography_Database"


def p(pth):
    return Path(pth)


def build_stages(args):
    """Single place defining every stage. Returns list of dicts."""
    data_root = p(args.data_root)
    out = p(args.output_dir)
    tb = data_root / TB_DIRNAME
    jepa_ckpt = out / "checkpoint_latest.pth"
    mae_enc = out / "baseline_comparison" / "mae_pretrain" / "encoder_final.pth"
    sup_best = out / "baseline_comparison" / "supervised_pretrain" / "best_model.pth"
    probe_dir = out / "linear_probe"
    probe_ckpt = probe_dir / "best_linear_probe.pth"
    split_file = probe_dir / "dataset_split.json"
    seed = ["--seed", str(args.seed)]
    bs = ["--batch-size", str(args.batch_size)]

    return [
        {
            "name": "jepa_pretrain",
            "cmd": [sys.executable, "cxr_ood/pretraining/jepa_pretrain.py",
                    "--config", str(JEPA_CONFIG)],
            "inputs": [str(JEPA_CONFIG)],
            "inputs_desc": "YAML config (controls data.root_path/logging.folder)",
            "outputs": [str(jepa_ckpt), str(out / "config.yaml")],
            "produced_by": "jepa_pretrain",
        },
        {
            "name": "mae_pretrain",
            "cmd": [sys.executable, "cxr_ood/pretraining/mae_pretrain.py",
                    "--data-root", str(tb),
                    "--output-dir", str(out / "baseline_comparison" / "mae_pretrain")] + seed,
            "inputs": [],
            "inputs_desc": "ID images under " + str(tb),
            "outputs": [str(mae_enc),
                        str(out / "baseline_comparison" / "mae_pretrain" / "best_model.pth")],
            "produced_by": "mae_pretrain",
        },
        {
            "name": "supervised_pretrain",
            "cmd": [sys.executable, "cxr_ood/pretraining/supervised_pretrain.py",
                    "--data-root", str(tb),
                    "--output-dir", str(out / "baseline_comparison" / "supervised_pretrain")] + seed,
            "inputs": [],
            "inputs_desc": "ID images under " + str(tb),
            "outputs": [str(sup_best),
                        str(out / "baseline_comparison" / "supervised_pretrain" / "encoder_final.pth")],
            "produced_by": "supervised_pretrain",
        },
        {
            "name": "linear_probe",
            "cmd": [sys.executable, "cxr_ood/evaluation/linear_probe.py",
                    "--checkpoint", str(jepa_ckpt),
                    "--data-path", str(tb),
                    "--output-dir", str(probe_dir)] + bs + seed,
            "inputs": [str(jepa_ckpt)],
            "inputs_desc": "JEPA checkpoint from jepa_pretrain",
            "outputs": [str(split_file), str(probe_ckpt)],
            "produced_by": "linear_probe",
        },
        {
            "name": "ood_detection",
            "cmd": [sys.executable, "cxr_ood/evaluation/ood_detection.py",
                    "--jepa-checkpoint", str(jepa_ckpt),
                    "--probe-checkpoint", str(probe_ckpt),
                    "--data-root", str(data_root),
                    "--split-file", str(split_file),
                    "--output-dir", str(out / "ood_detection")] + bs + seed,
            "inputs": [str(jepa_ckpt), str(probe_ckpt), str(split_file)],
            "inputs_desc": "JEPA ckpt + probe ckpt + split from linear_probe",
            "outputs": [str(out / "ood_detection" / "ood_detection_results.json")],
            "produced_by": "ood_detection",
            "ood_stage": True,
        },
        {
            "name": "baseline_comparison",
            "cmd": [sys.executable, "cxr_ood/analysis/baseline_comparison.py",
                    "--jepa-checkpoint", str(jepa_ckpt),
                    "--mae-checkpoint", str(mae_enc),
                    "--supervised-checkpoint", str(sup_best),
                    "--data-root", str(data_root),
                    "--output-dir", str(out / "baseline_comparison")] + bs + seed,
            "inputs": [str(jepa_ckpt), str(mae_enc), str(sup_best)],
            "inputs_desc": "JEPA+MAE+supervised checkpoints",
            "outputs": [str(out / "baseline_comparison" / "baseline_comparison.json")],
            "produced_by": "baseline_comparison",
            "ood_stage": True,
        },
        {
            "name": "robustness_ablation",
            "cmd": [sys.executable, "cxr_ood/analysis/robustness_ablation.py",
                    "--jepa-checkpoint", str(jepa_ckpt),
                    "--mae-checkpoint", str(mae_enc),
                    "--supervised-checkpoint", str(sup_best),
                    "--data-root", str(data_root),
                    "--output-dir", str(out / "robustness")] + bs + seed,
            "inputs": [str(jepa_ckpt), str(mae_enc), str(sup_best)],
            "inputs_desc": "JEPA+MAE+supervised checkpoints",
            "outputs": [str(out / "robustness" / "robustness_ablation_results.json")],
            "produced_by": "robustness_ablation",
            "ood_stage": True,
        },
    ]


def check_ood_folders(args, stage):
    """Pre-flight check only: OOD folders must exist for evaluation stages."""
    missing = []
    for name in [s.strip() for s in args.ood_datasets.split(",") if s.strip()]:
        if not (p(args.data_root) / name).exists():
            missing.append(str(p(args.data_root) / name))
    if missing:
        print(f"[SKIP-CHECK] stage '{stage['name']}' needs OOD folders "
              f"(from --ood-datasets, evaluation-only, never used for training):")
        for m in missing:
            print(f"  MISSING: {m}")
        print(f"  Produced by: external dataset download, not by any stage.")
        return False
    return True


def run_stage(stage, log_path, dry_run):
    """Run with subprocess (check=False), stream live + tee to log file."""
    cmd = stage["cmd"]
    print(f"\n=== [{stage['name']}] {' '.join(cmd)}")
    print(f"    inputs:  {stage['inputs'] or [stage['inputs_desc']]}")
    print(f"    outputs: {stage['outputs']}")
    if dry_run:
        return 0, 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    with open(log_path, "w", encoding="utf-8") as lf:
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
    proc.wait()
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Thin cxr_ood pipeline orchestrator (wrapper only)")
    ap.add_argument("--data-root", default="Datasets/CXR")
    ap.add_argument("--output-dir", default="experiments/cxr_jepa_pilot")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset, e.g. ood_detection,baseline_comparison")
    ap.add_argument("--from-stage", default=None)
    ap.add_argument("--to-stage", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ood-datasets", default="Montgomery TB CXR,Shenzhen TB CXR",
                    help="check-only for OOD stages; never passed to training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stages = build_stages(args)
    names = [s["name"] for s in stages]
    if args.stages:
        want = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [w for w in want if w not in names]
        if unknown:
            print(f"Unknown --stages: {unknown}. Valid: {names}")
            sys.exit(2)
        stages = [s for s in stages if s["name"] in want]
    if args.from_stage:
        stages = stages[names.index(args.from_stage):] if args.from_stage in names else stages
    if args.to_stage:
        idx = names.index(args.to_stage) if args.to_stage in names else len(names) - 1
        order = [s["name"] for s in stages]
        stages = [s for s in stages if names.index(s["name"]) <= idx]

    print(f"data-root={args.data_root} output-dir={args.output_dir} "
          f"dry-run={args.dry_run} force={args.force} seed={args.seed}")
    print(f"stages: {[s['name'] for s in stages]}")

    summary = []
    for st in stages:
        # 3. input check BEFORE run.
        missing = [f for f in st["inputs"] if not Path(f).exists()
                   and not (REPO_ROOT / f).exists()]
        # Resolve relative-to-root inputs too.
        missing = [f for f in missing if not (REPO_ROOT / f).exists() and not Path(f).exists()]
        if st.get("ood_stage") and not check_ood_folders(args, st):
            print(f"[FAIL] stage '{st['name']}': OOD folder missing (see above). Stopping.")
            summary.append((st["name"], "failed", 0.0))
            break
        if missing:
            print(f"[FAIL] stage '{st['name']}': missing required input(s):")
            for m in missing:
                print(f"  MISSING: {m}  (should have been produced by: {st['produced_by'] if m in st['inputs'] and st['name'] in ('linear_probe',) else 'earlier stage / ' + st['inputs_desc']})")
            print("Fix: run the producing stage first or check --output-dir/--data-root.")
            summary.append((st["name"], "failed", 0.0))
            break
        # 5. resume: skip if outputs exist.
        outs = [Path(f) for f in st["outputs"]]
        outs_rel = [(REPO_ROOT / f) if not Path(f).is_absolute() else Path(f) for f in st["outputs"]]
        exists = [o.exists() for o in outs_rel]
        if all(exists) and not args.force and not args.dry_run:
            print(f"[SKIP] {st['name']}: all outputs exist (use --force to re-run).")
            summary.append((st["name"], "skipped", 0.0))
            continue
        code, dur = run_stage(st, REPO_ROOT / "logs" / "pipeline" / f"{st['name']}.log",
                              args.dry_run)
        # 4. output check AFTER run (skipped in dry-run).
        if not args.dry_run:
            still_missing = [f for f in st["outputs"] if not Path(f).exists()
                             and not (REPO_ROOT / f).exists()]
            if code != 0 or still_missing:
                logp = REPO_ROOT / "logs" / "pipeline" / f"{st['name']}.log"
                print(f"\n[FAIL] stage '{st['name']}' failed. exit_code={code} log={logp}")
                if still_missing:
                    print(f"  missing outputs: {still_missing}")
                print(f"  resume with: python run_pipeline.py --from-stage {st['name']}")
                summary.append((st["name"], "failed", dur))
                break
            summary.append((st["name"], "ran", dur))
        else:
            summary.append((st["name"], "dry-run", 0.0))

    print("\n" + "=" * 60)
    print(f"{'stage':<22} | {'status':<10} | {'duration_s':>10}")
    print("-" * 60)
    for n, s, d in summary:
        print(f"{n:<22} | {s:<10} | {d:>10.1f}")
    print("=" * 60)

    if not args.dry_run and summary and all(s != "failed" for _, s, _ in summary):
        import json
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = "unknown (no git repo)"
        info = {"args": vars(args), "timestamp": datetime.now().isoformat(),
                "git_hash": commit, "python": sys.version,
                "summary": [{"stage": n, "status": s, "duration_s": d}
                            for n, s, d in summary]}
        for pkg in ("torch", "torchvision", "numpy", "sklearn", "PIL", "yaml", "matplotlib"):
            try:
                m = __import__(pkg)
                info[pkg] = getattr(m, "__version__", "installed")
            except Exception:
                info[pkg] = "not-installed"
        outp = Path(args.output_dir) / "pipeline_run.json"
        outp = outp if outp.is_absolute() else REPO_ROOT / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Run config saved to {outp}")


if __name__ == "__main__":
    main()
