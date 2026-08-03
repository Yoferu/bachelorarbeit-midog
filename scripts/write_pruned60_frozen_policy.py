#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze selected checkpoints and thresholds before final testing.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervised-run", type=Path)
    parser.add_argument("--distilled-run", type=Path)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--selected-checkpoint-json", type=Path)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Policy already exists, leaving unchanged: {args.output}")
        return
    thresholds = load_json(args.thresholds)
    if args.selected_checkpoint_json:
        selection = load_json(args.selected_checkpoint_json)
        if len(thresholds) != 1:
            raise SystemExit("Single-checkpoint policy expects exactly one calibrated model in thresholds JSON.")
        key, threshold_payload = next(iter(thresholds.items()))
        policy = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "nms_threshold": args.nms_thresh,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "models": {
                key: {
                    "checkpoint": selection["checkpoint_path"],
                    "selected_epoch": selection.get("epoch"),
                    "selected_global_optimizer_step": selection.get("global_optimizer_step"),
                    "selected_fractional_epoch": selection.get("fractional_epoch"),
                    "selected_threshold": threshold_payload["selected_threshold"],
                    "validation_AP": selection.get("validation_ap"),
                    "calibration_F1": threshold_payload["calibration_F1"],
                    "learning_rate": selection.get("learning_rate"),
                    "selection_rule": selection.get("selection_rule"),
                }
            },
            "frozen_policy": "Do not change checkpoint, threshold, NMS, patch settings, or model selection based on test results.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
        return

    if args.supervised_run is None or args.distilled_run is None:
        raise SystemExit("--supervised-run and --distilled-run are required unless --selected-checkpoint-json is used")
    sup = load_json(args.supervised_run / "best_epoch_selection.json")
    dist = load_json(args.distilled_run / "best_epoch_selection.json")
    policy = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "nms_threshold": args.nms_thresh,
        "patch_size": args.patch_size,
        "overlap": args.overlap,
        "models": {
            "baseline": {
                "checkpoint": None,
                "selected_epoch": None,
                "selected_threshold": thresholds["baseline"]["selected_threshold"],
                "calibration_F1": thresholds["baseline"]["calibration_F1"],
            },
            "pruned60": {
                "checkpoint": thresholds["pruned60"]["checkpoint"],
                "selected_epoch": 0,
                "selected_threshold": thresholds["pruned60"]["selected_threshold"],
                "calibration_F1": thresholds["pruned60"]["calibration_F1"],
            },
            "supervised": {
                "checkpoint": sup["best_ap_checkpoint"],
                "selected_epoch": sup["best_ap_epoch"],
                "selected_threshold": thresholds["supervised"]["selected_threshold"],
                "validation_AP": sup["best_ap"],
                "calibration_F1": thresholds["supervised"]["calibration_F1"],
            },
            "distilled": {
                "checkpoint": dist["best_ap_checkpoint"],
                "selected_epoch": dist["best_ap_epoch"],
                "selected_threshold": thresholds["distilled"]["selected_threshold"],
                "validation_AP": dist["best_ap"],
                "calibration_F1": thresholds["distilled"]["calibration_F1"],
            },
        },
        "frozen_policy": "Do not change epochs, checkpoints, thresholds, NMS, patch settings, or model selection based on test results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
