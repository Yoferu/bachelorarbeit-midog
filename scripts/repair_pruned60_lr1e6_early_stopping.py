#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.config import DistillationConfig
from src.distillation.early_stopping import migrate_legacy_state
from src.distillation.logging_utils import write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_best_row(run_dir: Path) -> dict[str, Any]:
    selection = read_json(run_dir / "dense_best_checkpoint.json")
    if selection:
        return {
            "best_validation_ap": float(selection["best_ap"]),
            "best_step": int(selection["best_ap_global_optimizer_step"]),
            "best_epoch_fraction": float(selection["best_ap_fractional_epoch"]),
            "best_checkpoint": str(selection["best_ap_source_checkpoint"]),
        }
    csv_path = run_dir / "dense_validation_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("inference_status") == "ok" and row.get("validation_ap")]
    if not rows:
        raise RuntimeError(f"No valid validation AP rows found in {csv_path}.")
    best = max(rows, key=lambda row: float(row["validation_ap"]))
    return {
        "best_validation_ap": float(best["validation_ap"]),
        "best_step": int(float(best["global_optimizer_step"])),
        "best_epoch_fraction": float(best["fractional_epoch"]),
        "best_checkpoint": str(best["checkpoint"]),
    }


def inspect_training_state(path: Path, expected_step: int) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing resumable training state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = ["student_state_dict", "optimizer_state_dict", "epoch_index", "next_batch_idx", "global_optimizer_step"]
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"Training state is missing required keys: {missing}")
    if int(state["global_optimizer_step"]) != expected_step:
        raise RuntimeError(f"Expected optimizer step {expected_step}, found {state['global_optimizer_step']}.")
    if int(state["epoch_index"]) != 1 or int(state["next_batch_idx"]) != 0:
        raise RuntimeError(
            f"Expected epoch_index=1 and next_batch_idx=0, found "
            f"epoch_index={state['epoch_index']} next_batch_idx={state['next_batch_idx']}."
        )
    return {
        "epoch_index": int(state["epoch_index"]),
        "next_batch_idx": int(state["next_batch_idx"]),
        "global_step": int(state.get("global_step", expected_step)),
        "global_optimizer_step": int(state["global_optimizer_step"]),
        "fractional_epoch": float(state.get("fractional_epoch", 1.0)),
        "has_optimizer_state": bool(state.get("optimizer_state_dict")),
        "has_student_state": bool(state.get("student_state_dict")),
        "has_rng_state": "torch_rng_state" in state,
    }


def backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.faulty_warmup_stop_{stamp}.bak")
    shutil.copy2(path, backup_path)
    return backup_path


def repair(run_dir: Path, config_path: Path, expected_step: int, dry_run: bool) -> dict[str, Any]:
    config = DistillationConfig.load(config_path)
    early_path = run_dir / "early_stopping.json"
    status_path = run_dir / "run_status.json"
    summary_path = run_dir / "run_summary.json"
    training_state_info = inspect_training_state(run_dir / "training_state_latest.pt", expected_step)
    raw_early = read_json(early_path)
    if not raw_early:
        raise RuntimeError(f"Missing early stopping state: {early_path}")
    best = read_best_row(run_dir)
    repaired = migrate_legacy_state(raw_early, config.training.early_stopping)
    repaired.best_validation_ap = best["best_validation_ap"]
    repaired.best_step = best["best_step"]
    repaired.best_epoch_fraction = best["best_epoch_fraction"]
    repaired.best_checkpoint = best["best_checkpoint"]
    repaired.phase = "active"
    repaired.warmup_completed = True
    repaired.patience_counter = 0
    repaired.non_improving_evaluations = 0
    repaired.stopped = False
    repaired.stopping_reason = None
    repaired.final_optimizer_step = None
    repaired.final_epoch_fraction = None
    repaired.last_evaluated_step = expected_step
    repaired.last_evaluated_fractional_epoch = 1.0
    repaired.last_evaluated_checkpoint = str(run_dir / "student_epoch_1.pt")

    status_payload = {
        "status": "interrupted",
        "reason": "repaired_faulty_warmup_patience_state_resume_from_epoch_2",
        "final_optimizer_step": expected_step,
        "final_epoch_fraction": 1.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary = read_json(summary_path)
    summary.update(
        {
            "status": "interrupted",
            "repair_status": "faulty_warmup_patience_repaired",
            "final_optimizer_step": expected_step,
            "final_epoch_fraction": 1.0,
            "early_stopping": repaired.__dict__,
        }
    )
    report = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "dry_run": dry_run,
        "training_state": training_state_info,
        "old_status": read_json(status_path).get("status"),
        "old_patience": raw_early.get("patience_counter", raw_early.get("non_improving_evaluations")),
        "new_status": status_payload["status"],
        "new_phase": repaired.phase,
        "new_warmup_completed": repaired.warmup_completed,
        "new_patience": repaired.patience_counter,
        "best_validation_ap": repaired.best_validation_ap,
        "best_step": repaired.best_step,
        "best_epoch_fraction": repaired.best_epoch_fraction,
        "best_checkpoint": repaired.best_checkpoint,
        "next_epoch_index": training_state_info["epoch_index"],
        "next_batch_idx": training_state_info["next_batch_idx"],
    }
    if dry_run:
        return report

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups = {
        "early_stopping": str(backup(early_path, stamp)),
        "run_status": str(backup(status_path, stamp)),
        "run_summary": str(backup(summary_path, stamp)),
    }
    repaired.save(early_path)
    write_json(status_path, status_payload)
    write_json(summary_path, summary)
    write_json(
        run_dir / "early_stopping_repair.json",
        {
            **report,
            "dry_run": False,
            "backups": backups,
            "repaired_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {**report, "backups": backups}


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair faulty epoch-1 warm-up early-stopping state for pruned60 lr1e-6 runs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = repair(args.run_dir, args.config, args.expected_step, args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
