#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


LR_RUNS = {
    "1e-4": "fcos18_pruned60_supervised_clean_dense_lr1e-4_seed42",
    "3e-5": "fcos18_pruned60_supervised_clean_dense_lr3e-5_seed42",
    "1e-5": "fcos18_pruned60_supervised_clean_dense_lr1e-5_seed42",
}
EXTENDED_RUNS = {
    "3e-5": "fcos18_pruned60_supervised_clean_dense_lr3e-5_seed42_extend5",
    "1e-5": "fcos18_pruned60_supervised_clean_dense_lr1e-5_seed42_extend10",
}
EXISTING_RUNS = {
    "supervised_clean": Path("experiments/distillation/runs/fcos18_pruned60_supervised_clean"),
    "distillation_clean": Path("experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation_clean"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def command_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None


def number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optimizer_step(row: dict[str, Any]) -> int:
    value = row.get("global_optimizer_step")
    if value not in {None, ""}:
        return int(float(value))
    epoch = row.get("epoch")
    return int(float(epoch)) if epoch not in {None, ""} else 10**12


def load_existing_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_type, run_dir in EXISTING_RUNS.items():
        reevaluated = root / "existing_clean_reevaluation" / run_type / "validation_results.csv"
        source_csv = reevaluated if reevaluated.exists() else run_dir / "validation_results.csv"
        source_kind = "reevaluated" if reevaluated.exists() else "existing_cached"
        for row in read_csv(source_csv):
            row = dict(row)
            row["run_type"] = run_type
            row["run_dir"] = str(run_dir)
            row["result_source"] = source_kind
            row["result_csv"] = str(source_csv)
            rows.append(row)
    return rows


def load_dense_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs_root = root / "runs"
    for lr, name in {**LR_RUNS, **EXTENDED_RUNS}.items():
        run_dir = runs_root / name
        for row in read_csv(run_dir / "dense_validation_results.csv"):
            row = dict(row)
            row["experiment_name"] = name
            row["learning_rate"] = lr
            row["run_dir"] = str(run_dir)
            row["is_extended_run"] = name.endswith(("extend5", "extend10"))
            row["is_best_checkpoint_of_run"] = False
            row["is_globally_selected_checkpoint"] = False
            rows.append(row)
    for name in sorted({row["experiment_name"] for row in rows}):
        run_rows = [row for row in rows if row["experiment_name"] == name and number(row.get("validation_ap")) is not None]
        if not run_rows:
            continue
        best = max(run_rows, key=lambda row: (number(row.get("validation_ap")) or -1.0, -optimizer_step(row)))
        best["is_best_checkpoint_of_run"] = True
    return rows


def select_global(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any] | None:
    ok = [row for row in rows if not row.get("is_extended_run") and number(row.get("validation_ap")) is not None]
    if not ok:
        return None
    best_ap = max(number(row["validation_ap"]) or -1.0 for row in ok)
    candidates = [row for row in ok if best_ap - (number(row["validation_ap"]) or -1.0) <= tolerance]
    return min(candidates, key=optimizer_step)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pruned60 dense LR sweep artifacts.")
    parser.add_argument("--root", type=Path, default=Path("experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep"))
    parser.add_argument("--ap-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)

    existing = load_existing_rows(args.root)
    write_csv(args.root / "existing_clean_epoch_comparison.csv", existing)
    write_json(args.root / "existing_clean_epoch_comparison.json", existing)
    existing_fields = ["run_type", "epoch", "checkpoint", "validation_ap", "f1", "precision", "recall", "diagnostic_threshold", "inference_status"]
    (args.root / "existing_clean_epoch_comparison.md").write_text(
        "# Existing Clean Epoch Comparison\n\n" + markdown_table(existing, existing_fields) + "\n",
        encoding="utf-8",
    )

    dense = load_dense_rows(args.root)
    selected = select_global(dense, args.ap_tolerance)
    if selected:
        selected["is_globally_selected_checkpoint"] = True
    dense_fields = [
        "experiment_name",
        "learning_rate",
        "checkpoint",
        "global_optimizer_step",
        "epoch",
        "fractional_epoch",
        "validation_ap",
        "validation_loss",
        "current_lr",
        "wall_clock_training_s",
        "evaluation_runtime_s",
        "is_best_checkpoint_of_run",
        "is_globally_selected_checkpoint",
        "inference_status",
    ]
    write_csv(args.root / "dense_checkpoint_results.csv", dense)
    write_json(args.root / "dense_checkpoint_results.json", dense)
    (args.root / "dense_checkpoint_results.md").write_text(
        "# Dense Checkpoint Results\n\n" + markdown_table(dense, dense_fields) + "\n",
        encoding="utf-8",
    )

    lr_rows = []
    for lr, name in LR_RUNS.items():
        run_rows = [row for row in dense if row.get("experiment_name") == name and number(row.get("validation_ap")) is not None]
        final_rows = sorted(run_rows, key=optimizer_step)
        best = max(run_rows, key=lambda row: (number(row.get("validation_ap")) or -1.0, -optimizer_step(row))) if run_rows else {}
        final = final_rows[-1] if final_rows else {}
        lr_rows.append(
            {
                "learning_rate": lr,
                "run": name,
                "status": "evaluated" if run_rows else "pending",
                "evaluated_checkpoints": len(run_rows),
                "best_validation_ap": best.get("validation_ap", ""),
                "best_optimizer_step": best.get("global_optimizer_step", ""),
                "best_fractional_epoch": best.get("fractional_epoch", ""),
                "final_validation_ap": final.get("validation_ap", ""),
            }
        )
    (args.root / "learning_rate_comparison.md").write_text(
        "# Learning Rate Comparison\n\n"
        "Primary comparison uses only the fixed two-epoch runs.\n\n"
        + markdown_table(
            lr_rows,
            [
                "learning_rate",
                "run",
                "status",
                "evaluated_checkpoints",
                "best_validation_ap",
                "best_optimizer_step",
                "best_fractional_epoch",
                "final_validation_ap",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    if selected:
        selection = {
            "checkpoint_path": selected["checkpoint"],
            "experiment_name": selected["experiment_name"],
            "learning_rate": selected["learning_rate"],
            "epoch": selected.get("epoch"),
            "global_optimizer_step": selected.get("global_optimizer_step"),
            "fractional_epoch": selected.get("fractional_epoch"),
            "validation_ap": selected.get("validation_ap"),
            "ap_tolerance": args.ap_tolerance,
            "selection_rule": "highest validation AP; ties within tolerance prefer fewer optimizer steps",
        }
    else:
        selection = {
            "status": "pending",
            "reason": "No dense validation results found.",
            "ap_tolerance": args.ap_tolerance,
            "selection_rule": "highest validation AP; ties within tolerance prefer fewer optimizer steps",
        }
    write_json(args.root / "selected_recovery_checkpoint.json", selection)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
        "packages": {"torch": torch.__version__},
        "paths": {
            "train_csv": "experiments/distillation/pruned60_clean_recovery/data/train_seed42.csv",
            "validation_csv": "experiments/distillation/pruned60_clean_recovery/data/validation_seed42.csv",
            "calibration_csv": "experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv",
            "final_test_csv": "data/midogpp_guide_eval_xvalidation split=test (not used here)",
        },
        "configs": {
            lr: f"experiments/distillation/configs/{name}.yaml" for lr, name in LR_RUNS.items()
        },
        "commands": {
            "tests": ".venv/bin/python -m pytest tests/test_prediction_json_serialization.py tests/test_distillation_loss_scaling.py tests/test_distilled_registry_reload.py",
            "summarize": ".venv/bin/python scripts/summarize_pruned60_dense_lr_sweep.py",
        },
    }
    write_json(args.root / "experiment_manifest.json", manifest)

    (args.root / "serialization_fix_audit.md").write_text(
        "# Serialization Fix Audit\n\n"
        "- Root cause: prediction dictionaries may contain NumPy arrays/scalars, but `_jsonable()` only converted Torch tensors and Python containers.\n"
        "- Fix: `src/benchmark/midog_guide_adapter.py` now converts Torch scalar tensors, non-scalar tensors, `numpy.ndarray`, and `numpy.generic` recursively.\n"
        "- Focused test: `tests/test_prediction_json_serialization.py`.\n"
        "- Final test data use: none.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
