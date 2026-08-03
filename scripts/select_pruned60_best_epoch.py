#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.stable_selection import write_stable_selection


EPOCH_RE = re.compile(r"student_epoch_(\d+)\.pt$")
STEP_RE = re.compile(r"student_step_(\d+)\.pt$")


def read_validation_losses(training_log: Path) -> dict[int, float]:
    if not training_log.exists():
        return {}
    losses: dict[int, float] = {}
    with training_log.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("batch") != "val":
                continue
            try:
                losses[int(row["epoch"]) + 1] = float(row["total"])
            except (KeyError, TypeError, ValueError):
                continue
    return losses


def metric(aggregates: dict, *names: str) -> float | None:
    for name in names:
        value = aggregates.get(name)
        if value is not None:
            return float(value)
    return None


def number(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def checkpoint_sort_key(path: Path) -> tuple[int, int]:
    if match := STEP_RE.match(path.name):
        return (0, int(match.group(1)))
    if match := EPOCH_RE.match(path.name):
        return (1, int(match.group(1)))
    return (2, 0)


def checkpoint_label(path: Path) -> str:
    if match := STEP_RE.match(path.name):
        return f"step_{int(match.group(1)):06d}"
    if match := EPOCH_RE.match(path.name):
        return f"epoch_{int(match.group(1)):03d}"
    return path.stem


def existing_rows_by_label(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("checkpoint_label")): dict(row) for row in csv.DictReader(handle)}


def checkpoint_metadata(path: Path) -> dict[str, object]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"checkpoint_metadata_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(artifact, dict):
        return {}
    return {
        "checkpoint_epoch": artifact.get("epoch"),
        "global_optimizer_step": artifact.get("global_optimizer_step"),
        "batch_index": artifact.get("batch_idx"),
        "fractional_epoch": artifact.get("fractional_epoch"),
        "current_lr": artifact.get("learning_rate"),
        "optimizer_learning_rates": artifact.get("optimizer_learning_rates"),
        "wall_clock_training_s": artifact.get("wall_clock_training_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-epoch checkpoints on validation and select best AP.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--config-file", type=Path, default=Path("experiments/eval_guide/configs/FCOS_18_eval.yaml"))
    parser.add_argument("--guide-repo", type=Path, default=Path("repos/MIDOG_2025_Guide"))
    parser.add_argument("--img-dir", type=Path, default=Path("data/midogpp"))
    parser.add_argument("--adapter", type=Path, default=Path("src/benchmark/midog_guide_adapter.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--det-thresh", type=float, default=0.584)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--checkpoint-glob", action="append", default=None)
    parser.add_argument("--metrics-subdir", default="validation_metrics")
    parser.add_argument("--results-output-name", default="validation_results.csv")
    parser.add_argument("--selection-output-name", default="best_epoch_selection.json")
    parser.add_argument("--best-checkpoint-name", default="student_best_ap.pt")
    parser.add_argument("--stable-selection-output-name", default="stable_checkpoint_selection.json")
    parser.add_argument("--stable-checkpoint-name", default="student_stable_selected.pt")
    parser.add_argument("--plateau-tolerance", type=float, default=0.0005)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    losses = read_validation_losses(args.run_dir / "training_log.csv")
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / args.metrics_subdir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / args.results_output_name
    existing_rows = existing_rows_by_label(out_csv)
    checkpoint_paths: dict[Path, None] = {}
    for pattern in args.checkpoint_glob or ["student_epoch_*.pt"]:
        for path in args.run_dir.glob(pattern):
            checkpoint_paths[path] = None
    checkpoints = sorted(checkpoint_paths, key=checkpoint_sort_key)
    if not checkpoints:
        raise SystemExit(f"No per-epoch checkpoints found in {args.run_dir}")

    for checkpoint in checkpoints:
        label = checkpoint_label(checkpoint)
        fallback_epoch = int(EPOCH_RE.match(checkpoint.name).group(1)) if EPOCH_RE.match(checkpoint.name) else None
        metadata = checkpoint_metadata(checkpoint)
        epoch_value = metadata.get("checkpoint_epoch") or fallback_epoch
        loss_epoch = int(epoch_value) if epoch_value not in {None, ""} else -1
        metrics_path = metrics_dir / f"{label}_metrics.json"
        runtime_path = metrics_dir / f"{label}_runtime.json"
        stat = checkpoint.stat()
        row: dict[str, object] = {
            "checkpoint_label": label,
            "epoch": epoch_value,
            "checkpoint": str(checkpoint),
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "checkpoint_size_bytes": stat.st_size,
            **metadata,
            "validation_loss": losses.get(loss_epoch),
            "diagnostic_threshold": args.det_thresh,
            "loading_status": "pending",
            "inference_status": "pending",
        }
        existing = existing_rows.pop(label, None)
        if (
            existing
            and not args.force
            and str(existing.get("checkpoint_mtime_ns")) == str(stat.st_mtime_ns)
            and str(existing.get("checkpoint_size_bytes")) == str(stat.st_size)
            and existing.get("inference_status") == "ok"
            and existing.get("validation_ap") not in {None, ""}
        ):
            rows.append(existing)
            continue
        if args.force or not metrics_path.exists() or not runtime_path.exists():
            cmd = [
                args.python,
                str(args.adapter),
                "--config_file",
                str(args.config_file),
                "--dataset",
                str(args.dataset),
                "--guide_repo",
                str(args.guide_repo),
                "--img_dir",
                str(args.img_dir),
                "--metrics_output",
                str(metrics_path),
                "--runtime_metadata_output",
                str(runtime_path),
                "--runtime_backend",
                "pytorch_eager",
                "--pruned_model_path",
                str(checkpoint),
                "--split",
                args.split,
                "--device",
                args.device,
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--overlap",
                str(args.overlap),
                "--nms_thresh",
                str(args.nms_thresh),
                "--det_thresh",
                str(args.det_thresh),
                "--overwrite",
            ]
            eval_start = time.perf_counter()
            completed = subprocess.run(cmd)
            row["evaluation_runtime_s"] = time.perf_counter() - eval_start
            if completed.returncode != 0:
                row["loading_status"] = "failed"
                row["inference_status"] = f"failed:{completed.returncode}"
                rows.append(row)
                continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        aggregates = metrics.get("aggregates", {})
        row.update(
            {
                "validation_ap": metric(aggregates, "AP", "ap"),
                "f1": metric(aggregates, "f1_score", "f1"),
                "precision": metric(aggregates, "precision"),
                "recall": metric(aggregates, "recall"),
                "loading_status": runtime.get("model_artifact_load_method") or "baseline",
                "inference_status": "ok",
                "evaluation_runtime_s": runtime.get("total_runtime_s", row.get("evaluation_runtime_s")),
            }
        )
        rows.append(row)

    rows.extend(existing_rows.values())
    rows = sorted(rows, key=lambda row: checkpoint_sort_key(Path(str(row.get("checkpoint", row.get("checkpoint_label", ""))))))
    fieldnames = [
        "checkpoint_label",
        "epoch",
        "checkpoint",
        "checkpoint_epoch",
        "global_optimizer_step",
        "batch_index",
        "fractional_epoch",
        "current_lr",
        "optimizer_learning_rates",
        "wall_clock_training_s",
        "checkpoint_mtime_ns",
        "checkpoint_size_bytes",
        "validation_ap",
        "validation_loss",
        "f1",
        "precision",
        "recall",
        "diagnostic_threshold",
        "evaluation_runtime_s",
        "loading_status",
        "inference_status",
        "checkpoint_metadata_error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row.get("validation_ap") is not None and row.get("inference_status") == "ok"]
    if not ok_rows:
        raise SystemExit("No successful validation evaluations; cannot select best epoch.")
    best_ap = max(
        ok_rows,
        key=lambda row: (
            float(row["validation_ap"]),
            -(int(row["global_optimizer_step"]) if row.get("global_optimizer_step") not in {None, ""} else int(row["epoch"])),
        ),
    )
    loss_rows = [row for row in ok_rows if number(row.get("validation_loss")) is not None]
    best_loss = min(loss_rows, key=lambda row: (number(row["validation_loss"]) or float("inf"), int(row["epoch"]))) if loss_rows else None
    shutil.copy2(best_ap["checkpoint"], output_dir / args.best_checkpoint_name)
    selection = {
        "best_ap_epoch": int(best_ap["epoch"]) if best_ap.get("epoch") not in {None, ""} else None,
        "best_ap_checkpoint_label": best_ap.get("checkpoint_label"),
        "best_ap_global_optimizer_step": (
            int(best_ap["global_optimizer_step"]) if best_ap.get("global_optimizer_step") not in {None, ""} else None
        ),
        "best_ap_fractional_epoch": (
            float(best_ap["fractional_epoch"]) if best_ap.get("fractional_epoch") not in {None, ""} else None
        ),
        "best_ap": best_ap["validation_ap"],
        "best_ap_source_checkpoint": best_ap["checkpoint"],
        "best_ap_checkpoint": str(output_dir / args.best_checkpoint_name),
        "lowest_validation_loss_epoch": int(best_loss["epoch"]) if best_loss else None,
        "lowest_validation_loss": best_loss.get("validation_loss") if best_loss else None,
        "criteria_agree": (int(best_ap["epoch"]) == int(best_loss["epoch"])) if best_loss else None,
        "validation_results_csv": str(out_csv),
    }
    (output_dir / args.selection_output_name).write_text(json.dumps(selection, indent=2), encoding="utf-8")
    stable_selection = write_stable_selection(
        run_dir=output_dir,
        results_csv=out_csv,
        output_name=args.stable_selection_output_name,
        artifact_name=args.stable_checkpoint_name,
        plateau_tolerance=args.plateau_tolerance,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps({"raw_best": selection, "stable_selection": stable_selection}, indent=2))


if __name__ == "__main__":
    main()
