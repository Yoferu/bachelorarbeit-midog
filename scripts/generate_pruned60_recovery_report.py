#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import statistics
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.distilled_model_loader import load_distilled_student
from src.distillation.model_registry import count_parameters


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def artifact_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def params_for(path: Path | None, fallback: int | None = None) -> int | None:
    if path is None or not path.exists():
        return fallback
    try:
        if path.name.endswith(".ckpt"):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else {}
            return int(sum(value.numel() for value in state.values() if hasattr(value, "numel")))
        model, _ = load_distilled_student(path)
        return count_parameters(model)
    except Exception:
        return fallback


def metrics_row(metrics_path: Path, runtime_path: Path) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    runtime = load_json(runtime_path)
    aggregates = metrics.get("aggregates", {})
    return {
        "quick_f1": aggregates.get("f1_score"),
        "precision": aggregates.get("precision"),
        "recall": aggregates.get("recall"),
        "tp": aggregates.get("true_positives"),
        "fp": aggregates.get("false_positives"),
        "fn": aggregates.get("false_negatives"),
        "detection_count": aggregates.get("total_detections"),
        "num_images": runtime.get("num_images"),
        "loading_method": runtime.get("model_artifact_load_method"),
        "architecture": runtime.get("model_name"),
    }


def timing_row(summary_path: Path) -> dict[str, Any]:
    row = read_summary_csv(summary_path)
    if not row:
        return {}

    def number(key: str) -> float | None:
        value = row.get(key)
        if value in {None, ""}:
            return None
        return float(value)

    measured = number("measured_inference_time_s")
    patches = number("num_patches")
    measured_files = sorted(summary_path.parent.glob("measured*_timing.json"))
    forward_values = []
    for path in measured_files:
        timing = load_json(path)
        forward = timing.get("stages", {}).get("forward_pass", {}).get("total_s")
        if forward is not None:
            forward_values.append(float(forward))

    return {
        "forward_latency_s": statistics.median(forward_values) if forward_values else None,
        "end_to_end_latency_s": number("total_end_to_end_s"),
        "mean_patch_latency_s": number("median_per_patch_s"),
        "throughput_patches_per_s": (patches / measured) if patches and measured else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pruned-60 recovery comparison report.")
    parser.add_argument("--output", type=Path, default=Path("experiments/distillation/pruned60_recovery_comparison.md"))
    parser.add_argument("--quality-dir", type=Path, default=Path("experiments/distillation/recovery_quality"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("experiments/distillation/recovery_cpu_benchmarks"))
    args = parser.parse_args()

    pruned_meta = load_json(Path("experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.meta.json"))
    pruning_result = pruned_meta.get("pruning_result", {})
    baseline_macs = pruning_result.get("macs_before")
    pruned_macs = pruning_result.get("macs_after")
    baseline_params = pruning_result.get("parameters_before_total")
    pruned_params = pruning_result.get("parameters_after_total")

    specs = [
        {
            "key": "fcos18",
            "label": "original FCOS_18",
            "training_method": "baseline",
            "teacher": "none",
            "initialization": "FCOS_18 checkpoint",
            "epochs": 0,
            "checkpoint": Path("repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt"),
            "params": baseline_params,
            "macs": baseline_macs,
        },
        {
            "key": "pruned60_untuned",
            "label": "untuned FCOS_18 FPN+Head 60%",
            "training_method": "none",
            "teacher": "none",
            "initialization": "DepGraph FPN+head 60% artifact",
            "epochs": 0,
            "checkpoint": Path("experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt"),
            "params": pruned_params,
            "macs": pruned_macs,
        },
        {
            "key": "supervised",
            "label": "60% pruned plus supervised fine-tuning",
            "training_method": "supervised_detection_loss",
            "teacher": "none",
            "initialization": "DepGraph FPN+head 60% artifact",
            "epochs": 20,
            "checkpoint": Path("experiments/distillation/runs/fcos18_pruned60_supervised_recovery/student_final.pt"),
            "params": pruned_params,
            "macs": pruned_macs,
        },
        {
            "key": "distilled",
            "label": "60% pruned plus FCOS_x101 distillation",
            "training_method": "supervised + output KD",
            "teacher": "FCOS_x101",
            "initialization": "DepGraph FPN+head 60% artifact",
            "epochs": 20,
            "checkpoint": Path("experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation/student_final.pt"),
            "params": pruned_params,
            "macs": pruned_macs,
        },
    ]

    rows = []
    for spec in specs:
        quality = metrics_row(
            args.quality_dir / spec["key"] / "quality_metrics.json",
            args.quality_dir / spec["key"] / "quality_runtime.json",
        )
        timing = timing_row(args.benchmark_dir / spec["key"] / "timing_median.csv")
        checkpoint = spec["checkpoint"]
        row = {
            **spec,
            **quality,
            **timing,
            "parameter_count": params_for(checkpoint, spec["params"]),
            "macs": spec["macs"],
            "artifact_size_mb": artifact_size_mb(checkpoint),
            "run_status": "complete" if checkpoint.exists() and quality and timing else "pending",
        }
        rows.append(row)

    baseline_forward = rows[0].get("mean_patch_latency_s")
    untuned_forward = rows[1].get("mean_patch_latency_s")
    for row in rows:
        latency = row.get("mean_patch_latency_s")
        row["speedup_vs_fcos18"] = (baseline_forward / latency) if baseline_forward and latency else None
        row["speedup_vs_untuned60"] = (untuned_forward / latency) if untuned_forward and latency else None

    by_key = {row["key"]: row for row in rows}
    untuned_f1 = by_key["pruned60_untuned"].get("quick_f1")
    baseline_f1 = by_key["fcos18"].get("quick_f1")
    supervised_f1 = by_key["supervised"].get("quick_f1")
    distilled_f1 = by_key["distilled"].get("quick_f1")

    def delta(a: Any, b: Any) -> float | None:
        return (float(a) - float(b)) if a is not None and b is not None else None

    conclusions = {
        "supervised_f1_recovery_over_untuned": delta(supervised_f1, untuned_f1),
        "distilled_f1_recovery_over_untuned": delta(distilled_f1, untuned_f1),
        "supervised_remaining_f1_gap_to_fcos18": delta(baseline_f1, supervised_f1),
        "distilled_remaining_f1_gap_to_fcos18": delta(baseline_f1, distilled_f1),
        "distillation_outperformed_supervised": (
            bool(distilled_f1 > supervised_f1) if distilled_f1 is not None and supervised_f1 is not None else None
        ),
        "structural_speed_advantage_preserved": (
            bool(by_key["distilled"].get("speedup_vs_fcos18", 0) > 1 and by_key["supervised"].get("speedup_vs_fcos18", 0) > 1)
            if by_key["distilled"].get("speedup_vs_fcos18") and by_key["supervised"].get("speedup_vs_fcos18")
            else None
        ),
    }

    columns = [
        "label",
        "training_method",
        "teacher",
        "initialization",
        "epochs",
        "parameter_count",
        "macs",
        "artifact_size_mb",
        "quick_f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "detection_count",
        "forward_latency_s",
        "end_to_end_latency_s",
        "mean_patch_latency_s",
        "throughput_patches_per_s",
        "speedup_vs_fcos18",
        "speedup_vs_untuned60",
        "checkpoint",
        "run_status",
    ]
    lines = [
        "# Pruned-60 Recovery Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    lines.extend(["", "## Recovery Calculations", ""])
    for key, value in conclusions.items():
        lines.append(f"- {key}: {fmt(value)}")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
