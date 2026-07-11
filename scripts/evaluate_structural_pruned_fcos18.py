#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments/pruning/evaluation/fcos18_depgraph_structural_10pct"


@dataclass(frozen=True)
class RunSpec:
    label: str
    pruned_model_path: Path | None = None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FCOS_18 baseline against the DepGraph structural pruning artifact."
    )
    parser.add_argument("--config_file", type=Path, default=PROJECT_ROOT / "experiments/eval_guide/configs/FCOS_18_eval.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/midogpp_guide_eval_xvalidation_smoke.csv")
    parser.add_argument("--guide_repo", type=Path, default=PROJECT_ROOT / "repos/MIDOG_2025_Guide")
    parser.add_argument("--img_dir", type=Path, default=PROJECT_ROOT / "data/midogpp")
    parser.add_argument(
        "--pruned_model_path",
        type=Path,
        default=PROJECT_ROOT / "experiments/pruning/models/FCOS_18_depgraph_structural_10pct.pt",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--runtime_backend", default="pytorch_eager")
    parser.add_argument("--skip_run", action="store_true", help="Only summarize existing outputs in output_dir.")
    return parser.parse_args()


def run_adapter(args: argparse.Namespace, spec: RunSpec) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = PROJECT_ROOT / "src/benchmark/midog_guide_adapter.py"
    metrics_output = args.output_dir / f"{spec.label}_metrics.json"
    runtime_output = args.output_dir / f"{spec.label}_runtime.json"
    timing_json = args.output_dir / f"{spec.label}_timing.json"
    timing_csv = args.output_dir / f"{spec.label}_timing.csv"
    command = [
        sys.executable,
        str(adapter),
        "--config_file",
        str(args.config_file),
        "--dataset",
        str(args.dataset),
        "--guide_repo",
        str(args.guide_repo),
        "--img_dir",
        str(args.img_dir),
        "--metrics_output",
        str(metrics_output),
        "--runtime_metadata_output",
        str(runtime_output),
        "--runtime_backend",
        args.runtime_backend,
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
        "--overwrite",
        "--profile_pipeline",
        "--timing_output_json",
        str(timing_json),
        "--timing_output_csv",
        str(timing_csv),
    ]
    if spec.pruned_model_path is not None:
        command.extend(["--pruned_model_path", str(spec.pruned_model_path)])

    log_path = args.output_dir / f"{spec.label}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_for(output_dir: Path, label: str) -> dict[str, Any]:
    metrics = load_json(output_dir / f"{label}_metrics.json")
    timing = load_json(output_dir / f"{label}_timing.json")
    aggregates = metrics["aggregates"]
    stages = timing.get("stages", {})
    forward_pass = stages.get("forward_pass", {})
    return {
        "model": label,
        "f1": float(aggregates["f1_score"]),
        "precision": float(aggregates["precision"]),
        "recall": float(aggregates["recall"]),
        "false_positives": int(aggregates["false_positives"]),
        "false_negatives": int(aggregates["false_negatives"]),
        "total_detections": int(aggregates["total_detections"]),
        "runtime_s": float(timing["total_end_to_end_s"]),
        "forward_pass_s": float(forward_pass["total_s"]),
        "mean_patch_latency_s": float(timing["mean_per_patch_s"]),
    }


def decide_finetuning(baseline: dict[str, Any], pruned: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    for metric in ["f1", "precision", "recall"]:
        drop = baseline[metric] - pruned[metric]
        if drop > 0.02:
            reasons.append(f"{metric} dropped by {drop:.4f}")

    baseline_detections = max(1, baseline["total_detections"])
    detection_shift = abs(pruned["total_detections"] - baseline["total_detections"]) / baseline_detections
    if detection_shift > 0.10:
        reasons.append(f"total detections shifted by {detection_shift:.1%}")

    speedup = (baseline["forward_pass_s"] - pruned["forward_pass_s"]) / baseline["forward_pass_s"]
    if reasons:
        if any("dropped" in reason for reason in reasons):
            return "fine-tuning needed", reasons
        return "fine-tuning likely needed", reasons
    if speedup > 0.01:
        return "no fine-tuning needed based on this evaluation", [f"forward pass improved by {speedup:.1%}"]
    return "inconclusive", ["accuracy is stable, but forward-pass speed did not improve meaningfully"]


def write_comparison(output_dir: Path, rows: list[dict[str, Any]], decision: str, reasons: list[str]) -> None:
    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"
    json_path = output_dir / "comparison.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({"runs": rows, "decision": decision, "reasons": reasons}, indent=2),
        encoding="utf-8",
    )

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in rows:
        values = []
        for key in fieldnames:
            value = row[key]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")

    md_path.write_text(
        "\n".join(
            [
                "# FCOS_18 Structural Pruning Evaluation",
                "",
                header,
                separator,
                *body,
                "",
                f"Decision: {decision}",
                "",
                "Reasons:",
                *[f"- {reason}" for reason in reasons],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = get_args()
    specs = [
        RunSpec("FCOS_18_baseline"),
        RunSpec("FCOS_18_depgraph_structural_10pct", args.pruned_model_path),
    ]
    if not args.skip_run:
        for spec in specs:
            run_adapter(args, spec)

    rows = [row_for(args.output_dir, spec.label) for spec in specs]
    decision, reasons = decide_finetuning(rows[0], rows[1])
    write_comparison(args.output_dir, rows, decision, reasons)

    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
