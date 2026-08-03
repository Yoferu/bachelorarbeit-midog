#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def aggregates(path: Path) -> dict[str, Any]:
    return load_json(path).get("aggregates", {})


def runtime(path: Path) -> dict[str, Any]:
    return load_json(path)


def timing_forward(path: Path) -> float | None:
    values = []
    for item in sorted(path.glob("measured*_timing.json")):
        value = load_json(item).get("stages", {}).get("forward_pass", {}).get("total_s")
        if value is not None:
            values.append(float(value))
    return statistics.median(values) if values else None


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def delta(a: Any, b: Any) -> float | None:
    return float(a) - float(b) if a is not None and b is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final pruned-60 clean recovery comparison report.")
    parser.add_argument("--root", type=Path, default=Path("experiments/distillation/pruned60_clean_recovery"))
    parser.add_argument("--output", type=Path, default=Path("experiments/distillation/pruned60_clean_recovery/final_comparison.md"))
    args = parser.parse_args()

    policy = load_json(args.root / "frozen_evaluation_policy.json")
    pruned_meta = load_json(Path("experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.meta.json"))
    pruning = pruned_meta.get("pruning_result", {})
    specs = [
        ("baseline", "original FCOS_18", "baseline", pruning.get("parameters_before_total"), pruning.get("macs_before")),
        ("pruned60", "untuned pruned 60%", "none", pruning.get("parameters_after_total"), pruning.get("macs_after")),
        ("supervised", "supervised-recovered 60%", "supervised fine-tuning", pruning.get("parameters_after_total"), pruning.get("macs_after")),
        ("distilled", "distilled-recovered 60%", "FCOS_x101 distillation", pruning.get("parameters_after_total"), pruning.get("macs_after")),
    ]
    rows = []
    baseline_forward = None
    for key, label, method, params, macs in specs:
        test_ag = aggregates(args.root / "final_test" / key / "test_metrics.json")
        run = runtime(args.root / "final_test" / key / "test_runtime.json")
        forward = timing_forward(args.root / "cpu_timing" / key)
        if key == "baseline":
            baseline_forward = forward
        model_policy = policy.get("models", {}).get(key, {})
        rows.append(
            {
                "Model": label,
                "Training method": method,
                "Best epoch": model_policy.get("selected_epoch"),
                "Calibration threshold": model_policy.get("selected_threshold"),
                "Test AP": test_ag.get("AP"),
                "Test F1": test_ag.get("f1_score"),
                "Precision": test_ag.get("precision"),
                "Recall": test_ag.get("recall"),
                "TP": test_ag.get("true_positives"),
                "FP": test_ag.get("false_positives"),
                "FN": test_ag.get("false_negatives"),
                "Params": params,
                "MACs": macs,
                "Forward time": forward,
                "Speedup": (baseline_forward / forward) if baseline_forward and forward else None,
                "checkpoint": model_policy.get("checkpoint"),
                "loading_method": run.get("model_artifact_load_method"),
                "num_images": run.get("num_images"),
            }
        )
    by_model = {row["Model"]: row for row in rows}
    base = by_model["original FCOS_18"]
    untuned = by_model["untuned pruned 60%"]
    supervised = by_model["supervised-recovered 60%"]
    distilled = by_model["distilled-recovered 60%"]

    lines = [
        "# Pruned-60 Clean Recovery Final Comparison",
        "",
        "| Model | Training method | Best epoch | Calibration threshold | Test AP | Test F1 | Precision | Recall | TP | FP | FN | Params | MACs | Forward time | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cols = [
            "Model",
            "Training method",
            "Best epoch",
            "Calibration threshold",
            "Test AP",
            "Test F1",
            "Precision",
            "Recall",
            "TP",
            "FP",
            "FN",
            "Params",
            "MACs",
            "Forward time",
            "Speedup",
        ]
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    lines.extend(
        [
            "",
            "## Recovery Calculations",
            "",
            f"- Supervised F1 recovery over untuned 60%: `{fmt(delta(supervised.get('Test F1'), untuned.get('Test F1')))}`",
            f"- Distilled F1 recovery over untuned 60%: `{fmt(delta(distilled.get('Test F1'), untuned.get('Test F1')))}`",
            f"- Supervised remaining F1 gap to FCOS_18: `{fmt(delta(base.get('Test F1'), supervised.get('Test F1')))}`",
            f"- Distilled remaining F1 gap to FCOS_18: `{fmt(delta(base.get('Test F1'), distilled.get('Test F1')))}`",
            f"- Supervised recovery helped: `{fmt((supervised.get('Test F1') or 0) > (untuned.get('Test F1') or 0))}`",
            f"- Distillation helped: `{fmt((distilled.get('Test F1') or 0) > (untuned.get('Test F1') or 0))}`",
            f"- Distillation outperformed supervised fine-tuning: `{fmt((distilled.get('Test F1') or 0) > (supervised.get('Test F1') or 0))}`",
            f"- Structural speed advantage preserved: `{fmt(all((row.get('Speedup') or 0) > 1 for row in rows[1:]))}`",
            "",
            "## Methodology Limitation",
            "",
            "32 images from the full test split were previously used for diagnostic pruning selection and quick comparisons. The complete test result is therefore not perfectly independent of all prior development decisions. Epochs and thresholds were nevertheless selected exclusively using train-derived validation and calibration subsets. The 50%/60%/70% comparison, when present, is reported only as a sensitivity analysis.",
            "",
            "## Result Roles",
            "",
            "- Previous pilot results: existing runs under `experiments/distillation/runs/*_recovery` and `experiments/distillation/runs/*_distillation` are pilot runs because they used the original train split without the new validation/calibration holdouts.",
            "- Validation results: per-epoch AP/loss are in each clean run's `validation_results.csv`.",
            "- Calibration results: threshold sweeps are in `threshold_calibration/*_threshold_results.csv`.",
            "- Full-test results: fixed-policy final metrics are in `final_test/*/test_metrics.json`.",
            "- Sensitivity-analysis results: optional outputs are in `sensitivity_analysis/` and must not redefine the main 60% candidate.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
