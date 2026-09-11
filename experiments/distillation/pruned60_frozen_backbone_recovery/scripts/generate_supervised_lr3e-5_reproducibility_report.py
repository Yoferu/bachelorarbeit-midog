#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.stable_selection import select_stable_checkpoint, write_stable_selection

BASE = PROJECT_ROOT / "experiments/distillation/pruned60_frozen_backbone_recovery"
REPORT_DIR = BASE / "reports"
SEEDS = [42, 43, 44]
RUNS = {seed: BASE / f"runs/supervised_lr3e-5_seed{seed}" for seed in SEEDS}
FULL_GRID = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 512]
COMMON_GRID = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 512]
STEP0_EXPECTED = 0.8665903211
TOLERANCE = 0.0005


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in {None, "", "nan"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def valid_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(path):
        if row.get("inference_status", "ok") != "ok":
            continue
        if as_float(row.get("validation_ap")) is None or as_int(row.get("global_optimizer_step")) is None:
            continue
        rows.append(row)
    return rows


def rows_by_step(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    by_step: dict[int, dict[str, str]] = {}
    for row in rows:
        step = as_int(row.get("global_optimizer_step"))
        if step is None:
            continue
        current = by_step.get(step)
        if current is None or str(row.get("checkpoint_label", "")).startswith("step_"):
            by_step[step] = row
    return by_step


def grid_rows(run_dir: Path, grid: list[int], *, source_name: str) -> list[dict[str, str]]:
    rows = valid_rows(run_dir / source_name)
    by_step = rows_by_step(rows)
    return [by_step[step] for step in grid if step in by_step]


def best(rows: list[dict[str, str]]) -> tuple[int | None, float | None]:
    if not rows:
        return None, None
    row = max(rows, key=lambda item: (as_float(item.get("validation_ap")) or -1.0, -(as_int(item.get("global_optimizer_step")) or 10**9)))
    return as_int(row.get("global_optimizer_step")), as_float(row.get("validation_ap"))


def stable_from_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) < 3:
        step, ap = best(rows)
        return {"stable_selection_status": "too_sparse_for_three_checkpoint_selection", "stable_selected_step": step, "stable_selected_ap": ap}
    return select_stable_checkpoint(rows, plateau_tolerance=TOLERANCE, project_root=PROJECT_ROOT)


def ensure_selection_artifacts(run_dir: Path) -> None:
    dense = run_dir / "dense_validation_results.csv"
    common = run_dir / "common_grid_validation_results.csv"
    if dense.exists() and not (run_dir / "stable_checkpoint_selection.json").exists():
        write_stable_selection(
            run_dir=run_dir,
            results_csv=dense,
            output_name="stable_checkpoint_selection.json",
            artifact_name="student_stable_selected.pt",
            plateau_tolerance=TOLERANCE,
            project_root=PROJECT_ROOT,
        )
    if common.exists() and not (run_dir / "common_stable_checkpoint_selection.json").exists():
        write_stable_selection(
            run_dir=run_dir,
            results_csv=common,
            output_name="common_stable_checkpoint_selection.json",
            artifact_name="student_common_stable_selected.pt",
            plateau_tolerance=TOLERANCE,
            project_root=PROJECT_ROOT,
        )


def gradient_summary(run_dir: Path) -> tuple[float | None, int | None]:
    best_value = None
    best_step = None
    for row in read_csv(run_dir / "training_log.csv"):
        value = as_float(row.get("gradient_norm"))
        step = as_int(row.get("global_optimizer_step"))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_step = step
    status = read_json(run_dir / "run_status.json")
    failure_value = as_float(status.get("gradient_norm"))
    failure_step = as_int(status.get("failing_optimizer_step"))
    if failure_value is not None and (best_value is None or failure_value > best_value):
        best_value = failure_value
        best_step = failure_step
    return best_value, best_step


def run_row(seed: int) -> dict[str, Any]:
    run_dir = RUNS[seed]
    ensure_selection_artifacts(run_dir)
    status = read_json(run_dir / "run_status.json")
    status_value = status.get("status") or ("not_run" if not run_dir.exists() else "missing_status")
    dense_rows = grid_rows(run_dir, FULL_GRID, source_name="dense_validation_results.csv")
    common_rows = grid_rows(run_dir, COMMON_GRID, source_name="common_grid_validation_results.csv")
    if not common_rows:
        common_rows = grid_rows(run_dir, COMMON_GRID, source_name="dense_validation_results.csv")
    step0 = as_float(rows_by_step(dense_rows or common_rows).get(0, {}).get("validation_ap"))
    common_raw_step, common_raw_ap = best(common_rows)
    full_raw_step, full_raw_ap = best(dense_rows)
    common_selection = stable_from_rows(common_rows)
    full_selection = read_json(run_dir / "stable_checkpoint_selection.json") or stable_from_rows(dense_rows)
    max_grad, max_grad_step = gradient_summary(run_dir)
    comparable = (
        status_value == "completed"
        and as_int(status.get("final_optimizer_step")) == 512
        and step0 is not None
        and abs(step0 - STEP0_EXPECTED) <= 1e-6
        and len({as_int(row.get("global_optimizer_step")) for row in common_rows}) == len(COMMON_GRID)
    )
    selected_ap = as_float(common_selection.get("stable_selected_ap"))
    return {
        "Seed": seed,
        "Status": status_value,
        "Final step": as_int(status.get("final_optimizer_step") or status.get("failing_optimizer_step")),
        "Step-0 AP": step0,
        "Common-grid raw-best AP": common_raw_ap,
        "Raw step": common_raw_step,
        "Common-grid selected AP": selected_ap,
        "Selected step": as_int(common_selection.get("stable_selected_step")),
        "Selection status": common_selection.get("stable_selection_status"),
        "Full-grid raw-best AP": full_raw_ap,
        "Full-grid raw-best step": full_raw_step,
        "Full-grid stable AP": as_float(full_selection.get("stable_selected_ap")),
        "Full-grid stable step": as_int(full_selection.get("stable_selected_step")),
        "Maximum gradient norm": max_grad,
        "Maximum gradient norm step": max_grad_step,
        "Delta over Step 0": selected_ap - step0 if selected_ap is not None and step0 is not None else None,
        "Comparable": comparable,
        "Run path": str(run_dir.relative_to(PROJECT_ROOT)),
        "Failure reason": status.get("reason") if status_value == "failed_numerical_instability" else None,
    }


def sample_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_int(values: list[int]) -> float | None:
    return statistics.median(values) if values else None


def seed42_outlier(rows: list[dict[str, Any]]) -> bool | None:
    values = [row["Common-grid selected AP"] for row in rows if row["Comparable"] and row["Common-grid selected AP"] is not None]
    if len(values) < 3:
        return None
    stdev = statistics.stdev(values)
    if stdev == 0:
        return False
    seed42 = next(row["Common-grid selected AP"] for row in rows if row["Seed"] == 42)
    others = [row["Common-grid selected AP"] for row in rows if row["Seed"] != 42 and row["Comparable"]]
    return abs(seed42 - statistics.mean(others)) > 2 * stdev


def interpretation(rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    valid = [row for row in rows if row["Comparable"]]
    improved = [row for row in valid if (row["Delta over Step 0"] or 0.0) > 0.0]
    stable_improved = [
        row for row in valid
        if row["Selection status"] == "stable_plateau_found" and (row["Delta over Step 0"] or 0.0) > 0.0
    ]
    if len(valid) < 3:
        return "incomplete: wait for all three comparable seed runs before interpreting reproducibility."
    if len(stable_improved) >= 2:
        return "strong robustness evidence: at least two seeds found a stable plateau above Step 0."
    if len(improved) >= 2 and stats["mean_improvement_over_step0"] and stats["mean_improvement_over_step0"] > 0:
        if stats["std_improvement_over_step0"] and stats["std_improvement_over_step0"] > stats["mean_improvement_over_step0"]:
            return "inconclusive: at least two seeds improve, but variance exceeds the mean improvement."
        if stats["fallback_isolated_raw_max_count"] >= 2:
            return "promising but not fully robust: at least two seeds improve, but most selections are isolated fallback maxima."
        return "reproducible improvement over Step 0 under the configured interpretation rules."
    return "inconclusive: improvement is not consistent across seeds."


def write_reports(rows: list[dict[str, Any]], stats: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    fields = [
        "Seed", "Status", "Final step", "Step-0 AP", "Common-grid raw-best AP", "Raw step",
        "Common-grid selected AP", "Selected step", "Selection status", "Full-grid raw-best AP",
        "Full-grid stable AP", "Maximum gradient norm", "Delta over Step 0",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    json_path.write_text(json.dumps({"rows": rows, "statistics": stats}, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Supervised LR 3e-5 Seed Reproducibility",
        "",
        "| Seed | Status | Final step | Step-0 AP | Common-grid raw-best AP | Raw step | Common-grid selected AP | Selected step | Selection status | Full-grid raw-best AP | Full-grid stable AP | Maximum gradient norm | Delta over Step 0 |",
        "| ---: | ------ | ---------: | --------: | ----------------------: | -------: | ----------------------: | ------------: | ---------------- | --------------------: | ------------------: | --------------------: | ----------------: |",
    ]
    for row in rows:
        value = {key: "" if row.get(key) is None else row.get(key) for key in fields}
        lines.append(
            f"| {value['Seed']} | {value['Status']} | {value['Final step']} | {value['Step-0 AP']} | "
            f"{value['Common-grid raw-best AP']} | {value['Raw step']} | {value['Common-grid selected AP']} | "
            f"{value['Selected step']} | {value['Selection status']} | {value['Full-grid raw-best AP']} | "
            f"{value['Full-grid stable AP']} | {value['Maximum gradient norm']} | {value['Delta over Step 0']} |"
        )
    lines.extend([
        "",
        "## Aggregate Statistics",
        "",
    ])
    for key, value in stats.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "Sample standard deviation is used for aggregate standard deviations when at least two comparable seeds are available.",
        "",
        "## Interpretation",
        "",
        str(stats["interpretation"]),
        "",
        "Final-test evaluation and threshold calibration remain blocked until this report has three comparable seed rows and is interpreted.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the supervised lr=3e-5 seed reproducibility report.")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = [run_row(seed) for seed in SEEDS]
    valid = [row for row in rows if row["Comparable"]]
    invalid = [row for row in rows if not row["Comparable"]]
    selected_values = [row["Common-grid selected AP"] for row in valid if row["Common-grid selected AP"] is not None]
    raw_values = [row["Common-grid raw-best AP"] for row in valid if row["Common-grid raw-best AP"] is not None]
    improvements = [row["Delta over Step 0"] for row in valid if row["Delta over Step 0"] is not None]
    selected_steps = [row["Selected step"] for row in valid if row["Selected step"] is not None]
    gradients = [row["Maximum gradient norm"] for row in valid if row["Maximum gradient norm"] is not None]
    stats = {
        "valid_completed_seed_count": len(valid),
        "failed_or_incomplete_seed_count": len(invalid),
        "mean_common_grid_raw_best_ap": mean(raw_values),
        "std_common_grid_raw_best_ap": sample_stdev(raw_values),
        "min_common_grid_raw_best_ap": min(raw_values) if raw_values else None,
        "max_common_grid_raw_best_ap": max(raw_values) if raw_values else None,
        "mean_common_grid_selected_ap": mean(selected_values),
        "std_common_grid_selected_ap": sample_stdev(selected_values),
        "min_common_grid_selected_ap": min(selected_values) if selected_values else None,
        "max_common_grid_selected_ap": max(selected_values) if selected_values else None,
        "mean_improvement_over_step0": mean(improvements),
        "std_improvement_over_step0": sample_stdev(improvements),
        "min_improvement_over_step0": min(improvements) if improvements else None,
        "max_improvement_over_step0": max(improvements) if improvements else None,
        "seeds_above_step0_count": sum(1 for value in improvements if value > 0.0),
        "seeds_above_step0_proportion": (sum(1 for value in improvements if value > 0.0) / len(improvements)) if improvements else None,
        "stable_plateau_found_count": sum(1 for row in valid if row["Selection status"] == "stable_plateau_found"),
        "fallback_isolated_raw_max_count": sum(1 for row in valid if row["Selection status"] == "fallback_isolated_raw_max"),
        "median_selected_optimizer_step": median_int(selected_steps),
        "selected_optimizer_step_range": [min(selected_steps), max(selected_steps)] if selected_steps else None,
        "mean_gradient_norm": mean(gradients),
        "maximum_gradient_norm": max(gradients) if gradients else None,
        "seed42_appears_outlier": seed42_outlier(rows),
    }
    stats["interpretation"] = interpretation(rows, stats)
    if invalid and not args.allow_incomplete:
        write_reports(rows, stats, REPORT_DIR / "supervised_lr3e-5_seed_reproducibility")
        raise SystemExit("Reproducibility report has incomplete or incomparable runs; rerun with --allow-incomplete to only write the partial report.")
    write_reports(rows, stats, REPORT_DIR / "supervised_lr3e-5_seed_reproducibility")
    print((REPORT_DIR / "supervised_lr3e-5_seed_reproducibility.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
