from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationPoint:
    row: dict[str, Any]
    step: int
    ap: float
    checkpoint: Path
    label: str
    fractional_epoch: float | None


def read_validation_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_stable_checkpoint(
    rows: list[dict[str, Any]],
    *,
    plateau_tolerance: float = 0.0005,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or Path.cwd()
    points, excluded = valid_points(rows, project_root=root)
    if not points:
        raise ValueError("No valid evaluated checkpoints are available for stable selection.")
    raw_best = max(points, key=lambda point: (point.ap, -point.step))
    raw_max_ap = raw_best.ap
    near_threshold = raw_max_ap - plateau_tolerance
    near_steps = [point.step for point in points if point.ap >= near_threshold]
    diagnostics: list[dict[str, Any]] = []
    stable_candidates: list[tuple[ValidationPoint, ValidationPoint, ValidationPoint, dict[str, float]]] = []
    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        following = points[index + 1]
        window_values = [previous.ap, current.ap, following.ap]
        diag = {
            "step": current.step,
            "validation_ap": current.ap,
            "previous_step": previous.step,
            "previous_ap": previous.ap,
            "next_step": following.step,
            "next_ap": following.ap,
            "window_mean_ap": sum(window_values) / 3,
            "window_min_ap": min(window_values),
            "window_max_ap": max(window_values),
            "window_range_ap": max(window_values) - min(window_values),
            "distance_from_raw_max": raw_max_ap - current.ap,
            "is_near_maximum": current.ap >= near_threshold,
            "has_supported_near_maximum_window": all(value >= near_threshold for value in window_values),
        }
        diagnostics.append(diag)
        if diag["has_supported_near_maximum_window"]:
            stable_candidates.append((previous, current, following, diag))

    if stable_candidates:
        previous, selected, following, diag = stable_candidates[0]
        status = "stable_plateau_found"
        no_plateau_reason = None
    else:
        previous = following = None
        selected = raw_best
        diag = {
            "window_mean_ap": None,
            "window_min_ap": None,
            "window_max_ap": None,
            "window_range_ap": None,
            "distance_from_raw_max": 0.0,
        }
        status = "fallback_isolated_raw_max"
        no_plateau_reason = "No interior three-checkpoint near-maximum plateau was found."

    selection = {
        "selection_method": "earliest_supported_near_maximum",
        "plateau_tolerance": plateau_tolerance,
        "near_maximum_threshold": near_threshold,
        "raw_best_ap": raw_best.ap,
        "raw_best_step": raw_best.step,
        "raw_best_fractional_epoch": raw_best.fractional_epoch,
        "raw_best_checkpoint": str(raw_best.checkpoint),
        "raw_best_checkpoint_label": raw_best.label,
        "raw_best_is_isolated": status == "fallback_isolated_raw_max",
        "near_maximum_steps": near_steps,
        "supported_candidate_steps": [candidate.step for _, candidate, _, _ in stable_candidates],
        "stable_selection_status": status,
        "stable_selected_ap": selected.ap,
        "stable_selected_step": selected.step,
        "stable_selected_fractional_epoch": selected.fractional_epoch,
        "stable_selected_checkpoint": str(selected.checkpoint),
        "stable_selected_checkpoint_label": selected.label,
        "previous_support_step": previous.step if previous else None,
        "previous_support_ap": previous.ap if previous else None,
        "next_support_step": following.step if following else None,
        "next_support_ap": following.ap if following else None,
        "window_mean_ap": diag["window_mean_ap"],
        "window_min_ap": diag["window_min_ap"],
        "window_max_ap": diag["window_max_ap"],
        "window_range_ap": diag["window_range_ap"],
        "distance_from_raw_max": diag["distance_from_raw_max"],
        "raw_to_stable_ap_delta": raw_best.ap - selected.ap,
        "valid_checkpoint_count": len(points),
        "excluded_checkpoint_count": len(excluded),
        "excluded_checkpoints": excluded,
        "interior_window_diagnostics": diagnostics,
        "no_stable_plateau_reason": no_plateau_reason,
        "recommendation": (
            "Proceed to calibration with the stable-selected checkpoint."
            if status == "stable_plateau_found"
            else "Treat the raw maximum as isolated; run additional validation, another seed, or a robustness check before final deployment evaluation."
        ),
    }
    return selection


def write_stable_selection(
    *,
    run_dir: Path,
    results_csv: Path | None = None,
    output_name: str = "stable_checkpoint_selection.json",
    artifact_name: str = "student_stable_selected.pt",
    plateau_tolerance: float = 0.0005,
    dry_run: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    csv_path = results_csv or run_dir / "dense_validation_results.csv"
    selection = select_stable_checkpoint(
        read_validation_csv(csv_path),
        plateau_tolerance=plateau_tolerance,
        project_root=project_root,
    )
    selection["validation_results_csv"] = str(csv_path)
    selection["stable_selected_artifact"] = str(run_dir / artifact_name)
    selection["dry_run"] = dry_run
    if not dry_run:
        (run_dir / output_name).write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
        shutil.copy2(selection["stable_selected_checkpoint"], run_dir / artifact_name)
    return selection


def valid_points(rows: list[dict[str, Any]], *, project_root: Path) -> tuple[list[ValidationPoint], list[dict[str, Any]]]:
    by_step: dict[int, ValidationPoint] = {}
    excluded: list[dict[str, Any]] = []
    for row in rows:
        point, reason = parse_point(row, project_root=project_root)
        if point is None:
            excluded.append({"checkpoint_label": row.get("checkpoint_label"), "checkpoint": row.get("checkpoint"), "reason": reason})
            continue
        existing = by_step.get(point.step)
        if existing is None or duplicate_preference_key(point) > duplicate_preference_key(existing):
            if existing is not None:
                excluded.append(
                    {
                        "checkpoint_label": existing.label,
                        "checkpoint": str(existing.checkpoint),
                        "reason": f"duplicate_step_{point.step}_lower_preference",
                    }
                )
            by_step[point.step] = point
        else:
            excluded.append(
                {
                    "checkpoint_label": point.label,
                    "checkpoint": str(point.checkpoint),
                    "reason": f"duplicate_step_{point.step}_lower_preference",
                }
            )
    return [by_step[step] for step in sorted(by_step)], excluded


def parse_point(row: dict[str, Any], *, project_root: Path) -> tuple[ValidationPoint | None, str | None]:
    if row.get("inference_status", "ok") != "ok":
        return None, f"inference_status={row.get('inference_status')}"
    try:
        ap = float(row.get("validation_ap"))
    except (TypeError, ValueError):
        return None, "validation_ap_not_numeric"
    if not math.isfinite(ap):
        return None, "validation_ap_not_finite"
    try:
        step = int(float(row.get("global_optimizer_step")))
    except (TypeError, ValueError):
        return None, "global_optimizer_step_not_numeric"
    checkpoint_raw = row.get("checkpoint")
    if not checkpoint_raw:
        return None, "checkpoint_missing"
    checkpoint = Path(str(checkpoint_raw))
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if not checkpoint.exists():
        return None, "checkpoint_file_missing"
    fractional_epoch = None
    if row.get("fractional_epoch") not in {None, ""}:
        try:
            fractional_epoch = float(row.get("fractional_epoch"))
        except (TypeError, ValueError):
            fractional_epoch = None
    return (
        ValidationPoint(
            row=row,
            step=step,
            ap=ap,
            checkpoint=checkpoint,
            label=str(row.get("checkpoint_label") or checkpoint.stem),
            fractional_epoch=fractional_epoch,
        ),
        None,
    )


def duplicate_preference_key(point: ValidationPoint) -> tuple[int, float, str]:
    is_dense_step = 1 if point.label.startswith("step_") or point.checkpoint.name.startswith("student_step_") else 0
    return (is_dense_step, point.ap, point.label)
