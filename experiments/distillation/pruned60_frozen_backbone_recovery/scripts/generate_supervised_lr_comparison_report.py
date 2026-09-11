#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.stable_selection import select_stable_checkpoint

BASE = PROJECT_ROOT / "experiments/distillation/pruned60_frozen_backbone_recovery"
REPORT_DIR = BASE / "reports"
RUNS = {
    "1e-6": {"role": "controlled_candidate", "run_dir": BASE / "runs/supervised_lr1e-6_seed42"},
    "3e-6": {"role": "controlled_candidate", "run_dir": BASE / "runs/supervised_lr3e-6_seed42"},
    "1e-5": {"role": "controlled_candidate", "run_dir": BASE / "runs/supervised_lr1e-5_seed42_20260725T212156Z"},
    "3e-5": {"role": "controlled_candidate", "run_dir": BASE / "runs/supervised_lr3e-5_seed42"},
    "5e-5": {"role": "controlled_candidate", "run_dir": BASE / "runs/supervised_lr5e-5_seed42"},
    "1e-4": {"role": "controlled_candidate", "run_dir": BASE / "runs/supervised_lr1e-4_seed42"},
    "1e-3": {"role": "exploratory_upper_boundary", "run_dir": BASE / "runs/supervised_lr1e-3_seed42_upper_boundary"},
}
PLATEAU_TOLERANCE = 0.0005


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


def valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("inference_status", "ok") == "ok"
        and as_float(row.get("validation_ap")) is not None
        and as_int(row.get("global_optimizer_step")) is not None
    ]


def rows_by_step(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for row in valid_rows(rows):
        step = as_int(row.get("global_optimizer_step"))
        if step is None:
            continue
        current = result.get(step)
        if current is None or str(row.get("checkpoint_label", "")).startswith("step_"):
            result[step] = row
    return result


def first_epoch_rows(run_dir: Path) -> list[dict[str, str]]:
    preferred = run_dir / "common_grid_validation_results.csv"
    fallback = run_dir / "dense_validation_results.csv"
    rows = read_csv(preferred if preferred.exists() else fallback)
    result = []
    for row in valid_rows(rows):
        step = as_int(row.get("global_optimizer_step"))
        if step is not None and step <= 512:
            result.append(row)
    return result


def full_rows(run_dir: Path) -> list[dict[str, str]]:
    preferred = run_dir / "dense_validation_results.csv"
    if not preferred.exists():
        preferred = run_dir / "common_grid_validation_results.csv"
    return valid_rows(read_csv(preferred))


def best(rows: list[dict[str, str]]) -> tuple[int | None, float | None]:
    if not rows:
        return None, None
    row = max(rows, key=lambda r: (as_float(r.get("validation_ap")) or -1.0, -(as_int(r.get("global_optimizer_step")) or 10**9)))
    return as_int(row.get("global_optimizer_step")), as_float(row.get("validation_ap"))


def selection(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) < 3:
        step, ap = best(rows)
        return {
            "stable_selection_status": "too_sparse_for_three_checkpoint_selection",
            "stable_selected_step": step,
            "stable_selected_ap": ap,
            "near_maximum_steps": [],
        }
    try:
        return select_stable_checkpoint(rows, plateau_tolerance=PLATEAU_TOLERANCE, project_root=PROJECT_ROOT)
    except ValueError:
        return {"stable_selection_status": "not_available", "stable_selected_step": None, "stable_selected_ap": None, "near_maximum_steps": []}


def grid_string(steps: list[int]) -> str:
    return ",".join(str(step) for step in steps)


def step_ap(rows: list[dict[str, str]], step: int) -> float | None:
    row = rows_by_step(rows).get(step)
    return as_float(row.get("validation_ap")) if row else None


def trend_after(rows: list[dict[str, str]], raw_step: int | None) -> str:
    if raw_step is None:
        return ""
    later = sorted([row for row in rows if (as_int(row.get("global_optimizer_step")) or -1) > raw_step], key=lambda r: as_int(r.get("global_optimizer_step")) or 0)
    if len(later) < 2:
        return "insufficient_later_points"
    first = as_float(later[0].get("validation_ap"))
    last = as_float(later[-1].get("validation_ap"))
    if first is None or last is None:
        return "unknown"
    if last > first + 0.0001:
        return "improving"
    if last < first - 0.0001:
        return "declining"
    return "flat"


def first_crossing(rows: list[dict[str, str]], reference: float | None, delta: float, *, direction: str) -> int | None:
    if reference is None:
        return None
    for row in sorted(rows, key=lambda r: as_int(r.get("global_optimizer_step")) or 0):
        step = as_int(row.get("global_optimizer_step"))
        ap = as_float(row.get("validation_ap"))
        if not step or ap is None:
            continue
        if direction == "above" and ap > reference + delta:
            return step
        if direction == "below" and ap < reference - delta:
            return step
    return None


def diagnostic_summary(run_dir: Path) -> dict[str, Any]:
    rows = valid_rows(read_csv(run_dir / "diagnostic_early_validation_results.csv"))
    step, ap = best(rows)
    return {
        "diagnostic_only_points": grid_string(sorted(rows_by_step(rows))),
        "diagnostic_only_validations": len(rows),
        "diagnostic_best_step": step,
        "diagnostic_best_ap": ap,
    }


def status_for(run_dir: Path) -> tuple[str, int | None, bool]:
    status = read_json(run_dir / "run_status.json")
    if status:
        return str(status.get("status", "unknown")), as_int(status.get("final_optimizer_step") or status.get("failing_optimizer_step")), status.get("status") == "failed_numerical_instability"
    if (run_dir / "student_final.pt").exists():
        return "completed", None, False
    return ("not_run" if not run_dir.exists() else "missing_status"), None, False


def numerical_summary(run_dir: Path) -> dict[str, Any]:
    status = read_json(run_dir / "run_status.json")
    training_rows = read_csv(run_dir / "training_log.csv")
    gradient_values = [
        value for value in (as_float(row.get("gradient_norm")) for row in training_rows)
        if value is not None
    ]
    status_gradient = as_float(status.get("gradient_norm"))
    if status_gradient is not None:
        gradient_values.append(status_gradient)
    return {
        "maximum_gradient_norm": max(gradient_values) if gradient_values else None,
        "failure_step": as_int(status.get("failing_optimizer_step") or status.get("failing_global_step")),
        "failure_reason": status.get("reason") if status.get("status") == "failed_numerical_instability" else None,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    first_epoch = {lr: first_epoch_rows(meta["run_dir"]) for lr, meta in RUNS.items()}
    available_step_sets = [set(rows_by_step(rows)) for rows in first_epoch.values() if rows]
    shared_steps = sorted(set.intersection(*available_step_sets)) if available_step_sets else []
    rows_out: list[dict[str, Any]] = []
    for lr, meta in RUNS.items():
        run_dir: Path = meta["run_dir"]
        first_rows = first_epoch[lr]
        first_by_step = rows_by_step(first_rows)
        common_rows = [first_by_step[step] for step in shared_steps if step in first_by_step]
        common_raw_step, common_raw_ap = best(common_rows)
        common_sel = selection(common_rows)
        full = full_rows(run_dir)
        full_raw_step, full_raw_ap = best(full)
        diagnostic = diagnostic_summary(run_dir)
        status, completed_steps, numerical_instability = status_for(run_dir)
        numerical = numerical_summary(run_dir)
        step0 = step_ap(first_rows, 0)
        first_below = first_crossing(common_rows, step0, 0.0, direction="below")
        near_count = 0 if common_raw_ap is None else sum(
            1 for row in common_rows if (as_float(row.get("validation_ap")) or -1.0) >= common_raw_ap - PLATEAU_TOLERANCE
        )
        row = {
            "LR": lr,
            "Role": meta["role"],
            "Status": status,
            "Completed steps": completed_steps,
            "Step-0 AP": step0,
            "Common-grid raw-best AP": common_raw_ap,
            "Raw-best step": common_raw_step,
            "Common-grid stable AP": common_sel.get("stable_selected_ap"),
            "Stable step": common_sel.get("stable_selected_step"),
            "Stable status": common_sel.get("stable_selection_status"),
            "First step below step 0": first_below,
            "Numerical instability": numerical_instability,
            "Maximum gradient norm": numerical["maximum_gradient_norm"],
            "Failure step": numerical["failure_step"],
            "Failure reason": numerical["failure_reason"],
            "Shared optimizer-step grid": grid_string(shared_steps),
            "Full available grid": grid_string(sorted(rows_by_step(full))),
            "Diagnostic-only points": diagnostic["diagnostic_only_points"],
            "Diagnostic-only validations": diagnostic["diagnostic_only_validations"],
            "First meaningful improvement": first_crossing(common_rows, step0, 0.0001, direction="above"),
            "First meaningful degradation": first_crossing(common_rows, step0, 0.0001, direction="below"),
            "Near-raw-max count": near_count,
            "AP trend after best": trend_after(common_rows, common_raw_step),
            "Completed common budget": step_ap(first_rows, 512) is not None,
            "Full-grid raw-best AP": full_raw_ap,
            "Full-grid raw-best step": full_raw_step,
            "Diagnostic best AP": diagnostic["diagnostic_best_ap"],
            "Diagnostic best step": diagnostic["diagnostic_best_step"],
            "comparison_phase": "exploratory_upper_boundary" if meta["role"] == "exploratory_upper_boundary" else "controlled_epoch_1_screening",
            "selection_grid": "shared_first_epoch_grid",
            "Run path": str(run_dir.relative_to(PROJECT_ROOT)),
        }
        rows_out.append(row)

    controlled = [row for row in rows_out if row["Role"] == "controlled_candidate" and row["Common-grid stable AP"] is not None]
    ranked = sorted(controlled, key=lambda row: row["Common-grid stable AP"], reverse=True)
    by_lr = {row["LR"]: row for row in rows_out}
    decision_notes: list[str] = []
    bracketed = False
    if by_lr.get("5e-5", {}).get("Status") == "completed" and by_lr.get("3e-5", {}).get("Common-grid stable AP") is not None:
        delta = by_lr["5e-5"]["Common-grid stable AP"] - by_lr["3e-5"]["Common-grid stable AP"]
        if abs(delta) < PLATEAU_TOLERANCE:
            decision_notes.append("5e-5 is practically equivalent to 3e-5 on common-grid stable AP; prefer the broader plateau and lower gradient sensitivity.")
        elif delta > 0 and by_lr["5e-5"].get("Stable status") == "stable_plateau_found":
            decision_notes.append("5e-5 is stable and better than 3e-5; recommend it for seed replication.")
        elif delta > 0:
            decision_notes.append("5e-5 improves over 3e-5 but remains isolated; test 4e-5 or 6e-5 before replication.")
        else:
            decision_notes.append("5e-5 is worse than 3e-5; retain 3e-5 for seed replication.")
    elif by_lr.get("5e-5", {}).get("Status") == "failed_numerical_instability":
        decision_notes.append("5e-5 failed numerically; the stability boundary lies between 3e-5 and 5e-5.")
    if by_lr.get("1e-4", {}).get("Common-grid stable AP") is not None and by_lr.get("3e-5", {}).get("Common-grid stable AP") is not None:
        if by_lr["1e-4"]["Common-grid stable AP"] < by_lr["3e-5"]["Common-grid stable AP"]:
            bracketed = True
            decision_notes.append("1e-4 is worse than 3e-5 on common-grid stable AP; optimum is bracketed between 1e-5 and 1e-4.")
        else:
            decision_notes.append("1e-4 is not worse than 3e-5; optimum is not yet bracketed by the controlled candidates.")
    if by_lr.get("1e-3", {}).get("Numerical instability"):
        decision_notes.append("1e-3 showed numerical instability; treat it as useful upper-bound evidence, not a deployment candidate.")
    if ranked:
        decision_notes.append(f"Primary controlled ranking leader: {ranked[0]['LR']} with stable AP {ranked[0]['Common-grid stable AP']:.10f}.")
    upper_lrs = ["3e-5", "5e-5", "1e-4"]
    upper_step_sets = [
        set(rows_by_step(first_epoch[lr]))
        for lr in upper_lrs
        if lr in first_epoch and first_epoch[lr]
    ]
    upper_shared_steps = sorted(set.intersection(*upper_step_sets)) if len(upper_step_sets) >= 2 else []
    payload = {
        "shared_optimizer_step_grid": shared_steps,
        "upper_lr_shared_completed_budget_grid": upper_shared_steps,
        "plateau_tolerance": PLATEAU_TOLERANCE,
        "ranking_basis": "common-grid stable-selected AP for controlled candidates",
        "controlled_ranked_lrs": [row["LR"] for row in ranked],
        "optimum_bracketed": bracketed,
        "intermediate_lr_justified": bool(by_lr.get("1e-4", {}).get("Common-grid stable AP") is not None),
        "decision_notes": decision_notes,
        "rows": rows_out,
    }
    columns = [
        "LR", "Role", "Status", "Completed steps", "Step-0 AP", "Common-grid raw-best AP", "Raw-best step",
        "Common-grid stable AP", "Stable step", "Stable status", "First step below step 0", "Numerical instability",
        "Maximum gradient norm", "Failure step", "Failure reason",
        "Shared optimizer-step grid", "Full available grid", "Diagnostic-only points", "Diagnostic-only validations",
        "First meaningful improvement", "First meaningful degradation", "Near-raw-max count", "AP trend after best",
        "Completed common budget", "Full-grid raw-best AP", "Full-grid raw-best step", "Diagnostic best AP",
        "Diagnostic best step", "comparison_phase", "selection_grid", "Run path",
    ]
    csv_path = REPORT_DIR / "supervised_learning_rate_comparison.csv"
    json_path = REPORT_DIR / "supervised_learning_rate_comparison.json"
    md_path = REPORT_DIR / "supervised_learning_rate_comparison.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_out)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    table_cols = columns[:14]
    lines = ["# Supervised Frozen-Backbone Learning-Rate Comparison", ""]
    lines.append(f"Shared optimizer-step grid: {grid_string(shared_steps) if shared_steps else 'not_available'}")
    lines.append(f"Upper-LR partial boundary grid for 3e-5/5e-5/1e-4 where available: {grid_string(upper_shared_steps) if upper_shared_steps else 'not_available'}")
    lines.append("")
    lines.append("| " + " | ".join(table_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(table_cols)) + " |")
    for row in rows_out:
        lines.append("| " + " | ".join(format_value(row.get(col)) for col in table_cols) + " |")
    lines.extend(["", "## Decision Notes", ""])
    lines.extend(f"- {note}" for note in decision_notes)
    lines.append("- 1e-3 is an exploratory upper-boundary run and is not an equal deployment candidate unless stability and plateau width unexpectedly justify it.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


if __name__ == "__main__":
    main()
