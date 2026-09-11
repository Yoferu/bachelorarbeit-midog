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

from src.distillation.config import DistillationConfig

BASE = PROJECT_ROOT / "experiments/distillation/pruned60_frozen_backbone_recovery"
REPORT_DIR = BASE / "reports"
HISTORICAL_DENSE = PROJECT_ROOT / "experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/runs"
HISTORICAL_DISTILL = PROJECT_ROOT / "experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation_clean"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def validation_ap(row: dict[str, Any]) -> float | None:
    return as_float(row.get("validation_ap"))


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [row for row in rows if validation_ap(row) is not None and row.get("inference_status", "ok") == "ok"]
    if not ok:
        return None
    return max(
        ok,
        key=lambda row: (
            validation_ap(row) or float("-inf"),
            -(as_int(row.get("global_optimizer_step")) or 10**12),
        ),
    )


def step0_ap(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if row.get("checkpoint_label") == "step_000000" or as_int(row.get("global_optimizer_step")) == 0:
            ap = validation_ap(row)
            if ap is not None:
                return ap
    return None


def run_status(run_dir: Path, config: DistillationConfig) -> tuple[str, str]:
    status = read_json(run_dir / "run_status.json")
    if status:
        return str(status.get("status", "unknown")), str(status.get("reason", ""))
    if (run_dir / "student_final.pt").exists():
        return "completed", "legacy_student_final"
    return ("not_started" if not run_dir.exists() else "interrupted"), ""


def epoch_bucket(epoch_fraction: float | None, final_step: int | None, best_step: int | None) -> str:
    if epoch_fraction is None:
        return ""
    final_marker = " at final available checkpoint" if final_step is not None and best_step == final_step else ""
    if epoch_fraction < 1.0:
        return "before the end of epoch 1" + final_marker
    if epoch_fraction < 2.0:
        return "during epoch 2" + final_marker
    if epoch_fraction <= 3.0:
        return "during epoch 3" + final_marker
    return "after epoch 3" + final_marker


def last_points_meaningfully_improve(rows: list[dict[str, Any]], min_delta: float = 0.0001) -> bool:
    ok = [row for row in rows if validation_ap(row) is not None and row.get("inference_status", "ok") == "ok"]
    ok = sorted(ok, key=lambda row: as_int(row.get("global_optimizer_step")) or 0)
    if len(ok) < 3:
        return False
    return (validation_ap(ok[-1]) or 0.0) > (validation_ap(ok[-3]) or 0.0) + min_delta


def evaluations_after_best(rows: list[dict[str, Any]], best: dict[str, Any] | None) -> int | None:
    if not best:
        return None
    best_step = as_int(best.get("global_optimizer_step"))
    if best_step is None:
        return None
    return sum(
        1
        for row in rows
        if validation_ap(row) is not None and (as_int(row.get("global_optimizer_step")) or -1) > best_step
    )


def earliest_degradation(rows: list[dict[str, Any]], reference: float | None) -> str:
    if reference is None:
        return ""
    sorted_rows = sorted(rows, key=lambda row: as_int(row.get("global_optimizer_step")) or 10**12)
    for row in sorted_rows:
        step = as_int(row.get("global_optimizer_step"))
        ap = validation_ap(row)
        if step is not None and step > 0 and ap is not None and ap < reference:
            return str(step)
    return ""


def config_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted((BASE / "configs").glob("*.yaml")):
        config = DistillationConfig.load(config_path)
        run_dir = Path(config.output_dir) / config.experiment_name
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not (run_dir / "dense_validation_results.csv").exists():
            timestamped = sorted((PROJECT_ROOT / config.output_dir).glob(f"{config.experiment_name}_20*"))
            timestamped = [path for path in timestamped if (path / "dense_validation_results.csv").exists() or path.exists()]
            if timestamped:
                run_dir = timestamped[-1]
        results = read_csv(run_dir / "dense_validation_results.csv")
        stable = read_json(run_dir / "stable_checkpoint_selection.json")
        audit = read_json(run_dir / "parameter_freezing_audit.json")
        status, reason = run_status(run_dir, config)
        early = read_json(run_dir / "early_stopping.json")
        best = best_row(results)
        step0 = step0_ap(results)
        best_ap = validation_ap(best) if best else None
        final_step = max((as_int(row.get("global_optimizer_step")) or 0 for row in results), default=None) if results else None
        absolute_delta = best_ap - step0 if best_ap is not None and step0 is not None else None
        relative_delta = absolute_delta / step0 if absolute_delta is not None and step0 not in {None, 0.0} else None
        method = "distillation" if config.distillation.enabled else "supervised"
        best_step = as_int(best.get("global_optimizer_step")) if best else None
        best_fraction = as_float(best.get("fractional_epoch")) if best else None
        rows.append(
            {
                "Method": method,
                "Backbone frozen": str(config.training.freeze_backbone),
                "Learning rate": config.training.lr,
                "LR": config.training.lr,
                "Max epochs": config.training.epochs,
                "Stop status": status,
                "Final step": final_step,
                "Best step": best_step,
                "Best epoch fraction": best_fraction,
                "Validation AP": best_ap,
                "Best AP": best_ap,
                "Raw-best step": best_step,
                "Raw-best AP": best_ap,
                "Raw-best checkpoint path": best.get("checkpoint", "") if best else "",
                "Stable-selected step": as_int(stable.get("stable_selected_step")),
                "Stable-selected AP": as_float(stable.get("stable_selected_ap")),
                "Stable-selected checkpoint path": stable.get("stable_selected_checkpoint", ""),
                "Selection status": stable.get("stable_selection_status", "not_available" if not stable else ""),
                "Previous support step": as_int(stable.get("previous_support_step")),
                "Next support step": as_int(stable.get("next_support_step")),
                "Raw-to-stable AP delta": as_float(stable.get("raw_to_stable_ap_delta")),
                "Stable recommendation": stable.get("recommendation", ""),
                "Step-0 AP": step0,
                "Absolute delta": absolute_delta,
                "Relative delta": relative_delta,
                "Early-stopping reason": early.get("stopping_reason") or reason,
                "Best checkpoint timing": epoch_bucket(best_fraction, final_step, best_step),
                "Improving near end": str(last_points_meaningfully_improve(results)),
                "More epoch justified": str(best_fraction is not None and best_fraction >= 2.8 and last_points_meaningfully_improve(results)),
                "Evaluations after best": evaluations_after_best(results, best),
                "Trainable parameters": audit.get("student_trainable_parameters", ""),
                "Status": "evaluated" if best else ("trained_not_evaluated" if run_dir.exists() else "not_run"),
                "Run path": str(run_dir.relative_to(PROJECT_ROOT)),
                "Config path": str(config_path.relative_to(PROJECT_ROOT)),
                "Best checkpoint path": best.get("checkpoint", "") if best else "",
                "Earliest degradation step": earliest_degradation(results, step0),
            }
        )
    return rows


def historical_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(HISTORICAL_DENSE.glob("fcos18_pruned60_supervised_clean_dense_lr*_seed42")):
        results = read_csv(path / "dense_validation_results.csv")
        best = best_row(results)
        step0 = step0_ap(results)
        best_ap = validation_ap(best) if best else None
        lr = path.name.split("_lr", 1)[-1].split("_seed", 1)[0]
        rows.append(
            {
                "Method": "historical_unfrozen_supervised",
                "Backbone frozen": "False",
                "Learning rate": lr,
                "Best step": as_int(best.get("global_optimizer_step")) if best else None,
                "Best epoch fraction": as_float(best.get("fractional_epoch")) if best else None,
                "Validation AP": best_ap,
                "Step-0 AP": step0,
                "Absolute delta": best_ap - step0 if best_ap is not None and step0 is not None else None,
                "Relative delta": (best_ap - step0) / step0 if best_ap is not None and step0 not in {None, 0.0} else None,
                "Trainable parameters": "",
                "Status": "historical_evaluated" if best else "historical_missing",
                "Run path": str(path.relative_to(PROJECT_ROOT)),
                "Config path": "",
                "Best checkpoint path": best.get("checkpoint", "") if best else "",
                "Earliest degradation step": earliest_degradation(results, step0),
            }
        )
    selection = read_json(HISTORICAL_DISTILL / "best_epoch_selection.json")
    if selection:
        rows.append(
            {
                "Method": "historical_unfrozen_distillation",
                "Backbone frozen": "False",
                "Learning rate": "1e-4",
                "Best step": "",
                "Best epoch fraction": selection.get("best_ap_epoch"),
                "Validation AP": as_float(selection.get("best_ap")),
                "Step-0 AP": "",
                "Absolute delta": "",
                "Relative delta": "",
                "Trainable parameters": "",
                "Status": "historical_epoch_evaluated",
                "Run path": str(HISTORICAL_DISTILL.relative_to(PROJECT_ROOT)),
                "Config path": "experiments/distillation/configs/fcos_x101_to_fcos18_pruned60_distillation_clean.yaml",
                "Best checkpoint path": selection.get("best_ap_source_checkpoint", ""),
                "Earliest degradation step": "",
            }
        )
    return rows


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "Method",
        "LR",
        "Raw-best step",
        "Raw-best AP",
        "Stable-selected step",
        "Stable-selected AP",
        "Selection status",
        "Previous support step",
        "Next support step",
        "Raw-to-stable AP delta",
    ]
    lines = ["# Frozen-Backbone Recovery Comparison", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(field)) for field in fields) + " |")
    lines.extend(["", "## Decisions", ""])
    lines.extend(recommendations(rows))
    lines.extend(["", "## Paths", ""])
    for row in rows:
        lines.append(f"- {row['Method']} lr={row['Learning rate']}: {row.get('Run path', '')}")
        if row.get("Best checkpoint path"):
            lines.append(f"  - best checkpoint: {row['Best checkpoint path']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_for(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("Method") == method and str(row.get("Backbone frozen")) == "True" and as_float(row.get("Stable-selected AP")) is not None
    ]
    return max(candidates, key=lambda row: as_float(row.get("Stable-selected AP")) or float("-inf")) if candidates else None


def recommendations(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    best_supervised = best_for(rows, "supervised")
    best_distill = best_for(rows, "distillation")
    if best_supervised:
        lines.append(
            f"- Best frozen supervised stable checkpoint: lr={best_supervised.get('Learning rate')} "
            f"step={best_supervised.get('Stable-selected step')} AP={format_cell(best_supervised.get('Stable-selected AP'))}."
        )
    if best_distill:
        lines.append(
            f"- Best frozen distillation stable checkpoint: lr={best_distill.get('Learning rate')} "
            f"step={best_distill.get('Stable-selected step')} AP={format_cell(best_distill.get('Stable-selected AP'))}."
        )
    by_key = {(row.get("Method"), str(row.get("Learning rate"))): row for row in rows}
    sup_1e6 = by_key.get(("supervised", "1e-06")) or by_key.get(("supervised", "1e-06"))
    dist_1e6 = by_key.get(("distillation", "1e-06")) or by_key.get(("distillation", "1e-06"))
    if sup_1e6 and dist_1e6:
        sup_ap = as_float(sup_1e6.get("Best AP"))
        dist_ap = as_float(dist_1e6.get("Best AP"))
        sup_stable = as_float(sup_1e6.get("Stable-selected AP"))
        dist_stable = as_float(dist_1e6.get("Stable-selected AP"))
        if sup_stable is not None and dist_stable is not None:
            stable_diff = dist_stable - sup_stable
            label = "practically equivalent" if abs(stable_diff) < 0.0005 else ("distillation higher" if stable_diff > 0 else "supervised higher")
            lines.append(f"- 1e-6 stable distillation minus supervised AP: {stable_diff:.10f} ({label}).")
            if abs(stable_diff) < 0.0005:
                lines.append("- Stable-selected AP is practically equivalent; prefer supervised fine-tuning unless another measured factor contradicts it.")
        if sup_ap is not None and dist_ap is not None:
            diff = dist_ap - sup_ap
            label = "practically equivalent" if abs(diff) < 0.0005 else ("distillation higher" if diff > 0 else "supervised higher")
            lines.append(f"- 1e-6 raw-best distillation minus supervised AP: {diff:.10f} ({label}).")
            if abs(diff) < 0.0005:
                lines.append("- Distillation does not currently justify its additional computational and implementation cost.")
    isolated = [
        row
        for row in rows
        if row.get("Selection status") == "fallback_isolated_raw_max" and str(row.get("Backbone frozen")) == "True"
    ]
    for row in isolated:
        lines.append(
            f"- {row.get('Method')} lr={row.get('Learning rate')} has no supported three-checkpoint plateau; "
            "treat its stable selection as a fallback and gather more evidence before final deployment evaluation."
        )
    best_1e5 = [
        row
        for row in rows
        if str(row.get("Backbone frozen")) == "True" and str(row.get("Learning rate")) == "1e-05" and as_float(row.get("Stable-selected AP")) is not None
    ]
    best_1e6 = [
        row
        for row in rows
        if str(row.get("Backbone frozen")) == "True" and str(row.get("Learning rate")) == "1e-06" and as_float(row.get("Stable-selected AP")) is not None
    ]
    if best_1e5 and best_1e6:
        if max(as_float(row.get("Stable-selected AP")) or 0.0 for row in best_1e6) <= max(as_float(row.get("Stable-selected AP")) or 0.0 for row in best_1e5):
            lines.append("- Neither 1e-6 stable-selected result exceeds the best 1e-5 stable-selected result; retain the best supported 1e-5 checkpoint.")
    for row in best_1e6:
        timing = str(row.get("Best checkpoint timing", ""))
        improving = row.get("Improving near end") == "True"
        if "before the end of epoch 1" in timing or "during epoch 2" in timing and not improving:
            lines.append(f"- {row.get('Method')} 1e-6: use the stable-selected checkpoint for downstream evaluation; more epochs require separate scientific justification.")
        elif row.get("More epoch justified") == "True":
            lines.append(f"- {row.get('Method')} 1e-6: one additional controlled epoch may be justified, but it should not be launched automatically.")
    return lines or ["- No evaluated 1e-6 results are available yet."]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = config_rows() + historical_rows()
    fields = [
        "Method",
        "Backbone frozen",
        "Learning rate",
        "LR",
        "Max epochs",
        "Stop status",
        "Final step",
        "Best step",
        "Best epoch fraction",
        "Validation AP",
        "Best AP",
        "Raw-best step",
        "Raw-best AP",
        "Raw-best checkpoint path",
        "Stable-selected step",
        "Stable-selected AP",
        "Stable-selected checkpoint path",
        "Selection status",
        "Previous support step",
        "Next support step",
        "Raw-to-stable AP delta",
        "Stable recommendation",
        "Step-0 AP",
        "Absolute delta",
        "Relative delta",
        "Early-stopping reason",
        "Best checkpoint timing",
        "Improving near end",
        "More epoch justified",
        "Evaluations after best",
        "Trainable parameters",
        "Status",
        "Run path",
        "Config path",
        "Best checkpoint path",
        "Earliest degradation step",
    ]
    csv_path = REPORT_DIR / "frozen_backbone_recovery_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_DIR / "frozen_backbone_recovery_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, REPORT_DIR / "frozen_backbone_recovery_summary.md")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
