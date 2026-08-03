from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from src.distillation.stable_selection import select_stable_checkpoint, write_stable_selection


def rows(tmp_path: Path, values: list[tuple[int, float]], *, missing_steps: set[int] | None = None) -> list[dict[str, str]]:
    missing_steps = missing_steps or set()
    result = []
    for step, ap in values:
        ckpt = tmp_path / f"student_step_{step:06d}.pt"
        if step not in missing_steps:
            ckpt.write_bytes(b"checkpoint")
        result.append(
            {
                "checkpoint_label": f"step_{step:06d}",
                "global_optimizer_step": str(step),
                "fractional_epoch": str(step / 512),
                "checkpoint": str(ckpt),
                "validation_ap": str(ap),
                "inference_status": "ok",
            }
        )
    return result


def test_stable_three_checkpoint_plateau(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(0, 0.8660), (50, 0.8698), (100, 0.8700), (150, 0.8699), (200, 0.8690)]))
    assert selection["raw_best_step"] == 100
    assert selection["near_maximum_threshold"] == pytest.approx(0.8695)
    assert selection["supported_candidate_steps"] == [100]
    assert selection["stable_selected_step"] == 100
    assert selection["stable_selection_status"] == "stable_plateau_found"


def test_earliest_candidate_in_longer_plateau(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(50, 0.8697), (100, 0.8698), (150, 0.8700), (200, 0.8699), (250, 0.8698)]))
    assert selection["raw_best_step"] == 150
    assert selection["supported_candidate_steps"] == [100, 150, 200]
    assert selection["stable_selected_step"] == 100


def test_isolated_raw_maximum_falls_back(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(50, 0.8680), (100, 0.8700), (150, 0.8681)]))
    assert selection["raw_best_step"] == 100
    assert selection["stable_selected_step"] == 100
    assert selection["stable_selection_status"] == "fallback_isolated_raw_max"
    assert selection["raw_best_is_isolated"] is True


def test_boundary_maximum_falls_back(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(0, 0.8700), (50, 0.8690), (100, 0.8688)]))
    assert selection["raw_best_step"] == 0
    assert selection["stable_selection_status"] == "fallback_isolated_raw_max"


def test_earlier_near_maximum_plateau_before_isolated_maximum(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(50, 0.8695), (100, 0.8696), (150, 0.8697), (200, 0.8700), (250, 0.8685)]))
    assert selection["near_maximum_steps"] == [50, 100, 150, 200]
    assert selection["raw_best_step"] == 200
    assert selection["stable_selected_step"] == 100


def test_duplicate_optimizer_steps_do_not_create_false_support(tmp_path: Path) -> None:
    data = rows(tmp_path, [(50, 0.8680), (100, 0.8700), (150, 0.8680)])
    duplicate = dict(data[1])
    duplicate["checkpoint_label"] = "epoch_001"
    duplicate["checkpoint"] = str(tmp_path / "student_epoch_1.pt")
    Path(duplicate["checkpoint"]).write_bytes(b"checkpoint")
    data.insert(2, duplicate)
    selection = select_stable_checkpoint(data)
    assert selection["valid_checkpoint_count"] == 3
    assert selection["stable_selection_status"] == "fallback_isolated_raw_max"


def test_failed_or_nan_rows_are_excluded(tmp_path: Path) -> None:
    data = rows(tmp_path, [(50, 0.8698), (100, 0.8700), (150, 0.8699)])
    data[0]["inference_status"] = "failed:1"
    data[1]["validation_ap"] = "nan"
    selection = select_stable_checkpoint(data)
    assert selection["valid_checkpoint_count"] == 1
    assert selection["raw_best_step"] == 150
    assert selection["stable_selection_status"] == "fallback_isolated_raw_max"


def test_missing_checkpoint_file_is_reported_and_excluded(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(50, 0.8698), (100, 0.8700), (150, 0.8699)], missing_steps={100}))
    assert selection["raw_best_step"] == 150
    assert any(item["reason"] == "checkpoint_file_missing" for item in selection["excluded_checkpoints"])


def test_numeric_step_sorting(tmp_path: Path) -> None:
    selection = select_stable_checkpoint(rows(tmp_path, [(1000, 0.8699), (25, 0.8698), (100, 0.8700)]))
    assert selection["near_maximum_steps"] == [25, 100, 1000]
    assert selection["stable_selected_step"] == 100


def test_existing_result_shape_writes_stable_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data = rows(run_dir, [(50, 0.8698), (100, 0.8700), (150, 0.8699)])
    csv_path = run_dir / "dense_validation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    (run_dir / "dense_best_checkpoint.json").write_text('{"best_ap_global_optimizer_step": 100}', encoding="utf-8")
    selection = write_stable_selection(run_dir=run_dir, project_root=Path.cwd())
    assert (run_dir / "stable_checkpoint_selection.json").exists()
    assert (run_dir / "student_stable_selected.pt").exists()
    assert (run_dir / "dense_best_checkpoint.json").read_text(encoding="utf-8") == '{"best_ap_global_optimizer_step": 100}'
    assert selection["stable_selected_step"] == 100
