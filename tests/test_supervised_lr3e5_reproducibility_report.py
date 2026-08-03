from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("experiments/distillation/pruned60_frozen_backbone_recovery/scripts/generate_supervised_lr3e-5_reproducibility_report.py")
    spec = importlib.util.spec_from_file_location("generate_supervised_lr3e5_reproducibility_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_results(run_dir: Path, points: list[tuple[int, float]], *, status: str = "completed") -> None:
    run_dir.mkdir(parents=True)
    fields = ["checkpoint_label", "global_optimizer_step", "fractional_epoch", "checkpoint", "validation_ap", "inference_status"]
    with (run_dir / "dense_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, ap in points:
            checkpoint = run_dir / f"student_step_{step:06d}.pt"
            checkpoint.write_bytes(b"checkpoint")
            writer.writerow(
                {
                    "checkpoint_label": f"step_{step:06d}",
                    "global_optimizer_step": step,
                    "fractional_epoch": step / 512,
                    "checkpoint": str(checkpoint),
                    "validation_ap": ap,
                    "inference_status": "ok",
                }
            )
    with (run_dir / "common_grid_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, ap in points:
            if step not in {0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 512}:
                continue
            writer.writerow(
                {
                    "checkpoint_label": f"step_{step:06d}",
                    "global_optimizer_step": step,
                    "fractional_epoch": step / 512,
                    "checkpoint": str(run_dir / f"student_step_{step:06d}.pt"),
                    "validation_ap": ap,
                    "inference_status": "ok",
                }
            )
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_optimizer_step", "gradient_norm"])
        writer.writeheader()
        writer.writerow({"global_optimizer_step": 1, "gradient_norm": 12.5})
    (run_dir / "run_status.json").write_text(json.dumps({"status": status, "final_optimizer_step": 512}), encoding="utf-8")


def test_reproducibility_report_fixture(tmp_path, monkeypatch) -> None:
    module = load_module()
    base = tmp_path / "exp"
    report_dir = base / "reports"
    runs = {seed: base / f"runs/supervised_lr3e-5_seed{seed}" for seed in [42, 43, 44]}
    grid = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 512]
    for seed, run_dir in runs.items():
        points = [(step, 0.8665903211 + (0.002 if step in {50, 100, 150} else 0.0)) for step in grid]
        write_results(run_dir, points)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(module, "RUNS", runs)

    rows = [module.run_row(seed) for seed in [42, 43, 44]]
    valid = [row for row in rows if row["Comparable"]]
    assert len(valid) == 3
    assert all(row["Delta over Step 0"] > 0 for row in valid)
    assert all(row["Selection status"] == "stable_plateau_found" for row in valid)

    selected_values = [row["Common-grid selected AP"] for row in valid]
    improvements = [row["Delta over Step 0"] for row in valid]
    stats = {
        "valid_completed_seed_count": len(valid),
        "failed_or_incomplete_seed_count": 0,
        "mean_common_grid_raw_best_ap": module.mean(selected_values),
        "std_common_grid_raw_best_ap": module.sample_stdev(selected_values),
        "min_common_grid_raw_best_ap": min(selected_values),
        "max_common_grid_raw_best_ap": max(selected_values),
        "mean_common_grid_selected_ap": module.mean(selected_values),
        "std_common_grid_selected_ap": module.sample_stdev(selected_values),
        "min_common_grid_selected_ap": min(selected_values),
        "max_common_grid_selected_ap": max(selected_values),
        "mean_improvement_over_step0": module.mean(improvements),
        "std_improvement_over_step0": module.sample_stdev(improvements),
        "min_improvement_over_step0": min(improvements),
        "max_improvement_over_step0": max(improvements),
        "seeds_above_step0_count": 3,
        "seeds_above_step0_proportion": 1.0,
        "stable_plateau_found_count": 3,
        "fallback_isolated_raw_max_count": 0,
        "median_selected_optimizer_step": module.median_int([50, 50, 50]),
        "selected_optimizer_step_range": [50, 50],
        "mean_gradient_norm": 12.5,
        "maximum_gradient_norm": 12.5,
        "seed42_appears_outlier": False,
    }
    stats["interpretation"] = module.interpretation(rows, stats)
    module.write_reports(rows, stats, report_dir / "supervised_lr3e-5_seed_reproducibility")

    payload = json.loads((report_dir / "supervised_lr3e-5_seed_reproducibility.json").read_text(encoding="utf-8"))
    assert payload["statistics"]["valid_completed_seed_count"] == 3
    assert "strong robustness evidence" in payload["statistics"]["interpretation"]
