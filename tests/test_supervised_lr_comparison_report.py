from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def load_report_module():
    path = Path("experiments/distillation/pruned60_frozen_backbone_recovery/scripts/generate_supervised_lr_comparison_report.py")
    spec = importlib.util.spec_from_file_location("generate_supervised_lr_comparison_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(
    run_dir: Path,
    points: list[tuple[int, float]],
    *,
    stable_step: int,
    status: str = "completed",
    diagnostic_points: list[tuple[int, float]] | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    fields = ["checkpoint_label", "global_optimizer_step", "fractional_epoch", "checkpoint", "validation_ap", "evaluation_runtime_s", "inference_status"]
    with (run_dir / "common_grid_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, ap in points:
            ckpt = run_dir / f"student_step_{step:06d}.pt"
            ckpt.write_bytes(b"checkpoint")
            writer.writerow(
                {
                    "checkpoint_label": f"step_{step:06d}",
                    "global_optimizer_step": step,
                    "fractional_epoch": step / 512,
                    "checkpoint": str(ckpt),
                    "validation_ap": ap,
                    "evaluation_runtime_s": 1.0,
                    "inference_status": "ok",
                }
            )
    with (run_dir / "dense_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, ap in points:
            ckpt = run_dir / f"student_step_{step:06d}.pt"
            writer.writerow(
                {
                    "checkpoint_label": f"step_{step:06d}",
                    "global_optimizer_step": step,
                    "fractional_epoch": step / 512,
                    "checkpoint": str(ckpt),
                    "validation_ap": ap,
                    "evaluation_runtime_s": 1.0,
                    "inference_status": "ok",
                }
            )
    if diagnostic_points:
        with (run_dir / "diagnostic_early_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step, ap in diagnostic_points:
                ckpt = run_dir / f"student_step_{step:06d}.pt"
                ckpt.write_bytes(b"checkpoint")
                writer.writerow(
                    {
                        "checkpoint_label": f"step_{step:06d}",
                        "global_optimizer_step": step,
                        "fractional_epoch": step / 512,
                        "checkpoint": str(ckpt),
                        "validation_ap": ap,
                        "evaluation_runtime_s": 1.0,
                        "inference_status": "ok",
                    }
                )
    raw_step, raw_ap = max(points, key=lambda item: item[1])
    stable_ap = dict(points)[stable_step]
    (run_dir / "stable_checkpoint_selection.json").write_text(
        json.dumps(
            {
                "stable_selection_status": "stable_plateau_found",
                "stable_selected_step": stable_step,
                "stable_selected_ap": stable_ap,
                "stable_selected_checkpoint": str(run_dir / f"student_step_{stable_step:06d}.pt"),
                "raw_best_step": raw_step,
                "raw_best_ap": raw_ap,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(json.dumps({"status": status, "final_optimizer_step": points[-1][0]}), encoding="utf-8")


def test_report_generation_with_fixture_runs(tmp_path, monkeypatch) -> None:
    module = load_report_module()
    base = tmp_path / "exp"
    report_dir = base / "reports"
    runs = {
        "1e-6": base / "runs/supervised_lr1e-6_seed42",
        "3e-6": base / "runs/supervised_lr3e-6_seed42",
        "1e-5": base / "runs/supervised_lr1e-5_seed42_20260725T212156Z",
        "3e-5": base / "runs/supervised_lr3e-5_seed42",
        "1e-4": base / "runs/supervised_lr1e-4_seed42",
        "1e-3": base / "runs/supervised_lr1e-3_seed42_upper_boundary",
    }
    for index, (lr, run_dir) in enumerate(runs.items()):
        diagnostics = [(10, 0.9), (20, 0.899)] if lr == "3e-5" else None
        write_run(
            run_dir,
            [(0, 0.8666), (25, 0.8668 + index * 0.001), (50, 0.867 + index * 0.001), (512, 0.8669 + index * 0.001)],
            stable_step=50,
            diagnostic_points=diagnostics,
        )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(module, "RUNS", runs)
    monkeypatch.setattr(
        module,
        "RUNS",
        {lr: {"role": "exploratory_upper_boundary" if lr == "1e-3" else "controlled_candidate", "run_dir": run_dir} for lr, run_dir in runs.items()},
    )

    module.main()

    assert (report_dir / "supervised_learning_rate_comparison.csv").exists()
    payload = json.loads((report_dir / "supervised_learning_rate_comparison.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 6
    assert payload["shared_optimizer_step_grid"] == [0, 25, 50, 512]
    row_3e5 = next(row for row in payload["rows"] if row["LR"] == "3e-5")
    assert row_3e5["Diagnostic-only validations"] == 2
    assert row_3e5["Raw-best step"] == 50
    assert row_3e5["Diagnostic best step"] == 10
    row_1e3 = next(row for row in payload["rows"] if row["LR"] == "1e-3")
    assert row_1e3["Role"] == "exploratory_upper_boundary"
