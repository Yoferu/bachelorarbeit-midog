from __future__ import annotations

import csv
import json

import torch

from scripts.repair_pruned60_lr1e6_early_stopping import repair


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_faulty_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment_name: supervised_lr1e-6_seed42
output_dir: runs
teacher: null
student:
  architecture: fcos18_depgraph_fpn_head_60pct
data:
  train_csv: train.csv
training:
  epochs: 3
  early_stopping:
    enabled: true
    min_epochs: 1
    patience_evaluations: 5
    min_delta: 0.0001
distillation:
  enabled: false
""",
        encoding="utf-8",
    )
    torch.save(
        {
            "student_state_dict": {"w": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {"x": 1}},
            "epoch_index": 1,
            "next_batch_idx": 0,
            "global_step": 512,
            "global_optimizer_step": 512,
            "fractional_epoch": 1.0,
        },
        run_dir / "training_state_latest.pt",
    )
    history = [
        {"validation_ap": 0.8665, "checkpoint": str(run_dir / "student_step_000000.pt"), "optimizer_step": 0, "completed_epoch": 0, "epoch_fraction": 0.0},
        {"validation_ap": 0.8686810731887817, "checkpoint": str(run_dir / "student_step_000450.pt"), "optimizer_step": 450, "completed_epoch": 0, "epoch_fraction": 0.87890625},
        {"validation_ap": 0.8676, "checkpoint": str(run_dir / "student_epoch_1.pt"), "optimizer_step": 512, "completed_epoch": 1, "epoch_fraction": 1.0},
    ]
    write_json(
        run_dir / "early_stopping.json",
        {
            "stopped": True,
            "final_optimizer_step": 512,
            "final_epoch_fraction": 1.0,
            "non_improving_evaluations": 5,
            "history": history,
        },
    )
    write_json(run_dir / "run_status.json", {"status": "early_stopped", "final_optimizer_step": 512})
    write_json(run_dir / "run_summary.json", {"status": "early_stopped", "final_optimizer_step": 512})
    write_json(
        run_dir / "dense_best_checkpoint.json",
        {
            "best_ap": "0.8686810731887817",
            "best_ap_global_optimizer_step": 450,
            "best_ap_fractional_epoch": 0.87890625,
            "best_ap_source_checkpoint": str(run_dir / "student_step_000450.pt"),
        },
    )
    with (run_dir / "dense_validation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint_label", "global_optimizer_step", "fractional_epoch", "checkpoint", "validation_ap", "inference_status"])
        writer.writeheader()
    return run_dir, config_path


def test_dry_run_repair_does_not_modify_files(tmp_path) -> None:
    run_dir, config_path = make_faulty_run(tmp_path)
    before = (run_dir / "run_status.json").read_text(encoding="utf-8")
    report = repair(run_dir, config_path, expected_step=512, dry_run=True)
    assert report["new_status"] == "interrupted"
    assert report["new_patience"] == 0
    assert report["best_step"] == 450
    assert (run_dir / "run_status.json").read_text(encoding="utf-8") == before


def test_repair_writes_resumable_state_and_backups(tmp_path) -> None:
    run_dir, config_path = make_faulty_run(tmp_path)
    report = repair(run_dir, config_path, expected_step=512, dry_run=False)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    early = json.loads((run_dir / "early_stopping.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    assert early["phase"] == "active"
    assert early["warmup_completed"] is True
    assert early["patience_counter"] == 0
    assert early["best_validation_ap"] == 0.8686810731887817
    assert early["best_step"] == 450
    assert report["next_epoch_index"] == 1
    assert report["next_batch_idx"] == 0
    assert any(run_dir.glob("early_stopping.json.faulty_warmup_stop_*.bak"))
