from __future__ import annotations

import json

from scripts.train_fcos_distillation import terminal_run_status


def test_terminal_run_status_detects_completed_and_early_stopped(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert terminal_run_status(run_dir) == "completed"

    (run_dir / "run_status.json").write_text(json.dumps({"status": "early_stopped"}), encoding="utf-8")
    assert terminal_run_status(run_dir) == "early_stopped"


def test_terminal_run_status_preserves_legacy_student_final_marker(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "student_final.pt").write_bytes(b"")

    assert terminal_run_status(run_dir) == "completed"


def test_terminal_run_status_allows_nonterminal_status_with_student_final(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "student_final.pt").write_bytes(b"")
    (run_dir / "run_status.json").write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")

    assert terminal_run_status(run_dir) is None
