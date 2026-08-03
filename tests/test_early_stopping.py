from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.distillation.config import EarlyStoppingConfig
from src.distillation.early_stopping import (
    EarlyStoppingState,
    migrate_legacy_state,
    record_validation,
    record_validation_error,
    should_validate_checkpoint,
)


def config() -> EarlyStoppingConfig:
    return EarlyStoppingConfig(
        enabled=True,
        min_epochs=1,
        patience_evaluations=5,
        min_delta=0.0001,
        validation_interval_steps=50,
    )


def apply(state: EarlyStoppingState, step: int, epoch_fraction: float, ap: float) -> tuple[EarlyStoppingState, bool]:
    return record_validation(
        state,
        validation_ap=ap,
        checkpoint=f"student_step_{step:06d}.pt",
        optimizer_step=step,
        completed_epoch=int(epoch_fraction),
        epoch_fraction=epoch_fraction,
        config=config(),
    )


def test_no_patience_accumulation_during_epoch_one() -> None:
    state = EarlyStoppingState()
    for step, ap in [(0, 0.90), (50, 0.89), (100, 0.88), (150, 0.87), (200, 0.86)]:
        state, stopped = apply(state, step, step / 512, ap)
        assert stopped is False
        assert state.patience_counter == 0
        assert state.non_improving_evaluations == 0
    assert state.best_validation_ap == 0.90
    assert state.phase == "warmup"


def test_epoch_one_end_transition_preserves_earlier_best_and_zeroes_patience() -> None:
    state = EarlyStoppingState()
    for step, fraction, ap in [(0, 0.0, 0.8665), (450, 0.87890625, 0.8686810731887817), (512, 1.0, 0.8676)]:
        state, stopped = apply(state, step, fraction, ap)
        assert stopped is False
    assert state.phase == "active"
    assert state.warmup_completed is True
    assert state.patience_counter == 0
    assert state.best_validation_ap == 0.8686810731887817
    assert state.best_step == 450
    assert state.best_epoch_fraction == 0.87890625


def test_first_validation_in_epoch_two_increments_patience_on_non_improvement() -> None:
    state = EarlyStoppingState(best_validation_ap=0.8686, best_step=450, warmup_completed=True, phase="active")
    state, stopped = apply(state, 550, 1.07421875, 0.8680)
    assert stopped is False
    assert state.patience_counter == 1


def test_five_failures_after_epoch_one_trigger_stop() -> None:
    state = EarlyStoppingState(best_validation_ap=0.8686, best_step=450, warmup_completed=True, phase="active")
    stopped = False
    for index, step in enumerate([550, 600, 650, 700, 750], start=1):
        state, stopped = apply(state, step, step / 512, 0.8680)
        assert state.patience_counter == index
    assert stopped is True
    assert state.phase == "stopped"
    assert state.stopped is True


def test_improvement_resets_patience() -> None:
    state = EarlyStoppingState(best_validation_ap=0.8686, best_step=450, warmup_completed=True, phase="active")
    state, _ = apply(state, 550, 1.07421875, 0.8680)
    state, _ = apply(state, 600, 1.171875, 0.8681)
    assert state.patience_counter == 2
    state, stopped = apply(state, 650, 1.26953125, 0.8688001)
    assert stopped is False
    assert state.best_validation_ap == 0.8688001
    assert state.patience_counter == 0


def test_improvement_exactly_equal_to_min_delta_is_not_meaningful() -> None:
    state = EarlyStoppingState(best_validation_ap=0.8686, best_step=450, warmup_completed=True, phase="active")
    state, stopped = apply(state, 550, 1.07421875, 0.8687)
    assert stopped is False
    assert state.best_validation_ap == 0.8686
    assert state.patience_counter == 1


def test_earlier_best_within_epoch_one_survives_lower_epoch_end_ap() -> None:
    state = EarlyStoppingState()
    for step, fraction, ap in [
        (0, 0.0, 0.8665903210639954),
        (300, 0.5859375, 0.8686071634292603),
        (450, 0.87890625, 0.8686810731887817),
        (512, 1.0, 0.8676483631134033),
    ]:
        state, _ = apply(state, step, fraction, ap)
    assert state.best_validation_ap == 0.8686810731887817
    assert state.best_step == 450
    assert state.patience_counter == 0
    assert state.warmup_completed is True


def test_faulty_state_migration_zeroes_warmup_patience_and_preserves_best() -> None:
    raw = {
        "stopped": True,
        "final_optimizer_step": 512,
        "final_epoch_fraction": 1.0,
        "non_improving_evaluations": 5,
        "history": [
            {"validation_ap": 0.8665, "checkpoint": "student_step_000000.pt", "optimizer_step": 0, "completed_epoch": 0, "epoch_fraction": 0.0},
            {"validation_ap": 0.8686810731887817, "checkpoint": "student_step_000450.pt", "optimizer_step": 450, "completed_epoch": 0, "epoch_fraction": 0.87890625},
            {"validation_ap": 0.8676, "checkpoint": "student_epoch_1.pt", "optimizer_step": 512, "completed_epoch": 1, "epoch_fraction": 1.0},
        ],
    }
    state = migrate_legacy_state(raw, config())
    assert state.stopped is False
    assert state.phase == "active"
    assert state.warmup_completed is True
    assert state.patience_counter == 0
    assert state.best_validation_ap == 0.8686810731887817
    assert state.best_step == 450
    assert state.best_checkpoint == "student_step_000450.pt"


def test_resume_after_repair_does_not_schedule_step_512_again() -> None:
    cfg = config()
    assert should_validate_checkpoint(
        optimizer_step=512,
        fractional_epoch=1.0,
        completed_epoch=1,
        batches_per_epoch=512,
        config=cfg,
    )
    assert not should_validate_checkpoint(
        optimizer_step=513,
        fractional_epoch=1.001953125,
        completed_epoch=1,
        batches_per_epoch=512,
        config=cfg,
    )
    assert should_validate_checkpoint(
        optimizer_step=550,
        fractional_epoch=550 / 512,
        completed_epoch=1,
        batches_per_epoch=512,
        config=cfg,
    )


def test_validation_error_does_not_alter_patience() -> None:
    state = EarlyStoppingState(best_validation_ap=0.8686, best_step=450, warmup_completed=True, phase="active", patience_counter=2)
    state = record_validation_error(
        state,
        checkpoint="student_step_000550.pt",
        optimizer_step=550,
        epoch_fraction=1.07421875,
        error="validation failed",
    )
    assert state.patience_counter == 2
    assert state.non_improving_evaluations == 2
    assert state.best_validation_ap == 0.8686
    assert state.history[-1]["patience_changed"] is False


def test_non_finite_ap_fails_immediately() -> None:
    with pytest.raises(ValueError):
        apply(EarlyStoppingState(), 0, 0.0, float("nan"))


def test_resume_state_round_trip(tmp_path: Path) -> None:
    state = EarlyStoppingState(
        best_validation_ap=0.8686810731887817,
        best_step=450,
        best_epoch_fraction=0.87890625,
        phase="active",
        warmup_completed=True,
    )
    path = tmp_path / "early_stopping.json"
    state.save(path)
    loaded = EarlyStoppingState.load(path)
    assert loaded.phase == "active"
    assert loaded.warmup_completed is True
    assert loaded.best_step == 450
    assert json.loads(path.read_text(encoding="utf-8"))["patience_counter"] == 0
