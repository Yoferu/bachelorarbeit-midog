from __future__ import annotations

from src.distillation.config import AggressiveLRSafetyStopConfig, EarlyStoppingConfig
from src.distillation.early_stopping import (
    EarlyStoppingState,
    record_validation,
    should_aggressive_lr_safety_stop,
)


def early_config() -> EarlyStoppingConfig:
    return EarlyStoppingConfig(enabled=True, min_epochs=1, validation_interval_steps=10)


def safety_config() -> AggressiveLRSafetyStopConfig:
    return AggressiveLRSafetyStopConfig(
        enabled=True,
        baseline_step=0,
        min_post_baseline_validations=5,
        min_delta=0.0001,
        degradation_margin=0.0005,
        trend_window=5,
    )


def add_point(state: EarlyStoppingState, step: int, ap: float) -> EarlyStoppingState:
    state, _ = record_validation(
        state,
        validation_ap=ap,
        checkpoint=f"student_step_{step:06d}.pt",
        optimizer_step=step,
        completed_epoch=0,
        epoch_fraction=step / 512,
        config=early_config(),
    )
    return state


def test_aggressive_safety_stop_triggers_on_early_consistent_degradation() -> None:
    state = EarlyStoppingState()
    for step, ap in [(0, 0.8666), (10, 0.8659), (20, 0.8658), (30, 0.8657), (40, 0.8656), (50, 0.8655)]:
        state = add_point(state, step, ap)

    should_stop, reason = should_aggressive_lr_safety_stop(state, config=safety_config())

    assert should_stop is True
    assert reason is not None
    assert "aggressive_lr_safety_stop" in reason


def test_aggressive_safety_stop_does_not_trigger_after_meaningful_improvement() -> None:
    state = EarlyStoppingState()
    for step, ap in [(0, 0.8666), (10, 0.8668), (20, 0.8658), (30, 0.8657), (40, 0.8656), (50, 0.8655)]:
        state = add_point(state, step, ap)

    should_stop, reason = should_aggressive_lr_safety_stop(state, config=safety_config())

    assert should_stop is False
    assert reason is None


def test_aggressive_safety_stop_requires_non_improving_recent_trend() -> None:
    state = EarlyStoppingState()
    for step, ap in [(0, 0.8666), (10, 0.8655), (20, 0.8654), (30, 0.8653), (40, 0.8652), (50, 0.8657)]:
        state = add_point(state, step, ap)

    should_stop, _ = should_aggressive_lr_safety_stop(state, config=safety_config())

    assert should_stop is False
