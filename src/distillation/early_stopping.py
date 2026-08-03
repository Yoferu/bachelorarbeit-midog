from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.distillation.config import AggressiveLRSafetyStopConfig, EarlyStoppingConfig
from src.distillation.logging_utils import write_json


@dataclass
class EarlyStoppingState:
    enabled: bool = True
    phase: str = "warmup"
    warmup_completed: bool = False
    min_epochs: int = 1
    patience_evaluations: int = 5
    patience_counter: int = 0
    min_delta: float = 0.0001
    best_validation_ap: float | None = None
    best_step: int | None = None
    best_epoch_fraction: float | None = None
    best_checkpoint: str | None = None
    non_improving_evaluations: int = 0
    validation_evaluations: int = 0
    evaluations_after_best: int = 0
    stopped: bool = False
    stopping_reason: str | None = None
    final_optimizer_step: int | None = None
    final_epoch_fraction: float | None = None
    last_evaluated_step: int | None = None
    last_evaluated_fractional_epoch: float | None = None
    last_evaluated_checkpoint: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "EarlyStoppingState":
        if not path.exists():
            return cls()
        raw = _read_json(path)
        if "phase" not in raw:
            return migrate_legacy_state(raw)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        state = cls(**{key: value for key, value in raw.items() if key in known})
        state.non_improving_evaluations = int(state.patience_counter)
        return state

    def save(self, path: Path) -> None:
        self.non_improving_evaluations = int(self.patience_counter)
        payload = asdict(self)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(path, payload)


def should_validate_checkpoint(
    *,
    optimizer_step: int,
    fractional_epoch: float,
    completed_epoch: int,
    batches_per_epoch: int,
    config: EarlyStoppingConfig,
) -> bool:
    if not config.enabled:
        return False
    if optimizer_step == 0:
        return True
    interval = int(config.validation_interval_steps)
    if interval > 0 and optimizer_step % interval == 0:
        return True
    expected_epoch_step = completed_epoch * batches_per_epoch
    return completed_epoch > 0 and optimizer_step == expected_epoch_step and math.isclose(
        fractional_epoch, float(completed_epoch)
    )


def record_validation(
    state: EarlyStoppingState,
    *,
    validation_ap: float,
    checkpoint: str,
    optimizer_step: int,
    completed_epoch: int,
    epoch_fraction: float,
    config: EarlyStoppingConfig,
) -> tuple[EarlyStoppingState, bool]:
    if not math.isfinite(validation_ap):
        raise ValueError(f"Validation AP must be finite, got {validation_ap!r}.")

    state.enabled = bool(config.enabled)
    state.min_epochs = int(config.min_epochs)
    state.patience_evaluations = int(config.patience_evaluations)
    state.min_delta = float(config.min_delta)
    state.last_evaluated_step = int(optimizer_step)
    state.last_evaluated_fractional_epoch = float(epoch_fraction)
    state.last_evaluated_checkpoint = checkpoint

    previous_best = state.best_validation_ap
    is_warmup_validation = float(epoch_fraction) <= float(config.min_epochs)
    patience_active = state.warmup_completed and not is_warmup_validation
    threshold = 0.0 if is_warmup_validation else float(config.min_delta)
    improved = previous_best is None or validation_ap > previous_best + threshold

    if improved:
        state.best_validation_ap = float(validation_ap)
        state.best_step = int(optimizer_step)
        state.best_epoch_fraction = float(epoch_fraction)
        state.best_checkpoint = checkpoint
        if patience_active:
            state.patience_counter = 0
        state.evaluations_after_best = 0
    else:
        if patience_active:
            state.patience_counter += 1
        state.evaluations_after_best += 1

    if is_warmup_validation:
        state.warmup_completed = math.isclose(float(epoch_fraction), float(config.min_epochs))
        state.phase = "active" if state.warmup_completed else "warmup"
        state.patience_counter = 0
    else:
        state.warmup_completed = True
        state.phase = "active"

    state.non_improving_evaluations = int(state.patience_counter)
    state.validation_evaluations += 1
    state.stopped = False
    state.stopping_reason = None
    state.final_optimizer_step = None
    state.final_epoch_fraction = None

    state.history.append(
        {
            "validation_ap": float(validation_ap),
            "checkpoint": checkpoint,
            "optimizer_step": int(optimizer_step),
            "completed_epoch": int(completed_epoch),
            "epoch_fraction": float(epoch_fraction),
            "previous_best_validation_ap": previous_best,
            "best_validation_ap": state.best_validation_ap,
            "improved": improved,
            "phase": state.phase,
            "warmup_completed": state.warmup_completed,
            "patience_active": patience_active,
            "patience_counter": state.patience_counter,
            "non_improving_evaluations": state.non_improving_evaluations,
        }
    )

    should_stop = patience_active and state.patience_counter >= int(config.patience_evaluations)
    if should_stop:
        state.stopped = True
        state.phase = "stopped"
        state.stopping_reason = (
            f"No validation AP improvement greater than {config.min_delta} for "
            f"{config.patience_evaluations} post-warm-up validation evaluations."
        )
        state.final_optimizer_step = int(optimizer_step)
        state.final_epoch_fraction = float(epoch_fraction)
    return state, should_stop


def record_validation_error(
    state: EarlyStoppingState,
    *,
    checkpoint: str,
    optimizer_step: int,
    epoch_fraction: float,
    error: str,
) -> EarlyStoppingState:
    state.history.append(
        {
            "checkpoint": checkpoint,
            "optimizer_step": int(optimizer_step),
            "epoch_fraction": float(epoch_fraction),
            "validation_error": error,
            "phase": state.phase,
            "warmup_completed": state.warmup_completed,
            "patience_counter": state.patience_counter,
            "patience_changed": False,
        }
    )
    state.non_improving_evaluations = int(state.patience_counter)
    return state


def should_aggressive_lr_safety_stop(
    state: EarlyStoppingState,
    *,
    config: AggressiveLRSafetyStopConfig,
) -> tuple[bool, str | None]:
    if not config.enabled:
        return False, None
    valid = [
        row
        for row in state.history
        if row.get("validation_ap") is not None and row.get("optimizer_step") is not None
    ]
    baseline_rows = [row for row in valid if int(row["optimizer_step"]) == int(config.baseline_step)]
    if not baseline_rows:
        return False, None
    latest = valid[-1]
    if float(latest.get("epoch_fraction", 0.0)) > float(config.max_epoch_fraction):
        return False, None
    baseline_ap = float(baseline_rows[-1]["validation_ap"])
    post_baseline = [row for row in valid if int(row["optimizer_step"]) > int(config.baseline_step)]
    if len(post_baseline) < int(config.min_post_baseline_validations):
        return False, None
    if any(float(row["validation_ap"]) > baseline_ap + float(config.min_delta) for row in post_baseline):
        return False, None
    window_size = int(config.trend_window)
    if len(post_baseline) < window_size:
        return False, None
    recent = post_baseline[-window_size:]
    if any(float(row["validation_ap"]) > baseline_ap - float(config.degradation_margin) for row in recent):
        return False, None
    first_ap = float(recent[0]["validation_ap"])
    latest_ap = float(recent[-1]["validation_ap"])
    if latest_ap > first_ap:
        return False, None
    reason = (
        f"{config.label}: after {len(post_baseline)} post-step-{config.baseline_step} validations, "
        f"no AP improved over step {config.baseline_step} by more than {config.min_delta}; "
        f"latest {window_size} AP values are at least {config.degradation_margin} below baseline "
        "with a non-improving trend."
    )
    return True, reason


def migrate_legacy_state(raw: dict[str, Any], config: EarlyStoppingConfig | None = None) -> EarlyStoppingState:
    cfg = config or EarlyStoppingConfig(enabled=True)
    state = EarlyStoppingState(
        enabled=bool(cfg.enabled),
        min_epochs=int(cfg.min_epochs),
        patience_evaluations=int(cfg.patience_evaluations),
        min_delta=float(cfg.min_delta),
    )
    for row in raw.get("history", []):
        if row.get("validation_ap") is None:
            continue
        state, _ = record_validation(
            state,
            validation_ap=float(row["validation_ap"]),
            checkpoint=str(row.get("checkpoint", "")),
            optimizer_step=int(row.get("optimizer_step", 0)),
            completed_epoch=int(row.get("completed_epoch", int(float(row.get("epoch_fraction", 0.0))))),
            epoch_fraction=float(row.get("epoch_fraction", 0.0)),
            config=cfg,
        )
    if not state.history:
        known = {field.name for field in EarlyStoppingState.__dataclass_fields__.values()}
        state = EarlyStoppingState(**{key: value for key, value in raw.items() if key in known})
    if raw.get("stopped") and float(raw.get("final_epoch_fraction") or 0.0) <= float(cfg.min_epochs):
        state.phase = "active"
        state.warmup_completed = True
        state.patience_counter = 0
        state.non_improving_evaluations = 0
        state.stopped = False
        state.stopping_reason = None
        state.final_optimizer_step = None
        state.final_epoch_fraction = None
    return state


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
