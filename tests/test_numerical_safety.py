from __future__ import annotations

import json

import pytest
import torch

from scripts.train_fcos_distillation import (
    check_gradients_finite_or_raise,
    check_parameters_finite_or_raise,
    finite_loss_components_or_raise,
)


def test_non_finite_loss_component_raises() -> None:
    with pytest.raises(RuntimeError, match="Non-finite loss components"):
        finite_loss_components_or_raise({"total": float("nan"), "supervised": 1.0}, step=3)


def test_non_finite_gradient_raises() -> None:
    model = torch.nn.Linear(2, 1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    next(model.parameters()).grad[0, 0] = float("inf")

    with pytest.raises(RuntimeError, match="Non-finite gradients"):
        check_gradients_finite_or_raise(model, step=4)


def test_non_finite_parameter_raises() -> None:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        next(model.parameters())[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="Non-finite parameters"):
        check_parameters_finite_or_raise(model, step=5)
