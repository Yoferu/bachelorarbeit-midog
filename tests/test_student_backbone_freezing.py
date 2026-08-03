from __future__ import annotations

import torch

from scripts.train_fcos_distillation import (
    apply_student_freezing,
    collect_parameter_audit,
    get_student_backbone_body,
    set_student_training_mode,
    validate_freezing_and_optimizer,
)
from src.distillation.config import DataConfig, DistillationConfig, DistillationLossConfig, ModelSpec, TrainingConfig


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 4))
        self.fpn = torch.nn.Linear(4, 4)


class DummyStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = DummyBackbone()
        self.head = torch.nn.Linear(4, 1)


def freeze_config() -> DistillationConfig:
    return DistillationConfig(
        experiment_name="dummy",
        output_dir="/tmp/dummy",
        teacher=None,
        student=ModelSpec(architecture="dummy"),
        data=DataConfig(train_csv="unused.csv"),
        training=TrainingConfig(lr=1e-5, weight_decay=1e-4, freeze_backbone=True),
        distillation=DistillationLossConfig(enabled=False),
    )


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def test_student_backbone_body_is_frozen_but_fpn_and_head_are_trainable() -> None:
    config = freeze_config()
    student = DummyStudent()
    apply_student_freezing(student, config)
    set_student_training_mode(student, config)

    assert all(not param.requires_grad for param in student.backbone.body.parameters())
    assert all(param.requires_grad for param in student.backbone.fpn.parameters())
    assert all(param.requires_grad for param in student.head.parameters())
    assert not get_student_backbone_body(student).training
    assert student.backbone.fpn.training
    assert student.head.training

    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    validate_freezing_and_optimizer(student=student, teacher=None, optimizer=optimizer, config=config)
    audit = collect_parameter_audit(student, optimizer, config)
    assert audit["optimizer_parameter_count"] == audit["student_trainable_parameters"]
    assert "backbone.body" in audit["frozen_top_level_modules"]
    assert "backbone.fpn" in audit["trainable_top_level_modules"]
    assert "head" in audit["trainable_top_level_modules"]


def test_frozen_backbone_body_tensors_do_not_change_after_optimizer_step() -> None:
    config = freeze_config()
    student = DummyStudent()
    apply_student_freezing(student, config)
    set_student_training_mode(student, config)
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    before_body = clone_state(student.backbone.body)
    before_fpn = clone_state(student.backbone.fpn)
    before_head = clone_state(student.head)

    x = torch.randn(8, 4)
    features = student.backbone.fpn(x)
    prediction = student.head(features)
    loss = prediction.square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    set_student_training_mode(student, config)

    after_body = clone_state(student.backbone.body)
    after_fpn = clone_state(student.backbone.fpn)
    after_head = clone_state(student.head)
    assert all(torch.equal(before_body[name], after_body[name]) for name in before_body)
    assert any(not torch.equal(before_fpn[name], after_fpn[name]) for name in before_fpn)
    assert any(not torch.equal(before_head[name], after_head[name]) for name in before_head)
    assert not student.backbone.body.training
