#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fcos_distillation import (
    apply_student_freezing,
    collect_parameter_audit,
    get_student_backbone_body,
    get_student_fpn,
    get_student_head,
    set_reproducibility,
    set_student_training_mode,
    validate_freezing_and_optimizer,
    verify_dataset_separation,
)
from src.distillation.config import DistillationConfig
from src.distillation.logging_utils import write_json
from src.distillation.model_registry import create_model, freeze_model


def clone_named_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def any_tensor_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name].detach().cpu()) for name in before)


def all_tensors_equal(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(before[name], after[name].detach().cpu()) for name in before)


def synthetic_optimizer_step(student: torch.nn.Module, config: DistillationConfig) -> dict[str, Any]:
    body = get_student_backbone_body(student)
    fpn = get_student_fpn(student)
    head = get_student_head(student)
    before_body = clone_named_tensors(body)
    before_fpn = clone_named_tensors(fpn)
    before_head = clone_named_tensors(head)

    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    set_student_training_mode(student, config)
    loss = torch.zeros((), dtype=torch.float32)
    for param in fpn.parameters():
        if param.requires_grad:
            loss = loss + param.float().sum() * 1e-8
    for param in head.parameters():
        if param.requires_grad:
            loss = loss + param.float().sum() * 1e-8
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    set_student_training_mode(student, config)

    body_after = {name: tensor.detach().cpu() for name, tensor in body.state_dict().items()}
    fpn_after = {name: tensor.detach().cpu() for name, tensor in fpn.state_dict().items()}
    head_after = {name: tensor.detach().cpu() for name, tensor in head.state_dict().items()}
    body_unchanged = all_tensors_equal(before_body, body_after)
    fpn_changed = any_tensor_changed(before_fpn, fpn_after)
    head_changed = any_tensor_changed(before_head, head_after)
    if not body_unchanged:
        raise RuntimeError("Frozen student backbone body changed after optimizer step.")
    if not (fpn_changed or head_changed):
        raise RuntimeError("No trainable FPN or head tensor changed after optimizer step.")
    return {
        "synthetic_step_executed": True,
        "backbone_body_unchanged": body_unchanged,
        "fpn_changed": fpn_changed,
        "head_changed": head_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify student-backbone freezing for a distillation config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--optimizer-step", action="store_true", help="Run a synthetic optimizer step on the real model.")
    args = parser.parse_args()

    config = DistillationConfig.load(args.config)
    config.training.device = args.device
    set_reproducibility(config.training.seed)

    device = torch.device(args.device)
    teacher = None
    if config.distillation.enabled:
        if config.teacher is None:
            raise RuntimeError("Distillation config is enabled but has no teacher.")
        teacher = create_model(config.teacher).to(device)
        freeze_model(teacher)

    student = create_model(config.student).to(device)
    apply_student_freezing(student, config)
    set_student_training_mode(student, config)
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    validate_freezing_and_optimizer(student=student, teacher=teacher, optimizer=optimizer, config=config)
    audit = collect_parameter_audit(student, optimizer, config)
    audit.update(verify_dataset_separation(config))
    audit["resolved_device"] = str(device)
    audit["optimizer_step_check"] = {"synthetic_step_executed": False}
    if args.optimizer_step:
        audit["optimizer_step_check"] = synthetic_optimizer_step(student, config)

    if args.audit_output is not None:
        write_json(args.audit_output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
