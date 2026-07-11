from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.distillation.config import DistillationLossConfig
from src.distillation.output_matching import MatchedOutput, match_outputs


@dataclass
class DistillationLossResult:
    total: torch.Tensor
    components: dict[str, float]
    active_terms: list[str]
    diagnostics: dict[str, object]


def compute_distillation_loss(
    *,
    supervised_loss: torch.Tensor,
    teacher_outputs: dict[str, torch.Tensor],
    student_outputs: dict[str, torch.Tensor],
    config: DistillationLossConfig,
    logger: logging.Logger,
) -> DistillationLossResult:
    total = supervised_loss * config.lambda_supervised
    components = {
        "supervised": float(supervised_loss.detach().cpu()),
        "supervised_weighted": float((supervised_loss * config.lambda_supervised).detach().cpu()),
    }
    active_terms = ["supervised"] if config.lambda_supervised else []

    if not config.enabled:
        components["total"] = float(total.detach().cpu())
        return DistillationLossResult(
            total=total, components=components, active_terms=active_terms, diagnostics={}
        )

    matched = match_outputs(
        teacher_outputs,
        student_outputs,
        skip_mismatched_shapes=config.skip_mismatched_shapes,
        logger=logger,
    )

    cls_loss = _classification_kd(matched.get("cls_logits", []), config.temperature)
    box_loss = _smooth_l1(matched.get("bbox_regression", []))
    ctr_loss = _smooth_l1(matched.get("bbox_ctrness", []))

    terms = [
        ("cls_distill", cls_loss, config.lambda_cls),
        ("box_distill", box_loss, config.lambda_box),
        ("centerness_distill", ctr_loss, config.lambda_centerness),
    ]
    diagnostics: dict[str, object] = {}
    for name, loss, weight in terms:
        output_key = {
            "cls_distill": "cls_logits",
            "box_distill": "bbox_regression",
            "centerness_distill": "bbox_ctrness",
        }[name]
        teacher_levels = list(teacher_outputs.get(output_key, []))
        student_levels = list(student_outputs.get(output_key, []))
        diagnostics[f"{name}_matched_levels"] = len(matched.get(output_key, []))
        diagnostics[f"{name}_skipped_levels"] = max(
            len(teacher_levels), len(student_levels)
        ) - len(matched.get(output_key, []))
        diagnostics[f"teacher_{output_key}_shapes"] = [list(value.shape) for value in teacher_levels]
        diagnostics[f"student_{output_key}_shapes"] = [list(value.shape) for value in student_levels]
        if loss is None or weight == 0:
            components[name] = 0.0
            components[f"{name}_weighted"] = 0.0
            continue
        total = total + weight * loss
        components[name] = float(loss.detach().cpu())
        components[f"{name}_weighted"] = float((weight * loss).detach().cpu())
        active_terms.append(name)

    components["total"] = float(total.detach().cpu())
    return DistillationLossResult(
        total=total,
        components=components,
        active_terms=active_terms,
        diagnostics=diagnostics,
    )


def _classification_kd(matches: list[MatchedOutput], temperature: float) -> torch.Tensor | None:
    if not matches:
        return None
    losses = []
    for match in matches:
        teacher = match.teacher.detach() / temperature
        student = match.student / temperature
        # Average the per-anchor categorical KL across batch and spatial positions.
        # ``batchmean`` would divide only by N for [N, anchors, classes], causing
        # the loss to scale linearly with the number of anchors.
        per_class_kl = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="none",
        )
        losses.append(per_class_kl.sum(dim=-1).mean() * (temperature**2))
    return torch.stack(losses).mean()


def _smooth_l1(matches: list[MatchedOutput]) -> torch.Tensor | None:
    if not matches:
        return None
    return torch.stack(
        [F.smooth_l1_loss(match.student, match.teacher.detach(), reduction="mean") for match in matches]
    ).mean()
