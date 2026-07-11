from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class MatchedOutput:
    key: str
    teacher: torch.Tensor
    student: torch.Tensor


def extract_fcos_head_outputs(model: nn.Module, images: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    try:
        transformed, _ = model.transform(images, None)
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        feature_list = list(features.values())
        outputs = model.head(feature_list)
        level_sizes = [feature.size(2) * feature.size(3) for feature in feature_list]
        return {key: list(value.split(level_sizes, dim=1)) for key, value in outputs.items()}
    finally:
        model.train(was_training)


def match_outputs(
    teacher_outputs: dict[str, Any],
    student_outputs: dict[str, Any],
    *,
    skip_mismatched_shapes: bool,
    logger: logging.Logger,
) -> dict[str, list[MatchedOutput]]:
    matched: dict[str, list[MatchedOutput]] = {}
    for key in ("cls_logits", "bbox_regression", "bbox_ctrness"):
        if key not in teacher_outputs or key not in student_outputs:
            message = f"Distillation output key missing for {key}."
            if skip_mismatched_shapes:
                logger.warning("%s Skipping.", message)
                matched[key] = []
                continue
            raise KeyError(message)
        t_value = teacher_outputs[key]
        s_value = student_outputs[key]
        t_items = _as_tensor_list(t_value)
        s_items = _as_tensor_list(s_value)
        count = min(len(t_items), len(s_items))
        if len(t_items) != len(s_items):
            logger.warning(
                "Teacher/student output level count differs for %s: %d vs %d. Matching first %d levels.",
                key,
                len(t_items),
                len(s_items),
                count,
            )
        rows: list[MatchedOutput] = []
        for idx in range(count):
            teacher = t_items[idx]
            student = s_items[idx]
            if teacher.shape != student.shape:
                message = (
                    f"Mismatched {key} level {idx} shapes: "
                    f"teacher={tuple(teacher.shape)} student={tuple(student.shape)}"
                )
                if skip_mismatched_shapes:
                    logger.warning("%s. Skipping this level.", message)
                    continue
                raise ValueError(message)
            rows.append(MatchedOutput(f"{key}_{idx}", teacher, student))
        matched[key] = rows
    return matched


def _as_tensor_list(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in value):
        return list(value)
    raise TypeError(f"Unsupported FCOS output value type: {type(value)!r}")

