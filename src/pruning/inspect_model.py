from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_EVAL_CONFIG_DIR = Path("experiments/eval_guide/configs")


@dataclass(frozen=True)
class ConvModuleInfo:
    name: str
    module_type: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    scope: str
    prunable: bool
    reason: str


def load_guide_model(
    *,
    model_name: str,
    guide_repo: Path = Path("repos/MIDOG_2025_Guide"),
    eval_config: Path | None = None,
    det_thresh: float = 0.05,
) -> tuple[Any, Any]:
    guide_repo = guide_repo.resolve()
    if not guide_repo.is_dir():
        raise FileNotFoundError(f"MIDOG guide repo not found: {guide_repo}")
    sys.path.insert(0, str(guide_repo))

    from utils.factory import ConfigCreator, ModelFactory

    config_path = eval_config or DEFAULT_EVAL_CONFIG_DIR / f"{model_name}_eval.yaml"
    config = ConfigCreator.load(str(config_path))
    model = ModelFactory.load(config, det_thresh=det_thresh)
    return model, config


def unwrap_detection_model(model: Any) -> Any:
    return getattr(model, "model", model)


def classify_scope(name: str) -> str:
    if name.startswith("backbone.fpn."):
        return "fpn"
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("head."):
        return "head"
    return "other"


def assess_prunable_conv(name: str, module: torch.nn.Conv2d) -> tuple[bool, str]:
    if name == "backbone.body.conv1":
        return False, "first input convolution"
    if name.startswith("backbone."):
        return False, "backbone/FPN channel changes require dependency-aware physical pruning"
    if name in {
        "head.classification_head.cls_logits",
        "head.regression_head.bbox_reg",
        "head.regression_head.bbox_ctrness",
    }:
        return False, "shape-sensitive FCOS output prediction layer"
    if name.startswith("head.") and ".conv." in name and module.in_channels == module.out_channels:
        return True, "safe masked baseline target: internal FCOS head Conv2d"
    if name.startswith("head."):
        return False, "head layer excluded because it is not an internal same-width conv"
    return False, "outside selected pruning scope"


def collect_conv_modules(model: Any) -> list[ConvModuleInfo]:
    detection_model = unwrap_detection_model(model)
    rows: list[ConvModuleInfo] = []
    for name, module in detection_model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        prunable, reason = assess_prunable_conv(name, module)
        rows.append(
            ConvModuleInfo(
                name=name,
                module_type=module.__class__.__name__,
                in_channels=int(module.in_channels),
                out_channels=int(module.out_channels),
                kernel_size=tuple(int(v) for v in module.kernel_size),
                scope=classify_scope(name),
                prunable=prunable,
                reason=reason,
            )
        )
    return rows


def get_module_by_name(model: Any, name: str) -> torch.nn.Module:
    modules = dict(unwrap_detection_model(model).named_modules())
    if name not in modules:
        raise KeyError(f"Module not found: {name}")
    return modules[name]


def format_conv_table(rows: list[ConvModuleInfo]) -> str:
    header = [
        "module",
        "type",
        "in",
        "out",
        "kernel",
        "scope",
        "prunable",
        "reason",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.name,
                    row.module_type,
                    str(row.in_channels),
                    str(row.out_channels),
                    "x".join(str(v) for v in row.kernel_size),
                    row.scope,
                    "yes" if row.prunable else "no",
                    row.reason,
                ]
            )
        )
    return "\n".join(lines)

