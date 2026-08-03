from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from src.pruning.inspect_model import collect_conv_modules, get_module_by_name, unwrap_detection_model
from src.pruning.pruning_config import PruningConfig
from src.pruning.structured_pruning import count_parameters


FIXED_OUTPUT_MODULES = {
    "head.classification_head.cls_logits": "fixed FCOS class logits output dimension",
    "head.regression_head.bbox_reg": "fixed FCOS box regression output dimension",
    "head.regression_head.bbox_ctrness": "fixed FCOS centerness output dimension",
}

DEFAULT_IGNORED_MODULES = {
    "backbone.body.conv1": "first RGB input convolution",
    **FIXED_OUTPUT_MODULES,
}

TARGET_SCOPES = {
    "depgraph_head_only": {"head"},
    "detection_head_only": {"head"},
    "depgraph_fpn_head": {"fpn", "head"},
    "fpn_head": {"fpn", "head"},
    "depgraph_backbone_fpn_head": {"backbone", "fpn", "head"},
    "backbone_fpn_head": {"backbone", "fpn", "head"},
    "backbone_fpn_head_convs": {"backbone", "fpn", "head"},
}


@dataclass(frozen=True)
class StructuralModuleChange:
    name: str
    scope: str
    in_channels_before: int
    out_channels_before: int
    in_channels_after: int
    out_channels_after: int
    pruned_in_channels: int
    pruned_out_channels: int


class DetectionOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model(images)
        return tuple(value for output in outputs for value in output.values())


def detection_output_transform(outputs: Any) -> tuple[torch.Tensor, ...]:
    return tuple(value for output in outputs for value in output.values())


def _load_torch_pruning():
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError(
            "Structural DepGraph pruning requires torch-pruning. "
            "Install it with: python -m pip install torch-pruning"
        ) from exc
    return tp


def build_example_input(config: PruningConfig, model_config: Any) -> torch.Tensor:
    size = config.example_input_size or int(getattr(model_config, "patch_size", 1024))
    return torch.zeros(1, 3, size, size, dtype=torch.float32, device=config.device)


def collect_structural_pruning_plan(
    model: Any,
    config: PruningConfig,
) -> dict[str, Any]:
    detection_model = unwrap_detection_model(model)
    if config.target_scope not in TARGET_SCOPES:
        raise ValueError(
            f"Unsupported DepGraph target_scope: {config.target_scope}. "
            f"Expected one of: {sorted(TARGET_SCOPES)}"
        )
    target_scopes = TARGET_SCOPES[config.target_scope]
    ignored_names = {**DEFAULT_IGNORED_MODULES}
    for name in config.ignored_modules:
        ignored_names[name] = "configured ignored module"
    for name in config.excluded_modules:
        ignored_names[name] = "configured excluded module"

    rows = collect_conv_modules(model)
    conv_names = {row.name for row in rows}
    missing_ignored = sorted(name for name in ignored_names if name not in conv_names)
    ignored_conv_names = [name for name in ignored_names if name in conv_names]

    candidates = []
    excluded = []
    for row in rows:
        if row.name in ignored_names:
            excluded.append({"name": row.name, "reason": ignored_names[row.name]})
            continue
        if row.scope not in target_scopes:
            ignored_names[row.name] = f"outside configured target_scope={config.target_scope}"
            ignored_conv_names.append(row.name)
            excluded.append({"name": row.name, "reason": ignored_names[row.name]})
            continue
        candidates.append(
            {
                "name": row.name,
                "scope": row.scope,
                "in_channels": row.in_channels,
                "out_channels": row.out_channels,
                "kernel_size": row.kernel_size,
            }
        )

    return {
        "candidate_modules": candidates,
        "ignored_modules": ignored_conv_names,
        "excluded_modules": excluded,
        "missing_ignored_modules": missing_ignored,
        "conv2d_count": len(rows),
        "candidate_count": len(candidates),
        "target_scope": config.target_scope,
        "target_scopes": sorted(target_scopes),
        "scope_counts": {
            scope: sum(1 for item in candidates if item["scope"] == scope)
            for scope in sorted({item["scope"] for item in candidates})
        },
        "detection_model_class": f"{type(detection_model).__module__}.{type(detection_model).__name__}",
    }


def _conv_shapes(model: Any) -> dict[str, tuple[int, int]]:
    detection_model = unwrap_detection_model(model)
    return {
        name: (int(module.in_channels), int(module.out_channels))
        for name, module in detection_model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def _module_changes(model: Any, before: dict[str, tuple[int, int]]) -> list[StructuralModuleChange]:
    rows = collect_conv_modules(model)
    changes = []
    for row in rows:
        before_in, before_out = before[row.name]
        if before_in == row.in_channels and before_out == row.out_channels:
            continue
        changes.append(
            StructuralModuleChange(
                name=row.name,
                scope=row.scope,
                in_channels_before=before_in,
                out_channels_before=before_out,
                in_channels_after=row.in_channels,
                out_channels_after=row.out_channels,
                pruned_in_channels=max(0, before_in - row.in_channels),
                pruned_out_channels=max(0, before_out - row.out_channels),
            )
        )
    return changes


def _count_macs_and_params(model: nn.Module, example_input: torch.Tensor) -> tuple[float | None, int | None, str | None]:
    tp = _load_torch_pruning()
    try:
        macs, params = tp.utils.count_ops_and_params(DetectionOutputWrapper(model), example_input)
        return float(macs), int(params), None
    except Exception as exc:
        return None, None, str(exc)


def _detection_count(outputs: list[dict[str, torch.Tensor]]) -> int:
    return int(sum(len(output.get("boxes", ())) for output in outputs))


def smoke_compare_detection_counts(
    *,
    original_model: Any,
    pruned_model: Any,
    example_input: torch.Tensor,
) -> dict[str, Any]:
    original_detection_model = unwrap_detection_model(original_model)
    pruned_detection_model = unwrap_detection_model(pruned_model)
    original_detection_model.eval()
    pruned_detection_model.eval()
    with torch.no_grad():
        original_outputs = original_detection_model(example_input)
        pruned_outputs = pruned_detection_model(example_input)
    original_count = _detection_count(original_outputs)
    pruned_count = _detection_count(pruned_outputs)
    return {
        "input_shape": list(example_input.shape),
        "original_detection_count": original_count,
        "pruned_detection_count": pruned_count,
        "absolute_difference": abs(original_count - pruned_count),
    }


def apply_depgraph_structural_pruning(
    *,
    model: Any,
    config: PruningConfig,
    example_input: torch.Tensor,
) -> dict[str, Any]:
    if config.pruning_mode != "depgraph_structural":
        raise ValueError(f"Expected pruning_mode=depgraph_structural, got {config.pruning_mode}")
    if config.pruning_type != "depgraph_magnitude_l2":
        raise ValueError(f"Unsupported pruning_type for DepGraph pruning: {config.pruning_type}")
    if not (0.0 < config.pruning_ratio < 1.0):
        raise ValueError("pruning_ratio must be between 0 and 1")

    tp = _load_torch_pruning()
    torch.manual_seed(config.seed)

    detection_model = unwrap_detection_model(model)
    detection_model.eval()
    detection_model.to(config.device)
    example_input = example_input.to(config.device)

    plan = collect_structural_pruning_plan(model, config)
    ignored_layers = [get_module_by_name(model, name) for name in plan["ignored_modules"]]
    before_shapes = _conv_shapes(model)
    parameters_before = count_parameters(detection_model, nonzero_only=False)
    macs_before, tp_params_before, macs_error_before = _count_macs_and_params(detection_model, example_input)

    pruner = tp.pruner.MagnitudePruner(
        detection_model,
        example_input,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=config.pruning_ratio,
        max_pruning_ratio=config.max_pruning_ratio,
        global_pruning=config.global_pruning,
        ignored_layers=ignored_layers,
        round_to=config.round_to,
        output_transform=detection_output_transform,
    )
    pruner.step()

    parameters_after = count_parameters(detection_model, nonzero_only=False)
    macs_after, tp_params_after, macs_error_after = _count_macs_and_params(detection_model, example_input)
    changes = _module_changes(model, before_shapes)

    return {
        "pruning_mode": "depgraph_structural",
        "physical_channel_removal": True,
        "pruning_library": "torch-pruning",
        "pruning_library_version": getattr(tp, "__version__", None),
        "target_modules": [item["name"] for item in plan["candidate_modules"]],
        "ignored_modules": plan["ignored_modules"],
        "excluded_modules": plan["excluded_modules"],
        "candidate_scope_counts": plan["scope_counts"],
        "parameters_before_total": parameters_before,
        "parameters_after_total": parameters_after,
        "parameters_removed_total": parameters_before - parameters_after,
        "parameters_removed_ratio": (parameters_before - parameters_after) / parameters_before,
        "macs_before": macs_before,
        "macs_after": macs_after,
        "macs_removed": None if macs_before is None or macs_after is None else macs_before - macs_after,
        "macs_removed_ratio": None
        if macs_before in {None, 0} or macs_after is None
        else (macs_before - macs_after) / macs_before,
        "torch_pruning_counted_params_before": tp_params_before,
        "torch_pruning_counted_params_after": tp_params_after,
        "macs_error_before": macs_error_before,
        "macs_error_after": macs_error_after,
        "module_results": [asdict(change) for change in changes],
        "pruned_layers": [change.name for change in changes],
        "plan": plan,
    }
