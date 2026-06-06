from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn.utils import prune

from src.pruning.inspect_model import collect_conv_modules, get_module_by_name
from src.pruning.pruning_config import PruningConfig


@dataclass(frozen=True)
class ModulePruningResult:
    name: str
    pruning_method: str
    amount: float
    out_channels: int
    zeroed_channels_before: int
    zeroed_channels_after: int
    newly_zeroed_channels: int


def count_parameters(model: torch.nn.Module, *, nonzero_only: bool = False) -> int:
    total = 0
    for parameter in model.parameters():
        if nonzero_only:
            total += int(torch.count_nonzero(parameter.detach()).item())
        else:
            total += int(parameter.numel())
    return total


def zeroed_output_channels(module: torch.nn.Conv2d) -> int:
    weight = module.weight.detach()
    flattened = weight.reshape(weight.shape[0], -1)
    return int((flattened.abs().sum(dim=1) == 0).sum().item())


def select_target_modules(model: Any, config: PruningConfig) -> list[str]:
    if config.target_scope not in {"detection_head_only", "safe_conv_layers_only"}:
        raise ValueError(
            "Only detection_head_only and safe_conv_layers_only are supported "
            "for the first masked pruning baseline."
        )
    rows = collect_conv_modules(model)
    return [
        row.name
        for row in rows
        if row.prunable and row.name not in set(config.excluded_modules)
    ]


def apply_masked_structured_pruning(
    *,
    model: Any,
    config: PruningConfig,
) -> dict[str, Any]:
    if config.pruning_type != "structured_channel_l2":
        raise ValueError(f"Unsupported pruning_type: {config.pruning_type}")
    if not (0.0 < config.pruning_ratio < 1.0):
        raise ValueError("pruning_ratio must be between 0 and 1")

    torch.manual_seed(config.seed)
    detection_model = getattr(model, "model", model)
    target_modules = select_target_modules(model, config)
    module_results: list[ModulePruningResult] = []

    for name in target_modules:
        module = get_module_by_name(model, name)
        if not isinstance(module, torch.nn.Conv2d):
            raise TypeError(f"Target module is not Conv2d: {name}")
        before = zeroed_output_channels(module)
        prune.ln_structured(module, name="weight", amount=config.pruning_ratio, n=2, dim=0)
        prune.remove(module, "weight")
        after = zeroed_output_channels(module)
        module_results.append(
            ModulePruningResult(
                name=name,
                pruning_method="torch.nn.utils.prune.ln_structured",
                amount=config.pruning_ratio,
                out_channels=int(module.out_channels),
                zeroed_channels_before=before,
                zeroed_channels_after=after,
                newly_zeroed_channels=max(0, after - before),
            )
        )

    return {
        "pruning_mode": "masked",
        "physical_channel_removal": False,
        "target_modules": target_modules,
        "module_results": [asdict(result) for result in module_results],
        "parameters_total": count_parameters(detection_model, nonzero_only=False),
        "parameters_nonzero": count_parameters(detection_model, nonzero_only=True),
        "zeroed_channels": sum(result.zeroed_channels_after for result in module_results),
        "newly_zeroed_channels": sum(result.newly_zeroed_channels for result in module_results),
    }

