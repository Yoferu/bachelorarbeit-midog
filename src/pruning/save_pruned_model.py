from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.pruning.inspect_model import collect_conv_modules
from src.pruning.pruning_config import PruningConfig
from src.pruning.structured_pruning import count_parameters


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def save_pruned_artifact(
    *,
    model: Any,
    config: PruningConfig,
    pruning_result: dict[str, Any],
    output: Path,
    eval_config_path: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_type": "masked_structured_pruned_detection_model",
        "model": config.model,
        "state_dict": model.state_dict(),
        "pruning_config": config.as_dict(),
        "eval_config": str(eval_config_path),
        "pruning_result": pruning_result,
    }
    torch.save(artifact, output)
    return output


def write_pruning_metadata(
    *,
    model: Any,
    config: PruningConfig,
    pruning_result: dict[str, Any],
    output: Path,
    eval_config_path: Path,
    base_checkpoint: str,
    parameters_before_total: int,
    parameters_before_nonzero: int,
) -> Path:
    metadata_path = output.with_suffix(".meta.json")
    rows = collect_conv_modules(model)
    metadata = {
        "artifact_path": str(output),
        "base_model": config.model,
        "base_eval_config": str(eval_config_path),
        "base_checkpoint": base_checkpoint,
        "pruning_method": config.pruning_type,
        "pruning_ratio": config.pruning_ratio,
        "target_scope": config.target_scope,
        "target_modules": pruning_result["target_modules"],
        "excluded_modules": [
            {"name": row.name, "reason": row.reason}
            for row in rows
            if not row.prunable or row.name in set(config.excluded_modules)
        ],
        "parameters_before_total": parameters_before_total,
        "parameters_after_total": count_parameters(model, nonzero_only=False),
        "parameters_before_nonzero": parameters_before_nonzero,
        "parameters_after_nonzero": count_parameters(model, nonzero_only=True),
        "zeroed_channels": pruning_result["zeroed_channels"],
        "newly_zeroed_channels": pruning_result["newly_zeroed_channels"],
        "pruning_mode": pruning_result["pruning_mode"],
        "physical_channel_removal": pruning_result["physical_channel_removal"],
        "module_results": pruning_result["module_results"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "notes": config.notes,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_pruned_state_dict(path: Path) -> dict[str, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and "state_dict" in artifact:
        return artifact["state_dict"]
    if isinstance(artifact, dict):
        return artifact
    raise TypeError(f"Unsupported pruned artifact format: {path}")

