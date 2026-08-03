from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from src.distillation.config import ModelSpec
from src.distillation.model_registry import _state_dict_from_checkpoint, create_model


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in model metadata file: {path}")
    return value


def resolve_student_spec(checkpoint_path: Path, checkpoint: Any) -> tuple[ModelSpec, str]:
    metadata_path = checkpoint_path.parent / "metadata.json"
    config_path = checkpoint_path.parent / "resolved_config.yaml"
    student: dict[str, Any] = {}
    sources: list[str] = []
    if config_path.is_file():
        config = _load_mapping(config_path)
        if isinstance(config.get("student"), dict):
            student.update(config["student"])
            sources.append(str(config_path))
    if metadata_path.is_file():
        metadata = _load_mapping(metadata_path)
        if metadata.get("student_architecture"):
            student["architecture"] = metadata["student_architecture"]
            sources.insert(0, str(metadata_path))
    if isinstance(checkpoint, dict):
        embedded = checkpoint.get("distillation_config")
        if isinstance(embedded, dict) and isinstance(embedded.get("student"), dict):
            for key, value in embedded["student"].items():
                student.setdefault(key, value)
            sources.append("checkpoint distillation_config")
        if checkpoint.get("student_architecture"):
            student.setdefault("architecture", checkpoint["student_architecture"])
            sources.append("checkpoint student_architecture")
    if not student.get("architecture"):
        raise ValueError(
            f"Could not resolve student architecture for checkpoint {checkpoint_path}. Checked "
            f"{metadata_path}, {config_path}, and checkpoint metadata. Add 'student_architecture' "
            "to metadata.json or 'student.architecture' to resolved_config.yaml."
        )
    student["checkpoint"] = None
    try:
        return ModelSpec(**student), ", ".join(dict.fromkeys(sources))
    except TypeError as exc:
        raise ValueError(f"Invalid student model metadata for {checkpoint_path}: {exc}") from exc


def load_distilled_student(checkpoint_path: Path) -> tuple[nn.Module, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_object"), nn.Module):
        return checkpoint["model_object"], "checkpoint model_object"
    spec, source = resolve_student_spec(checkpoint_path, checkpoint)
    try:
        model = create_model(spec)
    except ValueError as exc:
        raise ValueError(
            f"Could not recreate student architecture '{spec.architecture}' resolved from {source}: {exc}"
        ) from exc
    try:
        missing, unexpected = model.load_state_dict(
            _state_dict_from_checkpoint(checkpoint), strict=False
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint state_dict tensor shapes do not match registry architecture "
            f"'{spec.architecture}' resolved from {source}: {exc}"
        ) from exc
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state_dict does not match registry architecture '{spec.architecture}'. "
            f"Missing keys: {list(missing)[:10]}, unexpected keys: {list(unexpected)[:10]}"
        )
    return model, source
