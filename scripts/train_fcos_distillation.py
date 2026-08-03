#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.config import DistillationConfig
from src.distillation.early_stopping import (
    EarlyStoppingState,
    record_validation,
    record_validation_error,
    should_aggressive_lr_safety_stop,
    should_validate_checkpoint,
)
from src.distillation.logging_utils import (
    append_training_log,
    create_run_dir,
    current_git_commit,
    package_versions,
    setup_logger,
    write_json,
)
from src.distillation.losses import compute_distillation_loss
from src.distillation.model_registry import count_parameters, create_model, freeze_model
from src.distillation.output_matching import extract_fcos_head_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a config-driven FCOS teacher-student distillation run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--experiment-name-suffix", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-terminal", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--disable-early-stopping", action="store_true")
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_determinism(enabled: bool) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_order_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    payload = frame[available].astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_safe_sample(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe_sample(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_sample(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_sample_preview(dataset: Any, limit: int = 16) -> dict[str, Any]:
    samples = getattr(dataset, "samples", {})
    rows: list[Any] = []
    if isinstance(samples, dict):
        for key in sorted(samples)[:limit]:
            rows.append({"index": int(key), "sample": json_safe_sample(samples[key])})
    payload = json.dumps(rows, sort_keys=True)
    return {
        "preview": rows,
        "preview_count": len(rows),
        "preview_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def reproducibility_audit(
    *,
    config: DistillationConfig,
    dataset_df: pd.DataFrame,
    train_loader: Any,
    resolved_device: torch.device,
) -> dict[str, Any]:
    train_frame = dataset_df[dataset_df["split"].astype(str) == "train"].copy()
    val_frame = dataset_df[dataset_df["split"].astype(str) == "val"].copy()
    checkpoint = Path(config.student.checkpoint) if config.student.checkpoint else None
    checkpoint_hash = sha256_file(checkpoint) if checkpoint and checkpoint.exists() else None
    return {
        "configured_seed": int(config.training.seed),
        "resolved_seed": int(config.training.seed),
        "python_seed": int(config.training.seed),
        "numpy_seed": int(config.training.seed),
        "torch_initial_seed": int(torch.initial_seed()),
        "cuda_seed_all": int(config.training.seed) if torch.cuda.is_available() else None,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "resolved_device": str(resolved_device),
        "dataloader_generator": None,
        "dataloader_worker_init_fn": None,
        "dataloader_worker_seed_policy": (
            "No explicit generator or worker_init_fn is passed by the MIDOG guide datamodule; "
            "global Python/NumPy/Torch seeds are set before datamodule construction."
        ),
        "cuda_bitwise_determinism_claimed": False,
        "start_checkpoint": config.student.checkpoint,
        "start_checkpoint_sha256": checkpoint_hash,
        "train_csv": config.data.train_csv,
        "validation_csv": config.data.val_csv,
        "train_csv_order_sha256": dataframe_order_digest(train_frame, ["filename", "label", "x", "y", "split"]),
        "validation_csv_order_sha256": dataframe_order_digest(val_frame, ["filename", "label", "x", "y", "split"]),
        "training_dataset_sample_preview": dataset_sample_preview(getattr(train_loader, "dataset", None)),
    }


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda, but CUDA is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("training.device must be one of: auto, cpu, cuda")
    return torch.device(normalized)


def import_guide_datamodule(guide_repo: Path):
    guide_repo = guide_repo.resolve()
    if not guide_repo.is_dir():
        raise FileNotFoundError(f"MIDOG guide repo not found: {guide_repo}")
    if str(guide_repo) not in sys.path:
        sys.path.insert(0, str(guide_repo))
    from utils.datamodule import ObjectDetectionDataModule

    return ObjectDetectionDataModule


def build_dataset_frame(config: DistillationConfig) -> pd.DataFrame:
    train_df = pd.read_csv(config.data.train_csv)
    if "split" not in train_df.columns:
        train_df = train_df.assign(split="train")
    if config.data.val_csv:
        val_df = pd.read_csv(config.data.val_csv)
        val_df = val_df.assign(split="val")
        train_df = train_df[train_df["split"] != "val"]
        return pd.concat([train_df, val_df], ignore_index=True)
    if "val" not in set(train_df["split"].unique()):
        placeholder = train_df[train_df["split"] == "train"].head(1).copy()
        placeholder["split"] = "val"
        train_df = pd.concat([train_df, placeholder], ignore_index=True)
    return train_df


def move_batch_to_device(batch: Any, device: torch.device) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = batch
    images = [image.to(device) for image in images]
    moved_targets = []
    for target in targets:
        moved_targets.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return images, moved_targets


def finite_or_raise(loss: torch.Tensor, step: int) -> None:
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite distillation loss at global step {step}: {float(loss.detach().cpu())}")


def finite_loss_components_or_raise(components: dict[str, float], step: int) -> None:
    bad = {name: value for name, value in components.items() if isinstance(value, (int, float)) and not math.isfinite(float(value))}
    if bad:
        raise RuntimeError(f"Non-finite loss components at global step {step}: {bad}")


def check_gradients_finite_or_raise(model: torch.nn.Module, step: int) -> None:
    bad = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad.detach()).all()
    ]
    if bad:
        raise RuntimeError(f"Non-finite gradients before optimizer step {step}: {bad[:10]}")


def check_parameters_finite_or_raise(model: torch.nn.Module, step: int) -> None:
    bad = [name for name, param in model.named_parameters() if not torch.isfinite(param.detach()).all()]
    if bad:
        raise RuntimeError(f"Non-finite parameters after optimizer step {step}: {bad[:10]}")


def named_parameter_map(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return dict(model.named_parameters())


def get_student_backbone_body(student: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(student, "backbone", None)
    body = getattr(backbone, "body", None)
    if not isinstance(body, torch.nn.Module):
        raise RuntimeError("Could not locate student backbone body at student.backbone.body.")
    return body


def get_student_fpn(student: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(student, "backbone", None)
    fpn = getattr(backbone, "fpn", None)
    if not isinstance(fpn, torch.nn.Module):
        raise RuntimeError("Could not locate student FPN at student.backbone.fpn.")
    return fpn


def get_student_head(student: torch.nn.Module) -> torch.nn.Module:
    head = getattr(student, "head", None)
    if not isinstance(head, torch.nn.Module):
        raise RuntimeError("Could not locate student detection head at student.head.")
    return head


def apply_student_freezing(student: torch.nn.Module, config: DistillationConfig) -> None:
    if not config.training.freeze_backbone:
        return
    body = get_student_backbone_body(student)
    fpn = get_student_fpn(student)
    head = get_student_head(student)
    for param in body.parameters():
        param.requires_grad_(False)
    for param in fpn.parameters():
        param.requires_grad_(True)
    for param in head.parameters():
        param.requires_grad_(True)
    body.eval()


def set_student_training_mode(student: torch.nn.Module, config: DistillationConfig) -> None:
    student.train()
    if config.training.freeze_backbone:
        get_student_backbone_body(student).eval()


def parameter_prefix(name: str) -> str:
    if name.startswith("backbone.body."):
        return "backbone.body"
    if name.startswith("backbone.fpn."):
        return "backbone.fpn"
    return name.split(".", 1)[0]


def collect_parameter_audit(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: DistillationConfig,
) -> dict[str, Any]:
    params = named_parameter_map(student)
    total = sum(param.numel() for param in params.values())
    trainable = sum(param.numel() for param in params.values() if param.requires_grad)
    frozen = total - trainable
    optimizer_param_ids: set[int] = set()
    optimizer_param_count = 0
    optimizer_trainable_param_count = 0
    if optimizer is not None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                optimizer_param_ids.add(id(param))
                optimizer_param_count += param.numel()
                if param.requires_grad:
                    optimizer_trainable_param_count += param.numel()
    top: dict[str, dict[str, int]] = {}
    for name, param in params.items():
        key = parameter_prefix(name)
        bucket = top.setdefault(key, {"total": 0, "trainable": 0, "frozen": 0})
        bucket["total"] += param.numel()
        if param.requires_grad:
            bucket["trainable"] += param.numel()
        else:
            bucket["frozen"] += param.numel()
    return {
        "student_total_parameters": int(total),
        "student_trainable_parameters": int(trainable),
        "student_frozen_parameters": int(frozen),
        "student_trainable_parameter_percentage": float(trainable / total * 100.0) if total else 0.0,
        "frozen_top_level_modules": sorted(name for name, row in top.items() if row["frozen"] and not row["trainable"]),
        "trainable_top_level_modules": sorted(name for name, row in top.items() if row["trainable"]),
        "mixed_top_level_modules": sorted(name for name, row in top.items() if row["frozen"] and row["trainable"]),
        "top_level_parameter_counts": top,
        "optimizer_parameter_count": int(optimizer_param_count),
        "optimizer_trainable_parameter_count": int(optimizer_trainable_param_count),
        "configured_learning_rate": config.training.lr,
        "configured_weight_decay": config.training.weight_decay,
        "student_backbone_frozen": config.training.freeze_backbone,
        "distillation_enabled": config.distillation.enabled,
        "teacher_model_identifier": config.teacher.architecture if config.teacher else None,
        "random_seed": config.training.seed,
        "dataset_split_files": {
            "train_csv": config.data.train_csv,
            "val_csv": config.data.val_csv,
            "calibration_csv": config.data.calibration_csv,
            "final_csv": config.data.final_csv,
            "final_split": config.data.final_split,
        },
    }


def validate_freezing_and_optimizer(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    config: DistillationConfig,
) -> None:
    params = named_parameter_map(student)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    if config.training.freeze_backbone:
        body_bad = [name for name, param in params.items() if name.startswith("backbone.body.") and param.requires_grad]
        if body_bad:
            raise RuntimeError(f"freeze_backbone=true but backbone.body parameters remain trainable: {body_bad[:10]}")
        body_in_optimizer = [
            name for name, param in params.items() if name.startswith("backbone.body.") and id(param) in optimizer_param_ids
        ]
        if body_in_optimizer:
            raise RuntimeError(f"Frozen backbone.body parameters are present in optimizer: {body_in_optimizer[:10]}")
        fpn_bad = [name for name, param in params.items() if name.startswith("backbone.fpn.") and not param.requires_grad]
        head_bad = [name for name, param in params.items() if name.startswith("head.") and not param.requires_grad]
        if fpn_bad:
            raise RuntimeError(f"freeze_backbone=true but FPN parameters are frozen: {fpn_bad[:10]}")
        if head_bad:
            raise RuntimeError(f"freeze_backbone=true but detection head parameters are frozen: {head_bad[:10]}")
        if get_student_backbone_body(student).training:
            raise RuntimeError("freeze_backbone=true but student.backbone.body is not in eval mode.")
    if teacher is not None:
        teacher_bad = [name for name, param in teacher.named_parameters() if param.requires_grad]
        if teacher_bad:
            raise RuntimeError(f"Teacher has trainable parameters: {teacher_bad[:10]}")
        if teacher.training:
            raise RuntimeError("Teacher is not in eval mode after freezing.")


def split_image_ids(csv_path: str | None, split: str | None = None) -> set[str]:
    if not csv_path:
        return set()
    frame = pd.read_csv(csv_path)
    if "filename" not in frame.columns:
        raise RuntimeError(f"Dataset CSV lacks filename column: {csv_path}")
    if split is not None:
        if "split" not in frame.columns:
            raise RuntimeError(f"Dataset CSV lacks split column: {csv_path}")
        frame = frame[frame["split"].astype(str) == split]
    return set(frame["filename"].astype(str))


def verify_dataset_separation(config: DistillationConfig) -> dict[str, Any]:
    train_ids = split_image_ids(config.data.train_csv, None)
    val_ids = split_image_ids(config.data.val_csv, None)
    calibration_ids = split_image_ids(config.data.calibration_csv, None)
    final_ids = split_image_ids(config.data.final_csv, config.data.final_split)
    groups = {
        "train": train_ids,
        "validation": val_ids,
        "calibration": calibration_ids,
        "final": final_ids,
    }
    overlaps: dict[str, int] = {}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}_{right}"] = len(groups[left] & groups[right])
    bad = {key: value for key, value in overlaps.items() if value}
    if bad:
        raise RuntimeError(f"Dataset split overlap check failed: {bad}")
    return {
        "dataset_image_counts": {key: len(value) for key, value in groups.items()},
        "dataset_overlap_counts": overlaps,
    }



def evaluate_loss(
    *,
    teacher: torch.nn.Module | None,
    student: torch.nn.Module,
    dataloader: Any,
    config: DistillationConfig,
    device: torch.device,
    logger: Any,
) -> float:
    student.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            images, targets = move_batch_to_device(batch, device)
            set_student_training_mode(student, config)
            supervised_dict = student(images, targets)
            supervised_loss = sum(supervised_dict.values())
            student.eval()
            teacher_outputs = {}
            student_outputs = {}
            if config.distillation.enabled:
                if teacher is None:
                    raise RuntimeError("Validation requested distillation loss without a teacher.")
                teacher_outputs = extract_fcos_head_outputs(teacher, images)
                student_outputs = extract_fcos_head_outputs(student, images)
            loss_result = compute_distillation_loss(
                supervised_loss=supervised_loss,
                teacher_outputs=teacher_outputs,
                student_outputs=student_outputs,
                config=config.distillation,
                logger=logger,
            )
            losses.append(float(loss_result.total.detach().cpu()))
    student.train()
    if config.training.freeze_backbone:
        get_student_backbone_body(student).eval()
    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))

def save_student_checkpoint(
    path: Path,
    student: torch.nn.Module,
    config: DistillationConfig,
    epoch: int,
    *,
    global_optimizer_step: int | None = None,
    batch_idx: int | None = None,
    fractional_epoch: float | None = None,
    learning_rate: float | None = None,
    optimizer_learning_rates: list[float] | None = None,
    wall_clock_training_s: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_student = student.to("cpu")
    artifact = {
        "artifact_type": "fcos_distilled_student",
        "model_object": cpu_student,
        "state_dict": cpu_student.state_dict(),
        "student_architecture": config.student.architecture,
        "teacher_architecture": config.teacher.architecture if config.teacher else None,
        "distillation_config": config.as_dict(),
        "epoch": epoch,
        "global_optimizer_step": global_optimizer_step,
        "batch_idx": batch_idx,
        "fractional_epoch": fractional_epoch,
        "learning_rate": learning_rate,
        "optimizer_learning_rates": optimizer_learning_rates,
        "wall_clock_training_s": wall_clock_training_s,
    }
    torch.save(artifact, path)


def save_training_state(
    path: Path,
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DistillationConfig,
    epoch_index: int,
    next_batch_idx: int,
    global_step: int,
    global_optimizer_step: int,
    fractional_epoch_value: float,
    wall_clock_training_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "fcos_distillation_training_state",
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "distillation_config": config.as_dict(),
            "epoch_index": int(epoch_index),
            "next_batch_idx": int(next_batch_idx),
            "global_step": int(global_step),
            "global_optimizer_step": int(global_optimizer_step),
            "fractional_epoch": float(fractional_epoch_value),
            "wall_clock_training_s": float(wall_clock_training_s),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_random_state": random.getstate(),
        },
        path,
    )


def load_training_state(
    path: Path,
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or state.get("artifact_type") != "fcos_distillation_training_state":
        raise RuntimeError(f"Invalid training state checkpoint: {path}")
    student.load_state_dict(state["student_state_dict"], strict=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    return state


def write_status(run_dir: Path, status: str, **extra: Any) -> None:
    write_json(
        run_dir / "run_status.json",
        {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def handle_numerical_instability(
    *,
    run_dir: Path,
    training_state_path: Path,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DistillationConfig,
    epoch_index: int,
    next_batch_idx: int,
    global_step: int,
    global_optimizer_step: int,
    fractional_epoch_value: float,
    wall_clock_training_s: float,
    reason: str,
    loss_components: dict[str, float],
    gradient_norm: float | None,
) -> None:
    save_training_state(
        training_state_path,
        student=student,
        optimizer=optimizer,
        config=config,
        epoch_index=epoch_index,
        next_batch_idx=next_batch_idx,
        global_step=global_step,
        global_optimizer_step=global_optimizer_step,
        fractional_epoch_value=fractional_epoch_value,
        wall_clock_training_s=wall_clock_training_s,
    )
    write_status(
        run_dir,
        "failed_numerical_instability",
        reason=reason,
        failing_global_step=global_step,
        failing_optimizer_step=global_optimizer_step,
        fractional_epoch=fractional_epoch_value,
        loss_components=loss_components,
        gradient_norm=gradient_norm,
    )


def terminal_run_status(run_dir: Path) -> str | None:
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = None
        if status in {"completed", "early_stopped"}:
            return str(status)
        if status:
            return None
    if (run_dir / "student_final.pt").exists():
        return "completed"
    return None


def copy_latest_checkpoint(source: Path, run_dir: Path) -> None:
    if source.exists():
        shutil.copy2(source, run_dir / "student_latest.pt")


def validation_ap_from_csv(results_csv: Path, checkpoint_label_value: str) -> float:
    if not results_csv.exists():
        raise RuntimeError(f"Validation results file was not written: {results_csv}")
    with results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("checkpoint_label") != checkpoint_label_value:
                continue
            if row.get("inference_status") != "ok":
                raise RuntimeError(f"Validation failed for {checkpoint_label_value}: {row.get('inference_status')}")
            raw_ap = row.get("validation_ap")
            if raw_ap in {None, ""}:
                raise RuntimeError(f"Validation AP unavailable for {checkpoint_label_value}.")
            ap = float(raw_ap)
            if not np.isfinite(ap):
                raise RuntimeError(f"Validation AP is non-finite for {checkpoint_label_value}: {ap}")
            return ap
    raise RuntimeError(f"Validation row missing for checkpoint {checkpoint_label_value}.")


def checkpoint_label_for_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("student_step_"):
        return stem.removeprefix("student_")
    if stem.startswith("student_epoch_"):
        return f"epoch_{int(stem.removeprefix('student_epoch_')):03d}"
    return stem


def evaluate_validation_ap(
    *,
    config: DistillationConfig,
    run_dir: Path,
    checkpoint_path: Path,
    python_executable: str,
) -> float:
    early = config.training.early_stopping
    checkpoint_name = checkpoint_path.name
    cmd = [
        python_executable,
        "scripts/select_pruned60_best_epoch.py",
        "--run-dir",
        str(run_dir),
        "--dataset",
        str(config.data.val_csv),
        "--split",
        early.validation_split,
        "--config-file",
        early.validation_config_file,
        "--guide-repo",
        config.data.guide_repo,
        "--img-dir",
        config.data.image_root,
        "--adapter",
        early.validation_adapter,
        "--python",
        python_executable,
        "--device",
        early.validation_device,
        "--batch-size",
        str(early.validation_batch_size),
        "--num-workers",
        str(early.validation_num_workers),
        "--det-thresh",
        str(early.validation_det_thresh),
        "--nms-thresh",
        str(early.validation_nms_thresh),
        "--overlap",
        str(early.validation_overlap),
        "--checkpoint-glob",
        checkpoint_name,
        "--metrics-subdir",
        "dense_validation_metrics",
        "--results-output-name",
        "dense_validation_results.csv",
        "--selection-output-name",
        "dense_best_checkpoint.json",
        "--best-checkpoint-name",
        "student_dense_best_ap.pt",
    ]
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Validation command failed for {checkpoint_name} with exit code {completed.returncode}.")
    return validation_ap_from_csv(run_dir / "dense_validation_results.csv", checkpoint_label_for_path(checkpoint_path))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def current_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def fractional_epoch(epoch: int, batches_completed: int, batches_per_epoch: int) -> float:
    if batches_per_epoch <= 0:
        return float(epoch)
    return float(epoch + min(max(batches_completed, 0), batches_per_epoch) / batches_per_epoch)


def make_metadata(
    *,
    config: DistillationConfig,
    run_dir: Path,
    requested_device: str,
    resolved_device: torch.device,
    teacher: torch.nn.Module | None,
    student: torch.nn.Module,
    smoke_test: bool,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "teacher_architecture": config.teacher.architecture if config.teacher else None,
        "teacher_checkpoint": config.teacher.checkpoint if config.teacher else None,
        "student_architecture": config.student.architecture,
        "student_checkpoint": config.student.checkpoint,
        "student_initialization": config.student.init_mode,
        "dataset_paths": {
            "train_csv": config.data.train_csv,
            "val_csv": config.data.val_csv,
            "calibration_csv": config.data.calibration_csv,
            "final_csv": config.data.final_csv,
            "final_split": config.data.final_split,
            "image_root": config.data.image_root,
            "dataset_type": config.data.dataset_type,
        },
        "loss_weights": config.distillation.__dict__,
        "batch_size": config.training.batch_size,
        "epochs": config.training.epochs,
        "optimizer": {
            "name": config.training.optimizer,
            "lr": config.training.lr,
            "weight_decay": config.training.weight_decay,
        },
        "deterministic": config.training.deterministic,
        "gradient_clip_val": config.training.gradient_clip_val,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "freeze_backbone": config.training.freeze_backbone,
        "save_every_steps": config.logging.save_every_steps,
        "checkpoint_steps": list(config.logging.checkpoint_steps),
        "seed": config.training.seed,
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": current_git_commit(),
        "package_versions": package_versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(run_dir),
        "teacher_parameters": count_parameters(teacher) if teacher else None,
        "student_parameters": count_parameters(student),
        "smoke_test": smoke_test,
    }


def run() -> None:
    args = parse_args()
    config = DistillationConfig.load(args.config)
    if args.experiment_name_suffix:
        config.experiment_name = f"{config.experiment_name}{args.experiment_name_suffix}"
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.device is not None:
        config.training.device = args.device
    if args.smoke_test:
        config.training.max_batches_per_epoch = min(config.training.max_batches_per_epoch or 2, 2)
        config.training.epochs = min(config.training.epochs, 1)
        config.data.num_train_samples = min(config.data.num_train_samples, 2)
        config.data.num_val_samples = min(config.data.num_val_samples, 1)
    if args.disable_early_stopping:
        config.training.early_stopping.enabled = False

    set_reproducibility(config.training.seed)
    set_determinism(config.training.deterministic)
    device = resolve_device(config.training.device)
    base_output_dir = Path(config.output_dir)
    preferred_run_dir = base_output_dir / config.experiment_name
    if args.force:
        args.overwrite = True
    terminal_status = terminal_run_status(preferred_run_dir)
    if args.resume_terminal:
        args.resume = True
    if terminal_status and not args.force and not args.resume_terminal:
        print(f"SKIP: {config.experiment_name} is already {terminal_status} in {preferred_run_dir}.")
        return
    if args.resume and preferred_run_dir.exists() and not args.overwrite:
        run_dir = preferred_run_dir
    else:
        run_dir = create_run_dir(base_output_dir, config.experiment_name, args.overwrite)
    logger = setup_logger(run_dir)
    write_status(run_dir, "running", config=str(args.config), resume=bool(args.resume))
    config.save_yaml(run_dir / "resolved_config.yaml")
    logger.info("Starting distillation run in %s", run_dir)

    teacher = None
    if config.distillation.enabled:
        if config.teacher is None:
            raise RuntimeError("Distillation is enabled, but no teacher model is configured.")
        teacher = create_model(config.teacher).to(device)
        freeze_model(teacher)
        if any(param.requires_grad for param in teacher.parameters()):
            raise RuntimeError("Teacher freeze check failed: at least one parameter requires gradients.")

    student = create_model(config.student).to(device)
    apply_student_freezing(student, config)
    set_student_training_mode(student, config)
    if not any(param.requires_grad for param in student.parameters()):
        raise RuntimeError("Student gradient check failed: no trainable parameters.")

    metadata = make_metadata(
        config=config,
        run_dir=run_dir,
        requested_device=config.training.device,
        resolved_device=device,
        teacher=teacher,
        student=student,
        smoke_test=args.smoke_test,
    )
    write_json(run_dir / "metadata.json", metadata)

    ObjectDetectionDataModule = import_guide_datamodule(Path(config.data.guide_repo))
    dataset_df = build_dataset_frame(config)
    datamodule = ObjectDetectionDataModule(
        img_dir=config.data.image_root,
        dataset=dataset_df,
        domain_col=config.data.domain_col,
        box_format=config.data.box_format,
        num_train_samples=config.data.num_train_samples,
        num_val_samples=config.data.num_val_samples,
        fg_prob=config.data.fg_prob,
        arb_prob=config.data.arb_prob,
        patch_size=config.student.patch_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader() if config.logging.evaluate_every else None
    write_json(
        run_dir / "reproducibility_audit.json",
        reproducibility_audit(
            config=config,
            dataset_df=dataset_df,
            train_loader=train_loader,
            resolved_device=device,
        ),
    )

    if config.training.optimizer.lower() != "adamw":
        raise ValueError("Only optimizer=AdamW is currently supported by this trainer.")
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    validate_freezing_and_optimizer(student=student, teacher=teacher, optimizer=optimizer, config=config)
    parameter_audit = collect_parameter_audit(student, optimizer, config)
    dataset_audit = verify_dataset_separation(config)
    parameter_audit.update(dataset_audit)
    parameter_audit["resolved_device"] = str(device)
    write_json(run_dir / "parameter_freezing_audit.json", parameter_audit)
    metadata.update(parameter_audit)
    write_json(run_dir / "metadata.json", metadata)
    logger.info(
        "Parameter audit: total=%d trainable=%d frozen=%d freeze_backbone=%s optimizer_params=%d",
        parameter_audit["student_total_parameters"],
        parameter_audit["student_trainable_parameters"],
        parameter_audit["student_frozen_parameters"],
        parameter_audit["student_backbone_frozen"],
        parameter_audit["optimizer_parameter_count"],
    )
    accumulation_steps = max(int(config.training.gradient_accumulation_steps), 1)
    global_step = 0
    global_optimizer_step = 0
    start_epoch = 0
    start_batch_idx = 0
    training_state_path = run_dir / "training_state_latest.pt"
    resume_state = load_training_state(training_state_path, student=student, optimizer=optimizer, device=device) if args.resume else None
    if resume_state:
        global_step = int(resume_state["global_step"])
        global_optimizer_step = int(resume_state["global_optimizer_step"])
        start_epoch = int(resume_state["epoch_index"])
        start_batch_idx = int(resume_state["next_batch_idx"])
        set_student_training_mode(student, config)
        logger.info(
            "Resumed training state from %s at epoch_index=%d next_batch_idx=%d optimizer_step=%d",
            training_state_path,
            start_epoch,
            start_batch_idx,
            global_optimizer_step,
        )
    active_terms_seen: set[str] = set()
    best_val_loss = float("inf")
    best_checkpoint = run_dir / "student_best.pt"
    training_started = time.perf_counter()
    early_state_path = run_dir / "early_stopping.json"
    early_state = EarlyStoppingState.load(early_state_path) if args.resume else EarlyStoppingState()
    checkpoint_steps = {int(step) for step in config.logging.checkpoint_steps}
    save_every_steps = config.logging.save_every_steps
    if save_every_steps is not None and int(save_every_steps) <= 0:
        raise ValueError("logging.save_every_steps must be a positive integer when set.")
    save_every_steps = int(save_every_steps) if save_every_steps is not None else None
    train_batches_per_epoch = len(train_loader)
    if config.training.max_batches_per_epoch is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, int(config.training.max_batches_per_epoch))

    def checkpoint_and_maybe_validate(
        checkpoint_path: Path,
        *,
        checkpoint_epoch: int,
        epoch_index: int,
        next_batch_idx: int,
        batch_idx: int | None,
        epoch_fraction_value: float,
    ) -> bool:
        save_student_checkpoint(
            checkpoint_path,
            student,
            config,
            checkpoint_epoch,
            global_optimizer_step=global_optimizer_step,
            batch_idx=batch_idx,
            fractional_epoch=epoch_fraction_value,
            learning_rate=current_learning_rate(optimizer),
            optimizer_learning_rates=current_learning_rates(optimizer),
            wall_clock_training_s=time.perf_counter() - training_started,
        )
        copy_latest_checkpoint(checkpoint_path, run_dir)
        student.to(device)
        save_training_state(
            training_state_path,
            student=student,
            optimizer=optimizer,
            config=config,
            epoch_index=epoch_index,
            next_batch_idx=next_batch_idx,
            global_step=global_step,
            global_optimizer_step=global_optimizer_step,
            fractional_epoch_value=epoch_fraction_value,
            wall_clock_training_s=time.perf_counter() - training_started,
        )
        append_jsonl(
            run_dir / "learning_rate_history.jsonl",
            {
                "checkpoint": str(checkpoint_path),
                "global_optimizer_step": global_optimizer_step,
                "fractional_epoch": epoch_fraction_value,
                "optimizer_learning_rates": current_learning_rates(optimizer),
            },
        )
        completed_epoch_for_stopping = (
            int(round(epoch_fraction_value))
            if math.isclose(epoch_fraction_value, round(epoch_fraction_value))
            else int(math.floor(epoch_fraction_value))
        )
        if should_validate_checkpoint(
            optimizer_step=global_optimizer_step,
            fractional_epoch=epoch_fraction_value,
            completed_epoch=completed_epoch_for_stopping,
            batches_per_epoch=train_batches_per_epoch,
            config=config.training.early_stopping,
        ):
            try:
                ap = evaluate_validation_ap(
                    config=config,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    python_executable=sys.executable,
                )
            except Exception as exc:
                record_validation_error(
                    early_state,
                    checkpoint=str(checkpoint_path),
                    optimizer_step=global_optimizer_step,
                    epoch_fraction=epoch_fraction_value,
                    error=f"{type(exc).__name__}: {exc}",
                ).save(early_state_path)
                append_jsonl(
                    run_dir / "validation_errors.jsonl",
                    {
                        "checkpoint": str(checkpoint_path),
                        "global_optimizer_step": global_optimizer_step,
                        "fractional_epoch": epoch_fraction_value,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                write_status(run_dir, "failed", reason="validation_failed", error=f"{type(exc).__name__}: {exc}")
                raise
            early_state_updated, should_stop = record_validation(
                early_state,
                validation_ap=ap,
                checkpoint=str(checkpoint_path),
                optimizer_step=global_optimizer_step,
                completed_epoch=completed_epoch_for_stopping,
                epoch_fraction=epoch_fraction_value,
                config=config.training.early_stopping,
            )
            early_state_updated.save(early_state_path)
            append_jsonl(
                run_dir / "validation_history.jsonl",
                {
                    "checkpoint": str(checkpoint_path),
                    "global_optimizer_step": global_optimizer_step,
                    "fractional_epoch": epoch_fraction_value,
                    "validation_ap": ap,
                    "best_validation_ap": early_state.best_validation_ap,
                    "phase": early_state.phase,
                    "warmup_completed": early_state.warmup_completed,
                    "patience_counter": early_state.patience_counter,
                    "non_improving_evaluations": early_state.non_improving_evaluations,
                },
            )
            if early_state.best_checkpoint:
                best_source = Path(early_state.best_checkpoint)
                if best_source.exists():
                    shutil.copy2(best_source, run_dir / "student_dense_best_ap.pt")
            if not should_stop:
                safety_stop, safety_reason = should_aggressive_lr_safety_stop(
                    early_state,
                    config=config.training.aggressive_lr_safety_stop,
                )
                if safety_stop:
                    early_state.stopped = True
                    early_state.phase = "stopped"
                    early_state.stopping_reason = safety_reason
                    early_state.final_optimizer_step = int(global_optimizer_step)
                    early_state.final_epoch_fraction = float(epoch_fraction_value)
                    early_state.save(early_state_path)
                    should_stop = True
            return should_stop
        return False

    if 0 in checkpoint_steps and global_optimizer_step == 0 and not (run_dir / "student_step_000000.pt").exists():
        should_stop = checkpoint_and_maybe_validate(
            run_dir / "student_step_000000.pt",
            checkpoint_epoch=0,
            epoch_index=0,
            next_batch_idx=0,
            batch_idx=None,
            epoch_fraction_value=0.0,
        )
        if should_stop:
            raise RuntimeError("Early stopping triggered at step 0, which should be impossible with min_epochs >= 1.")

    early_stopped = False
    for epoch in range(start_epoch, config.training.epochs):
        set_student_training_mode(student, config)
        optimizer.zero_grad(set_to_none=True)
        batches_in_epoch = 0
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue
            if config.training.max_batches_per_epoch is not None and batch_idx >= config.training.max_batches_per_epoch:
                break
            batches_in_epoch += 1
            images, targets = move_batch_to_device(batch, device)

            supervised_dict = student(images, targets)
            supervised_loss = sum(supervised_dict.values())

            teacher_outputs = {}
            student_outputs = {}
            if config.distillation.enabled:
                assert teacher is not None
                with torch.no_grad():
                    teacher_outputs = extract_fcos_head_outputs(teacher, images)
                student_outputs = extract_fcos_head_outputs(student, images)
            loss_result = compute_distillation_loss(
                supervised_loss=supervised_loss,
                teacher_outputs=teacher_outputs,
                student_outputs=student_outputs,
                config=config.distillation,
                logger=logger,
            )
            current_fractional_epoch = fractional_epoch(epoch, batch_idx + 1, train_batches_per_epoch)
            gradient_norm_value: float | None = None
            try:
                finite_or_raise(loss_result.total, global_step)
                finite_loss_components_or_raise(loss_result.components, global_step)
            except RuntimeError as exc:
                handle_numerical_instability(
                    run_dir=run_dir,
                    training_state_path=training_state_path,
                    student=student,
                    optimizer=optimizer,
                    config=config,
                    epoch_index=epoch,
                    next_batch_idx=batch_idx,
                    global_step=global_step,
                    global_optimizer_step=global_optimizer_step,
                    fractional_epoch_value=current_fractional_epoch,
                    wall_clock_training_s=time.perf_counter() - training_started,
                    reason=str(exc),
                    loss_components=loss_result.components,
                    gradient_norm=gradient_norm_value,
                )
                raise
            (loss_result.total / accumulation_steps).backward()
            should_step = (batches_in_epoch % accumulation_steps) == 0
            if should_step:
                try:
                    if config.training.numerical_safety.enabled:
                        check_gradients_finite_or_raise(student, global_step)
                    trainable_parameters = [param for param in student.parameters() if param.requires_grad]
                    if config.training.gradient_clip_val is not None and config.training.gradient_clip_val > 0:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_parameters,
                            max_norm=float(config.training.gradient_clip_val),
                        )
                        gradient_norm_value = float(gradient_norm.detach().cpu() if torch.is_tensor(gradient_norm) else gradient_norm)
                    else:
                        gradient_norm_value = float(
                            torch.linalg.vector_norm(
                                torch.stack(
                                    [
                                        param.grad.detach().norm()
                                        for param in trainable_parameters
                                        if param.grad is not None
                                    ]
                                )
                            ).detach().cpu()
                        )
                    if (
                        config.training.numerical_safety.enabled
                        and gradient_norm_value > float(config.training.numerical_safety.max_gradient_norm)
                    ):
                        raise RuntimeError(
                            f"Exploding gradient norm {gradient_norm_value:.6g} exceeds "
                            f"threshold {config.training.numerical_safety.max_gradient_norm}."
                        )
                except RuntimeError as exc:
                    handle_numerical_instability(
                        run_dir=run_dir,
                        training_state_path=training_state_path,
                        student=student,
                        optimizer=optimizer,
                        config=config,
                        epoch_index=epoch,
                        next_batch_idx=batch_idx,
                        global_step=global_step,
                        global_optimizer_step=global_optimizer_step,
                        fractional_epoch_value=current_fractional_epoch,
                        wall_clock_training_s=time.perf_counter() - training_started,
                        reason=str(exc),
                        loss_components=loss_result.components,
                        gradient_norm=gradient_norm_value,
                    )
                    raise
                optimizer.step()
                global_optimizer_step += 1
                try:
                    if config.training.numerical_safety.enabled:
                        check_parameters_finite_or_raise(student, global_optimizer_step)
                except RuntimeError as exc:
                    handle_numerical_instability(
                        run_dir=run_dir,
                        training_state_path=training_state_path,
                        student=student,
                        optimizer=optimizer,
                        config=config,
                        epoch_index=epoch,
                        next_batch_idx=batch_idx + 1,
                        global_step=global_step,
                        global_optimizer_step=global_optimizer_step,
                        fractional_epoch_value=current_fractional_epoch,
                        wall_clock_training_s=time.perf_counter() - training_started,
                        reason=str(exc),
                        loss_components=loss_result.components,
                        gradient_norm=gradient_norm_value,
                    )
                    raise
                optimizer.zero_grad(set_to_none=True)
                if config.training.freeze_backbone:
                    get_student_backbone_body(student).eval()
                should_save_step = (
                    global_optimizer_step in checkpoint_steps
                    or (save_every_steps is not None and global_optimizer_step % save_every_steps == 0)
                )
                if should_save_step:
                    early_stopped = checkpoint_and_maybe_validate(
                        run_dir / f"student_step_{global_optimizer_step:06d}.pt",
                        checkpoint_epoch=epoch + 1,
                        epoch_index=epoch,
                        next_batch_idx=batch_idx + 1,
                        batch_idx=batch_idx,
                        epoch_fraction_value=current_fractional_epoch,
                    )
                    if early_stopped:
                        break

            active_terms_seen.update(loss_result.active_terms)
            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                "global_optimizer_step": global_optimizer_step,
                "fractional_epoch": current_fractional_epoch,
                "learning_rate": current_learning_rate(optimizer),
                "optimizer_learning_rates": json.dumps(current_learning_rates(optimizer)),
                "gradient_norm": gradient_norm_value if gradient_norm_value is not None else "",
                **loss_result.components,
                **loss_result.diagnostics,
                "active_terms": "|".join(loss_result.active_terms),
                "optimizer_step": int(should_step),
                "gradient_accumulation_steps": accumulation_steps,
            }
            append_training_log(run_dir / "training_log.csv", row)
            append_jsonl(run_dir / "loss_history.jsonl", row)
            if global_step % max(config.logging.log_interval, 1) == 0:
                logger.info("epoch=%d batch=%d loss=%.6f active=%s", epoch, batch_idx, loss_result.components["total"], row["active_terms"])
            global_step += 1
        start_batch_idx = 0
        if early_stopped:
            break
        if batches_in_epoch and (batches_in_epoch % accumulation_steps) != 0:
            if config.training.gradient_clip_val is not None and config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in student.parameters() if param.requires_grad],
                    max_norm=float(config.training.gradient_clip_val),
                )
            optimizer.step()
            global_optimizer_step += 1
            optimizer.zero_grad(set_to_none=True)
            if config.training.freeze_backbone:
                get_student_backbone_body(student).eval()
            should_save_step = (
                global_optimizer_step in checkpoint_steps
                or (save_every_steps is not None and global_optimizer_step % save_every_steps == 0)
            )
            if should_save_step:
                early_stopped = checkpoint_and_maybe_validate(
                    run_dir / f"student_step_{global_optimizer_step:06d}.pt",
                    checkpoint_epoch=epoch + 1,
                    epoch_index=epoch + 1,
                    next_batch_idx=0,
                    batch_idx=batches_in_epoch - 1,
                    epoch_fraction_value=fractional_epoch(epoch, train_batches_per_epoch, train_batches_per_epoch),
                )
                if early_stopped:
                    break

        if config.logging.save_every and (epoch + 1) % config.logging.save_every == 0:
            early_stopped = checkpoint_and_maybe_validate(
                run_dir / f"student_epoch_{epoch + 1}.pt",
                checkpoint_epoch=epoch + 1,
                epoch_index=epoch + 1,
                next_batch_idx=0,
                batch_idx=batches_in_epoch - 1 if batches_in_epoch else None,
                epoch_fraction_value=float(epoch + 1),
            )
            if early_stopped:
                break

        if val_loader is not None and (epoch + 1) % config.logging.evaluate_every == 0:
            val_loss = evaluate_loss(
                teacher=teacher,
                student=student,
                dataloader=val_loader,
                config=config,
                device=device,
                logger=logger,
            )
            append_training_log(
                run_dir / "training_log.csv",
                {
                    "epoch": epoch,
                    "batch": "val",
                    "global_step": global_step,
                    "global_optimizer_step": global_optimizer_step,
                    "fractional_epoch": float(epoch + 1),
                    "learning_rate": current_learning_rate(optimizer),
                    "optimizer_learning_rates": json.dumps(current_learning_rates(optimizer)),
                    "gradient_norm": "",
                    "supervised": "",
                    "cls_distill": "",
                    "box_distill": "",
                    "centerness_distill": "",
                    "total": val_loss,
                    "active_terms": "validation",
                },
            )
            logger.info("epoch=%d val_loss=%.6f", epoch, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_student_checkpoint(
                    best_checkpoint,
                    student,
                    config,
                    epoch + 1,
                    global_optimizer_step=global_optimizer_step,
                    batch_idx=batches_in_epoch - 1 if batches_in_epoch else None,
                    fractional_epoch=float(epoch + 1),
                    learning_rate=current_learning_rate(optimizer),
                    optimizer_learning_rates=current_learning_rates(optimizer),
                    wall_clock_training_s=time.perf_counter() - training_started,
                )
                student.to(device)

    final_checkpoint = run_dir / "student_final.pt"
    final_epoch_fraction_value = (
        float(early_state.final_epoch_fraction)
        if early_stopped and early_state.final_epoch_fraction is not None
        else float(min(config.training.epochs, global_optimizer_step / train_batches_per_epoch if train_batches_per_epoch else 0.0))
    )
    final_epoch_value = int(final_epoch_fraction_value) if float(final_epoch_fraction_value).is_integer() else int(final_epoch_fraction_value) + 1
    save_student_checkpoint(
        final_checkpoint,
        student,
        config,
        final_epoch_value,
        global_optimizer_step=global_optimizer_step,
        batch_idx=None,
        fractional_epoch=final_epoch_fraction_value,
        learning_rate=current_learning_rate(optimizer),
        optimizer_learning_rates=current_learning_rates(optimizer),
        wall_clock_training_s=time.perf_counter() - training_started,
    )
    copy_latest_checkpoint(final_checkpoint, run_dir)
    save_training_state(
        training_state_path,
        student=student,
        optimizer=optimizer,
        config=config,
        epoch_index=min(final_epoch_value, config.training.epochs),
        next_batch_idx=0,
        global_step=global_step,
        global_optimizer_step=global_optimizer_step,
        fractional_epoch_value=final_epoch_fraction_value,
        wall_clock_training_s=time.perf_counter() - training_started,
    )

    smoke_report = {
        "teacher_loaded": teacher is not None,
        "teacher_parameters_frozen": None if teacher is None else not any(param.requires_grad for param in teacher.parameters()),
        "student_has_trainable_parameters": any(param.requires_grad for param in student.parameters()),
        "batches_completed": global_step,
        "optimizer_steps_completed": global_optimizer_step,
        "active_distillation_terms": sorted(active_terms_seen),
        "final_checkpoint": str(final_checkpoint),
        "metadata_written": (run_dir / "metadata.json").exists(),
        "training_log_written": (run_dir / "training_log.csv").exists(),
    }
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    reloaded = checkpoint.get("model_object") or create_model(config.student)
    if "model_object" not in checkpoint:
        reloaded.load_state_dict(checkpoint["state_dict"], strict=False)
    reloaded.eval()
    with torch.no_grad():
        inference_output = reloaded([torch.zeros(3, config.student.patch_size, config.student.patch_size)])
    smoke_report["saved_student_checkpoint_loads"] = True
    smoke_report["saved_student_checkpoint_runs_inference"] = isinstance(inference_output, list)
    if args.smoke_test:
        write_json(run_dir / "smoke_test_report.json", smoke_report)
    metadata["final_checkpoint"] = str(final_checkpoint)
    metadata["best_checkpoint"] = str(best_checkpoint) if best_checkpoint.exists() else None
    metadata["best_ap_checkpoint"] = str(run_dir / "student_dense_best_ap.pt") if (run_dir / "student_dense_best_ap.pt").exists() else None
    metadata["best_val_loss"] = best_val_loss if best_checkpoint.exists() else None
    metadata["active_distillation_terms"] = sorted(active_terms_seen)
    metadata["batches_completed"] = global_step
    metadata["optimizer_steps_completed"] = global_optimizer_step
    metadata["final_epoch_fraction"] = final_epoch_fraction_value
    metadata["run_status"] = "early_stopped" if early_stopped else "completed"
    metadata["early_stopping"] = config.training.early_stopping.__dict__
    metadata["aggressive_lr_safety_stop"] = config.training.aggressive_lr_safety_stop.__dict__
    write_json(run_dir / "metadata.json", metadata)
    run_summary = {
        "experiment_name": config.experiment_name,
        "status": "early_stopped" if early_stopped else "completed",
        "final_checkpoint": str(final_checkpoint),
        "latest_checkpoint": str(run_dir / "student_latest.pt"),
        "best_checkpoint": str(run_dir / "student_dense_best_ap.pt") if (run_dir / "student_dense_best_ap.pt").exists() else None,
        "final_optimizer_step": global_optimizer_step,
        "final_epoch_fraction": final_epoch_fraction_value,
        "max_epochs": config.training.epochs,
        "early_stopping": early_state.__dict__,
    }
    write_json(run_dir / "run_summary.json", run_summary)
    write_status(
        run_dir,
        "early_stopped" if early_stopped else "completed",
        final_optimizer_step=global_optimizer_step,
        final_epoch_fraction=final_epoch_fraction_value,
        reason=early_state.stopping_reason if early_stopped else "max_epochs_reached",
    )
    logger.info("Finished distillation run. Final checkpoint: %s", final_checkpoint)


if __name__ == "__main__":
    run()
