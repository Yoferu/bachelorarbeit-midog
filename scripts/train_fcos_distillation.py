#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
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
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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



def evaluate_loss(
    *,
    teacher: torch.nn.Module,
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
            student.train()
            supervised_dict = student(images, targets)
            supervised_loss = sum(supervised_dict.values())
            student.eval()
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
    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))

def save_student_checkpoint(path: Path, student: torch.nn.Module, config: DistillationConfig, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_student = student.to("cpu")
    artifact = {
        "artifact_type": "fcos_distilled_student",
        "model_object": cpu_student,
        "state_dict": cpu_student.state_dict(),
        "student_architecture": config.student.architecture,
        "teacher_architecture": config.teacher.architecture,
        "distillation_config": config.as_dict(),
        "epoch": epoch,
    }
    torch.save(artifact, path)


def make_metadata(
    *,
    config: DistillationConfig,
    run_dir: Path,
    requested_device: str,
    resolved_device: torch.device,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    smoke_test: bool,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "teacher_architecture": config.teacher.architecture,
        "teacher_checkpoint": config.teacher.checkpoint,
        "student_architecture": config.student.architecture,
        "student_checkpoint": config.student.checkpoint,
        "student_initialization": config.student.init_mode,
        "dataset_paths": {
            "train_csv": config.data.train_csv,
            "val_csv": config.data.val_csv,
            "image_root": config.data.image_root,
            "dataset_type": config.data.dataset_type,
        },
        "loss_weights": config.distillation.__dict__,
        "batch_size": config.training.batch_size,
        "epochs": config.training.epochs,
        "optimizer": {"name": "AdamW", "lr": config.training.lr, "weight_decay": config.training.weight_decay},
        "seed": config.training.seed,
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": current_git_commit(),
        "package_versions": package_versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(run_dir),
        "teacher_parameters": count_parameters(teacher),
        "student_parameters": count_parameters(student),
        "smoke_test": smoke_test,
    }


def run() -> None:
    args = parse_args()
    config = DistillationConfig.load(args.config)
    if args.smoke_test:
        config.training.max_batches_per_epoch = min(config.training.max_batches_per_epoch or 2, 2)
        config.training.epochs = min(config.training.epochs, 1)
        config.data.num_train_samples = min(config.data.num_train_samples, 2)
        config.data.num_val_samples = min(config.data.num_val_samples, 1)

    set_reproducibility(config.training.seed)
    device = resolve_device(config.training.device)
    run_dir = create_run_dir(Path(config.output_dir), config.experiment_name, args.overwrite)
    logger = setup_logger(run_dir)
    config.save_yaml(run_dir / "resolved_config.yaml")
    logger.info("Starting distillation run in %s", run_dir)

    teacher = create_model(config.teacher).to(device)
    freeze_model(teacher)
    if any(param.requires_grad for param in teacher.parameters()):
        raise RuntimeError("Teacher freeze check failed: at least one parameter requires gradients.")

    student = create_model(config.student).to(device)
    student.train()
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

    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    global_step = 0
    active_terms_seen: set[str] = set()
    best_val_loss = float("inf")
    best_checkpoint = run_dir / "student_best.pt"

    for epoch in range(config.training.epochs):
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            if config.training.max_batches_per_epoch is not None and batch_idx >= config.training.max_batches_per_epoch:
                break
            images, targets = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            supervised_dict = student(images, targets)
            supervised_loss = sum(supervised_dict.values())

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
            finite_or_raise(loss_result.total, global_step)
            loss_result.total.backward()
            optimizer.step()

            active_terms_seen.update(loss_result.active_terms)
            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                **loss_result.components,
                "active_terms": "|".join(loss_result.active_terms),
            }
            append_training_log(run_dir / "training_log.csv", row)
            if global_step % max(config.logging.log_interval, 1) == 0:
                logger.info("epoch=%d batch=%d loss=%.6f active=%s", epoch, batch_idx, loss_result.components["total"], row["active_terms"])
            global_step += 1

        if config.logging.save_every and (epoch + 1) % config.logging.save_every == 0:
            save_student_checkpoint(run_dir / f"student_epoch_{epoch + 1}.pt", student, config, epoch + 1)
            student.to(device)

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
                save_student_checkpoint(best_checkpoint, student, config, epoch + 1)
                student.to(device)

    final_checkpoint = run_dir / "student_final.pt"
    save_student_checkpoint(final_checkpoint, student, config, config.training.epochs)

    smoke_report = {
        "teacher_loaded": True,
        "teacher_parameters_frozen": not any(param.requires_grad for param in teacher.parameters()),
        "student_has_trainable_parameters": any(param.requires_grad for param in student.parameters()),
        "batches_completed": global_step,
        "active_distillation_terms": sorted(active_terms_seen),
        "final_checkpoint": str(final_checkpoint),
        "metadata_written": (run_dir / "metadata.json").exists(),
        "training_log_written": (run_dir / "training_log.csv").exists(),
    }
    reloaded = create_model(config.student)
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
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
    metadata["best_val_loss"] = best_val_loss if best_checkpoint.exists() else None
    metadata["active_distillation_terms"] = sorted(active_terms_seen)
    metadata["batches_completed"] = global_step
    write_json(run_dir / "metadata.json", metadata)
    logger.info("Finished distillation run. Final checkpoint: %s", final_checkpoint)


if __name__ == "__main__":
    run()
