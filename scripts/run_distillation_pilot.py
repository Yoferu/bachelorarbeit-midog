#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_fcos_distillation import (  # noqa: E402
    build_dataset_frame, import_guide_datamodule, move_batch_to_device,
    resolve_device, save_student_checkpoint, set_reproducibility,
)
from src.benchmark.distilled_model_loader import load_distilled_student  # noqa: E402
from src.distillation.config import DistillationConfig  # noqa: E402
from src.distillation.logging_utils import append_training_log, create_run_dir, setup_logger, write_json  # noqa: E402
from src.distillation.losses import compute_distillation_loss  # noqa: E402
from src.distillation.model_registry import create_model, freeze_model  # noqa: E402
from src.distillation.output_matching import extract_fcos_head_outputs  # noqa: E402


def gradient_norm(model):
    values = [p.grad.detach().float().norm(2).square() for p in model.parameters() if p.grad is not None]
    norm = torch.stack(values).sum().sqrt() if values else torch.zeros(())
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    return float(norm.cpu()), finite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = DistillationConfig.load(args.config)
    set_reproducibility(config.training.seed)
    device = resolve_device(config.training.device)
    run_dir = create_run_dir(Path(config.output_dir), config.experiment_name, overwrite=False)
    logger = setup_logger(run_dir)
    config.save_yaml(run_dir / "resolved_config.yaml")

    teacher = create_model(config.teacher).to(device)
    freeze_model(teacher)
    student = create_model(config.student).to(device).train()
    DataModule = import_guide_datamodule(Path(config.data.guide_repo))
    dm = DataModule(
        img_dir=config.data.image_root, dataset=build_dataset_frame(config),
        domain_col=config.data.domain_col, box_format=config.data.box_format,
        num_train_samples=config.data.num_train_samples, num_val_samples=config.data.num_val_samples,
        fg_prob=config.data.fg_prob, arb_prob=config.data.arb_prob,
        patch_size=config.student.patch_size, batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    totals = []
    matched = {}
    skipped = {}
    for step, batch in enumerate(dm.train_dataloader()):
        if step >= (config.training.max_batches_per_epoch or 20):
            break
        images, targets = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        supervised = sum(student(images, targets).values())
        with torch.no_grad():
            teacher_outputs = extract_fcos_head_outputs(teacher, images)
        student_outputs = extract_fcos_head_outputs(student, images)
        result = compute_distillation_loss(
            supervised_loss=supervised, teacher_outputs=teacher_outputs,
            student_outputs=student_outputs, config=config.distillation, logger=logger,
        )
        if not bool(torch.isfinite(result.total)):
            raise RuntimeError(f"Non-finite loss at step {step}")
        result.total.backward()
        grad, grad_finite = gradient_norm(student)
        if not grad_finite or not math.isfinite(grad):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        optimizer.step()
        totals.append(result.components["total"])
        row = {"epoch": 0, "batch": step, "global_step": step, **result.components,
               "gradient_norm": grad, "losses_finite": True, "gradients_finite": grad_finite,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        for key, value in result.diagnostics.items():
            row[key] = json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
            if key.endswith("_matched_levels"): matched[key] = value
            if key.endswith("_skipped_levels"): skipped[key] = value
        append_training_log(run_dir / "training_log.csv", row)
        logger.info("step=%d total=%.6f supervised=%.6f cls=%.6f box=%.6f ctr=%.6f grad=%.6f",
                    step, result.components["total"], result.components["supervised"],
                    result.components["cls_distill"], result.components["box_distill"],
                    result.components["centerness_distill"], grad)

    validation_totals = []
    for val_step, batch in enumerate(dm.val_dataloader()):
        if val_step >= config.data.num_val_samples:
            break
        images, targets = move_batch_to_device(batch, device)
        student.train()
        with torch.no_grad():
            supervised = sum(student(images, targets).values())
            teacher_outputs = extract_fcos_head_outputs(teacher, images)
            student_outputs = extract_fcos_head_outputs(student, images)
            result = compute_distillation_loss(
                supervised_loss=supervised, teacher_outputs=teacher_outputs,
                student_outputs=student_outputs, config=config.distillation, logger=logger,
            )
        validation_totals.append(result.components["total"])

    checkpoint = run_dir / "student_final.pt"
    save_student_checkpoint(checkpoint, student, config, 1)
    reloaded, source = load_distilled_student(checkpoint)
    reloaded.eval()
    with torch.inference_mode(): output = reloaded([torch.zeros(3, config.student.patch_size, config.student.patch_size)])
    stable = all(math.isfinite(v) for v in totals)
    report = {
        "student_architecture": config.student.architecture, "teacher_architecture": config.teacher.architecture,
        "requested_device": config.training.device, "resolved_device": str(device), "batches_completed": len(totals),
        "teacher_parameters_frozen": not any(p.requires_grad for p in teacher.parameters()),
        "student_has_trainable_parameters": any(p.requires_grad for p in student.parameters()),
        "all_losses_finite": stable, "initial_total_loss": totals[0], "final_total_loss": totals[-1],
        "minimum_total_loss": min(totals), "maximum_total_loss": max(totals),
        "validation_batches": len(validation_totals),
        "validation_total_loss_mean": float(np.mean(validation_totals)) if validation_totals else None,
        "matched_levels": matched, "skipped_levels": skipped, "checkpoint": str(checkpoint),
        "loading_method": "registry_state_dict", "metadata_source": source,
        "checkpoint_reload_successful": True, "cpu_inference_successful": isinstance(output, list),
    }
    write_json(run_dir / "pilot_report.json", report)
    write_json(run_dir / "runtime_metadata.json", {
        "loading_method": "registry_state_dict", "metadata_source": source, "device": "cpu",
        "checkpoint_reload_successful": True, "inference_successful": isinstance(output, list),
        "full_evaluation_run": False,
    })
    print(run_dir)


if __name__ == "__main__":
    main()
