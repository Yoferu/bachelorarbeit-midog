#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fcos_distillation import save_student_checkpoint
from src.benchmark.distilled_model_loader import load_distilled_student
from src.distillation.config import DistillationConfig
from src.distillation.losses import compute_distillation_loss
from src.distillation.model_registry import count_parameters, create_model, freeze_model
from src.distillation.output_matching import extract_fcos_head_outputs, match_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU smoke checks for pruned-60 recovery configs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pruned60_recovery_smoke"))
    parser.add_argument("--include-teacher", action="store_true")
    parser.add_argument("--patch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DistillationConfig.load(args.config)
    config.training.device = "cpu"

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    student = create_model(config.student).cpu()
    student.train()
    student_parameters = count_parameters(student)
    if not (15_500_000 <= student_parameters <= 15_700_000):
        raise RuntimeError(f"Unexpected structural student parameter count: {student_parameters}")
    if not any(param.requires_grad for param in student.parameters()):
        raise RuntimeError("Student has no trainable parameters.")

    image = torch.zeros(3, args.patch_size, args.patch_size)
    target = {
        "boxes": torch.tensor([[32.0, 32.0, 96.0, 96.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    loss_dict = student([image], [target])
    supervised_loss = sum(loss_dict.values())
    if not torch.isfinite(supervised_loss):
        raise RuntimeError(f"Non-finite supervised loss: {float(supervised_loss.detach())}")

    with torch.no_grad():
        student_outputs = extract_fcos_head_outputs(student, [image])
    student_level_counts = {key: len(value) for key, value in student_outputs.items()}
    if any(count != 5 for count in student_level_counts.values()):
        raise RuntimeError(f"Expected five FCOS output levels, got {student_level_counts}")

    teacher_loaded = False
    teacher_frozen = None
    matched_level_counts = {}
    distillation_total = None
    if config.distillation.enabled and args.include_teacher:
        if config.teacher is None:
            raise RuntimeError("Distillation config has no teacher.")
        teacher = create_model(config.teacher).cpu()
        freeze_model(teacher)
        teacher_loaded = True
        teacher_frozen = not any(param.requires_grad for param in teacher.parameters())
        with torch.no_grad():
            teacher_outputs = extract_fcos_head_outputs(teacher, [image])
        matched = match_outputs(
            teacher_outputs,
            student_outputs,
            skip_mismatched_shapes=config.distillation.skip_mismatched_shapes,
            logger=type("SmokeLogger", (), {"warning": lambda *a, **k: None})(),
        )
        matched_level_counts = {key: len(value) for key, value in matched.items()}
        result = compute_distillation_loss(
            supervised_loss=supervised_loss,
            teacher_outputs=teacher_outputs,
            student_outputs=student_outputs,
            config=config.distillation,
            logger=type("SmokeLogger", (), {"warning": lambda *a, **k: None})(),
        )
        if not torch.isfinite(result.total):
            raise RuntimeError(f"Non-finite distillation loss: {float(result.total.detach())}")
        distillation_total = float(result.total.detach())

    checkpoint_path = args.output_dir / "student_final.pt"
    save_student_checkpoint(checkpoint_path, student, config, epoch=0)
    reloaded, source = load_distilled_student(checkpoint_path)
    reloaded.eval()
    with torch.no_grad():
        inference = reloaded([image])
    if not isinstance(inference, list) or not {"boxes", "labels", "scores"}.issubset(inference[0]):
        raise RuntimeError("Reloaded checkpoint did not produce FCOS inference output.")

    report = {
        "config": str(args.config),
        "student_parameters": student_parameters,
        "supervised_loss": float(supervised_loss.detach()),
        "student_trainable": True,
        "student_output_levels": student_level_counts,
        "checkpoint_path": str(checkpoint_path),
        "reload_source": source,
        "reloaded_inference_ok": True,
        "teacher_loaded": teacher_loaded,
        "teacher_frozen": teacher_frozen,
        "matched_level_counts": matched_level_counts,
        "distillation_total": distillation_total,
    }
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
