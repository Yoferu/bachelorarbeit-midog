import json

import pytest
import torch
import yaml

from src.benchmark.distilled_model_loader import load_distilled_student
from src.distillation.config import ModelSpec
from src.distillation.model_registry import create_model


@pytest.mark.parametrize(
    "architecture",
    ["fcos_mobilenetv3_small_fpn", "fcos_mobilenetv3_large_fpn"],
)
def test_mobilenetv3_student_registry_reload_and_inference(tmp_path, architecture):
    spec = ModelSpec(architecture=architecture, pretrained_backbone=False, init_mode="random", patch_size=64)
    original = create_model(spec)
    checkpoint = tmp_path / "student_final.pt"
    torch.save({"artifact_type": "fcos_distilled_student", "state_dict": original.state_dict()}, checkpoint)
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump({"student": vars(spec)}, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"student_architecture": architecture}), encoding="utf-8"
    )

    reloaded, source = load_distilled_student(checkpoint)
    assert "metadata.json" in source
    reloaded.eval()
    with torch.inference_mode():
        outputs = reloaded([torch.rand(3, 64, 64)])
    assert len(outputs) == 1
    assert {"boxes", "labels", "scores"}.issubset(outputs[0])
