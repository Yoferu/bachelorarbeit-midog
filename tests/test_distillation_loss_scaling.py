import torch

from src.distillation.losses import _classification_kd
from src.distillation.output_matching import MatchedOutput


def test_classification_kd_is_invariant_to_repeated_anchors():
    torch.manual_seed(42)
    teacher = torch.randn(1, 100, 2)
    student = torch.randn(1, 100, 2)
    base = _classification_kd([MatchedOutput("cls", teacher, student)], temperature=2.0)
    repeated = _classification_kd(
        [MatchedOutput("cls", teacher.repeat(1, 10, 1), student.repeat(1, 10, 1))],
        temperature=2.0,
    )
    assert torch.allclose(base, repeated, rtol=1e-5, atol=1e-7)
