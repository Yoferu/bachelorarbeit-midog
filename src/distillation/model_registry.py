from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.fcos import FCOS
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models import resnet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from src.distillation.config import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_REPO = PROJECT_ROOT / "repos" / "MIDOG_2025_Guide"

ALIASES = {
    "fcos18": "fcos18",
    "fcos_18": "fcos18",
    "FCOS_18": "fcos18",
    "fcos_x50": "fcos_x50",
    "FCOS_x50": "fcos_x50",
    "fcos_x101": "fcos_x101",
    "FCOS_x101": "fcos_x101",
    "fcos_mobilenetv3_small_fpn": "fcos_mobilenetv3_small_fpn",
    "fcos_mobilenetv3_large_fpn": "fcos_mobilenetv3_large_fpn",
}

RESNET_ARCH = {
    "fcos18": "resnet18",
    "fcos_x50": "resnext50_32x4d",
    "fcos_x101": "resnext101_32x8d",
}


def canonical_architecture(name: str) -> str:
    if name not in ALIASES:
        raise ValueError(f"Unknown model architecture '{name}'. Known: {sorted(ALIASES)}")
    return ALIASES[name]


class MobileNetV3FPNBackbone(nn.Module):
    """MobileNetV3 features adapted to 256-channel FPN outputs.

    FPN inputs:
    - MobileNetV3-Small: feature layers 1 (stride 4), 2 (stride 8), 8 (stride 16), 12 (stride 32)
    - MobileNetV3-Large: feature layers 2 (stride 4), 4 (stride 8), 12 (stride 16), 16 (stride 32)
    A LastLevelMaxPool block adds a fifth coarser FCOS level.
    """

    out_channels = 256

    def __init__(self, variant: str, pretrained: bool) -> None:
        super().__init__()
        if variant == "small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v3_small(weights=weights).features
            return_layers = {"1": "0", "2": "1", "8": "2", "12": "3"}
            in_channels = [16, 24, 48, 576]
        elif variant == "large":
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = mobilenet_v3_large(weights=weights).features
            return_layers = {"2": "0", "4": "1", "12": "2", "16": "3"}
            in_channels = [24, 40, 112, 960]
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.fpn(self.body(x))


def make_resnet_fcos(spec: ModelSpec) -> FCOS:
    arch = canonical_architecture(spec.architecture)
    backbone_name = RESNET_ARCH[arch]
    weights = spec.weights if spec.pretrained_backbone else None
    body = resnet.__dict__[backbone_name](weights=weights)
    backbone = _resnet_fpn_extractor(
        backbone=body,
        trainable_layers=5,
        returned_layers=spec.returned_layers,
        extra_blocks=LastLevelMaxPool(),
    )
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    anchor_sizes = tuple([anchor_sizes[0]] + [anchor_sizes[i] for i in spec.returned_layers])
    anchor_generator = AnchorGenerator(anchor_sizes, ((1.0,),) * len(anchor_sizes))
    return FCOS(
        backbone,
        spec.num_classes,
        anchor_generator=anchor_generator,
        min_size=spec.patch_size,
        max_size=spec.patch_size,
        score_thresh=spec.det_thresh,
        nms_thresh=0.6,
        detections_per_img=300,
        topk_candidates=1000,
    )


def make_mobilenet_fcos(spec: ModelSpec, variant: str) -> FCOS:
    pretrained = bool(spec.pretrained_backbone and spec.init_mode.lower() in {"imagenet", "pretrained"})
    backbone = MobileNetV3FPNBackbone(variant=variant, pretrained=pretrained)
    anchor_generator = AnchorGenerator(((8,), (16,), (32,), (64,), (128,)), ((1.0,),) * 5)
    return FCOS(
        backbone,
        spec.num_classes,
        anchor_generator=anchor_generator,
        min_size=spec.patch_size,
        max_size=spec.patch_size,
        score_thresh=spec.det_thresh,
        nms_thresh=0.6,
        detections_per_img=300,
        topk_candidates=1000,
    )


def create_model(spec: ModelSpec) -> nn.Module:
    arch = canonical_architecture(spec.architecture)
    if arch in RESNET_ARCH:
        model = make_resnet_fcos(spec)
    elif arch == "fcos_mobilenetv3_small_fpn":
        model = make_mobilenet_fcos(spec, "small")
    elif arch == "fcos_mobilenetv3_large_fpn":
        model = make_mobilenet_fcos(spec, "large")
    else:
        raise ValueError(f"Unsupported architecture: {spec.architecture}")
    if spec.checkpoint:
        load_checkpoint(model, Path(spec.checkpoint))
    return model


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
    else:
        raise TypeError("Checkpoint is not a state_dict-like object.")

    normalized = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> tuple[list[str], list[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _state_dict_from_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)


def freeze_model(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

