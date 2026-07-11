from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    architecture: str
    checkpoint: str | None = None
    freeze: bool = False
    pretrained_backbone: bool = True
    init_mode: str = "imagenet"
    eval_config: str | None = None
    num_classes: int = 2
    patch_size: int = 1024
    det_thresh: float = 0.05
    weights: str | None = "IMAGENET1K_V2"
    returned_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    extra_blocks: bool = False


@dataclass
class DataConfig:
    train_csv: str
    val_csv: str | None = None
    image_root: str = "data/midogpp"
    dataset_type: str = "midog_guide"
    box_format: str = "cxcy"
    domain_col: str | None = None
    num_train_samples: int = 1024
    num_val_samples: int = 128
    fg_prob: float = 0.5
    arb_prob: float = 0.25
    guide_repo: str = "repos/MIDOG_2025_Guide"


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 123
    device: str = "auto"
    max_batches_per_epoch: int | None = None


@dataclass
class DistillationLossConfig:
    enabled: bool = True
    temperature: float = 2.0
    lambda_supervised: float = 1.0
    lambda_cls: float = 0.5
    lambda_box: float = 0.25
    lambda_centerness: float = 0.25
    match_outputs: bool = True
    skip_mismatched_shapes: bool = True


@dataclass
class LoggingConfig:
    save_every: int = 1
    evaluate_every: int = 0
    log_interval: int = 10


@dataclass
class DistillationConfig:
    experiment_name: str
    output_dir: str
    teacher: ModelSpec
    student: ModelSpec
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: Path) -> "DistillationConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DistillationConfig":
        return cls(
            experiment_name=raw["experiment_name"],
            output_dir=raw["output_dir"],
            teacher=ModelSpec(**raw["teacher"]),
            student=ModelSpec(**raw["student"]),
            data=DataConfig(**raw["data"]),
            training=TrainingConfig(**raw.get("training", {})),
            distillation=DistillationLossConfig(**raw.get("distillation", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.as_dict(), sort_keys=False), encoding="utf-8")

