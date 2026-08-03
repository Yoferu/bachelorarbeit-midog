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
    calibration_csv: str | None = None
    final_csv: str | None = None
    final_split: str = "test"
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
    optimizer: str = "AdamW"
    deterministic: bool = False
    gradient_clip_val: float | None = 1.0
    gradient_accumulation_steps: int = 1
    freeze_backbone: bool = False
    early_stopping: "EarlyStoppingConfig" = field(default_factory=lambda: EarlyStoppingConfig())
    aggressive_lr_safety_stop: "AggressiveLRSafetyStopConfig" = field(default_factory=lambda: AggressiveLRSafetyStopConfig())
    numerical_safety: "NumericalSafetyConfig" = field(default_factory=lambda: NumericalSafetyConfig())


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    metric: str = "validation_ap"
    mode: str = "max"
    min_epochs: int = 1
    patience_evaluations: int = 5
    min_delta: float = 0.0001
    validation_interval_steps: int = 50
    restore_best_checkpoint: bool = False
    validation_device: str = "cuda"
    validation_batch_size: int = 1
    validation_num_workers: int = 4
    validation_det_thresh: float = 0.584
    validation_nms_thresh: float = 0.3
    validation_overlap: float = 0.3
    validation_config_file: str = "experiments/eval_guide/configs/FCOS_18_eval.yaml"
    validation_adapter: str = "src/benchmark/midog_guide_adapter.py"
    validation_split: str = "val"


@dataclass
class AggressiveLRSafetyStopConfig:
    enabled: bool = False
    label: str = "aggressive_lr_safety_stop"
    max_epoch_fraction: float = 1.0
    baseline_step: int = 0
    min_post_baseline_validations: int = 5
    min_delta: float = 0.0001
    degradation_margin: float = 0.0005
    trend_window: int = 5


@dataclass
class NumericalSafetyConfig:
    enabled: bool = True
    max_gradient_norm: float = 1000.0


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
    save_every_steps: int | None = None
    checkpoint_steps: list[int] = field(default_factory=list)


@dataclass
class DistillationConfig:
    experiment_name: str
    output_dir: str
    teacher: ModelSpec | None
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
        training_raw = dict(raw.get("training", {}))
        if isinstance(training_raw.get("early_stopping"), dict):
            training_raw["early_stopping"] = EarlyStoppingConfig(**training_raw["early_stopping"])
        if isinstance(training_raw.get("aggressive_lr_safety_stop"), dict):
            training_raw["aggressive_lr_safety_stop"] = AggressiveLRSafetyStopConfig(
                **training_raw["aggressive_lr_safety_stop"]
            )
        if isinstance(training_raw.get("numerical_safety"), dict):
            training_raw["numerical_safety"] = NumericalSafetyConfig(**training_raw["numerical_safety"])
        return cls(
            experiment_name=raw["experiment_name"],
            output_dir=raw["output_dir"],
            teacher=ModelSpec(**raw["teacher"]) if raw.get("teacher") is not None else None,
            student=ModelSpec(**raw["student"]),
            data=DataConfig(**raw["data"]),
            training=TrainingConfig(**training_raw),
            distillation=DistillationLossConfig(**raw.get("distillation", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.as_dict(), sort_keys=False), encoding="utf-8")
