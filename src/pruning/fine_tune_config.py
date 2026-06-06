from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FineTuneConfig:
    model: str
    base_eval_config: Path
    pruned_model_path: Path
    dataset: Path
    img_dir: Path
    output_model_path: Path
    run_name: str = "fcos18_structured_head_l2_10pct_finetune_smoke"
    guide_repo: Path = Path("repos/MIDOG_2025_Guide")
    results_dir: Path = Path("experiments/pruning/results")
    logs_dir: Path = Path("experiments/pruning/logs")
    device: str = "auto"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42
    deterministic: bool = False
    cudnn_benchmark: bool = False
    learning_rate: float = 1e-5
    optimizer: str = "AdamW"
    scheduler: str | None = None
    max_epochs: int = 1
    max_steps: int = 2
    batch_size: int = 1
    num_workers: int = 0
    num_train_samples: int = 4
    num_val_samples: int = 2
    fg_prob: float = 0.75
    arb_prob: float = 0.0
    box_format: str = "cxcy"
    domain_col: str = "tumortype"
    limit_val_batches: float = 0.0
    gradient_clip_val: float = 1.0
    preserve_zeroed_pruning_masks: bool = True
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "FineTuneConfig":
        with path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        for key in (
            "base_eval_config",
            "pruned_model_path",
            "dataset",
            "img_dir",
            "output_model_path",
            "guide_repo",
            "results_dir",
            "logs_dir",
        ):
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data
