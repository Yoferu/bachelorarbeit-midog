from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PruningConfig:
    model: str
    pruning_type: str
    pruning_ratio: float
    target_scope: str
    excluded_modules: list[str] = field(default_factory=list)
    device: str = "cpu"
    seed: int = 42
    guide_repo: Path = Path("repos/MIDOG_2025_Guide")
    eval_config: Path | None = None
    pruning_mode: str = "masked"
    notes: str | None = None

    @classmethod
    def load(cls, path: Path) -> "PruningConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("eval_config") is not None:
            data["eval_config"] = Path(data["eval_config"])
        if data.get("guide_repo") is not None:
            data["guide_repo"] = Path(data["guide_repo"])
        return cls(**data)

    def as_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["guide_repo"] = str(self.guide_repo)
        data["eval_config"] = str(self.eval_config) if self.eval_config else None
        return data

