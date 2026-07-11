from __future__ import annotations

import csv
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fcos_distillation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def create_run_dir(base_dir: Path, experiment_name: str, overwrite: bool) -> Path:
    run_dir = base_dir / experiment_name
    if run_dir.exists() and not overwrite:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = base_dir / f"{experiment_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=overwrite)
    return run_dir


def append_training_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception:
        versions["torchvision"] = None
    return versions

