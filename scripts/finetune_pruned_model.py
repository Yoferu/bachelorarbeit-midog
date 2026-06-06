#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning.fine_tune_config import FineTuneConfig
from src.pruning.inspect_model import get_module_by_name, load_guide_model, unwrap_detection_model
from src.pruning.save_pruned_model import current_git_commit, load_pruned_state_dict
from src.pruning.structured_pruning import count_parameters


class PreserveZeroedChannelWeights(Callback):
    def __init__(self, zeroed_channels: dict[str, list[int]]) -> None:
        self.zeroed_channels = zeroed_channels

    def _apply_masks(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for module_name, channels in self.zeroed_channels.items():
                if not channels:
                    continue
                module = get_module_by_name(model, module_name)
                indices = torch.as_tensor(channels, device=module.weight.device, dtype=torch.long)
                module.weight.index_fill_(0, indices, 0.0)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._apply_masks(pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self._apply_masks(pl_module)


def zeroed_output_channel_indices(module: torch.nn.Conv2d) -> list[int]:
    weight = module.weight.detach()
    flattened = weight.reshape(weight.shape[0], -1)
    return torch.where(flattened.abs().sum(dim=1) == 0)[0].cpu().tolist()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def resolve_device(requested_device: str) -> str:
    requested = requested_device.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Fine-tuning requested device=cuda, but torch.cuda.is_available() is false.")
        return requested
    raise ValueError("device must be one of: auto, cpu, cuda")


def cuda_device_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def setup_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pruning_finetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def import_guide_datamodule(guide_repo: Path):
    guide_path = str(guide_repo.resolve())
    if guide_path not in sys.path:
        sys.path.insert(0, guide_path)
    from utils.datamodule import ObjectDetectionDataModule

    return ObjectDetectionDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a masked structured-pruned FCOS artifact.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = FineTuneConfig.load(args.config)
    set_reproducibility(config.seed)
    resolved_device = resolve_device(config.device)
    resolved_accelerator = "gpu" if resolved_device == "cuda" else "cpu"
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.logs_dir / f"{config.run_name}_{timestamp}.log"
    logger = setup_file_logger(log_path)
    logger.info("Loaded fine-tune config: %s", config.as_dict())

    model, model_config = load_guide_model(
        model_name=config.model,
        guide_repo=config.guide_repo,
        eval_config=config.base_eval_config,
    )
    state_dict = load_pruned_state_dict(config.pruned_model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Pruned model state_dict did not match the loaded base model. "
            f"Missing keys: {list(missing)[:10]}, unexpected keys: {list(unexpected)[:10]}"
        )

    model.hparams.lr = config.learning_rate
    model.hparams.optimizer = config.optimizer
    model.hparams.scheduler = config.scheduler
    model.hparams.batch_size = config.batch_size
    model.to(resolved_device)

    detection_model = unwrap_detection_model(model)
    zeroed_channels = {
        name: zeroed_output_channel_indices(module)
        for name, module in detection_model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }
    zeroed_channels = {name: channels for name, channels in zeroed_channels.items() if channels}
    logger.info("Preserving zeroed channel masks for %d modules.", len(zeroed_channels))

    ObjectDetectionDataModule = import_guide_datamodule(config.guide_repo)
    dataset_df = pd.read_csv(config.dataset)
    num_val_samples = config.num_val_samples
    if config.limit_val_batches == 0 and "val" not in set(dataset_df["split"].unique()):
        placeholder = dataset_df[dataset_df["split"] == "train"].head(1).copy()
        placeholder["split"] = "val"
        dataset_df = pd.concat([dataset_df, placeholder], ignore_index=True)
        num_val_samples = max(1, config.num_val_samples)
        logger.info(
            "No val split found and validation is disabled; added an in-memory val placeholder "
            "for upstream datamodule construction."
        )
    datamodule = ObjectDetectionDataModule(
        img_dir=config.img_dir,
        dataset=dataset_df,
        domain_col=config.domain_col,
        box_format=config.box_format,
        num_train_samples=config.num_train_samples,
        num_val_samples=num_val_samples,
        fg_prob=config.fg_prob,
        arb_prob=config.arb_prob,
        patch_size=model_config.patch_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    csv_logger = CSVLogger(save_dir=str(config.results_dir), name=config.run_name, version=timestamp)
    callbacks: list[Callback] = []
    if config.preserve_zeroed_pruning_masks:
        callbacks.append(PreserveZeroedChannelWeights(zeroed_channels))

    trainer = pl.Trainer(
        accelerator=resolved_accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        logger=csv_logger,
        callbacks=callbacks,
        gradient_clip_val=config.gradient_clip_val,
        limit_val_batches=config.limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_checkpointing=False,
        deterministic=config.deterministic,
    )

    train_started_at = datetime.now(timezone.utc)
    trainer.fit(model, datamodule=datamodule)
    train_finished_at = datetime.now(timezone.utc)

    config.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_type": "masked_structured_pruned_finetuned_detection_model",
        "model": config.model,
        "state_dict": model.state_dict(),
        "fine_tune_config": config.as_dict(),
        "source_pruned_model_path": str(config.pruned_model_path),
        "base_eval_config": str(config.base_eval_config),
        "zeroed_channels_preserved": config.preserve_zeroed_pruning_masks,
        "zeroed_channels": zeroed_channels,
    }
    torch.save(artifact, config.output_model_path)

    metadata = {
        "base_model": config.model,
        "base_checkpoint": model_config.checkpoint,
        "base_eval_config": str(config.base_eval_config),
        "pruned_checkpoint": str(config.pruned_model_path),
        "fine_tuned_checkpoint": str(config.output_model_path),
        "training_dataset": str(config.dataset),
        "training_split": "train",
        "learning_rate": config.learning_rate,
        "max_epochs": config.max_epochs,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "cudnn_benchmark": config.cudnn_benchmark,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "optimizer": config.optimizer,
        "scheduler": config.scheduler,
        "requested_device": config.device,
        "resolved_device": resolved_device,
        "device": resolved_device,
        "requested_accelerator": config.accelerator,
        "accelerator": resolved_accelerator,
        "devices": config.devices,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": cuda_device_name(),
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "num_train_samples": config.num_train_samples,
        "num_val_samples": num_val_samples,
        "preserve_zeroed_pruning_masks": config.preserve_zeroed_pruning_masks,
        "zeroed_channels": sum(len(channels) for channels in zeroed_channels.values()),
        "parameters_total": count_parameters(model, nonzero_only=False),
        "parameters_nonzero": count_parameters(model, nonzero_only=True),
        "train_started_at": train_started_at.isoformat(),
        "train_finished_at": train_finished_at.isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "log_path": str(log_path),
        "lightning_log_dir": str(Path(csv_logger.log_dir)),
        "notes": config.notes,
        "final_smoke_metrics": None,
    }
    metadata_path = config.output_model_path.with_suffix(".meta.json")
    results_metadata_path = config.results_dir / f"{config.output_model_path.stem}.meta.json"
    results_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata["results_metadata_path"] = str(results_metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved fine-tuned checkpoint: %s", config.output_model_path)
    logger.info("Saved fine-tune metadata: %s", metadata_path)
    logger.info("Saved fine-tune results metadata: %s", results_metadata_path)
    print(f"Saved fine-tuned checkpoint: {config.output_model_path}")
    print(f"Saved fine-tune metadata: {metadata_path}")
    print(f"Saved fine-tune results metadata: {results_metadata_path}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
