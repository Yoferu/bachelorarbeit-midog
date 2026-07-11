#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning.save_pruned_model import load_pruned_model_object, load_pruned_state_dict


VARIANTS = {
    "fcos18_depgraph_fpn_head_20pct": PROJECT_ROOT
    / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_20pct.pt",
    "fcos18_depgraph_fpn_head_30pct": PROJECT_ROOT
    / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_30pct.pt",
    "fcos18_depgraph_fpn_head_40pct": PROJECT_ROOT
    / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_40pct.pt",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose FPN+head DepGraph prediction equality.")
    parser.add_argument("--config_file", type=Path, default=PROJECT_ROOT / "experiments/eval_guide/configs/FCOS_18_eval.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/midogpp_guide_eval_xvalidation_smoke.csv")
    parser.add_argument("--guide_repo", type=Path, default=PROJECT_ROOT / "repos/MIDOG_2025_Guide")
    parser.add_argument("--img_dir", type=Path, default=PROJECT_ROOT / "data/midogpp")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/pruning/diagnostics/fpn_head_prediction_comparison",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(config_file: Path, guide_repo: Path, artifact: Path):
    sys.path.insert(0, str(guide_repo.resolve()))
    from utils.factory import ConfigCreator, ModelFactory

    config = ConfigCreator.load(str(config_file))
    model = ModelFactory.load(config, det_thresh=0.05)
    object_model = load_pruned_model_object(artifact)
    if object_model is not None:
        return object_model, config
    state_dict = load_pruned_state_dict(artifact)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch for {artifact}: missing={missing[:10]}, unexpected={unexpected[:10]}")
    return model, config


def jsonable_predictions(preds: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "boxes": np.asarray(preds["boxes"]).tolist(),
        "scores": np.asarray(preds["scores"]).tolist(),
        "labels": np.asarray(preds["labels"]).tolist(),
    }


def tensors_to_jsonable(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, list[Any]]:
    return {
        "boxes": boxes.cpu().numpy().tolist(),
        "scores": scores.cpu().numpy().tolist(),
        "labels": labels.cpu().numpy().tolist(),
    }


def run_image_with_intermediates(processor: Any, image_path: Path, patch_config: Any) -> dict[str, Any]:
    strategy = processor.strategy
    strategy.model.eval()
    strategy.model.to(strategy.device)
    dataloader = strategy._create_dataloader(image_path, patch_config)

    all_predictions = []
    all_coords = []
    for batch_images, batch_x, batch_y in dataloader:
        predictions = strategy._process_batch(batch_images)
        all_predictions.extend(predictions)
        all_coords.extend(zip(batch_x, batch_y))

    pre_boxes = []
    pre_scores = []
    pre_labels = []
    for pred, (x_orig, y_orig) in zip(all_predictions, all_coords):
        if len(pred["boxes"]) == 0:
            continue
        offset = torch.tensor([x_orig, y_orig, x_orig, y_orig], device=pred["boxes"].device)
        pre_boxes.append((pred["boxes"] + offset).cpu())
        pre_scores.append(pred["scores"].cpu())
        pre_labels.append(pred["labels"].cpu())

    if pre_boxes:
        boxes = torch.cat(pre_boxes)
        scores = torch.cat(pre_scores)
        labels = torch.cat(pre_labels)
    else:
        boxes = torch.empty((0, 4), device="cpu")
        scores = torch.empty(0, device="cpu")
        labels = torch.empty(0, dtype=torch.long, device="cpu")

    return {
        "patch_count": len(dataloader.dataset),
        "pre_nms": {"boxes": boxes, "scores": scores, "labels": labels},
        "post_nms": strategy._post_process_predictions(all_predictions, all_coords),
    }


def score_summary(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if scores.size == 0:
        return {
            "count_before_threshold_after_nms": 0,
            "count_after_threshold_after_nms": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "quantiles": {},
        }
    return {
        "count_before_threshold_after_nms": int(scores.size),
        "count_after_threshold_after_nms": int(np.sum(scores > threshold)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "quantiles": {
            "p10": float(np.quantile(scores, 0.10)),
            "p25": float(np.quantile(scores, 0.25)),
            "p75": float(np.quantile(scores, 0.75)),
            "p90": float(np.quantile(scores, 0.90)),
        },
    }


def top_detections(preds: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
    boxes = np.asarray(preds["boxes"])
    scores = np.asarray(preds["scores"])
    labels = np.asarray(preds["labels"])
    order = np.argsort(-scores)[:n]
    return [
        {
            "rank": int(rank + 1),
            "score": float(scores[idx]),
            "label": int(labels[idx]),
            "box": [float(v) for v in boxes[idx]],
        }
        for rank, idx in enumerate(order)
    ]


def case_counts(gt_rows: pd.DataFrame, preds: dict[str, Any], threshold: float, radius: int = 25) -> dict[str, int]:
    sys.path.insert(0, str((PROJECT_ROOT / "repos/MIDOG_2025_Guide").resolve()))
    from utils.eval_utils import _F1_core_balltree

    annos = gt_rows[["xmin", "ymin", "xmax", "ymax"]].values
    boxes = np.asarray(preds["boxes"])
    scores = np.asarray(preds["scores"])
    _, tp, fp, fn = _F1_core_balltree(annos=annos, boxes=boxes, scores=scores, det_thresh=threshold, radius=radius)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn)}


def prediction_file_hash(prediction_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(prediction_dir.glob("*_detections.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.guide_repo.resolve()))
    from utils import inference as guide_inference

    dataset = pd.read_csv(args.dataset)
    radius = 25
    dataset = dataset.assign(xmin=dataset["x"] - radius)
    dataset = dataset.assign(ymin=dataset["y"] - radius)
    dataset = dataset.assign(xmax=dataset["x"] + radius)
    dataset = dataset.assign(ymax=dataset["y"] + radius)
    test_dataset = dataset.query("split == @args.split")
    gt_dataset = test_dataset.query("label == 1")
    filenames = list(test_dataset.filename.unique())

    variant_summaries = {}
    for name, artifact in VARIANTS.items():
        model, config = load_model(args.config_file, args.guide_repo, artifact)
        processor, patch_config = guide_inference.setup_inference(
            model=model,
            is_wsi=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            nms_thresh=args.nms_thresh,
            device=args.device,
            patch_size=config.patch_size,
            overlap=args.overlap,
            overwrite=True,
        )
        pred_dir = args.output_dir / name
        pred_dir.mkdir(parents=True, exist_ok=True)

        image_summaries = {}
        for filename in filenames:
            intermediate = run_image_with_intermediates(processor, args.img_dir / filename, patch_config)
            post = intermediate["post_nms"]
            pre = intermediate["pre_nms"]
            preds = {
                "boxes": post["boxes"].numpy(),
                "scores": post["scores"].numpy(),
                "labels": post["labels"].numpy(),
            }
            output_path = pred_dir / f"{Path(filename).stem}_detections.json"
            output_path.write_text(json.dumps(jsonable_predictions(preds), indent=2), encoding="utf-8")
            pre_output_path = pred_dir / f"{Path(filename).stem}_pre_nms_detections.json"
            pre_output_path.write_text(
                json.dumps(tensors_to_jsonable(pre["boxes"], pre["scores"], pre["labels"]), indent=2),
                encoding="utf-8",
            )

            scores = np.asarray(preds["scores"])
            pre_scores = pre["scores"].numpy()
            image_summaries[filename] = {
                **score_summary(scores, config.det_thresh),
                "patch_count": int(intermediate["patch_count"]),
                "count_before_threshold_before_nms": int(pre_scores.size),
                "count_after_threshold_before_nms": int(np.sum(pre_scores > config.det_thresh)),
                "count_after_nms": int(scores.size),
                "top10": top_detections(preds),
                "case_counts": case_counts(gt_dataset.query("filename == @filename"), preds, config.det_thresh),
            }

        variant_summaries[name] = {
            "artifact_path": str(artifact),
            "artifact_sha256": sha256(artifact),
            "prediction_dir": str(pred_dir),
            "prediction_files_sha256": prediction_file_hash(pred_dir),
            "det_thresh": float(config.det_thresh),
            "nms_thresh": args.nms_thresh,
            "images": image_summaries,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(variant_summaries, indent=2), encoding="utf-8")

    report_lines = [
        "# FPN+Head Prediction Diagnostic",
        "",
        "Pre-NMS counts are after torchvision FCOS model-internal filtering per patch and before patch-merge NMS. Post-NMS counts are after patch coordinate merge and class-wise NMS.",
        "",
        "| Variant | Artifact SHA256 | Prediction SHA256 | Before Threshold, Pre-NMS | After Threshold, Pre-NMS | After NMS | After Threshold, Post-NMS | TP | FP | FN |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in variant_summaries.items():
        before_pre_nms = sum(image["count_before_threshold_before_nms"] for image in summary["images"].values())
        after_pre_nms = sum(image["count_after_threshold_before_nms"] for image in summary["images"].values())
        after_nms = sum(image["count_after_nms"] for image in summary["images"].values())
        after = sum(image["count_after_threshold_after_nms"] for image in summary["images"].values())
        tp = sum(image["case_counts"]["tp"] for image in summary["images"].values())
        fp = sum(image["case_counts"]["fp"] for image in summary["images"].values())
        fn = sum(image["case_counts"]["fn"] for image in summary["images"].values())
        report_lines.append(
            f"| `{name}` | `{summary['artifact_sha256'][:12]}` | `{summary['prediction_files_sha256'][:12]}` | "
            f"{before_pre_nms} | {after_pre_nms} | {after_nms} | {after} | {tp} | {fp} | {fn} |"
        )
    report_lines.append("")
    report_lines.append("## Per-Image Counts")
    report_lines.append("")
    report_lines.append("| Variant | Image | Before Threshold, Pre-NMS | After Threshold, Pre-NMS | After NMS | After Threshold, Post-NMS | Score Min | Score Median | Score Max | TP | FP | FN |")
    report_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, summary in variant_summaries.items():
        for filename, image in summary["images"].items():
            counts = image["case_counts"]
            report_lines.append(
                f"| `{name}` | `{filename}` | {image['count_before_threshold_before_nms']} | "
                f"{image['count_after_threshold_before_nms']} | {image['count_after_nms']} | "
                f"{image['count_after_threshold_after_nms']} | {image['min']:.4f} | "
                f"{image['median']:.4f} | {image['max']:.4f} | "
                f"{counts['tp']} | {counts['fp']} | {counts['fn']} |"
            )
    report_lines.append("")
    report_lines.append("Top-10 post-NMS boxes/scores for each image are stored in `summary.json` under each image's `top10` field.")
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print((args.output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
