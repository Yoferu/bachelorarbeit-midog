#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning.save_pruned_model import load_pruned_model_object, load_pruned_state_dict


VARIANTS = {
    "baseline": None,
    "fcos18_depgraph_fpn_head_50pct": PROJECT_ROOT
    / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_50pct.pt",
    "fcos18_depgraph_fpn_head_60pct": PROJECT_ROOT
    / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt",
    "fcos18_depgraph_fpn_head_70pct": PROJECT_ROOT
    / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_70pct.pt",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FPN+head threshold sweep on cached post-NMS predictions.")
    parser.add_argument("--config_file", type=Path, default=PROJECT_ROOT / "experiments/eval_guide/configs/FCOS_18_eval.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/midogpp_guide_eval_xvalidation_quick.csv")
    parser.add_argument("--guide_repo", type=Path, default=PROJECT_ROOT / "repos/MIDOG_2025_Guide")
    parser.add_argument("--img_dir", type=Path, default=PROJECT_ROOT / "data/midogpp")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/pruning/sweeps/fcos18_depgraph_fpn_head_threshold_sweep_quick",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.584, 0.60, 0.65, 0.70, 0.75, 0.80],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(config_file: Path, guide_repo: Path, artifact: Path | None):
    sys.path.insert(0, str(guide_repo.resolve()))
    from utils.factory import ConfigCreator, ModelFactory

    config = ConfigCreator.load(str(config_file))
    model = ModelFactory.load(config, det_thresh=0.05)
    if artifact is None:
        return model, config

    object_model = load_pruned_model_object(artifact)
    if object_model is not None:
        return object_model, config

    state_dict = load_pruned_state_dict(artifact)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch for {artifact}: missing={missing[:10]}, unexpected={unexpected[:10]}")
    return model, config


def to_jsonable(preds: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "boxes": np.asarray(preds["boxes"]).tolist(),
        "scores": np.asarray(preds["scores"]).tolist(),
        "labels": np.asarray(preds["labels"]).tolist(),
    }


def run_predictions(
    *,
    name: str,
    artifact: Path | None,
    args: argparse.Namespace,
    filenames: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(args.guide_repo.resolve()))
    from utils import inference as guide_inference

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

    pred_dir = args.output_dir / "predictions" / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for filename in filenames:
        result = processor.process_single(args.img_dir / filename, patch_config)
        preds[filename] = {
            "boxes": np.asarray(result["boxes"]),
            "scores": np.asarray(result["scores"]),
            "labels": np.asarray(result["labels"]),
        }
        (pred_dir / f"{Path(filename).stem}_detections.json").write_text(
            json.dumps(to_jsonable(preds[filename]), indent=2),
            encoding="utf-8",
        )

    scores = np.concatenate([case["scores"] for case in preds.values() if len(case["scores"]) > 0])
    score_summary = {
        "count": int(scores.size),
        "min": float(np.min(scores)) if scores.size else None,
        "p10": float(np.quantile(scores, 0.10)) if scores.size else None,
        "p25": float(np.quantile(scores, 0.25)) if scores.size else None,
        "median": float(np.median(scores)) if scores.size else None,
        "p75": float(np.quantile(scores, 0.75)) if scores.size else None,
        "p90": float(np.quantile(scores, 0.90)) if scores.size else None,
        "max": float(np.max(scores)) if scores.size else None,
    }
    metadata = {
        "artifact_path": str(artifact) if artifact else None,
        "artifact_sha256": sha256(artifact) if artifact else None,
        "post_nms_score_summary": score_summary,
    }
    return preds, metadata


def evaluate_thresholds(
    *,
    name: str,
    preds: dict[str, dict[str, Any]],
    gt_dataset: pd.DataFrame,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    sys.path.insert(0, str((PROJECT_ROOT / "repos/MIDOG_2025_Guide").resolve()))
    from utils.eval_utils import _F1_core_balltree

    rows = []
    for threshold in thresholds:
        tp = fp = fn = total_detections = 0
        for filename, case_preds in preds.items():
            boxes = np.asarray(case_preds["boxes"])
            scores = np.asarray(case_preds["scores"])
            total_detections += int(np.sum(scores > threshold))
            annos = gt_dataset.query("filename == @filename")[["xmin", "ymin", "xmax", "ymax"]].values
            _, case_tp, case_fp, case_fn = _F1_core_balltree(
                annos=annos,
                boxes=boxes,
                scores=scores,
                det_thresh=threshold,
            )
            tp += int(case_tp)
            fp += int(case_fp)
            fn += int(case_fn)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-12)
        rows.append(
            {
                "variant": name,
                "threshold": float(threshold),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "total_detections": total_detections,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return rows


def write_outputs(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "threshold_sweep.csv"
    json_path = args.output_dir / "threshold_sweep.json"
    md_path = args.output_dir / "threshold_sweep.md"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_by_variant = {}
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        best_by_variant[variant] = max(
            variant_rows,
            key=lambda row: (row["f1"], row["recall"], -row["false_positives"]),
        )
    json_path.write_text(
        json.dumps({"rows": rows, "best_by_variant": best_by_variant, "metadata": metadata}, indent=2),
        encoding="utf-8",
    )

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = ["| " + " | ".join(fmt(row[key]) for key in fieldnames) + " |" for row in rows]
    best_lines = [
        "| variant | best_threshold | best_f1 | precision | recall | detections | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in best_by_variant.items():
        best_lines.append(
            f"| {variant} | {row['threshold']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['total_detections']} | {row['false_positives']} | {row['false_negatives']} |"
        )

    score_lines = [
        "| variant | post-NMS count | min | p10 | p25 | median | p75 | p90 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, info in metadata.items():
        scores = info["post_nms_score_summary"]
        score_lines.append(
            f"| {variant} | {scores['count']} | {fmt(scores['min'])} | {fmt(scores['p10'])} | "
            f"{fmt(scores['p25'])} | {fmt(scores['median'])} | {fmt(scores['p75'])} | "
            f"{fmt(scores['p90'])} | {fmt(scores['max'])} |"
        )

    md_path.write_text(
        "\n".join(
            [
                "# FPN+Head Threshold Sweep",
                "",
                "## Best Threshold Per Variant",
                "",
                *best_lines,
                "",
                "## Post-NMS Score Distributions",
                "",
                *score_lines,
                "",
                "## Full Sweep",
                "",
                header,
                separator,
                *body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.dataset)
    radius = 25
    dataset = dataset.assign(xmin=dataset["x"] - radius)
    dataset = dataset.assign(ymin=dataset["y"] - radius)
    dataset = dataset.assign(xmax=dataset["x"] + radius)
    dataset = dataset.assign(ymax=dataset["y"] + radius)
    test_dataset = dataset.query("split == @args.split")
    gt_dataset = test_dataset.query("label == 1")
    filenames = list(test_dataset.filename.unique())

    all_rows = []
    metadata = {}
    thresholds = sorted(set(args.thresholds))
    for name, artifact in VARIANTS.items():
        preds, variant_metadata = run_predictions(name=name, artifact=artifact, args=args, filenames=filenames)
        metadata[name] = variant_metadata
        all_rows.extend(evaluate_thresholds(name=name, preds=preds, gt_dataset=gt_dataset, thresholds=thresholds))

    write_outputs(args=args, rows=all_rows, metadata=metadata)
    print((args.output_dir / "threshold_sweep.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
