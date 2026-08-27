#!/usr/bin/env python3
"""Dense MIDOG threshold sweep over cached post-NMS predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--guide-repo", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--minimum", type=float, default=0.300)
    parser.add_argument("--maximum", type=float, default=0.800)
    parser.add_argument("--step", type=float, default=0.001)
    args = parser.parse_args()

    sys.path.insert(0, str(args.guide_repo.resolve()))
    from utils.eval_utils import MIDOGEvaluation, _F1_core_balltree

    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    preds = {name: {k: np.asarray(v) for k, v in values.items()} for name, values in raw.items()}
    dataset = pd.read_csv(args.dataset)
    dataset = dataset.assign(
        xmin=dataset.x - 25, ymin=dataset.y - 25, xmax=dataset.x + 25, ymax=dataset.y + 25
    )
    gt = dataset.query("split == @args.split and label == 1")
    expected = list(dataset.query("split == @args.split").filename.unique())
    if list(preds) != expected:
        raise SystemExit("Prediction ordering/content does not exactly match the calibration manifest")
    thresholds = np.round(np.arange(args.minimum, args.maximum + args.step / 2, args.step), 3)
    rows = []
    for threshold in thresholds:
        tp = fp = fn = detections = 0
        for filename, values in preds.items():
            scores = values["scores"]
            detections += int(np.sum(scores > threshold))
            annos = gt.query("filename == @filename")[["xmin", "ymin", "xmax", "ymax"]].values
            _, ctp, cfp, cfn = _F1_core_balltree(annos, values["boxes"], scores, float(threshold))
            tp += int(ctp); fp += int(cfp); fn += int(cfn)
        rows.append({
            "variant": args.variant, "backend": args.backend, "precision": args.precision,
            "threshold": float(threshold), "f1": 2 * tp / (2 * tp + fp + fn + 1e-12),
            "precision_metric": tp / (tp + fp + 1e-12), "recall": tp / (tp + fn + 1e-12),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "total_detections": detections,
        })
    # Existing guide optimize_threshold uses np.argmax, hence the lowest threshold wins exact F1 ties.
    best = dict(max(rows, key=lambda r: (r["f1"], -r["threshold"])))
    evaluator = MIDOGEvaluation(
        gt_file=gt,
        preds=preds,
        output_file="unused.json",
        det_thresh=best["threshold"],
        split=args.split,
    )
    evaluator.score()
    best["ap"] = float(evaluator._metrics["aggregates"]["AP"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    selection = {
        **best,
        "selection_criterion": "maximum aggregate F1; exact ties select lowest threshold (repository np.argmax policy)",
        "grid": {"minimum": args.minimum, "maximum": args.maximum, "step": args.step, "count": len(rows)},
        "nms_threshold": 0.3, "split": args.split,
        "calibration_images": len(expected), "calibration_annotations": len(gt),
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "predictions": str(args.predictions.resolve()), "predictions_sha256": sha256(args.predictions),
        "model_artifact": str(args.model_artifact.resolve()), "model_artifact_sha256": sha256(args.model_artifact),
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
