#!/usr/bin/env python3
"""Score a frozen threshold against cached post-NMS MIDOG predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--guide-repo", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.guide_repo.resolve()))
    from utils.eval_utils import MIDOGEvaluation

    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    preds = {
        name: {key: np.asarray(value) for key, value in values.items()}
        for name, values in raw.items()
    }
    dataset = pd.read_csv(args.dataset)
    selected = dataset.query("split == @args.split")
    expected = list(selected.filename.unique())
    if list(preds) != expected:
        raise SystemExit("Prediction ordering/content does not exactly match the selected manifest split")
    gt = selected.query("label == 1").assign(
        xmin=lambda df: df.x - 25,
        ymin=lambda df: df.y - 25,
        xmax=lambda df: df.x + 25,
        ymax=lambda df: df.y + 25,
    )
    evaluator = MIDOGEvaluation(
        gt_file=gt,
        preds=preds,
        output_file=args.output,
        det_thresh=args.threshold,
        split=args.split,
    )
    evaluator.score()
    case_metrics = evaluator._metrics["case"]
    aggregates = evaluator._metrics["aggregates"]
    aggregates["true_positives"] = int(sum(case["tp"] for case in case_metrics.values()))
    aggregates["false_positives"] = int(sum(case["fp"] for case in case_metrics.values()))
    aggregates["false_negatives"] = int(sum(case["fn"] for case in case_metrics.values()))
    aggregates["total_detections"] = int(
        sum(np.sum(values["scores"] > args.threshold) for values in preds.values())
    )
    aggregates["det_thresh"] = args.threshold
    aggregates["split"] = args.split
    aggregates["num_images"] = len(expected)
    evaluator.save()
    print(json.dumps(aggregates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
