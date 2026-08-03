#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


def threshold_values(best: float | None = None) -> list[float]:
    if best is None:
        return [round(i / 100, 2) for i in range(5, 91, 5)]
    lo = max(0.05, best - 0.05)
    hi = min(0.90, best + 0.05)
    count = int(round((hi - lo) / 0.01))
    return [round(lo + i * 0.01, 2) for i in range(count + 1)]


def load_aggregates(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("aggregates", {})


def number(value) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def select(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            number(row.get("F1")) or -1.0,
            number(row.get("recall")) or -1.0,
            -abs(float(row["confidence_threshold"]) - 0.584),
        ),
    )


def load_guide_evaluator(guide_repo: Path):
    guide_repo = guide_repo.resolve()
    if str(guide_repo) not in sys.path:
        sys.path.insert(0, str(guide_repo))
    from utils.eval_utils import MIDOGEvaluation

    return MIDOGEvaluation


def run_low_threshold_inference(args, model_key: str, checkpoint: str | None, metrics_dir: Path) -> tuple[Path, Path]:
    predictions_path = metrics_dir / f"{model_key}_raw_predictions.json"
    runtime_path = metrics_dir / f"{model_key}_raw_runtime.json"
    metrics_path = metrics_dir / f"{model_key}_raw_metrics.json"
    if not predictions_path.exists() or not runtime_path.exists():
        cmd = [
            args.python,
            str(args.adapter),
            "--config_file",
            str(args.config_file),
            "--dataset",
            str(args.dataset),
            "--guide_repo",
            str(args.guide_repo),
            "--img_dir",
            str(args.img_dir),
            "--metrics_output",
            str(metrics_path),
            "--runtime_metadata_output",
            str(runtime_path),
            "--predictions_output",
            str(predictions_path),
            "--runtime_backend",
            "pytorch_eager",
            "--split",
            args.split,
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--overlap",
            str(args.overlap),
            "--nms_thresh",
            str(args.nms_thresh),
            "--det_thresh",
            "0.05",
            "--overwrite",
        ]
        if checkpoint:
            cmd.extend(["--pruned_model_path", checkpoint])
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            raise SystemExit(f"Low-threshold inference failed for {model_key}: {completed.returncode}")
    return predictions_path, runtime_path


def checkpoint_selection_type(checkpoint: str | None, allow_raw_best: bool) -> str:
    if not checkpoint:
        return "baseline"
    path = Path(checkpoint)
    if path.name == "student_stable_selected.pt":
        return "stable_selected"
    stable_artifact = path.parent / "student_stable_selected.pt"
    stable_json = path.parent / "stable_checkpoint_selection.json"
    if path.name in {"student_dense_best_ap.pt", "student_best_ap.pt"} and stable_artifact.exists() and stable_json.exists():
        if not allow_raw_best:
            raise SystemExit(
                f"{checkpoint} is a raw-best artifact while {stable_artifact} exists. "
                "Use the stable-selected checkpoint for calibration or pass --allow-raw-best-checkpoint."
            )
        return "raw_best_explicit_override"
    if stable_artifact.exists() and path.resolve() == stable_artifact.resolve():
        return "stable_selected"
    return "explicit_checkpoint"


def evaluate_threshold(args, model_key: str, threshold: float, metrics_dir: Path, gt_file: pd.DataFrame, preds: dict, evaluator_cls) -> dict:
    metrics_path = metrics_dir / f"{model_key}_thr_{threshold:.2f}_metrics.json"
    if not metrics_path.exists():
        evaluation = evaluator_cls(
            gt_file=gt_file,
            preds=preds,
            output_file=metrics_path,
            det_thresh=threshold,
            split=args.split,
        )
        evaluation.score()
        case_metrics = evaluation._metrics["case"]
        aggregates = evaluation._metrics["aggregates"]
        aggregates["true_positives"] = int(sum(case["tp"] for case in case_metrics.values()))
        aggregates["false_positives"] = int(sum(case["fp"] for case in case_metrics.values()))
        aggregates["false_negatives"] = int(sum(case["fn"] for case in case_metrics.values()))
        aggregates["total_detections"] = sum(
            int(sum(float(score) > threshold for score in case_preds.get("scores", [])))
            for case_preds in preds.values()
        )
        evaluation.save()
    ag = load_aggregates(metrics_path)
    return {
        "confidence_threshold": f"{threshold:.2f}",
        "F1": ag.get("f1_score"),
        "precision": ag.get("precision"),
        "recall": ag.get("recall"),
        "TP": ag.get("true_positives"),
        "FP": ag.get("false_positives"),
        "FN": ag.get("false_negatives"),
        "detection_count": ag.get("total_detections"),
        "AP": ag.get("AP"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate confidence thresholds on calibration data.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--pruned60-checkpoint", default="")
    parser.add_argument("--supervised-checkpoint", default="")
    parser.add_argument("--distilled-checkpoint", default="")
    parser.add_argument("--single-model-key", default="")
    parser.add_argument("--single-checkpoint", default="")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--config-file", type=Path, default=Path("experiments/eval_guide/configs/FCOS_18_eval.yaml"))
    parser.add_argument("--guide-repo", type=Path, default=Path("repos/MIDOG_2025_Guide"))
    parser.add_argument("--img-dir", type=Path, default=Path("data/midogpp"))
    parser.add_argument("--adapter", type=Path, default=Path("src/benchmark/midog_guide_adapter.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--allow-raw-best-checkpoint", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "threshold_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.single_model_key:
        if not args.single_checkpoint:
            raise SystemExit("--single-checkpoint is required when --single-model-key is set")
        models = {args.single_model_key: args.single_checkpoint}
    else:
        missing = [
            name
            for name, value in {
                "--pruned60-checkpoint": args.pruned60_checkpoint,
                "--supervised-checkpoint": args.supervised_checkpoint,
                "--distilled-checkpoint": args.distilled_checkpoint,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing required checkpoint arguments: {', '.join(missing)}")
        models = {
            "baseline": args.baseline_checkpoint or None,
            "pruned60": args.pruned60_checkpoint,
            "supervised": args.supervised_checkpoint,
            "distilled": args.distilled_checkpoint,
        }
    output_names = {
        "baseline": "baseline_threshold_results.csv",
        "pruned60": "pruned60_threshold_results.csv",
        "supervised": "supervised_threshold_results.csv",
        "distilled": "distilled_threshold_results.csv",
    }
    selected = {}
    evaluator_cls = load_guide_evaluator(args.guide_repo)
    dataset = pd.read_csv(args.dataset)
    if {"x", "y"}.issubset(dataset.columns):
        radius = 25
        dataset = dataset.assign(xmin=dataset["x"] - radius)
        dataset = dataset.assign(ymin=dataset["y"] - radius)
        dataset = dataset.assign(xmax=dataset["x"] + radius)
        dataset = dataset.assign(ymax=dataset["y"] + radius)
    gt_file = dataset[(dataset["split"].astype(str) == args.split) & (dataset["label"] == 1)].copy()
    for key, checkpoint in models.items():
        selection_type = checkpoint_selection_type(checkpoint, args.allow_raw_best_checkpoint)
        print(f"{key}: using checkpoint selection type {selection_type}: {checkpoint}")
        predictions_path, runtime_path = run_low_threshold_inference(args, key, checkpoint, metrics_dir)
        preds = json.loads(predictions_path.read_text(encoding="utf-8"))
        rows = [evaluate_threshold(args, key, threshold, metrics_dir, gt_file, preds, evaluator_cls) for threshold in threshold_values()]
        broad_best = select(rows)
        seen = {row["confidence_threshold"] for row in rows}
        for threshold in threshold_values(float(broad_best["confidence_threshold"])):
            label = f"{threshold:.2f}"
            if label not in seen:
                rows.append(evaluate_threshold(args, key, threshold, metrics_dir, gt_file, preds, evaluator_cls))
                seen.add(label)
        rows = sorted(rows, key=lambda row: float(row["confidence_threshold"]))
        final_best = select(rows)
        output_name = output_names.get(key, f"{key}_threshold_results.csv")
        with (args.output_dir / output_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selected[key] = {
            "checkpoint": checkpoint,
            "checkpoint_selection_type": selection_type,
            "raw_predictions": str(predictions_path),
            "raw_runtime": str(runtime_path),
            "selected_threshold": float(final_best["confidence_threshold"]),
            "calibration_F1": final_best["F1"],
            "calibration_precision": final_best["precision"],
            "calibration_recall": final_best["recall"],
            "tie_break_rule": ["highest F1", "highest recall", "threshold closest to 0.584"],
        }
    (args.output_dir / "selected_thresholds.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
