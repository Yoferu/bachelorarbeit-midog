#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def image_distribution(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "annotation_count": int(len(df)),
        "unique_image_count": int(df["filename"].nunique()),
    }
    if "label" in df.columns:
        result["label_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
    if "tumortype" in df.columns:
        result["tumortype_counts"] = {
            str(k): int(v) for k, v in df["tumortype"].value_counts().sort_index().items()
        }
    return result


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def subset(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return df[df["filename"].astype(str).isin(ids)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clean image-level train/val/calibration splits.")
    parser.add_argument("--source", type=Path, default=Path("data/midogpp_guide_eval_xvalidation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/distillation/pruned60_clean_recovery/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    required = {"filename", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {args.source}: {sorted(missing)}")

    train_df = df[df["split"].astype(str) == "train"].copy()
    test_df = df[df["split"].astype(str) == "test"].copy()
    image_ids = sorted(train_df["filename"].astype(str).unique())
    rng = random.Random(args.seed)
    rng.shuffle(image_ids)

    n_images = len(image_ids)
    n_validation = round(n_images * 0.10)
    n_calibration = round(n_images * 0.10)
    validation_ids = sorted(image_ids[:n_validation])
    calibration_ids = sorted(image_ids[n_validation : n_validation + n_calibration])
    recovery_train_ids = sorted(image_ids[n_validation + n_calibration :])

    groups = {
        "train": recovery_train_ids,
        "validation": validation_ids,
        "calibration": calibration_ids,
    }
    overlaps = {
        "train_validation": len(set(recovery_train_ids) & set(validation_ids)),
        "train_calibration": len(set(recovery_train_ids) & set(calibration_ids)),
        "validation_calibration": len(set(validation_ids) & set(calibration_ids)),
        "train_test": len(set(recovery_train_ids) & set(test_df["filename"].astype(str).unique())),
        "validation_test": len(set(validation_ids) & set(test_df["filename"].astype(str).unique())),
        "calibration_test": len(set(calibration_ids) & set(test_df["filename"].astype(str).unique())),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Dataset split overlap check failed: {overlaps}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_frames = {
        "train": subset(train_df, recovery_train_ids).assign(split="train"),
        "validation": subset(train_df, validation_ids).assign(split="val"),
        "calibration": subset(train_df, calibration_ids).assign(split="calibration"),
    }
    split_frames["train"].to_csv(args.output_dir / "train_seed42.csv", index=False)
    split_frames["validation"].to_csv(args.output_dir / "validation_seed42.csv", index=False)
    split_frames["calibration"].to_csv(args.output_dir / "calibration_seed42.csv", index=False)
    write_ids(args.output_dir / "train_image_ids.txt", recovery_train_ids)
    write_ids(args.output_dir / "validation_image_ids.txt", validation_ids)
    write_ids(args.output_dir / "calibration_image_ids.txt", calibration_ids)

    manifest = {
        "source_csv": str(args.source),
        "source_split": "train",
        "seed": args.seed,
        "ratios": {"train": 0.80, "validation": 0.10, "calibration": 0.10},
        "groups": {
            name: {
                "image_ids": ids,
                **image_distribution(split_frames[name]),
            }
            for name, ids in groups.items()
        },
        "complete_original_test_split": {
            "source_csv": str(args.source),
            "split": "test",
            **image_distribution(test_df),
        },
        "pairwise_overlap_counts": overlaps,
        "notes": [
            "data/midogpp_guide_eval_xvalidation_quick.csv is a subset of the full test split.",
            "Some test images were previously used for diagnostic quick evaluations and initial pruning-candidate selection.",
            "The final test result is therefore not perfectly independent of all earlier development decisions.",
            "No epoch or confidence threshold may be selected using the full test split.",
        ],
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Pruned-60 Clean Recovery Dataset Split Audit",
        "",
        f"- Source CSV: `{args.source}`",
        "- Source split for recovery data: `train`",
        f"- Seed: `{args.seed}`",
        f"- Complete original test split images: `{test_df['filename'].nunique()}`",
        f"- Complete original test split annotations: `{len(test_df)}`",
        "",
        "## Split Summary",
        "",
        "| Role | Split label | Images | Annotations |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, frame in split_frames.items():
        label = "train" if name == "train" else ("val" if name == "validation" else "calibration")
        lines.append(f"| {name} | {label} | {frame['filename'].nunique()} | {len(frame)} |")
    lines.extend(["", "## Pairwise Overlap Counts", "", "| Pair | Overlap |", "| --- | ---: |"])
    for key, value in overlaps.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Class And Annotation Distribution", ""])
    for name, frame in split_frames.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Annotation count: `{len(frame)}`")
        lines.append(f"- Unique image count: `{frame['filename'].nunique()}`")
        if "label" in frame.columns:
            lines.append(f"- Label counts: `{frame['label'].value_counts().sort_index().to_dict()}`")
        if "tumortype" in frame.columns:
            lines.append(f"- Tumortype counts: `{frame['tumortype'].value_counts().sort_index().to_dict()}`")
        lines.append(f"- Image identifiers: `{', '.join(groups[name])}`")
        lines.append("")
    lines.extend(
        [
            "## Test-Set Note",
            "",
            "`data/midogpp_guide_eval_xvalidation_quick.csv` is a subset of the full test split. Some test images were previously used for diagnostic quick evaluations and initial pruning-candidate selection. The final full-test result is therefore not perfectly independent of all earlier development decisions, but epoch selection and threshold calibration are restricted to train-derived validation and calibration subsets.",
            "",
            "All overlap counts above must remain zero before final evaluation.",
            "",
        ]
    )
    (args.output_dir / "dataset_split_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: image_distribution(v) for k, v in split_frames.items()}, indent=2))


if __name__ == "__main__":
    main()
