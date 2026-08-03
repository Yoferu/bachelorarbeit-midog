#!/usr/bin/env python3
"""Fail fast when epoch/threshold selection datasets leak into final eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def image_ids(csv_path: Path, split: str) -> set[str]:
    df = pd.read_csv(csv_path)
    required = {"filename", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {csv_path} is missing required columns: {sorted(missing)}")
    selected = df[df["split"].astype(str) == split]
    if selected.empty:
        declared = sorted(df["split"].astype(str).unique())
        raise SystemExit(
            f"ERROR: {csv_path} has no rows for split={split!r}; declared splits={declared}"
        )
    return set(selected["filename"].astype(str).unique())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify threshold/epoch selection data is disjoint from final held-out data."
    )
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--final-split", default="test")
    parser.add_argument("--diagnostic-csv", type=Path, action="append", default=[])
    parser.add_argument("--diagnostic-split", default="test")
    args = parser.parse_args()

    calibration = image_ids(args.calibration_csv, args.calibration_split)
    final = image_ids(args.final_csv, args.final_split)
    overlap = sorted(calibration & final)
    if overlap:
        preview = ", ".join(overlap[:20])
        raise SystemExit(
            "ERROR: calibration and final evaluation images overlap "
            f"({len(overlap)} images): {preview}"
        )
    if args.calibration_csv.resolve() == args.final_csv.resolve() and args.calibration_split == args.final_split:
        raise SystemExit("ERROR: script would tune and evaluate on the same CSV split.")

    for diagnostic_csv in args.diagnostic_csv:
        diagnostic = image_ids(diagnostic_csv, args.diagnostic_split)
        diagnostic_final_overlap = sorted(diagnostic & final)
        if diagnostic_final_overlap:
            print(
                "WARNING: diagnostic dataset overlaps final held-out images "
                f"({diagnostic_csv}, {len(diagnostic_final_overlap)} images). "
                "Use it for diagnostic development only, not threshold or epoch selection."
            )

    print(
        "Dataset role check passed: "
        f"calibration_images={len(calibration)}, final_images={len(final)}, overlap=0"
    )


if __name__ == "__main__":
    main()
