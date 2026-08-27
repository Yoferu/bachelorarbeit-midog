from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


VARIANTS = [
    ("fcos18", "pytorch", "fp32"),
    ("fcos18", "onnx", "fp32"),
    ("fcos18", "onnx", "int8"),
    ("pruned60", "pytorch", "fp32"),
    ("pruned60", "onnx", "fp32"),
    ("pruned60", "onnx", "int8"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    return dirs[-1] if dirs else None


def parse_peak_rss(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"Maximum resident set size \\(kbytes\\):\\s*(\\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def collect_variant(results_root: Path, model: str, runtime: str, precision: str) -> dict[str, Any]:
    variant_dir = results_root / model / f"{runtime}_{precision}"
    run_dir = latest_run_dir(variant_dir)
    row: dict[str, Any] = {
        "Model": model,
        "Runtime": runtime,
        "Precision": precision,
        "Run dir": str(run_dir) if run_dir else "",
    }
    if not run_dir:
        return row

    measured_timing = sorted(run_dir.glob("measured*_timing.json"))
    timing = read_json(measured_timing[-1]) if measured_timing else read_json(run_dir / "smoke_timing.json")
    measured_metrics = sorted(run_dir.glob("measured*_metrics.json"))
    metrics = read_json(measured_metrics[-1]) if measured_metrics else read_json(run_dir / "smoke_metrics.json")
    metadata = read_json(run_dir / "variant_metadata.json")
    peak_rss_values = [parse_peak_rss(p) for p in sorted(run_dir.glob("*_time.txt"))]
    peak_rss_values = [v for v in peak_rss_values if v is not None]

    forward_time = timing.get("stage_forward_pass_s")
    if forward_time is None:
        forward_time = timing.get("stages", {}).get("forward_pass", {}).get("total_s")
    e2e = timing.get("total_end_to_end_s")
    patches = timing.get("num_patches")
    patches_s = (float(patches) / float(forward_time)) if patches and forward_time else None
    aggregate = metrics.get("aggregates", metrics)

    row.update(
        {
            "Model size": metadata.get("model_size_bytes"),
            "Forward time": forward_time,
            "E2E time": e2e,
            "Patches/s": patches_s,
            "Peak RAM": max(peak_rss_values) if peak_rss_values else metadata.get("peak_rss_kb"),
            "Metric": aggregate.get("f1_score"),
            "AP": aggregate.get("AP"),
            "Detections": aggregate.get("total_detections"),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Raspberry Pi FCOS benchmark results.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/rpi4_benchmark_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("results/rpi4_benchmark_comparison.md"))
    args = parser.parse_args()

    rows = [collect_variant(args.results_root, *variant) for variant in VARIANTS]
    baseline = next((row for row in rows if row["Model"] == "fcos18" and row["Runtime"] == "pytorch" and row["Precision"] == "fp32"), {})
    baseline_forward = baseline.get("Forward time")
    for row in rows:
        forward = row.get("Forward time")
        if baseline_forward and forward:
            row["Speedup vs FCOS18 PyTorch"] = float(baseline_forward) / float(forward)
            row["Runtime reduction %"] = (1.0 - (float(forward) / float(baseline_forward))) * 100.0
        else:
            row["Speedup vs FCOS18 PyTorch"] = None
            row["Runtime reduction %"] = None

    fieldnames = [
        "Model",
        "Runtime",
        "Precision",
        "Model size",
        "Forward time",
        "E2E time",
        "Patches/s",
        "Speedup vs FCOS18 PyTorch",
        "Runtime reduction %",
        "Peak RAM",
        "Metric",
        "AP",
        "Detections",
        "Run dir",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Model | Runtime | Precision | Model size | Forward time | E2E time | Patches/s | Speedup vs FCOS18 PyTorch | Peak RAM | Metric |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("Model")),
                    fmt(row.get("Runtime")),
                    fmt(row.get("Precision")),
                    fmt(row.get("Model size")),
                    fmt(row.get("Forward time")),
                    fmt(row.get("E2E time")),
                    fmt(row.get("Patches/s")),
                    fmt(row.get("Speedup vs FCOS18 PyTorch")),
                    fmt(row.get("Peak RAM")),
                    fmt(row.get("Metric")),
                ]
            )
            + " |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
