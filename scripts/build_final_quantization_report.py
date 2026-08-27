#!/usr/bin/env python3
"""Build the final OpenVINO/ORT fixed and calibrated quantization reports."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OV = ROOT / "experiments/quantization/openvino_ptq_20260819T0000Z"
ORT = ROOT / "experiments/quantization/ptq_20260816T1815Z"
OUT = ROOT / "experiments/combined_optimization/final_quantization_comparison_20260819T0000Z"
FIXED_THRESHOLD = 0.584
ORT_BASELINE_FORWARD = 3780.1057181580213
ORT_BASELINE_E2E = 3913.469515539


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_ram(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(match.group(1)) * 1024 if match else None


def artifact_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def fixed_row(model, runtime, precision, result_dir, artifacts, ram_override=None):
    metrics = read(result_dir / "metrics.json")["aggregates"]
    timing = read(result_dir / "timing.json")
    forward = timing["stages"]["forward_pass"]["total_s"]
    e2e = timing["total_end_to_end_s"]
    ram = ram_override if ram_override is not None else peak_ram(result_dir / "resource_usage.txt")
    return {
        "model": model, "runtime": runtime, "precision": precision,
        "operating_point": "fixed", "threshold": FIXED_THRESHOLD,
        "f1": metrics["f1_score"], "precision_metric": metrics["precision"],
        "recall": metrics["recall"], "ap": metrics["AP"],
        "forward_time_s": forward, "e2e_time_s": e2e,
        "throughput_patches_s": 8500 / e2e,
        "speedup_vs_ort_baseline_forward": ORT_BASELINE_FORWARD / forward,
        "speedup_vs_ort_baseline_e2e": ORT_BASELINE_E2E / e2e,
        "peak_ram_bytes": ram, "model_size_bytes": artifact_size(artifacts),
        "image_count": timing["num_images"], "patch_count": timing["num_patches"],
        "result_dir": str(result_dir.relative_to(ROOT)),
    }


def calibrated_row(fixed, metrics_path):
    metrics = read(metrics_path)["aggregates"]
    row = dict(fixed)
    row.update({
        "operating_point": "calibrated", "threshold": metrics["det_thresh"],
        "f1": metrics["f1_score"], "precision_metric": metrics["precision"],
        "recall": metrics["recall"], "ap": metrics["AP"],
        "result_dir": str(metrics_path.parent.relative_to(ROOT)),
    })
    return row


def write_csv(path: Path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ort_ref = read(ORT / "onnx_fp32_int8_reference/onnx_fp32_int8_comparison.json")["rows"]
    ram_lookup = {(r["model"], r["precision"]): r["peak_ram_bytes"] for r in ort_ref}
    specs = [
        ("Baseline", "ORT", "FP32", ORT/"baseline/final_test_fp32_fixed", [ORT/"baseline/export/FCOS_18_patch1024_opset18_d5892925a8.onnx"], ram_lookup[("FCOS_18 baseline", "FP32")]),
        ("Baseline", "ORT", "INT8", ORT/"baseline/final_test_fixed", [ORT/"baseline/int8/FCOS_18_qdq_int8.onnx"], ram_lookup[("FCOS_18 baseline", "INT8")]),
        ("Baseline", "OpenVINO", "FP32", OV/"baseline/final_test_fixed_fp32", [OV/"baseline/fp32/FCOS_18_fp32.xml", OV/"baseline/fp32/FCOS_18_fp32.bin"], None),
        ("Baseline", "OpenVINO", "INT8", OV/"baseline/final_test_fixed_int8", [OV/"baseline/int8/FCOS_18_int8.xml", OV/"baseline/int8/FCOS_18_int8.bin"], None),
        ("Pruned/recovered", "ORT", "FP32", ORT/"pruned_recovered/final_test_fp32_fixed", [ORT/"pruned_recovered/export/FCOS_18_patch1024_opset18_d5892925a8.onnx"], ram_lookup[("Pruned/recovered", "FP32")]),
        ("Pruned/recovered", "ORT", "INT8", ORT/"pruned_recovered/final_test_fixed", [ORT/"pruned_recovered/int8/FCOS_18_pruned60_recovered_qdq_int8.onnx"], ram_lookup[("Pruned/recovered", "INT8")]),
        ("Pruned/recovered", "OpenVINO", "FP32", OV/"pruned_recovered/final_test_fixed_fp32", [OV/"pruned_recovered/fp32/FCOS_18_pruned60_recovered_fp32.xml", OV/"pruned_recovered/fp32/FCOS_18_pruned60_recovered_fp32.bin"], None),
        ("Pruned/recovered", "OpenVINO", "INT8", OV/"pruned_recovered/final_test_fixed_int8", [OV/"pruned_recovered/int8/FCOS_18_pruned60_recovered_int8.xml", OV/"pruned_recovered/int8/FCOS_18_pruned60_recovered_int8.bin"], None),
    ]
    fixed = [fixed_row(*spec) for spec in specs]
    index = {(r["model"], r["runtime"], r["precision"]): r for r in fixed}
    cal_specs = [
        (("Baseline", "ORT", "INT8"), OV/"baseline/final_test_calibrated/onnxruntime_int8/metrics.json"),
        (("Pruned/recovered", "ORT", "INT8"), OV/"pruned_recovered/final_test_calibrated/onnxruntime_int8/metrics.json"),
        (("Baseline", "OpenVINO", "INT8"), OV/"baseline/final_test_calibrated/openvino_int8/metrics.json"),
        (("Pruned/recovered", "OpenVINO", "INT8"), OV/"pruned_recovered/final_test_calibrated/openvino_int8/metrics.json"),
    ]
    calibrated = [calibrated_row(index[key], path) for key, path in cal_specs]
    all_rows = fixed + calibrated
    write_csv(OUT/"cross_runtime_quantization_comparison.csv", fixed)
    write_csv(OUT/"fixed_vs_calibrated_comparison.csv", [r for r in all_rows if r["precision"] == "INT8"])

    ov_rows = [r for r in fixed if r["runtime"] == "OpenVINO"]
    write_csv(OUT/"openvino_quantization_comparison.csv", ov_rows)
    (OUT/"openvino_quantization_comparison.json").write_text(json.dumps({"rows": ov_rows}, indent=2), encoding="utf-8")
    ov_md = ["# OpenVINO fixed-threshold quantization comparison", "", "| Model | Precision | F1 | Precision | Recall | AP | Forward s | E2E s | RAM MiB | Size MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ov_rows:
        ov_md.append(f"| {r['model']} | {r['precision']} | {r['f1']:.6f} | {r['precision_metric']:.6f} | {r['recall']:.6f} | {r['ap']:.6f} | {r['forward_time_s']:.2f} | {r['e2e_time_s']:.2f} | {r['peak_ram_bytes']/2**20:.1f} | {r['model_size_bytes']/2**20:.2f} |")
    (OUT/"openvino_quantization_comparison.md").write_text("\n".join(ov_md)+"\n", encoding="utf-8")

    selections = []
    sweeps = []
    for path in sorted(OV.glob("*/calibration/*/threshold_selection.json")):
        selections.append(read(path))
        sweeps.append(pd.read_csv(path.with_name("threshold_sweep.csv")))
    pd.concat(sweeps, ignore_index=True).to_csv(OUT/"threshold_sweep_results.csv", index=False)
    (OUT/"threshold_selection.json").write_text(json.dumps({"selections": selections}, indent=2), encoding="utf-8")

    candidates = [r for r in fixed if r["precision"] == "FP32"] + calibrated
    frontier = []
    for row in candidates:
        dominated = any(
            other is not row
            and other["f1"] >= row["f1"] and other["ap"] >= row["ap"] and other["e2e_time_s"] <= row["e2e_time_s"]
            and (other["f1"] > row["f1"] or other["ap"] > row["ap"] or other["e2e_time_s"] < row["e2e_time_s"])
            for other in candidates
        )
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda r: r["e2e_time_s"])
    write_csv(OUT/"pareto_frontier.csv", frontier)
    pareto_md = ["# Pareto frontier (F1, AP, E2E)", "", "| Model | Runtime | Precision | Threshold | F1 | AP | E2E s | Size MiB | RAM MiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in frontier:
        pareto_md.append(f"| {r['model']} | {r['runtime']} | {r['precision']} | {r['threshold']:.3f} | {r['f1']:.6f} | {r['ap']:.6f} | {r['e2e_time_s']:.2f} | {r['model_size_bytes']/2**20:.2f} | {fmt(None if r['peak_ram_bytes'] is None else r['peak_ram_bytes']/2**20, 1)} |")
    (OUT/"pareto_frontier.md").write_text("\n".join(pareto_md)+"\n", encoding="utf-8")

    artifacts = [p for spec in specs for p in spec[4]]
    artifacts += [ROOT/"experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr3e-5_seed42/student_stable_selected.pt"]
    model_hashes = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "size_bytes": p.stat().st_size} for p in artifacts]
    cal_df = pd.read_csv(ROOT/"experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv")
    test_df = pd.read_csv(ROOT/"data/midogpp_guide_eval_xvalidation.csv")
    cal_names = set(cal_df.query("split == 'calibration'").filename)
    test_names = set(test_df.query("split == 'test'").filename)
    env = {
        "python": platform.python_version(), "platform": platform.platform(),
        "openvino": "2026.1.0", "nncf": "3.2.0", "onnxruntime": "1.26.0",
        "cpu_affinity": "0-3", "device": "CPU only", "model_artifacts": model_hashes,
        "calibration_split": {"images": len(cal_names), "rows": len(cal_df), "positive_annotations": int((cal_df.label == 1).sum()), "patches_available_and_inference": 2922, "ptq_subset_size": 300, "nncf_transform_calls": 600},
        "final_test": {"images": len(test_names), "annotations": len(test_df.query("split == 'test'")), "patches": 8500},
        "calibration_test_overlap": sorted(cal_names & test_names),
    }
    (OUT/"environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    status = {"status": "complete", "phase_a_fixed_openvino": True, "phase_b_threshold_calibration": True, "phase_c_frozen_threshold_scoring": True, "partial_results_reported_as_complete": False, "validation": {"calibration_test_overlap_count": len(cal_names & test_names), "all_fixed_runs_images": [r["image_count"] for r in fixed], "all_fixed_runs_patches": [r["patch_count"] for r in fixed]}}
    (OUT/"run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    summary = {"fixed_rows": fixed, "calibrated_rows": calibrated, "pareto_frontier": frontier}
    (OUT/"final_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "fixed_rows": len(fixed), "calibrated_rows": len(calibrated), "pareto_rows": len(frontier)}, indent=2))


if __name__ == "__main__":
    main()
