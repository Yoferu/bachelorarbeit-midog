#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/quantization/ptq_20260816T1815Z"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, model: str, precision: str, folder: str, smoke: str, artifact: Path):
    metrics = load(RUN / folder / smoke / ("fp32_metrics.json" if precision == "FP32" else "metrics.json"))["aggregates"]
    timing = load(RUN / folder / smoke / ("fp32_timing.json" if precision == "FP32" else "timing.json"))
    return {
        "variant": name, "model": model, "pruning": folder == "pruned_recovered",
        "quantization": precision, "runtime": "ONNX Runtime CPU", "scope": "3-image smoke / 264 patches",
        "f1": metrics["f1_score"], "precision": metrics["precision"], "recall": metrics["recall"], "ap": metrics["AP"],
        "forward_time_s": timing["stages"]["forward_pass"]["total_s"],
        "e2e_time_s": timing["total_end_to_end_s"], "ram_bytes": None,
        "artifact_size_bytes": artifact.stat().st_size, "artifact": str(artifact.relative_to(ROOT)),
    }


def write_table(rows, csv_path: Path, json_path: Path, md_path: Path, title: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", "All numeric rows below are deterministic three-image smoke measurements on 264 patches. They are not final-test claims.", "", "| Variant | Model | Quant. | Runtime | F1 | AP | Forward s | E2E s | Size MiB |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['model']} | {r['quantization']} | {r['runtime']} | {r['f1']:.4f} | {r['ap']:.4f} | {r['forward_time_s']:.2f} | {r['e2e_time_s']:.2f} | {r['artifact_size_bytes']/1048576:.2f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    bfp = RUN / "baseline/export/FCOS_18_patch1024_opset18_d5892925a8.onnx"
    bi8 = RUN / "baseline/int8/FCOS_18_qdq_int8.onnx"
    pfp = RUN / "pruned_recovered/export/FCOS_18_patch1024_opset18_d5892925a8.onnx"
    pi8 = RUN / "pruned_recovered/int8/FCOS_18_pruned60_recovered_qdq_int8.onnx"
    rows = [
        row("Q1-FP32", "FCOS_18", "FP32", "baseline", "smoke", bfp),
        row("Q1-INT8", "FCOS_18", "INT8 QDQ", "baseline", "smoke_int8", bi8),
        row("Q2-FP32", "FCOS_18 pruned60 recovered", "FP32", "pruned_recovered", "smoke", pfp),
        row("Q2-INT8", "FCOS_18 pruned60 recovered", "INT8 QDQ", "pruned_recovered", "smoke_int8", pi8),
    ]
    for r in rows: r["speedup_vs_q1_fp32_e2e"] = rows[0]["e2e_time_s"] / r["e2e_time_s"]
    write_table(rows, RUN / "quantization_comparison.csv", RUN / "quantization_comparison.json", RUN / "quantization_comparison.md", "Quantization Comparison")
    combined = RUN.parent.parent / "combined_optimization/combined_20260816T1815Z"
    write_table(rows, combined / "combined_optimization_comparison.csv", combined / "combined_optimization_comparison.json", combined / "combined_optimization_comparison.md", "Combined Optimization Comparison")
    pareto = [rows[1], rows[3]]
    write_table(pareto, combined / "pareto_frontier.csv", combined / "pareto_frontier.json", combined / "pareto_frontier.md", "Smoke-Level Pareto Frontier")

    bf, bi, pf, pi = rows
    pruning_f = pf["forward_time_s"] / bf["forward_time_s"]
    quant_f = bi["forward_time_s"] / bf["forward_time_s"]
    expected_f = pruning_f * quant_f
    actual_f = pi["forward_time_s"] / bf["forward_time_s"]
    pruning_e = pf["e2e_time_s"] / bf["e2e_time_s"]
    quant_e = bi["e2e_time_s"] / bf["e2e_time_s"]
    expected_e = pruning_e * quant_e
    actual_e = pi["e2e_time_s"] / bf["e2e_time_s"]
    analysis = {
        "scope": "3-image smoke / 264 patches", "forward": {"pruning_factor": pruning_f, "quantization_factor": quant_f, "expected_combined_factor": expected_f, "actual_combined_factor": actual_f, "relative_deviation": actual_f/expected_f-1},
        "end_to_end": {"pruning_factor": pruning_e, "quantization_factor": quant_e, "expected_combined_factor": expected_e, "actual_combined_factor": actual_e, "relative_deviation": actual_e/expected_e-1},
    }
    (combined / "additivity_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    statuses = {
        "pytorch_int8_static": "not_implemented: PyTorch 2.11 recommends torchao/PT2E; torchao is absent and eager quantization cannot safely cover this FCOS graph",
        "pytorch_int8_compile": "not_supported: no compatible PyTorch quantized representation was established",
        "onnxruntime_int8": "success: static QDQ, per-channel QInt8 weights, QUInt8 activations, CPUExecutionProvider",
        "openvino_int8": "unsupported_environment: OpenVINO is installed but NNCF is absent; no dependency was silently installed",
        "threshold_recalibration": "not_run: full fixed-threshold result did not complete, so recalibration was not justified",
        "full_test": "interrupted_after_4_of_111_images: no partial metrics retained",
    }
    (RUN / "run_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
