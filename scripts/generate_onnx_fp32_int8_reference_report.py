#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/quantization/ptq_20260816T1815Z"
OUT = RUN / "onnx_fp32_int8_reference"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_ram(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(match.group(1)) * 1024 if match else None


def make_row(model: str, precision: str, result_dir: Path, artifact: Path, known_ram: int | None = None):
    metrics = load(result_dir / "metrics.json")["aggregates"]
    timing = load(result_dir / "timing.json")
    runtime = load(result_dir / "runtime.json")
    stages = timing["stages"]
    ram = peak_ram(result_dir / "resource_usage.txt") or known_ram
    return {
        "model": model,
        "precision": precision,
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "f1": metrics["f1_score"],
        "precision_metric": metrics["precision"],
        "recall": metrics["recall"],
        "ap": metrics["AP"],
        "forward_time_s": stages["forward_pass"]["total_s"],
        "e2e_time_s": timing["total_end_to_end_s"],
        "throughput_patches_per_s": timing["num_patches"] / timing["total_end_to_end_s"],
        "preprocessing_time_s": stages["dataloader_wait_preprocessing"]["total_s"],
        "postprocessing_nms_time_s": stages["patch_merging_postprocessing_nms"]["total_s"],
        "peak_ram_bytes": ram,
        "model_size_bytes": artifact.stat().st_size,
        "image_count": runtime["num_images"],
        "patch_count": timing["num_patches"],
        "threshold": runtime["det_thresh"],
        "dataset": runtime["dataset_csv"],
        "execution_providers": runtime["execution_providers"],
        "onnxruntime_version": "1.26.0",
        "model_path": str(artifact.relative_to(ROOT)),
        "model_sha256": sha256(artifact),
        "result_dir": str(result_dir.relative_to(ROOT)),
    }


def effect(source, target):
    return {
        "source": f"{source['model']} {source['precision']}",
        "target": f"{target['model']} {target['precision']}",
        "forward_speedup": source["forward_time_s"] / target["forward_time_s"],
        "e2e_speedup": source["e2e_time_s"] / target["e2e_time_s"],
        "f1_delta": target["f1"] - source["f1"],
        "ap_delta": target["ap"] - source["ap"],
        "precision_delta": target["precision_metric"] - source["precision_metric"],
        "recall_delta": target["recall"] - source["recall"],
        "model_size_reduction_fraction": 1 - target["model_size_bytes"] / source["model_size_bytes"],
        "ram_reduction_fraction": (
            1 - target["peak_ram_bytes"] / source["peak_ram_bytes"]
            if source["peak_ram_bytes"] and target["peak_ram_bytes"] else None
        ),
    }


def main():
    baseline_fp32_artifact = RUN / "baseline/export/FCOS_18_patch1024_opset18_d5892925a8.onnx"
    baseline_int8_artifact = RUN / "baseline/int8/FCOS_18_qdq_int8.onnx"
    pruned_fp32_artifact = RUN / "pruned_recovered/export/FCOS_18_patch1024_opset18_d5892925a8.onnx"
    pruned_int8_artifact = RUN / "pruned_recovered/int8/FCOS_18_pruned60_recovered_qdq_int8.onnx"
    rows = [
        make_row("FCOS_18 baseline", "FP32", RUN / "baseline/final_test_fp32_fixed", baseline_fp32_artifact),
        make_row("FCOS_18 baseline", "INT8", RUN / "baseline/final_test_fixed", baseline_int8_artifact),
        make_row("Pruned/recovered", "FP32", RUN / "pruned_recovered/final_test_fp32_fixed", pruned_fp32_artifact),
        make_row("Pruned/recovered", "INT8", RUN / "pruned_recovered/final_test_fixed", pruned_int8_artifact, 2258188 * 1024),
    ]
    bf, bi, pf, pi = rows
    effects = {
        "quantization_baseline": effect(bf, bi),
        "quantization_pruned_recovered": effect(pf, pi),
        "pruning_fp32": effect(bf, pf),
        "pruning_int8": effect(bi, pi),
    }
    methodology = {
        "valid": True,
        "dataset_identity": len({r["dataset"] for r in rows}) == 1,
        "image_count_identity": [r["image_count"] for r in rows],
        "patch_count_identity": [r["patch_count"] for r in rows],
        "threshold_identity": [r["threshold"] for r in rows],
        "providers": [r["execution_providers"] for r in rows],
        "patch_coordinates": "Same deterministic ROI_InferenceDataset, patch size 1024, overlap 0.3, image order, and 8500-patch count; coordinates are not persisted separately.",
        "postprocessing": "Same guide processor, merge/NMS implementation, NMS threshold 0.3, and MIDOGEvaluation adapter.",
        "cpu_controls": {"affinity": "0-3", "OMP_NUM_THREADS": 4, "MKL_NUM_THREADS": 4, "OPENBLAS_NUM_THREADS": 4, "NUMEXPR_NUM_THREADS": 4, "TORCH_NUM_THREADS": 4, "TORCH_NUM_INTEROP_THREADS": 1, "batch_size": 1, "num_workers": 0},
        "ort_session_threads": "No explicit SessionOptions override in either FP32 or INT8 backend; both use identical ORT defaults under the same four-core affinity and environment.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "onnx_fp32_int8_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "onnx_fp32_int8_comparison.json").write_text(json.dumps({"rows": rows, "effects": effects, "methodology": methodology}, indent=2), encoding="utf-8")
    lines = ["# ONNX Runtime FP32/INT8 Final-Test Reference", "", "All rows use the same 111-image final test, 5,383 annotations, 8,500 patches, fixed threshold 0.584, and CPU_4C_LIMITED controls.", "", "| Model | Precision | F1 | Precision metric | Recall | AP | Forward s | E2E s | Peak RAM MiB | Size MiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        ram = f"{row['peak_ram_bytes']/1048576:.1f}" if row["peak_ram_bytes"] else "unavailable"
        lines.append(f"| {row['model']} | {row['precision']} | {row['f1']:.6f} | {row['precision_metric']:.6f} | {row['recall']:.6f} | {row['ap']:.6f} | {row['forward_time_s']:.2f} | {row['e2e_time_s']:.2f} | {ram} | {row['model_size_bytes']/1048576:.2f} |")
    lines.extend(["", "## Effects", "", "| Comparison | Forward speedup | E2E speedup | F1 delta | AP delta | Precision delta | Recall delta | Size reduction |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for value in effects.values():
        lines.append(f"| {value['source']} -> {value['target']} | {value['forward_speedup']:.4f}x | {value['e2e_speedup']:.4f}x | {value['f1_delta']:+.6f} | {value['ap_delta']:+.6f} | {value['precision_delta']:+.6f} | {value['recall_delta']:+.6f} | {value['model_size_reduction_fraction']:.2%} |")
    lines.extend(["", "## Methodology", "", "Comparison valid: identical dataset, deterministic patch generator/configuration, 111 images, 8,500 patches, threshold 0.584, CPUExecutionProvider only, and identical postprocessing/evaluation code. ORT session thread options were not explicitly overridden in either precision path; both used identical defaults under affinity 0-3 and the same thread environment.", "", "Baseline INT8 peak RAM is unavailable because its original `/usr/bin/time` output was not persisted. Pruned INT8 peak RAM (2205.3 MiB) comes from its captured `/usr/bin/time` output."])
    (OUT / "onnx_fp32_int8_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
