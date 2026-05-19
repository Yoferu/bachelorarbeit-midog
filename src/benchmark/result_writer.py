import csv
import json
from pathlib import Path
from typing import Any


def write_runtime_metadata(metadata: dict[str, Any], path: Path | None) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Runtime metadata saved to: {path}")


def write_timing_outputs(
    summary: dict[str, Any],
    json_path: Path | None,
    csv_path: Path | None,
) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Timing JSON saved to: {json_path}")

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        flat = {
            "run_name": summary["run_name"],
            "config_file": summary["config_file"],
            "dataset_csv": summary["dataset_csv"],
            "runtime_backend": summary.get("runtime_backend"),
            "effective_runtime_backend": summary.get("effective_runtime_backend"),
            "compile_backend": summary.get("compile_backend"),
            "compile_mode": summary.get("compile_mode"),
            "onnx_opset": summary.get("onnx_opset"),
            "openvino_device": summary.get("openvino_device"),
            "export_path": summary.get("export_path"),
            "export_reused": summary.get("export_reused"),
            "runtime_warnings": "; ".join(summary.get("runtime_warnings", [])),
            "model_name": summary.get("model_name"),
            "num_images": summary["num_images"],
            "num_patches": summary["num_patches"],
            "device": summary["device"],
            "batch_size": summary["batch_size"],
            "num_workers": summary["num_workers"],
            "overlap": summary["overlap"],
            "nms_thresh": summary["nms_thresh"],
            "total_end_to_end_s": summary["total_end_to_end_s"],
            "startup_time_s": summary["startup_time_s"],
            "measured_inference_time_s": summary["measured_inference_time_s"],
            "evaluation_metrics_s": summary["evaluation_metrics_s"],
            "output_serialization_s": summary["output_serialization_s"],
            "mean_per_image_s": summary["mean_per_image_s"],
            "median_per_image_s": summary["median_per_image_s"],
            "mean_per_patch_s": summary["mean_per_patch_s"],
            "median_per_patch_s": summary["median_per_patch_s"],
        }
        for stage, values in summary["stages"].items():
            flat[f"stage_{stage}_s"] = values["total_s"]
            flat[f"stage_{stage}_pct"] = values["percent_of_measured_inference"]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
        print(f"Timing CSV saved to: {csv_path}")
