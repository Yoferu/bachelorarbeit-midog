import argparse
import csv
import glob
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize measured quick benchmark timing JSON files.")
    parser.add_argument("--timing_glob", required=True, help="Glob for measured timing JSON files.")
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    files = [Path(path) for path in sorted(glob.glob(args.timing_glob))]
    if not files:
        raise FileNotFoundError(f"No timing JSON files matched: {args.timing_glob}")

    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "timing_file": str(path),
            "run_name": data.get("run_name"),
            "dataset_csv": data.get("dataset_csv"),
            "runtime_backend": data.get("runtime_backend"),
            "effective_runtime_backend": data.get("effective_runtime_backend"),
            "compile_backend": data.get("compile_backend"),
            "compile_mode": data.get("compile_mode"),
            "onnx_opset": data.get("onnx_opset"),
            "openvino_device": data.get("openvino_device"),
            "openvino_version": data.get("openvino_version"),
            "available_devices": "; ".join(data.get("available_devices") or []),
            "selected_device": data.get("selected_device"),
            "source_onnx_path": data.get("source_onnx_path"),
            "source_onnx_reused": data.get("source_onnx_reused"),
            "openvino_model_path": data.get("openvino_model_path"),
            "openvino_weights_path": data.get("openvino_weights_path"),
            "openvino_ir_reused": data.get("openvino_ir_reused"),
            "openvino_compress_to_fp16": data.get("openvino_compress_to_fp16"),
            "export_path": data.get("export_path"),
            "export_reused": data.get("export_reused"),
            "num_images": data.get("num_images"),
            "num_patches": data.get("num_patches"),
            "device": data.get("device"),
            "batch_size": data.get("batch_size"),
            "num_workers": data.get("num_workers"),
            "overlap": data.get("overlap"),
            "nms_thresh": data.get("nms_thresh"),
            "measured_inference_time_s": data.get("measured_inference_time_s"),
            "total_end_to_end_s": data.get("total_end_to_end_s"),
            "median_per_image_s": data.get("median_per_image_s"),
            "median_per_patch_s": data.get("median_per_patch_s"),
        })

    measured = [row["measured_inference_time_s"] for row in rows if row["measured_inference_time_s"] is not None]
    totals = [row["total_end_to_end_s"] for row in rows if row["total_end_to_end_s"] is not None]

    summary = {
        "timing_file": "SUMMARY",
        "run_name": "median_measured_repeats",
        "dataset_csv": rows[0]["dataset_csv"],
        "runtime_backend": rows[0]["runtime_backend"],
        "effective_runtime_backend": rows[0]["effective_runtime_backend"],
        "compile_backend": rows[0]["compile_backend"],
        "compile_mode": rows[0]["compile_mode"],
        "onnx_opset": rows[0]["onnx_opset"],
        "openvino_device": rows[0]["openvino_device"],
        "openvino_version": rows[0]["openvino_version"],
        "available_devices": rows[0]["available_devices"],
        "selected_device": rows[0]["selected_device"],
        "source_onnx_path": rows[0]["source_onnx_path"],
        "source_onnx_reused": rows[0]["source_onnx_reused"],
        "openvino_model_path": rows[0]["openvino_model_path"],
        "openvino_weights_path": rows[0]["openvino_weights_path"],
        "openvino_ir_reused": rows[0]["openvino_ir_reused"],
        "openvino_compress_to_fp16": rows[0]["openvino_compress_to_fp16"],
        "export_path": rows[0]["export_path"],
        "export_reused": rows[0]["export_reused"],
        "num_images": rows[0]["num_images"],
        "num_patches": rows[0]["num_patches"],
        "device": rows[0]["device"],
        "batch_size": rows[0]["batch_size"],
        "num_workers": rows[0]["num_workers"],
        "overlap": rows[0]["overlap"],
        "nms_thresh": rows[0]["nms_thresh"],
        "measured_inference_time_s": statistics.median(measured) if measured else None,
        "total_end_to_end_s": statistics.median(totals) if totals else None,
        "median_per_image_s": statistics.median(
            row["median_per_image_s"] for row in rows if row["median_per_image_s"] is not None
        ),
        "median_per_patch_s": statistics.median(
            row["median_per_patch_s"] for row in rows if row["median_per_patch_s"] is not None
        ),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
        writer.writerows(rows)

    print(f"Wrote timing summary: {args.output_csv}")


if __name__ == "__main__":
    main()
