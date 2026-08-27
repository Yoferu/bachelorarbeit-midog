#!/usr/bin/env python3
"""Static QDQ PTQ for an exported FCOS graph using the clean recovery calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--final-test-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--guide-repo", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--max-patches-per-image", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    cfg = args()
    for path in (cfg.fp32_model, cfg.calibration_csv, cfg.split_manifest, cfg.final_test_csv):
        if not path.is_file():
            raise SystemExit(f"Required input does not exist: {path}")
    if cfg.output_model.exists() or cfg.metadata_output.exists():
        raise SystemExit("Refusing to overwrite an existing quantization artifact or metadata file")

    calibration = pd.read_csv(cfg.calibration_csv)
    final_test = pd.read_csv(cfg.final_test_csv)
    calibration_ids = set(calibration.filename.astype(str))
    final_test_ids = set(final_test.query("split == 'test'").filename.astype(str))
    overlap = sorted(calibration_ids & final_test_ids)
    manifest = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    declared = manifest["pairwise_overlap_counts"]
    if overlap or declared.get("calibration_test") != 0:
        raise SystemExit(f"Calibration/final-test leakage detected: {overlap[:10]}")
    if len(calibration_ids) != 39:
        raise SystemExit(f"Expected audited 39-image calibration split, found {len(calibration_ids)}")

    sys.path.insert(0, str(cfg.guide_repo.resolve()))
    from utils import inference as guide_inference
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    import onnx
    import onnxruntime as ort

    input_name = ort.InferenceSession(str(cfg.fp32_model), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    patch_config = guide_inference.PatchConfig(size=cfg.patch_size, overlap=cfg.overlap)
    samples = []
    for filename in sorted(calibration_ids):
        dataset = guide_inference.ROI_InferenceDataset(cfg.image_dir / filename, patch_config=patch_config)
        if not len(dataset):
            raise SystemExit(f"Calibration image yielded no patches: {filename}")
        for index in range(min(len(dataset), cfg.max_patches_per_image)):
            patch, *_ = dataset[index]
            value = patch.detach().cpu().numpy().astype(np.float32, copy=False)
            if value.shape != (3, cfg.patch_size, cfg.patch_size) or not np.isfinite(value).all():
                raise SystemExit(f"Invalid calibration patch for {filename}: {value.shape}")
            samples.append({input_name: value})

    class Reader(CalibrationDataReader):
        def __init__(self, rows):
            self.rows = rows
            self.iterator = iter(rows)

        def get_next(self):
            return next(self.iterator, None)

        def rewind(self):
            self.iterator = iter(self.rows)

    cfg.output_model.parent.mkdir(parents=True, exist_ok=False)
    quantize_static(
        str(cfg.fp32_model),
        str(cfg.output_model),
        Reader(samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    graph = onnx.load(str(cfg.output_model), load_external_data=False)
    ops = Counter(node.op_type for node in graph.graph.node)
    int8_weights = sum(i.data_type in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8} for i in graph.graph.initializer)
    fp32_graph = onnx.load(str(cfg.fp32_model), load_external_data=False)
    fp32_ops = Counter(node.op_type for node in fp32_graph.graph.node)
    quantized_conv_weights = sum(
        node.op_type == "DequantizeLinear" and any("conv" in output.lower() for output in node.output)
        for node in graph.graph.node
    )
    if int8_weights == 0 or ops["QuantizeLinear"] == 0 or ops["DequantizeLinear"] == 0:
        cfg.output_model.unlink(missing_ok=True)
        raise SystemExit("Quantizer produced no verified QDQ INT8 compute")
    metadata = {
        "status": "success",
        "representation": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel_weights": True,
        "calibration_method": "MinMax",
        "calibration_csv": str(cfg.calibration_csv.resolve()),
        "calibration_csv_sha256": sha256(cfg.calibration_csv),
        "calibration_images": len(calibration_ids),
        "calibration_annotations": int(len(calibration)),
        "calibration_patches": len(samples),
        "calibration_test_overlap": overlap,
        "source_model": str(cfg.fp32_model.resolve()),
        "source_model_sha256": sha256(cfg.fp32_model),
        "output_model": str(cfg.output_model.resolve()),
        "output_model_sha256": sha256(cfg.output_model),
        "source_size_bytes": cfg.fp32_model.stat().st_size,
        "output_size_bytes": cfg.output_model.stat().st_size,
        "source_operator_counts": dict(fp32_ops),
        "quantized_operator_counts": dict(ops),
        "int8_uint8_initializer_count": int8_weights,
        "qdq_node_count": ops["QuantizeLinear"] + ops["DequantizeLinear"],
        "diagnostic_conv_named_dequantize_count": quantized_conv_weights,
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    cfg.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
