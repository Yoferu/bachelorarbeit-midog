#!/usr/bin/env python3
"""Standard NNCF static PTQ for an OpenVINO FCOS IR using the audited calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--subset-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    output_bin = cfg.output_model.with_suffix(".bin")
    for path in (cfg.fp32_model, cfg.fp32_model.with_suffix(".bin"), cfg.calibration_csv, cfg.split_manifest, cfg.final_test_csv):
        if not path.is_file():
            raise SystemExit(f"Required input does not exist: {path}")
    if cfg.output_model.exists() or output_bin.exists() or cfg.metadata_output.exists():
        raise SystemExit("Refusing to overwrite an existing quantization artifact or metadata file")

    calibration = pd.read_csv(cfg.calibration_csv)
    final_test = pd.read_csv(cfg.final_test_csv)
    calibration_ids = sorted(set(calibration.filename.astype(str)))
    final_test_ids = set(final_test.query("split == 'test'").filename.astype(str))
    overlap_ids = sorted(set(calibration_ids) & final_test_ids)
    manifest = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    if overlap_ids or manifest["pairwise_overlap_counts"].get("calibration_test") != 0:
        raise SystemExit(f"Calibration/final-test leakage detected: {overlap_ids[:10]}")
    if len(calibration_ids) != 39 or len(calibration) != 2289:
        raise SystemExit(f"Unexpected calibration split: {len(calibration_ids)} images, {len(calibration)} annotations")

    sys.path.insert(0, str(cfg.guide_repo.resolve()))
    from utils import inference as guide_inference
    import nncf
    import openvino as ov

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    patch_config = guide_inference.PatchConfig(size=cfg.patch_size, overlap=cfg.overlap)
    patch_refs = []
    per_image_patch_counts = {}
    for filename in calibration_ids:
        dataset = guide_inference.ROI_InferenceDataset(cfg.image_dir / filename, patch_config=patch_config)
        count = len(dataset)
        if count == 0:
            raise SystemExit(f"Calibration image yielded no patches: {filename}")
        per_image_patch_counts[filename] = count
        patch_refs.extend((filename, index) for index in range(count))
    if len(patch_refs) < cfg.subset_size:
        raise SystemExit(f"Only {len(patch_refs)} calibration patches for subset_size={cfg.subset_size}")

    # Deterministic round-robin order exposes all 39 images before repeating images.
    ordered_refs = []
    max_count = max(per_image_patch_counts.values())
    for index in range(max_count):
        for filename in calibration_ids:
            if index < per_image_patch_counts[filename]:
                ordered_refs.append((filename, index))
    loaded = Counter()

    def transform(ref):
        filename, index = ref
        dataset = guide_inference.ROI_InferenceDataset(cfg.image_dir / filename, patch_config=patch_config)
        patch, *_ = dataset[index]
        value = patch.detach().cpu().numpy().astype(np.float32, copy=False)
        if value.shape != (3, cfg.patch_size, cfg.patch_size) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid calibration patch for {filename}[{index}]: {value.shape}")
        loaded[filename] += 1
        return value

    core = ov.Core()
    fp32_model = core.read_model(str(cfg.fp32_model))
    dataset = nncf.Dataset(ordered_refs, transform)
    quantized_model = nncf.quantize(
        fp32_model,
        dataset,
        subset_size=cfg.subset_size,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
    )
    cfg.output_model.parent.mkdir(parents=True, exist_ok=False)
    ov.save_model(quantized_model, str(cfg.output_model), compress_to_fp16=False)

    saved = core.read_model(str(cfg.output_model))
    ops = Counter(op.get_type_name() for op in saved.get_ops())
    element_types = Counter(str(out.get_element_type()) for op in saved.get_ops() for out in op.outputs())
    fq_count = ops["FakeQuantize"]
    low_precision_outputs = element_types["i8"] + element_types["u8"]
    if fq_count == 0 and low_precision_outputs == 0:
        cfg.output_model.unlink(missing_ok=True)
        output_bin.unlink(missing_ok=True)
        raise SystemExit("NNCF produced no verified low-precision OpenVINO graph")
    rt_info = {str(k): str(v) for k, v in saved.get_rt_info().items()}
    metadata = {
        "status": "success",
        "workflow": "OpenVINO FP32 IR -> NNCF static PTQ -> OpenVINO INT8 IR",
        "source_model": str(cfg.fp32_model.resolve()),
        "source_model_sha256": sha256(cfg.fp32_model),
        "source_weights_sha256": sha256(cfg.fp32_model.with_suffix('.bin')),
        "output_model": str(cfg.output_model.resolve()),
        "output_model_sha256": sha256(cfg.output_model),
        "output_weights_sha256": sha256(output_bin),
        "source_size_bytes": cfg.fp32_model.stat().st_size + cfg.fp32_model.with_suffix('.bin').stat().st_size,
        "output_size_bytes": cfg.output_model.stat().st_size + output_bin.stat().st_size,
        "calibration_csv": str(cfg.calibration_csv.resolve()),
        "calibration_csv_sha256": sha256(cfg.calibration_csv),
        "calibration_images": len(calibration_ids),
        "calibration_annotations": len(calibration),
        "available_calibration_patches": len(patch_refs),
        "nncf_subset_size": cfg.subset_size,
        "actual_transform_calls": sum(loaded.values()),
        "calibration_images_presented": len(loaded),
        "per_image_transform_calls": dict(loaded),
        "calibration_test_overlap": overlap_ids,
        "input_shape": [3, cfg.patch_size, cfg.patch_size],
        "input_dtype": "float32",
        "preprocessing": "MIDOG Guide ROI_InferenceDataset patch tensor; same 1024/0.3 path as evaluation",
        "seed": cfg.seed,
        "nncf_parameters": {
            "algorithm": "nncf.quantize standard PTQ",
            "subset_size": cfg.subset_size,
            "model_type": None,
            "preset": "PERFORMANCE",
            "target_device": "CPU",
            "fast_bias_correction": True,
            "advanced_parameters": None,
        },
        "operator_counts": dict(ops),
        "element_type_counts": dict(element_types),
        "fake_quantize_count": fq_count,
        "low_precision_output_count": low_precision_outputs,
        "rt_info": rt_info,
        "versions": {"openvino": ov.__version__, "nncf": nncf.__version__, "numpy": np.__version__},
    }
    cfg.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
