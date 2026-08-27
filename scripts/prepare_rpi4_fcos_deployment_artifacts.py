from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_REPO = REPO_ROOT / "repos/MIDOG_2025_Guide"
if str(GUIDE_REPO) not in sys.path:
    sys.path.insert(0, str(GUIDE_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import inference as guide_inference  # noqa: E402


class PatchCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        *,
        onnx_path: Path,
        calibration_csv: Path,
        image_root: Path,
        patch_size: int,
        overlap: float,
        max_images: int,
        max_patches: int,
    ) -> None:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        self.samples = self._collect_samples(
            calibration_csv=calibration_csv,
            image_root=image_root,
            patch_size=patch_size,
            overlap=overlap,
            max_images=max_images,
            max_patches=max_patches,
        )
        self._iter = iter(self.samples)

    def _collect_samples(
        self,
        *,
        calibration_csv: Path,
        image_root: Path,
        patch_size: int,
        overlap: float,
        max_images: int,
        max_patches: int,
    ) -> list[dict[str, np.ndarray]]:
        with calibration_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        filenames = sorted({row["filename"] for row in rows if row.get("split") == "calibration"})
        if max_images > 0:
            filenames = filenames[:max_images]

        samples: list[dict[str, np.ndarray]] = []
        patch_config = guide_inference.PatchConfig(size=patch_size, overlap=overlap)
        for filename in filenames:
            dataset = guide_inference.ROI_InferenceDataset(image_root / filename, patch_config=patch_config)
            for idx in range(len(dataset)):
                patch, *_ = dataset[idx]
                samples.append({self.input_name: patch.numpy().astype(np.float32, copy=False)})
                if len(samples) >= max_patches:
                    return samples
        return samples

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self.samples)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env={**os.environ, **(env or {})})


def write_config(path: Path, *, model_name: str, checkpoint: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "backbone: resnet18",
                f"checkpoint: {checkpoint.as_posix()}",
                "det_thresh: 0.584",
                "detector: FCOS",
                "extra_blocks: false",
                f"model_name: {model_name}",
                "num_classes: 2",
                "patch_size: 1024",
                "returned_layers:",
                "- 1",
                "- 2",
                "- 3",
                "- 4",
                "weights: null",
                "means: null",
                "stds: null",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_pruned_onnx(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if paths["pruned_fp32"].exists() and not args.force:
        return
    tmp_export = paths["work"] / "models"
    run(
        [
            str(args.python),
            "src/benchmark/midog_guide_adapter.py",
            "--config_file",
            str(paths["pruned_config"]),
            "--dataset",
            str(args.test_csv),
            "--guide_repo",
            str(args.guide_repo),
            "--img_dir",
            str(args.image_root),
            "--metrics_output",
            str(paths["results"] / "pruned60_fp32_onnx_export_metrics.json"),
            "--runtime_metadata_output",
            str(paths["results"] / "pruned60_fp32_onnx_export_runtime.json"),
            "--runtime_backend",
            "onnxruntime_cpu",
            "--pruned_model_path",
            str(paths["pruned_pt"]),
            "--onnx_opset",
            "18",
            "--export_dir",
            str(tmp_export),
            "--split",
            "test",
            "--device",
            "cpu",
            "--batch_size",
            "1",
            "--num_workers",
            "0",
            "--overlap",
            "0.3",
            "--nms_thresh",
            "0.3",
            "--overwrite",
        ],
        env={"MPLCONFIGDIR": "/tmp/matplotlib-rpi4-artifacts"},
    )
    runtime = json.loads((paths["results"] / "pruned60_fp32_onnx_export_runtime.json").read_text(encoding="utf-8"))
    exported = Path(runtime["export_path"])
    if not exported.is_absolute():
        exported = REPO_ROOT / exported
    shutil.copy2(exported, paths["pruned_fp32"])


def quantize_model(
    *,
    fp32_path: Path,
    int8_path: Path,
    calibration_csv: Path,
    image_root: Path,
    max_images: int,
    max_patches: int,
    force: bool,
) -> dict[str, Any]:
    if int8_path.exists() and not force:
        return inspect_quantized_model(int8_path)

    reader = PatchCalibrationDataReader(
        onnx_path=fp32_path,
        calibration_csv=calibration_csv,
        image_root=image_root,
        patch_size=1024,
        overlap=0.3,
        max_images=max_images,
        max_patches=max_patches,
    )
    if not reader.samples:
        raise RuntimeError(f"No calibration patches collected from {calibration_csv}")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=True,
        reduce_range=False,
    )
    return inspect_quantized_model(int8_path) | {"calibration_patches": len(reader.samples)}


def inspect_quantized_model(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    quantized = sorted(k for k in ops if k in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "ConvInteger", "MatMulInteger"})
    fp32_left = sorted(k for k in ops if k in {"Conv", "MatMul", "Gemm"})
    return {
        "onnx_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "operator_counts": ops,
        "quantized_operator_types": quantized,
        "float_operator_types_remaining": fp32_left,
    }


def count_parameters_from_checkpoint(config_path: Path, guide_repo: Path) -> int:
    sys.path.insert(0, str(guide_repo))
    from utils.factory import ConfigCreator, ModelFactory

    config = ConfigCreator.load(str(config_path))
    model = ModelFactory.load(config, det_thresh=0.05)
    return int(sum(param.numel() for param in model.parameters()))


def count_parameters_from_pruned(path: Path, guide_repo: Path) -> int:
    sys.path.insert(0, str(guide_repo))
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = artifact["model_object"]
    return int(sum(param.numel() for param in model.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FCOS_18/Pruned60 ONNX and INT8 deployment artifacts.")
    parser.add_argument("--work-dir", type=Path, default=Path("dist/rpi4_fcos_artifacts"))
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--guide-repo", type=Path, default=Path("repos/MIDOG_2025_Guide"))
    parser.add_argument("--image-root", type=Path, default=Path("data/midogpp"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/midogpp_guide_eval_xvalidation_3img.csv"))
    parser.add_argument("--calibration-csv", type=Path, default=Path("experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv"))
    parser.add_argument("--max-calibration-images", type=int, default=8)
    parser.add_argument("--max-calibration-patches", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    work = args.work_dir
    paths = {
        "work": work,
        "models": work / "models",
        "configs": work / "configs",
        "results": work / "results",
        "checkpoint": work / "models/fcos18_checkpoint.ckpt",
        "pruned_pt": work / "models/fcos18_pruned60.pt",
        "fcos18_fp32": work / "models/fcos18_fp32.onnx",
        "fcos18_int8": work / "models/fcos18_int8.onnx",
        "pruned_fp32": work / "models/fcos18_pruned60_fp32.onnx",
        "pruned_int8": work / "models/fcos18_pruned60_int8.onnx",
        "fcos18_config": work / "configs/FCOS_18_eval.yaml",
        "pruned_config": work / "configs/FCOS_18_pruned60_eval.yaml",
    }
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["configs"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    shutil.copy2(REPO_ROOT / "repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt", paths["checkpoint"])
    shutil.copy2(REPO_ROOT / "experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx", paths["fcos18_fp32"])
    shutil.copy2(REPO_ROOT / "experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt", paths["pruned_pt"])
    write_config(paths["fcos18_config"], model_name="FCOS_18", checkpoint=paths["checkpoint"])
    write_config(paths["pruned_config"], model_name="FCOS_18_pruned60", checkpoint=paths["checkpoint"])

    export_pruned_onnx(args, paths)
    quant_reports = {
        "fcos18_int8": quantize_model(
            fp32_path=paths["fcos18_fp32"],
            int8_path=paths["fcos18_int8"],
            calibration_csv=args.calibration_csv,
            image_root=args.image_root,
            max_images=args.max_calibration_images,
            max_patches=args.max_calibration_patches,
            force=args.force,
        ),
        "pruned60_int8": quantize_model(
            fp32_path=paths["pruned_fp32"],
            int8_path=paths["pruned_int8"],
            calibration_csv=args.calibration_csv,
            image_root=args.image_root,
            max_images=args.max_calibration_images,
            max_patches=args.max_calibration_patches,
            force=args.force,
        ),
    }
    report = {
        "source_artifacts": {
            "fcos18_checkpoint": "repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt",
            "fcos18_fp32_onnx": "experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx",
            "pruned60": "experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt",
        },
        "model_files": {
            key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in paths.items()
            if key in {"checkpoint", "pruned_pt", "fcos18_fp32", "fcos18_int8", "pruned_fp32", "pruned_int8"}
        },
        "parameter_counts": {
            "fcos18": count_parameters_from_checkpoint(paths["fcos18_config"], args.guide_repo),
            "pruned60": count_parameters_from_pruned(paths["pruned_pt"], args.guide_repo),
        },
        "quantization": {
            "method": "ONNX Runtime static post-training quantization",
            "format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "op_types_to_quantize": ["Conv", "MatMul", "Gemm"],
            "calibration_csv": str(args.calibration_csv),
            "max_calibration_images": args.max_calibration_images,
            "max_calibration_patches": args.max_calibration_patches,
            "preprocessing": "existing MIDOG ROI_InferenceDataset patch extraction, 1024 patch size, 0.3 overlap, tensor scaling to [0,1]",
            "reports": quant_reports,
        },
    }
    (paths["results"] / "deployment_artifacts_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
