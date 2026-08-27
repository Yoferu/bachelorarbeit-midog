import argparse
import logging
import os
import pprint
import sys
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.benchmark_config import BenchmarkConfig
from src.benchmark.distilled_model_loader import load_distilled_student
from src.benchmark.pipeline_timing import build_timing_summary, instrument_guide_inference
from src.benchmark.result_writer import write_runtime_metadata, write_timing_outputs
from src.benchmark.runtime_backends import RuntimeBackendUnavailable, configure_runtime_backend
from src.pruning.save_pruned_model import load_pruned_model_object, load_pruned_state_dict


RUNTIME_BACKENDS = ("pytorch_eager", "pytorch_compile", "onnxruntime_cpu", "onnxruntime_int8", "openvino_cpu", "openvino_int8", "tensorrt")


def _parse_bool_flag(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: 1/0, true/false, yes/no, on/off")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracked adapter for MIDOG_2025_Guide evaluation.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--box_format", type=str, default="cxcy")
    parser.add_argument("--compile_backend", type=str, default="inductor")
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--det_thresh", type=float, default=None)
    parser.add_argument("--export_dir", type=Path, default=None)
    parser.add_argument("--guide_repo", type=Path, default=Path("repos/MIDOG_2025_Guide"))
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--int8_model_path", type=Path, default=None)
    parser.add_argument("--metrics_output", type=Path, default=None)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--onnx_opset", type=int, default=18)
    parser.add_argument("--openvino_device", type=str, default="CPU")
    parser.add_argument(
        "--openvino_compress_to_fp16",
        type=_parse_bool_flag,
        default=_parse_bool_flag(os.environ.get("OPENVINO_COMPRESS_TO_FP16", "1")),
        help="Use OpenVINO save_model FP16 weight compression for IR export (1/0).",
    )
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile_pipeline", action="store_true")
    parser.add_argument("--predictions_output", type=Path, default=None)
    parser.add_argument("--pruned_model_path", type=Path, default=os.environ.get("PRUNED_MODEL_PATH"))
    parser.add_argument(
        "--runtime_backend",
        choices=RUNTIME_BACKENDS,
        default=os.environ.get("RUNTIME_BACKEND", "pytorch_eager"),
    )
    parser.add_argument("--runtime_metadata_output", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--timing_output_csv", type=Path, default=None)
    parser.add_argument("--timing_output_json", type=Path, default=None)
    parser.add_argument("--wsi", action="store_true")
    return parser.parse_args()


def _load_guide_modules(guide_repo: Path, profile_pipeline: bool):
    guide_repo = guide_repo.resolve()
    if not guide_repo.is_dir():
        raise FileNotFoundError(f"MIDOG_2025_Guide repo not found: {guide_repo}")
    sys.path.insert(0, str(guide_repo))

    from utils import inference as guide_inference
    from utils.eval_utils import MIDOGEvaluation
    from utils.factory import ConfigCreator, ModelFactory

    if profile_pipeline:
        instrument_guide_inference(guide_inference)

    return guide_inference, MIDOGEvaluation, ConfigCreator, ModelFactory


def _build_runtime_example_input(
    *,
    guide_inference,
    image_path: Path,
    patch_size: int,
    overlap: float,
    is_wsi: bool,
):
    dataset_class = guide_inference.WSI_InferenceDataset if is_wsi else guide_inference.ROI_InferenceDataset
    patch_config = guide_inference.PatchConfig(size=patch_size, overlap=overlap)
    dataset = dataset_class(image_path, patch_config=patch_config)
    if len(dataset) == 0:
        return None
    patch, *_ = dataset[0]
    return patch


def _prediction_count_at_threshold(preds: dict, det_thresh: float) -> int:
    total = 0
    for case_preds in preds.values():
        scores = case_preds.get("scores", [])
        total += int(sum(float(score) > det_thresh for score in scores))
    return total


def _jsonable(value):
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_model_artifact(path: Path, guide_model):
    try:
        registry_model, source = load_distilled_student(path)
        return registry_model, source, "registry"
    except (ValueError, RuntimeError) as registry_error:
        model_object = load_pruned_model_object(path)
        if model_object is not None:
            return model_object, f"model_object fallback ({registry_error})", "model_object"
        state_dict = load_pruned_state_dict(path)
        missing, unexpected = guide_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Could not resolve an architecture for model artifact {path}: {registry_error} "
                "The legacy state_dict also did not match the guide model. "
                f"Missing keys: {list(missing)[:10]}, unexpected keys: {list(unexpected)[:10]}"
            ) from registry_error
        return guide_model, f"guide model state_dict fallback ({registry_error})", "guide_state_dict"


def evaluate(config: BenchmarkConfig, logger: logging.Logger | None = None) -> None:
    run_start = time.perf_counter()
    startup_times = {}
    runtime_warning_prefix = "Runtime warning:"
    if config.device == "cpu":
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))))
        try:
            torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
        except RuntimeError:
            pass

    guide_inference, MIDOGEvaluation, ConfigCreator, ModelFactory = _load_guide_modules(
        config.guide_repo,
        config.profile_pipeline,
    )

    if not config.config_file.exists():
        raise FileNotFoundError(f"Could not find config_file: {config.config_file}")
    if not config.dataset.exists():
        raise FileNotFoundError(f"Could not find dataset: {config.dataset}")
    if not config.img_dir.is_dir():
        raise ValueError(f"This is not a directory: {config.img_dir}")

    stage_start = time.perf_counter()
    dataset = pd.read_csv(config.dataset)
    startup_times["dataset_loading"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if config.box_format == "cxcy":
        radius = 25
        dataset = dataset.assign(xmin=dataset["x"] - radius)
        dataset = dataset.assign(ymin=dataset["y"] - radius)
        dataset = dataset.assign(xmax=dataset["x"] + radius)
        dataset = dataset.assign(ymax=dataset["y"] + radius)
    startup_times["dataset_box_conversion"] = time.perf_counter() - stage_start

    test_dataset = dataset.query("split == @config.split")
    filenames = test_dataset.filename.unique()

    stage_start = time.perf_counter()
    model_config = ConfigCreator.load(str(config.config_file))
    startup_times["config_loading"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    model = ModelFactory.load(model_config, det_thresh=0.05)
    startup_times["model_loading"] = time.perf_counter() - stage_start

    pruned_model_path = Path(config.pruned_model_path) if config.pruned_model_path else None
    pruned_model_loaded = False
    if pruned_model_path:
        if not pruned_model_path.exists():
            raise FileNotFoundError(f"Could not find pruned model artifact: {pruned_model_path}")
        stage_start = time.perf_counter()
        model, model_artifact_source, model_artifact_load_method = _load_model_artifact(
            pruned_model_path, model
        )
        pruned_model_loaded = True
        startup_times["pruned_model_loading"] = time.perf_counter() - stage_start

    export_dir = config.export_dir or Path("experiments/eval_guide/exported_models")
    default_runtime_metadata = (
        config.metrics_output.with_name(f"{config.metrics_output.stem}_runtime.json")
        if config.metrics_output
        else config.config_file.with_name(f"{config.config_file.stem}_runtime.json")
    )
    runtime_metadata_output = config.runtime_metadata_output or default_runtime_metadata
    validation_suffix = {
        "onnxruntime_cpu": "onnx_validation",
        "onnxruntime_int8": "onnx_int8_validation",
        "openvino_cpu": "openvino_validation",
        "openvino_int8": "openvino_int8_validation",
        "tensorrt": "tensorrt_validation",
    }.get(config.runtime_backend)
    validation_output = (
        runtime_metadata_output.with_name(f"{runtime_metadata_output.stem}_{validation_suffix}.json")
        if validation_suffix
        else None
    )
    runtime_example_input = None
    if config.runtime_backend in {"onnxruntime_cpu", "onnxruntime_int8", "openvino_cpu", "openvino_int8", "tensorrt"} and len(filenames) > 0:
        stage_start = time.perf_counter()
        runtime_example_input = _build_runtime_example_input(
            guide_inference=guide_inference,
            image_path=config.img_dir / filenames[0],
            patch_size=model_config.patch_size,
            overlap=config.overlap,
            is_wsi=config.wsi,
        )
        startup_times["runtime_validation_input"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    try:
        runtime_result = configure_runtime_backend(
            model=model,
            runtime_backend=config.runtime_backend,
            compile_backend=config.compile_backend,
            compile_mode=config.compile_mode,
            onnx_opset=config.onnx_opset,
            openvino_device=config.openvino_device,
            openvino_compress_to_fp16=config.openvino_compress_to_fp16,
            export_dir=export_dir,
            model_name=model_config.model_name,
            patch_size=model_config.patch_size,
            config_file=config.config_file,
            validation_output=validation_output,
            example_input=runtime_example_input,
            int8_model_path=config.int8_model_path,
        )
    except RuntimeBackendUnavailable as exc:
        print(f"{runtime_warning_prefix} {exc}")
        raise SystemExit(2) from exc
    startup_times["runtime_backend_setup"] = time.perf_counter() - stage_start
    model = runtime_result.model

    stage_start = time.perf_counter()
    processor, patch_config = guide_inference.setup_inference(
        model=model,
        is_wsi=config.wsi,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        nms_thresh=config.nms_thresh,
        device=config.device,
        patch_size=model_config.patch_size,
        overlap=config.overlap,
        overwrite=config.overwrite,
        logger=logger,
    )
    startup_times["inference_setup"] = time.perf_counter() - stage_start

    print("Loaded model configurations:")
    pprint.pprint(model_config)
    print()
    print(f"Runtime backend: {runtime_result.runtime_backend}")
    print(f"Effective runtime backend: {runtime_result.effective_runtime_backend}")
    for warning in runtime_result.runtime_warnings:
        print(f"{runtime_warning_prefix} {warning}")
    print()

    preds = {}
    image_timings = []
    stage_totals = {}
    patches_per_image = []

    for filename in tqdm(filenames, desc="Running inference"):
        image_path = config.img_dir / filename

        image_start = time.perf_counter()
        process_kwargs = {"profile_pipeline": True} if config.profile_pipeline else {}
        results = processor.process_single(
            image_path,
            patch_config,
            **process_kwargs,
        )
        image_total = time.perf_counter() - image_start

        if config.profile_pipeline:
            timing = results.pop("_timing", {"stages": {}, "num_patches": 0})
            image_timings.append(image_total)
            patches_per_image.append(int(timing.get("num_patches", 0)))
            for stage, seconds in timing.get("stages", {}).items():
                stage_totals[stage] = stage_totals.get(stage, 0.0) + float(seconds)

        preds[filename] = results

    if config.predictions_output:
        config.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        config.predictions_output.write_text(json.dumps(_jsonable(preds)), encoding="utf-8")

    filtered_dataset = test_dataset.query("label == 1")
    eval_det_thresh = model_config.det_thresh if config.det_thresh is None else float(config.det_thresh)
    evaluation = MIDOGEvaluation(
        gt_file=filtered_dataset,
        preds=preds,
        output_file=config.metrics_output or config.config_file.with_suffix(".json"),
        det_thresh=eval_det_thresh,
        split=config.split,
    )

    metrics_start = time.perf_counter()
    evaluation.score()
    aggregate_metrics = evaluation._metrics["aggregates"]
    case_metrics = evaluation._metrics["case"]
    aggregate_metrics["true_positives"] = int(sum(case["tp"] for case in case_metrics.values()))
    aggregate_metrics["false_positives"] = int(sum(case["fp"] for case in case_metrics.values()))
    aggregate_metrics["false_negatives"] = int(sum(case["fn"] for case in case_metrics.values()))
    aggregate_metrics["total_detections"] = _prediction_count_at_threshold(preds, eval_det_thresh)
    metrics_time = time.perf_counter() - metrics_start

    serialization_start = time.perf_counter()
    evaluation.save()
    serialization_time = time.perf_counter() - serialization_start

    print("Evaluation done.")
    print(f"Evaluation results for {config.split} split")
    pprint.pprint(evaluation._metrics["aggregates"])

    total_end_to_end_s = time.perf_counter() - run_start
    runtime_metadata = {
        **runtime_result.as_dict(),
        "model_name": model_config.model_name,
        "pruned_model_path": str(pruned_model_path) if pruned_model_path else None,
        "pruned_model_loaded": pruned_model_loaded,
        "model_artifact_load_method": model_artifact_load_method if pruned_model_path else None,
        "model_artifact_source": model_artifact_source if pruned_model_path else None,
        "config_file": str(config.config_file),
        "dataset_csv": str(config.dataset),
        "device": config.device,
        "det_thresh": eval_det_thresh,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "overlap": config.overlap,
        "nms_thresh": config.nms_thresh,
        "num_images": int(len(filenames)),
        "total_runtime_s": total_end_to_end_s,
        "pipeline_timing_enabled": config.profile_pipeline,
    }
    write_runtime_metadata(runtime_metadata, runtime_metadata_output)

    if config.profile_pipeline:
        default_json = config.config_file.with_name(f"{config.config_file.stem}_timing.json")
        default_csv = config.config_file.with_name(f"{config.config_file.stem}_timing.csv")
        summary = build_timing_summary(
            run_start=run_start,
            startup_times=startup_times,
            stage_totals=stage_totals,
            image_timings=image_timings,
            patches_per_image=patches_per_image,
            config_file=config.config_file,
            dataset_csv=config.dataset,
            num_images=len(filenames),
            device=config.device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            overlap=config.overlap,
            nms_thresh=config.nms_thresh,
            metrics_time=metrics_time,
            output_serialization_time=serialization_time,
            runtime_metadata=runtime_metadata,
        )
        write_timing_outputs(
            summary,
            config.timing_output_json or default_json,
            config.timing_output_csv or default_csv,
        )


def main() -> None:
    args = get_args()
    evaluate(BenchmarkConfig(**vars(args)))
    print("End of script.")


if __name__ == "__main__":
    main()
