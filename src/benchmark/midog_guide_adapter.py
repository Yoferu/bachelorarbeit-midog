import argparse
import logging
import pprint
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm.autonotebook import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.benchmark_config import BenchmarkConfig
from src.benchmark.pipeline_timing import build_timing_summary, instrument_guide_inference
from src.benchmark.result_writer import write_timing_outputs


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracked adapter for MIDOG_2025_Guide evaluation.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--box_format", type=str, default="cxcy")
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guide_repo", type=Path, default=Path("repos/MIDOG_2025_Guide"))
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--metrics_output", type=Path, default=None)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile_pipeline", action="store_true")
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


def evaluate(config: BenchmarkConfig, logger: logging.Logger | None = None) -> None:
    run_start = time.perf_counter()
    startup_times = {}

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

    stage_start = time.perf_counter()
    model_config = ConfigCreator.load(str(config.config_file))
    startup_times["config_loading"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    model = ModelFactory.load(model_config, det_thresh=0.05)
    startup_times["model_loading"] = time.perf_counter() - stage_start

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

    test_dataset = dataset.query("split == @config.split")
    filenames = test_dataset.filename.unique()

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

    filtered_dataset = test_dataset.query("label == 1")
    evaluation = MIDOGEvaluation(
        gt_file=filtered_dataset,
        preds=preds,
        output_file=config.metrics_output or config.config_file.with_suffix(".json"),
        det_thresh=model_config.det_thresh,
        split=config.split,
    )

    metrics_start = time.perf_counter()
    evaluation.score()
    metrics_time = time.perf_counter() - metrics_start

    serialization_start = time.perf_counter()
    evaluation.save()
    serialization_time = time.perf_counter() - serialization_start

    print("Evaluation done.")
    print(f"Evaluation results for {config.split} split")
    pprint.pprint(evaluation._metrics["aggregates"])

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
