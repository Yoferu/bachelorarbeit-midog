import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def percent(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (value / total) * 100.0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def build_timing_summary(
    *,
    run_start: float,
    startup_times: dict[str, float],
    stage_totals: dict[str, float],
    image_timings: list[float],
    patches_per_image: list[int],
    config_file: Path,
    dataset_csv: Path,
    num_images: int,
    device: str,
    batch_size: int,
    num_workers: int,
    overlap: float,
    nms_thresh: float,
    metrics_time: float,
    output_serialization_time: float,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measured_inference_time = sum(image_timings)
    num_patches = sum(patches_per_image)

    summary = {
        "run_name": config_file.stem,
        "config_file": str(config_file),
        "dataset_csv": str(dataset_csv),
        "num_images": int(num_images),
        "num_patches": int(num_patches),
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "overlap": overlap,
        "nms_thresh": nms_thresh,
        "total_end_to_end_s": time.perf_counter() - run_start,
        "startup_time_s": sum(startup_times.values()),
        "startup_stages_s": startup_times,
        "measured_inference_time_s": measured_inference_time,
        "evaluation_metrics_s": metrics_time,
        "output_serialization_s": output_serialization_time,
        "stages": {
            stage: {
                "total_s": seconds,
                "percent_of_measured_inference": percent(seconds, measured_inference_time),
            }
            for stage, seconds in sorted(stage_totals.items())
        },
        "per_image_s": image_timings,
        "patches_per_image": patches_per_image,
        "mean_per_image_s": float(np.mean(image_timings)) if image_timings else None,
        "median_per_image_s": median(image_timings),
        "mean_per_patch_s": (measured_inference_time / num_patches) if num_patches else None,
        "median_per_patch_s": median([
            image_time / patch_count
            for image_time, patch_count in zip(image_timings, patches_per_image)
            if patch_count
        ]),
    }
    if runtime_metadata:
        summary.update(runtime_metadata)
    return summary


def instrument_guide_inference(guide_inference: Any) -> None:
    def process_image(self, image_path, patch_config=None, **kwargs):
        profile_pipeline = bool(kwargs.pop("profile_pipeline", False))
        timing = {"stages": {}, "num_patches": 0} if profile_pipeline else None

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        patch_config = patch_config or guide_inference.PatchConfig()

        stage_start = time.perf_counter()
        self.model.eval()
        self.model.to(self.device)
        if profile_pipeline:
            timing["stages"]["model_device_setup"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        dataloader = self._create_dataloader(image_path, patch_config)
        if profile_pipeline:
            timing["stages"]["image_io_patch_extraction_setup"] = time.perf_counter() - stage_start
            timing["num_patches"] = len(dataloader.dataset)

        all_predictions = []
        all_coords = []

        with guide_inference.tqdm(dataloader, desc="Processing batches") as pbar:
            batch_iter = iter(pbar)
            while True:
                if profile_pipeline:
                    stage_start = time.perf_counter()
                try:
                    batch_images, batch_x, batch_y = next(batch_iter)
                except StopIteration:
                    break
                if profile_pipeline:
                    timing["stages"]["dataloader_wait_preprocessing"] = (
                        timing["stages"].get("dataloader_wait_preprocessing", 0.0)
                        + time.perf_counter() - stage_start
                    )

                if profile_pipeline:
                    stage_start = time.perf_counter()
                predictions = self._process_batch(batch_images)
                if profile_pipeline:
                    timing["stages"]["forward_pass"] = (
                        timing["stages"].get("forward_pass", 0.0)
                        + time.perf_counter() - stage_start
                    )

                all_predictions.extend(predictions)
                all_coords.extend(zip(batch_x, batch_y))

        stage_start = time.perf_counter()
        results = self._post_process_predictions(all_predictions, all_coords)
        if profile_pipeline:
            timing["stages"]["patch_merging_postprocessing_nms"] = time.perf_counter() - stage_start

        output = {
            "boxes": results["boxes"].numpy(),
            "scores": results["scores"].numpy(),
            "labels": results["labels"].numpy(),
        }
        if profile_pipeline:
            output["_timing"] = timing
        return output

    def process_single(self, image_path, patch_config=None, output_dir=None, **kwargs):
        results = self.strategy.process_image(
            image_path,
            patch_config=patch_config,
            **kwargs,
        )
        if output_dir or self.config.save_dir:
            stage_start = time.perf_counter()
            self._save_results(results, image_path, output_dir)
            if "_timing" in results:
                results["_timing"]["stages"]["output_serialization"] = (
                    results["_timing"]["stages"].get("output_serialization", 0.0)
                    + time.perf_counter() - stage_start
                )
        return results

    guide_inference.Torchvision_Inference.process_image = process_image
    guide_inference.ImageProcessor.process_single = process_single
