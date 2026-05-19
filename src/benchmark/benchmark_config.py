from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    config_file: Path
    dataset: Path
    img_dir: Path
    guide_repo: Path
    batch_size: int = 8
    box_format: str = "cxcy"
    device: str = "cuda"
    metrics_output: Path | None = None
    nms_thresh: float = 0.3
    num_workers: int = 8
    overlap: float = 0.3
    overwrite: bool = False
    profile_pipeline: bool = False
    split: str = "test"
    timing_output_csv: Path | None = None
    timing_output_json: Path | None = None
    wsi: bool = False
