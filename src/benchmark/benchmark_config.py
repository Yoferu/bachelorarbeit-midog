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
    compile_backend: str = "inductor"
    compile_mode: str | None = None
    device: str = "cuda"
    det_thresh: float | None = None
    export_dir: Path | None = None
    metrics_output: Path | None = None
    nms_thresh: float = 0.3
    num_workers: int = 8
    onnx_opset: int = 18
    int8_model_path: Path | None = None
    openvino_device: str = "CPU"
    openvino_compress_to_fp16: bool = True
    openvino_performance_hint: str = "LATENCY"
    openvino_num_streams: int = 1
    runtime_intra_op_threads: int = 0
    runtime_inter_op_threads: int = 1
    openvino_inference_num_threads: int = 4
    openvino_inference_precision: str = "f32"
    warmup_iterations: int = 0
    overlap: float = 0.3
    overwrite: bool = False
    profile_pipeline: bool = False
    predictions_output: Path | None = None
    pruned_model_path: Path | None = None
    runtime_backend: str = "pytorch_eager"
    runtime_metadata_output: Path | None = None
    split: str = "test"
    timing_output_csv: Path | None = None
    timing_output_json: Path | None = None
    wsi: bool = False
