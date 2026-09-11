# Benchmark controls

See [experiment commands](experiments.md) for the execution sequence. All paths
below are relative to the repository root.

The smoke/quick/full wrappers support `--help`. They default to `.venv/bin/python`,
CPU, batch size 1, 4 data-loader workers, overlap 0.3, NMS 0.3, and PyTorch eager.
The quick wrapper uses one warm-up run and three measured repetitions.
These defaults alone do **not** constrain execution to four physical cores.

| Environment variable | Purpose |
| --- | --- |
| `PROJECT`, `PYTHON`, `GUIDE_REPO`, `IMG_DIR` | Checkout, interpreter, guide clone, TIFF directory |
| `EXP`, `ADAPTER` | Evaluation directory (including configs), adapter path |
| `SMOKE_DATASET`, `QUICK_DATASET`, `FULL_DATASET` | Input CSV for the corresponding wrapper |
| `DEVICE`, `BATCH_SIZE`, `NUM_WORKERS` | Device, batch size, data-loader workers |
| `OVERLAP`, `NMS_THRESH` | Patch overlap and NMS threshold |
| `PROFILE_PIPELINE=1` | Write pipeline-stage timing JSON/CSV |
| `WARMUP_RUNS`, `REPEATS` | Quick benchmark repetitions |
| `PRUNED_MODEL_PATH` | Serialized pruning/recovery artifact |
| `RUNTIME_BACKEND` | `pytorch_eager`, `pytorch_compile`, `onnxruntime_cpu`, `openvino_cpu` |
| `COMPILE_BACKEND`, `COMPILE_MODE` | torch.compile options |
| `EXPORT_DIR`, `ONNX_OPSET` | Export cache and ONNX opset (default 18) |
| `OPENVINO_DEVICE`, `OPENVINO_COMPRESS_TO_FP16` | OpenVINO device and IR weight compression |

For INT8 models and explicit runtime thread/precision controls, use
`src/benchmark/midog_guide_adapter.py --help` and the corrected CPU launchers.
Set ORT intra-op threads explicitly: automatic sizing can create a host-wide
thread pool and override inherited affinity. The September launcher uses intra-op
4, inter-op 1, OpenVINO threads 4, one stream, workers 0, and five warm-up patches.
Choose four distinct physical cores using `lscpu -e=CPU,CORE,SOCKET`; CPU IDs are
host-specific. Run benchmarks sequentially on an otherwise idle machine.

`OPENVINO_COMPRESS_TO_FP16=0` preserves FP32 weights but does not enforce FP32
execution. Strict FP32 additionally needs adapter option
`--openvino_inference_precision f32`. Read runtime metadata, not filenames, when
classifying precision. The original FP32-storage OpenVINO runs may execute BF16.
No completed FP16 benchmark is established by the final provenance index.

Export validation may report postprocessing/NMS differences. Keep validation
metadata with comparisons. Cached-prediction rescoring changes threshold metrics
without producing new inference timings. Historical 105-image, quick 32-image,
three-image device, and final 111-image workloads must stay separately labelled.
