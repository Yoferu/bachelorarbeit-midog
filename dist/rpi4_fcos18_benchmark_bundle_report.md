# Raspberry Pi 5 FCOS_18 Benchmark Bundle Report (legacy `rpi4_*` path)

The physical target was a **Raspberry Pi 5 with four Cortex-A76 CPU cores**.
Occurrences of `rpi4` or Raspberry Pi 4 below are retained historical package
identifiers/instructions, not the identity of the measured hardware. This is a
pre-test preparation report, not proof of a completed run.

## Included Files

- `MANIFEST.txt` (4421 bytes)
- `README_RPI4.md` (2971 bytes)
- `SHA256SUMS` (4571 bytes)
- `configs/FCOS_18_eval.yaml` (236 bytes)
- `data/annotations/midogpp_guide_eval_xvalidation_3img.csv` (4122 bytes)
- `data/images/366.tiff` (156190526 bytes)
- `data/images/413.tiff` (129777570 bytes)
- `data/images/487.tiff` (129777570 bytes)
- `data/rpi4_smoke_subset.csv` (2177 bytes)
- `data/rpi4_test_subset.csv` (4122 bytes)
- `install_rpi4.sh` (634 bytes)
- `models/fcos18_checkpoint.ckpt` (222496424 bytes)
- `models/fcos18_fp32.onnx` (74444978 bytes)
- `repos/MIDOG_2025_Guide/README.md` (3596 bytes)
- `repos/MIDOG_2025_Guide/requirements.txt` (203 bytes)
- `repos/MIDOG_2025_Guide/utils/eval_utils.py` (10914 bytes)
- `repos/MIDOG_2025_Guide/utils/factory.py` (8834 bytes)
- `repos/MIDOG_2025_Guide/utils/inference.py` (22838 bytes)
- `repos/MIDOG_2025_Guide/utils/litmodel.py` (6625 bytes)
- `repos/MIDOG_2025_Guide/utils/model.py` (15304 bytes)
- `requirements-rpi4.txt` (139 bytes)
- `run_all_benchmarks.sh` (137 bytes)
- `run_onnxruntime_benchmark.sh` (2635 bytes)
- `run_pytorch_benchmark.sh` (2685 bytes)
- `run_smoke_test.sh` (2589 bytes)
- `scripts/run_full_benchmark.sh` (2155 bytes)
- `scripts/run_quick_benchmark.sh` (3884 bytes)
- `scripts/summarize_quick_benchmark_timing.py` (6374 bytes)
- `src/benchmark/__init__.py` (67 bytes)
- `src/benchmark/benchmark_config.py` (998 bytes)
- `src/benchmark/distilled_model_loader.py` (3725 bytes)
- `src/benchmark/midog_guide_adapter.py` (17072 bytes)
- `src/benchmark/pipeline_timing.py` (6082 bytes)
- `src/benchmark/result_writer.py` (4283 bytes)
- `src/benchmark/runtime_backends.py` (64313 bytes)
- `src/distillation/__init__.py` (61 bytes)
- `src/distillation/config.py` (5442 bytes)
- `src/distillation/early_stopping.py` (10132 bytes)
- `src/distillation/logging_utils.py` (3078 bytes)
- `src/distillation/losses.py` (4222 bytes)
- `src/distillation/model_registry.py` (8223 bytes)
- `src/distillation/output_matching.py` (3086 bytes)
- `src/distillation/stable_selection.py` (9123 bytes)
- `src/pruning/__init__.py` (64 bytes)
- `src/pruning/depgraph_structural_pruning.py` (10546 bytes)
- `src/pruning/fine_tune_config.py` (2023 bytes)
- `src/pruning/inspect_model.py` (4071 bytes)
- `src/pruning/pruning_config.py` (1336 bytes)
- `src/pruning/save_pruned_model.py` (3706 bytes)
- `src/pruning/structured_pruning.py` (3522 bytes)
- `verify_bundle.py` (4869 bytes)

## Large Files Over 100 MB

- `data/images/366.tiff`: 148.95 MiB
- `data/images/413.tiff`: 123.77 MiB
- `data/images/487.tiff`: 123.77 MiB
- `models/fcos18_checkpoint.ckpt`: 212.19 MiB

## Excluded Categories

- `.git/`, `.github/`, virtual environments, caches, notebook checkpoints, build outputs
- complete MIDOG++ dataset except `413.tiff`, `366.tiff`, `487.tiff`
- FCOS_x50 and FCOS_x101 checkpoints
- pruning, distillation, fine-tuning, optimizer, and training-state artifacts
- old benchmark logs, results, profiling traces, TensorBoard and W&B outputs
- OpenVINO IR files, TensorRT exports and engines, duplicate ONNX exports

## Size

- Total uncompressed size: 679.94 MiB
- ZIP size: 552.92 MiB

## Exact Model Artifacts

- PyTorch checkpoint source: `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt`
- Bundle checkpoint: `models/fcos18_checkpoint.ckpt`
- ONNX source: `experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx`
- Bundle ONNX: `models/fcos18_fp32.onnx`

## Exact Test Images

- `data/images/413.tiff`
- `data/images/366.tiff`
- `data/images/487.tiff`

## Path Modifications

- `configs/FCOS_18_eval.yaml` uses `checkpoint: models/fcos18_checkpoint.ckpt`.
- `configs/FCOS_18_eval.yaml` uses `weights: null` to avoid an ImageNet weight download before checkpoint loading.
- The copied ONNX Runtime backend honors `PACKAGED_ONNX_MODEL`, set by the launchers to `models/fcos18_fp32.onnx`, so the Raspberry Pi reuses the validated export.

## Portability Problems

- Full ARM64 wheel compatibility and thermal behavior must be validated on Raspberry Pi 4 hardware.
- The existing guide inference module imports OpenSlide and OpenCV even for ROI TIFF input.

## Build And Verification Commands

- `scripts/create_rpi4_fcos18_bundle.sh`
- `python verify_bundle.py`
- ZIP creation: Python `zipfile` with deflate compression level 6
