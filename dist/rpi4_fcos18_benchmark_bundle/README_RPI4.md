# Raspberry Pi 5 FCOS_18 Benchmark Bundle (legacy `rpi4_*` path)

> The physical target was a **Raspberry Pi 5 with four Cortex-A76 CPU cores**.
> `rpi4` and any “Raspberry Pi 4” text below are legacy package instructions,
> retained for provenance. This pre-test bundle is not a completed run.

This bundle runs the existing MIDOG FCOS benchmark adapter with two CPU backends:

1. PyTorch eager inference
2. ONNX Runtime with `CPUExecutionProvider`

Historical packaging text described Raspberry Pi 4 compatibility. The measured
project device was Raspberry Pi 5. Check the architecture first:

```bash
uname -m
```

It must print `aarch64`.

## Included Model And Data

- PyTorch checkpoint: `models/fcos18_checkpoint.ckpt`, copied from `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt`
- ONNX model: `models/fcos18_fp32.onnx`, copied from the validated export `experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx`
- Config: `configs/FCOS_18_eval.yaml`
- Test subset: `data/rpi4_test_subset.csv`
- Images: `413.tiff`, `366.tiff`, `487.tiff`

The bundled config uses the same FCOS_18 architecture, checkpoint, detection threshold `0.584`, patch size `1024`, overlap `0.3`, and patch-merge NMS threshold `0.3`. The checkpoint path is made relative to the bundle, and `weights` is set to `null` so torchvision does not try to download ImageNet weights before loading the checkpoint.

## Install

```bash
unzip rpi4_fcos18_benchmark_bundle.zip
cd rpi4_fcos18_benchmark_bundle
chmod +x install_rpi4.sh run_*.sh
./install_rpi4.sh
```

`install_rpi4.sh` installs Debian runtime libraries, creates `.venv`, installs CPU PyTorch/Torchvision, installs `requirements-rpi4.txt`, and runs:

```bash
python verify_bundle.py
```

## Run

```bash
./run_smoke_test.sh
./run_pytorch_benchmark.sh
./run_onnxruntime_benchmark.sh
```

To run everything:

```bash
./run_all_benchmarks.sh
```

Results are written to timestamped directories under:

```text
results/smoke/
results/pytorch_eager/
results/onnxruntime_cpu/
```

The launchers set:

```bash
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
TORCH_NUM_THREADS=4
TORCH_NUM_INTEROP_THREADS=1
```

They use `taskset -c 0-3` when available.

## Monitoring

Temperature and throttling:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
vcgencmd measure_clock arm
```

Peak RAM and runtime:

```bash
/usr/bin/time -v ./run_pytorch_benchmark.sh
/usr/bin/time -v ./run_onnxruntime_benchmark.sh
```

## Regenerate

From the original repository root:

```bash
scripts/create_rpi4_fcos18_bundle.sh
```

The script rebuilds `dist/rpi4_fcos18_benchmark_bundle/`, verifies it, writes `MANIFEST.txt`, `SHA256SUMS`, a size report, and creates `dist/rpi4_fcos18_benchmark_bundle.zip`.

## Known Limitations

- ARM64 installation is documented but cannot be fully validated on non-ARM development hardware.
- Runtime depends on ARM64 wheels being available for the installed Python version.
- OpenSlide and OpenCV are imported by the existing guide inference module even when using ROI TIFF images rather than WSI mode.
- The ONNX model is reused from the included validated FP32 export; the Raspberry Pi should not export ONNX during benchmarking.
