# Raspberry Pi 5 FCOS Benchmark Bundle (legacy `rpi4_*` path)

> The physical target was a **Raspberry Pi 5 with four Cortex-A76 CPU cores**.
> `rpi4` is legacy package naming retained for provenance. This pre-test bundle
> is preparation material, not evidence of a completed run.

This bundle benchmarks six CPU deployment variants on the unchanged three-image diagnostic set: FCOS_18 and FPN+Head 60% Pruned60, each as PyTorch FP32, ONNX Runtime FP32, and ONNX Runtime INT8.

Install on a 64-bit Raspberry Pi OS (`uname -m` must be `aarch64`):

```bash
unzip rpi4_fcos_benchmark_bundle.zip
cd rpi4_fcos_benchmark_bundle
chmod +x install_rpi4.sh run_*.sh
./install_rpi4.sh
```

Run all smoke tests and benchmarks:

```bash
./run_all_benchmarks.sh
```

Run one variant:

```bash
./run_benchmark.sh --model fcos18 --runtime pytorch --precision fp32
./run_benchmark.sh --model fcos18 --runtime onnx --precision fp32
./run_benchmark.sh --model fcos18 --runtime onnx --precision int8
./run_benchmark.sh --model pruned60 --runtime pytorch --precision fp32
./run_benchmark.sh --model pruned60 --runtime onnx --precision fp32
./run_benchmark.sh --model pruned60 --runtime onnx --precision int8
```

Results are timestamped under `results/<model>/<runtime>_<precision>/`. Generate the comparison table with:

```bash
python scripts/summarize_rpi4_benchmarks.py
```

The launchers use batch size 1, patch size 1024, overlap 0.3, NMS 0.3, FCOS detection threshold 0.584, four CPU threads, and one inter-op thread. PyTorch INT8 is intentionally unsupported.

Monitor Raspberry Pi thermals and throttling:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
vcgencmd measure_clock arm
/usr/bin/time -v ./run_benchmark.sh --model pruned60 --runtime onnx --precision int8
```

Model provenance and quantization metadata are in `models/deployment_artifacts_report.json`. Quantization used ONNX Runtime static QDQ INT8 with deterministic calibration patches from `experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv`; calibration images are not packaged.
