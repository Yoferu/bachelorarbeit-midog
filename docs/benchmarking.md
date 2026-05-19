# Benchmarking

## Smoke Test

The smoke test checks that one model and the evaluation path run on a tiny fixed
CSV. It is not a runtime or accuracy claim.

```bash
PROJECT="$HOME/bachelorarbeit-midog"
"$PROJECT/scripts/run_smoke_test.sh"
```

In restricted sandboxes where multiprocessing sockets are blocked, use:

```bash
NUM_WORKERS=0 "$HOME/bachelorarbeit-midog/scripts/run_smoke_test.sh"
```

## Quick Benchmark

The quick benchmark uses a fixed subset generated from the official MIDOG++
xvalidation test split. It is for optimization iteration and regression checks,
not final reporting.

Generate or regenerate the subset:

```bash
PROJECT="$HOME/bachelorarbeit-midog"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/create_quick_benchmark_subset.py" \
  --input_csv "$PROJECT/data/midogpp_guide_eval_xvalidation.csv" \
  --output_csv "$PROJECT/data/midogpp_guide_eval_xvalidation_quick.csv" \
  --n 32 \
  --seed 42 \
  --split test \
  --stratify_auto
```

Run one model with one warmup run and three measured repeats:

```bash
"$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

Run with optional pipeline timing:

```bash
PROFILE_PIPELINE=1 "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

Timed quick runs write per-run JSON/CSV summaries and a
`*_timing_median.csv` summary under `experiments/eval_guide/results/`.

## Runtime and Deployment Optimization

`pytorch_eager` is the baseline runtime and remains the default for all benchmark
scripts.

`pytorch_compile` is the first low-risk runtime optimization. It wraps the loaded
model with `torch.compile` when available. Configure it with `COMPILE_BACKEND`
and `COMPILE_MODE`; the default backend is `inductor`.

`onnxruntime_cpu` is the first real deployment runtime target. In the current
repository state it is a documented placeholder: `onnx` and `onnxruntime` are
not installed in the local environment, and the FCOS detection model needs a
validated forward-pass export before ONNX Runtime results can be reported.

`openvino_cpu` is the CPU-focused deployment target. It is also a documented
placeholder until a validated ONNX export exists and OpenVINO is installed.

TorchScript is not prioritized because recent PyTorch releases direct users
toward `torch.export`/`torch.compile` flows rather than TorchScript for new
deployment work.

Final runtime and accuracy claims must use the Full Benchmark. Use the Quick
Benchmark for iteration and regression detection.

Default smoke:

```bash
"$HOME/bachelorarbeit-midog/scripts/run_smoke_test.sh"
```

Compiled quick benchmark:

```bash
RUNTIME_BACKEND=pytorch_compile PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

ONNX Runtime quick benchmark:

```bash
RUNTIME_BACKEND=onnxruntime_cpu PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

OpenVINO quick benchmark:

```bash
RUNTIME_BACKEND=openvino_cpu PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

Exported runtime artifacts belong under:

```text
experiments/eval_guide/exported_models/
```

The directory is ignored except for its `.gitignore`, so generated model files
are not committed.

## Full Benchmark

The full benchmark runs the complete official MIDOG++ xvalidation test split.
Use this for final runtime and accuracy reporting.

```bash
"$HOME/bachelorarbeit-midog/scripts/run_full_benchmark.sh"
```

For a single model:

```bash
"$HOME/bachelorarbeit-midog/scripts/run_full_benchmark.sh" FCOS_18
```

## Adapter Boundary

All benchmark scripts call the tracked adapter:

```text
src/benchmark/midog_guide_adapter.py
```

The adapter uses `repos/MIDOG_2025_Guide` as a read-only dependency and applies
optional pipeline-stage timing in memory. The gitignored guide clone does not
need local modifications for smoke, quick, timed quick, or full benchmarks.
