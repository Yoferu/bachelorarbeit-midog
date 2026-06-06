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

Install optional deployment runtime dependencies with:

```bash
python -m pip install -r requirements-runtime.txt
```

`pytorch_eager` is the baseline runtime and remains the default for all benchmark
scripts.

`pytorch_compile` is the first low-risk runtime optimization. It wraps the loaded
model with `torch.compile` when available. Configure it with `COMPILE_BACKEND`
and `COMPILE_MODE`; the default backend is `inductor`.

`onnxruntime_cpu` is the first real deployment runtime target. It exports the
torchvision FCOS inference graph to ONNX under
`experiments/eval_guide/exported_models/` and runs patch inference with ONNX
Runtime `CPUExecutionProvider`. Existing matching `.onnx` files are reused.
Validation metadata is written next to the benchmark runtime JSON under
`experiments/eval_guide/results/`.

The ONNX exporter first tries `torch.onnx.export(..., dynamo=True)`. With the
current PyTorch/torchvision FCOS stack, dynamo export can fail on
postprocessing/NMS data-dependent shapes, so the backend falls back to the
legacy exporter and records that warning in the result metadata. Patch
extraction, surrounding Python postprocessing/merging, output serialization, and
metrics remain in Python.

`openvino_cpu` is the CPU-focused deployment target. It uses the existing ONNX
export as the intermediate representation, converts that model with
`openvino.convert_model`, saves OpenVINO IR with `openvino.save_model`, and runs
patch inference through `openvino.Core().compile_model(..., "CPU")`.
By default the adapter keeps OpenVINO's default IR save behavior, including FP16
weight compression. Set `OPENVINO_COMPRESS_TO_FP16=0` or pass
`--openvino_compress_to_fp16 0` to save a separate FP32 IR with
`openvino.save_model(..., compress_to_fp16=False)` for accuracy validation.
The FP32 IR filename includes `_fp32`, and runtime, validation, and timing
metadata record `openvino_compress_to_fp16`.

OpenVINO generated artifacts are stored under:

```text
experiments/eval_guide/exported_models/
```

The generated `.onnx`, `.xml`, and `.bin` files are ignored by git and should
not be committed.

OpenVINO validation compares PyTorch eager, ONNX Runtime CPU, and OpenVINO CPU
on a real smoke patch. With the current exported FCOS graph, OpenVINO can differ
slightly after exported filtering/NMS, so the adapter records
`passed_with_warnings` when the output structure is valid but detection counts or
numeric values differ. Treat OpenVINO as a real deployment backend, but use the
Full Benchmark for final runtime and accuracy claims.

TorchScript is not prioritized because recent PyTorch releases direct users
toward `torch.export`/`torch.compile` flows rather than TorchScript for new
deployment work.

Final runtime and accuracy claims must use the Full Benchmark. Use the Quick
Benchmark for iteration and regression detection.
Existing quick-run results for `pytorch_eager`, `pytorch_compile`, and
`onnxruntime_cpu` remain valid as long as their benchmark code paths and inputs
are not changed.

## Pruned Model Evaluation

Pruning is evaluated separately from runtime/deployment optimization at first.
The current `FCOS_18` pruning artifact is a masked structured pruning baseline:
it zeros full detection-head output-channel filters but does not physically
remove channels from the graph. Treat it as a feasibility and accuracy check,
not as evidence of real compute reduction.

Load a pruned artifact with `PRUNED_MODEL_PATH`:

```bash
NUM_WORKERS=0 \
PRUNED_MODEL_PATH="$HOME/bachelorarbeit-midog/experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt" \
RUNTIME_BACKEND=pytorch_eager \
"$HOME/bachelorarbeit-midog/scripts/run_smoke_test.sh" FCOS_18
```

For quick timing:

```bash
NUM_WORKERS=0 \
PRUNED_MODEL_PATH="$HOME/bachelorarbeit-midog/experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt" \
RUNTIME_BACKEND=pytorch_eager \
PROFILE_PIPELINE=1 \
"$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

The benchmark scripts append the pruned artifact stem to run names, so baseline
runtime benchmark outputs are not overwritten. See `docs/pruning.md` for the
structured pruning workflow and limitations.

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

Use `ONNX_OPSET` to override the default opset, currently 18:

```bash
ONNX_OPSET=18 RUNTIME_BACKEND=onnxruntime_cpu \
  "$HOME/bachelorarbeit-midog/scripts/run_smoke_test.sh"
```

OpenVINO quick benchmark:

```bash
RUNTIME_BACKEND=openvino_cpu PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

Final runtime comparisons should include:

```text
pytorch_eager
pytorch_compile
onnxruntime_cpu
openvino_cpu
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
