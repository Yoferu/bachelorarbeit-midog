# MIDOG Evaluation Artifacts

This repository contains the local scripts, configuration files, logs, and JSON/CSV evaluation outputs used for MIDOG-related FCOS experiments.

Large datasets and cloned upstream repositories are intentionally excluded from version control:

- `data/`
- `repos/`
- `.venv/`

## Contents

- `scripts/` - stable CLI entrypoints for preparing MIDOG++ subsets, benchmark runs, and result summaries.
- `src/benchmark/` - tracked benchmark adapter and timing helpers. This is the project-owned boundary to the gitignored MIDOG guide clone.
- `src/pruning/` - tracked pruning utilities for project-owned structured pruning experiments.
- `logs/` - local FCOS inference logs.
- `results/raw_predictions/` - JSON detection outputs for the evaluated FCOS variants.
- `experiments/eval_guide/` - evaluation configs, logs, and summarized metrics.

## Python Dependencies

The scripts use Python 3 and the packages listed in `requirements.txt`.
Optional deployment runtime dependencies are recorded separately:

```bash
python -m pip install -r requirements-runtime.txt
```

## Benchmark Levels

### 1. Smoke Test

The smoke test is a very small technical check that the MIDOG guide evaluation
command and model can run. It is not a performance or accuracy claim.

```bash
PROJECT="$HOME/bachelorarbeit-midog"
"$PROJECT/scripts/run_smoke_test.sh"
```

### 2. Quick Benchmark

The quick benchmark is a fixed 32-image subset of the official MIDOG++
xvalidation test split. It is intended for fast optimization iteration and
regression detection, not for final accuracy claims. The subset is generated
once and reused; benchmark runs do not resample it.

Create or regenerate the fixed quick subset:

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

Run the quick benchmark with one warmup run and three measured repeats:

```bash
PROJECT="$HOME/bachelorarbeit-midog"
"$PROJECT/scripts/run_quick_benchmark.sh" FCOS_18
```

Run the quick benchmark with optional pipeline-stage timing:

```bash
PROJECT="$HOME/bachelorarbeit-midog"
PROFILE_PIPELINE=1 "$PROJECT/scripts/run_quick_benchmark.sh" FCOS_18
```

For all three FCOS variants, omit the model argument:

```bash
PROJECT="$HOME/bachelorarbeit-midog"
"$PROJECT/scripts/run_quick_benchmark.sh"
```

Quick benchmark logs are written to `experiments/eval_guide/logs/` with
`quick` in the filename. Timing summaries, when enabled, are written as JSON and
CSV files under `experiments/eval_guide/results/`. The timed quick benchmark
also writes `*_timing_median.csv`; use the median measured repeat as the primary
runtime comparison value.

### 3. Full Benchmark

The full benchmark remains the complete official MIDOG++ xvalidation test split
evaluation and is the source for final runtime and accuracy reporting.

```bash
PROJECT="$HOME/bachelorarbeit-midog"
"$PROJECT/scripts/run_full_benchmark.sh"
```

### Optional Timing Fields

Pipeline timing is disabled by default. Passing `PROFILE_PIPELINE=1` to the
benchmark scripts enables coarse `time.perf_counter()` timing through the
tracked adapter in `src/benchmark/midog_guide_adapter.py`. The guide clone under
`repos/` is treated as a read-only dependency; benchmark functionality does not
depend on uncommitted changes inside that clone.

## Runtime and Deployment Optimization

`pytorch_eager` is the default baseline. `pytorch_compile` is the first low-risk
optimization path and can be enabled without changing benchmark commands:

```bash
RUNTIME_BACKEND=pytorch_compile PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

`onnxruntime_cpu` exports the torchvision FCOS inference graph to ONNX and runs
patch inference with ONNX Runtime `CPUExecutionProvider`. Generated `.onnx`
files are written to `experiments/eval_guide/exported_models/`, which is ignored
by git. The exporter first tries the modern dynamo path and falls back to the
legacy ONNX exporter when torchvision FCOS postprocessing/NMS blocks dynamo.
Patch extraction, Python postprocessing/merging around patches, and metrics stay
in the tracked adapter/evaluator flow.

`openvino_cpu` uses the validated ONNX export as an intermediate format, converts
it to OpenVINO IR, and runs patch inference on the OpenVINO CPU plugin. Generated
`.xml` and `.bin` files are written next to the ONNX files under
`experiments/eval_guide/exported_models/` and are not committed.
The default OpenVINO IR keeps `openvino.save_model` defaults, including FP16
weight compression. For a controlled FP32 comparison, run with
`OPENVINO_COMPRESS_TO_FP16=0`; this calls
`openvino.save_model(..., compress_to_fp16=False)`, writes a distinct `_fp32`
IR, and records `openvino_compress_to_fp16` in runtime, validation, and timing
metadata.

OpenVINO can produce slightly different FCOS postprocessing/NMS results than
PyTorch/ONNX Runtime because exported detection filtering is part of the model
graph. The adapter records this as validation metadata and warnings; final
runtime and accuracy comparisons should include all four backends on the same
Full Benchmark before making claims.

TorchScript is not prioritized for new deployment work because recent PyTorch
versions favor `torch.export`/`torch.compile` flows.

Example deployment-target commands:

```bash
RUNTIME_BACKEND=onnxruntime_cpu PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18

RUNTIME_BACKEND=openvino_cpu PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18

OPENVINO_COMPRESS_TO_FP16=0 RUNTIME_BACKEND=openvino_cpu PROFILE_PIPELINE=1 \
  "$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

## Structured Pruning

The first pruning milestone is a conservative `FCOS_18` 10% masked structured
channel-L2 baseline on internal detection-head convolutions. It produces a
loadable PyTorch artifact and metadata under `experiments/pruning/models/`.
Because this baseline zeros channels without physically removing them, it is
mainly for feasibility and accuracy validation; large CPU speedups require a
future dependency-aware physical pruning pass.

See `docs/pruning.md` for inspection, artifact creation, smoke test, and quick
benchmark commands. Use `PRUNED_MODEL_PATH=...` to evaluate the pruned artifact
without changing baseline configs or weights.
