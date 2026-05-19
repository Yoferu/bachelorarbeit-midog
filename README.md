# MIDOG Evaluation Artifacts

This repository contains the local scripts, configuration files, logs, and JSON/CSV evaluation outputs used for MIDOG-related FCOS experiments.

Large datasets and cloned upstream repositories are intentionally excluded from version control:

- `data/`
- `repos/`
- `.venv/`

## Contents

- `scripts/` - stable CLI entrypoints for preparing MIDOG++ subsets, benchmark runs, and result summaries.
- `src/benchmark/` - tracked benchmark adapter and timing helpers. This is the project-owned boundary to the gitignored MIDOG guide clone.
- `logs/` - local FCOS inference logs.
- `results/raw_predictions/` - JSON detection outputs for the evaluated FCOS variants.
- `experiments/eval_guide/` - evaluation configs, logs, and summarized metrics.

## Python Dependencies

The scripts use Python 3 and the packages listed in `requirements.txt`.

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
