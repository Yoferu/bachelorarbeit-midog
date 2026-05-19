# MIDOG Evaluation Artifacts

This repository contains the local scripts, configuration files, logs, and JSON/CSV evaluation outputs used for MIDOG-related FCOS experiments.

Large datasets and cloned upstream repositories are intentionally excluded from version control:

- `data/`
- `repos/`
- `.venv/`

## Contents

- `scripts/` - helper scripts for preparing MIDOG++ subsets, converting annotations, preparing FCOS configs, and collecting evaluation metrics.
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
EXP="$PROJECT/experiments/eval_guide"
cd "$PROJECT/repos/MIDOG_2025_Guide"

"$PROJECT/.venv/bin/python" evaluate.py \
  --config_file "$EXP/configs/FCOS_18_eval.yaml" \
  --dataset "$PROJECT/data/midogpp_guide_eval_xvalidation_smoke.csv" \
  --img_dir "$PROJECT/data/midogpp" \
  --split test \
  --device cpu \
  --batch_size 1 \
  --num_workers 4 \
  --overlap 0.3 \
  --nms_thresh 0.3 \
  --overwrite
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
evaluation and is the source for final runtime and accuracy reporting. The
existing command style is unchanged:

```bash
PROJECT="$HOME/bachelorarbeit-midog"
EXP="$PROJECT/experiments/eval_guide"
cd "$PROJECT/repos/MIDOG_2025_Guide"

for MODEL in FCOS_18 FCOS_x50 FCOS_x101; do
  /usr/bin/time -v "$PROJECT/.venv/bin/python" evaluate.py \
    --config_file "$EXP/configs/${MODEL}_eval.yaml" \
    --dataset "$PROJECT/data/midogpp_guide_eval_xvalidation.csv" \
    --img_dir "$PROJECT/data/midogpp" \
    --split test \
    --device cpu \
    --batch_size 1 \
    --num_workers 4 \
    --overlap 0.3 \
    --nms_thresh 0.3 \
    --overwrite \
    2>&1 | tee "$EXP/logs/${MODEL}_midogpp_xvalidation_test_full_cpu.log"
done
```

### Optional Timing Fields

Pipeline timing is disabled by default. Passing `--profile_pipeline` to
`repos/MIDOG_2025_Guide/evaluate.py` adds coarse `time.perf_counter()` timing
for startup, image loading and patch setup, dataloader/preprocessing wait,
forward pass, patch merging/postprocessing/NMS, metric calculation, and output
serialization. JSON and CSV output paths can be controlled with
`--timing_output_json` and `--timing_output_csv`.
