#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/bachelorarbeit-midog}"
EXP="${EXP:-$PROJECT/experiments/eval_guide}"
GUIDE_REPO="${GUIDE_REPO:-$PROJECT/repos/MIDOG_2025_Guide}"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
ADAPTER="${ADAPTER:-$PROJECT/src/benchmark/midog_guide_adapter.py}"
SMOKE_DATASET="${SMOKE_DATASET:-$PROJECT/data/midogpp_guide_eval_xvalidation_smoke.csv}"
IMG_DIR="${IMG_DIR:-$PROJECT/data/midogpp}"
MODEL="${MODEL:-FCOS_18}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OVERLAP="${OVERLAP:-0.3}"
NMS_THRESH="${NMS_THRESH:-0.3}"
PROFILE_PIPELINE="${PROFILE_PIPELINE:-0}"

mkdir -p "$EXP/logs" "$EXP/results"

RUN_NAME="${MODEL}_midogpp_xvalidation_smoke_${DEVICE}"
LOG_FILE="$EXP/logs/${RUN_NAME}.log"
METRICS_FILE="$EXP/results/${RUN_NAME}_metrics.json"
TIMING_ARGS=()

if [ "$PROFILE_PIPELINE" = "1" ]; then
  TIMING_ARGS=(
    --profile_pipeline
    --timing_output_json "$EXP/results/${RUN_NAME}_timing.json"
    --timing_output_csv "$EXP/results/${RUN_NAME}_timing.csv"
  )
fi

/usr/bin/time -v "$PYTHON" "$ADAPTER" \
  --config_file "$EXP/configs/${MODEL}_eval.yaml" \
  --dataset "$SMOKE_DATASET" \
  --guide_repo "$GUIDE_REPO" \
  --img_dir "$IMG_DIR" \
  --metrics_output "$METRICS_FILE" \
  --split test \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --overlap "$OVERLAP" \
  --nms_thresh "$NMS_THRESH" \
  --overwrite \
  "${TIMING_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
