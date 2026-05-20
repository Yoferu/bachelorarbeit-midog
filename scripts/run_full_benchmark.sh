#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/bachelorarbeit-midog}"
EXP="${EXP:-$PROJECT/experiments/eval_guide}"
GUIDE_REPO="${GUIDE_REPO:-$PROJECT/repos/MIDOG_2025_Guide}"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
ADAPTER="${ADAPTER:-$PROJECT/src/benchmark/midog_guide_adapter.py}"
FULL_DATASET="${FULL_DATASET:-$PROJECT/data/midogpp_guide_eval_xvalidation.csv}"
IMG_DIR="${IMG_DIR:-$PROJECT/data/midogpp}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OVERLAP="${OVERLAP:-0.3}"
NMS_THRESH="${NMS_THRESH:-0.3}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-pytorch_eager}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
COMPILE_MODE="${COMPILE_MODE:-}"
ONNX_OPSET="${ONNX_OPSET:-18}"
OPENVINO_DEVICE="${OPENVINO_DEVICE:-CPU}"
EXPORT_DIR="${EXPORT_DIR:-$EXP/exported_models}"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(FCOS_18 FCOS_x50 FCOS_x101)
fi

mkdir -p "$EXP/logs" "$EXP/results"

RUNTIME_LABEL="$DEVICE"
if [ "$RUNTIME_BACKEND" != "pytorch_eager" ]; then
  RUNTIME_LABEL="${DEVICE}_${RUNTIME_BACKEND}"
fi

for MODEL in "${MODELS[@]}"; do
  RUN_NAME="${MODEL}_midogpp_xvalidation_test_full_${RUNTIME_LABEL}"
  METRICS_FILE="$EXP/results/${RUN_NAME}_metrics.json"
  RUNTIME_METADATA_FILE="$EXP/results/${RUN_NAME}_runtime.json"
  COMPILE_ARGS=()

  if [ -n "$COMPILE_MODE" ]; then
    COMPILE_ARGS=(--compile_mode "$COMPILE_MODE")
  fi

  /usr/bin/time -v "$PYTHON" "$ADAPTER" \
    --config_file "$EXP/configs/${MODEL}_eval.yaml" \
    --dataset "$FULL_DATASET" \
    --guide_repo "$GUIDE_REPO" \
    --img_dir "$IMG_DIR" \
    --metrics_output "$METRICS_FILE" \
    --runtime_metadata_output "$RUNTIME_METADATA_FILE" \
    --runtime_backend "$RUNTIME_BACKEND" \
    --compile_backend "$COMPILE_BACKEND" \
    "${COMPILE_ARGS[@]}" \
    --onnx_opset "$ONNX_OPSET" \
    --openvino_device "$OPENVINO_DEVICE" \
    --export_dir "$EXPORT_DIR" \
    --split test \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --overlap "$OVERLAP" \
    --nms_thresh "$NMS_THRESH" \
    --overwrite \
    2>&1 | tee "$EXP/logs/${RUN_NAME}.log"
done
