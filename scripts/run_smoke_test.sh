#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [FCOS_18 FCOS_x50 FCOS_x101]"
  echo "Environment: PROJECT, PYTHON, GUIDE_REPO, IMG_DIR, DEVICE, NUM_WORKERS,"
  echo "RUNTIME_BACKEND, PRUNED_MODEL_PATH, PROFILE_PIPELINE; see docs/benchmarking.md."
  exit 0
fi

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
PRUNED_MODEL_PATH="${PRUNED_MODEL_PATH:-}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-pytorch_eager}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
COMPILE_MODE="${COMPILE_MODE:-}"
ONNX_OPSET="${ONNX_OPSET:-18}"
OPENVINO_DEVICE="${OPENVINO_DEVICE:-CPU}"
OPENVINO_COMPRESS_TO_FP16="${OPENVINO_COMPRESS_TO_FP16:-1}"
EXPORT_DIR="${EXPORT_DIR:-$EXP/exported_models}"

if [ "$#" -gt 0 ]; then
  MODEL="$1"
fi

mkdir -p "$EXP/logs" "$EXP/results"

RUNTIME_LABEL="$DEVICE"
if [ "$RUNTIME_BACKEND" != "pytorch_eager" ]; then
  RUNTIME_LABEL="${DEVICE}_${RUNTIME_BACKEND}"
fi
if [ "$RUNTIME_BACKEND" = "openvino_cpu" ] && [ "$OPENVINO_COMPRESS_TO_FP16" = "0" ]; then
  RUNTIME_LABEL="${RUNTIME_LABEL}_fp32"
fi
PRUNED_ARGS=()
if [ -n "$PRUNED_MODEL_PATH" ]; then
  PRUNED_STEM="$(basename "$PRUNED_MODEL_PATH")"
  PRUNED_STEM="${PRUNED_STEM%.*}"
  RUNTIME_LABEL="${RUNTIME_LABEL}_pruned_${PRUNED_STEM}"
  PRUNED_ARGS=(--pruned_model_path "$PRUNED_MODEL_PATH")
fi

RUN_NAME="${MODEL}_midogpp_xvalidation_smoke_${RUNTIME_LABEL}"
LOG_FILE="$EXP/logs/${RUN_NAME}.log"
METRICS_FILE="$EXP/results/${RUN_NAME}_metrics.json"
RUNTIME_METADATA_FILE="$EXP/results/${RUN_NAME}_runtime.json"
TIMING_ARGS=()
COMPILE_ARGS=()

if [ -n "$COMPILE_MODE" ]; then
  COMPILE_ARGS=(--compile_mode "$COMPILE_MODE")
fi

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
  --runtime_metadata_output "$RUNTIME_METADATA_FILE" \
  --runtime_backend "$RUNTIME_BACKEND" \
  "${PRUNED_ARGS[@]}" \
  --compile_backend "$COMPILE_BACKEND" \
  "${COMPILE_ARGS[@]}" \
  --onnx_opset "$ONNX_OPSET" \
  --openvino_device "$OPENVINO_DEVICE" \
  --openvino_compress_to_fp16 "$OPENVINO_COMPRESS_TO_FP16" \
  --export_dir "$EXPORT_DIR" \
  --split test \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --overlap "$OVERLAP" \
  --nms_thresh "$NMS_THRESH" \
  --overwrite \
  "${TIMING_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
