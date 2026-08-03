#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PIPELINE_LOG="${PIPELINE_LOG:-$PROJECT_ROOT/experiments/distillation/pruned60_recovery_pipeline.log}"
MARKER_DIR="$PROJECT_ROOT/experiments/distillation/.pruned60_recovery_stages"

SUP_CONFIG="$PROJECT_ROOT/experiments/distillation/configs/fcos18_pruned60_supervised_recovery.yaml"
DISTILL_CONFIG="$PROJECT_ROOT/experiments/distillation/configs/fcos_x101_to_fcos18_pruned60_distillation.yaml"
PRUNED60="$PROJECT_ROOT/experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt"
TEACHER="$PROJECT_ROOT/repos/FCOS_Inference_CLI/checkpoints/FCOS_x101.ckpt"
QUICK_DATASET="$PROJECT_ROOT/data/midogpp_guide_eval_xvalidation_quick.csv"
IMG_DIR="$PROJECT_ROOT/data/midogpp"
GUIDE_REPO="$PROJECT_ROOT/repos/MIDOG_2025_Guide"
ADAPTER="$PROJECT_ROOT/src/benchmark/midog_guide_adapter.py"
FCOS18_CONFIG="$PROJECT_ROOT/experiments/eval_guide/configs/FCOS_18_eval.yaml"

SUP_RUN="$PROJECT_ROOT/experiments/distillation/runs/fcos18_pruned60_supervised_recovery"
DISTILL_RUN="$PROJECT_ROOT/experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation"
SUP_FINAL="$SUP_RUN/student_final.pt"
DISTILL_FINAL="$DISTILL_RUN/student_final.pt"

QUALITY_DIR="$PROJECT_ROOT/experiments/distillation/recovery_quality"
BENCH_DIR="$PROJECT_ROOT/experiments/distillation/recovery_cpu_benchmarks"
REPORT="$PROJECT_ROOT/experiments/distillation/pruned60_recovery_comparison.md"

RECOVERY_BATCH_SIZE="${RECOVERY_BATCH_SIZE:-2}"
RECOVERY_ACCUM_STEPS="${RECOVERY_ACCUM_STEPS:-1}"
RECOVERY_NUM_WORKERS="${RECOVERY_NUM_WORKERS:-4}"

mkdir -p "$(dirname "$PIPELINE_LOG")" "$MARKER_DIR"

timestamp() {
  date -Is
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$PIPELINE_LOG"
}

run_logged() {
  log "CMD $*"
  set +e
  "$@" 2>&1 | tee -a "$PIPELINE_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

start_stage() {
  CURRENT_STAGE="$1"
  log "START $CURRENT_STAGE"
}

done_stage() {
  local stage="$1"
  touch "$MARKER_DIR/$stage.done"
  log "DONE $stage"
}

fail_stage() {
  local rc="$1"
  log "FAILED ${CURRENT_STAGE:-unknown} exit_code=$rc"
  exit "$rc"
}

trap 'rc=$?; fail_stage "$rc"' ERR

is_done() {
  [ -f "$MARKER_DIR/$1.done" ]
}

require_file() {
  if [ ! -f "$1" ]; then
    log "Missing required file: $1"
    return 2
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    log "Missing required directory: $1"
    return 2
  fi
}

run_training_stage() {
  local stage="$1"
  local config="$2"
  local run_dir="$3"
  local final_ckpt="$4"
  if is_done "$stage" && [ -f "$final_ckpt" ]; then
    log "SKIP $stage already complete"
    return
  fi
  if [ -f "$final_ckpt" ]; then
    done_stage "$stage"
    return
  fi
  if [ -d "$run_dir" ] && [ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    log "Refusing to overwrite partial run directory without final checkpoint: $run_dir"
    return 3
  fi
  start_stage "$stage"
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/train_fcos_distillation.py" \
    --config "$config" \
    --overwrite \
    --batch-size "$RECOVERY_BATCH_SIZE" \
    --gradient-accumulation-steps "$RECOVERY_ACCUM_STEPS" \
    --device cuda
  require_file "$final_ckpt"
  done_stage "$stage"
}

quality_eval() {
  local stage="$1"
  local key="$2"
  local ckpt="${3:-}"
  local out_dir="$QUALITY_DIR/$key"
  local metrics="$out_dir/quality_metrics.json"
  local runtime="$out_dir/quality_runtime.json"
  if is_done "$stage" && [ -f "$metrics" ] && [ -f "$runtime" ]; then
    log "SKIP $stage already complete"
    return
  fi
  start_stage "$stage"
  mkdir -p "$out_dir"
  local pruned_args=()
  if [ -n "$ckpt" ]; then
    pruned_args=(--pruned_model_path "$ckpt")
  fi
  run_logged "$PYTHON" "$ADAPTER" \
    --config_file "$FCOS18_CONFIG" \
    --dataset "$QUICK_DATASET" \
    --guide_repo "$GUIDE_REPO" \
    --img_dir "$IMG_DIR" \
    --metrics_output "$metrics" \
    --runtime_metadata_output "$runtime" \
    --runtime_backend pytorch_eager \
    "${pruned_args[@]}" \
    --split test \
    --device cuda \
    --batch_size 1 \
    --num_workers "$RECOVERY_NUM_WORKERS" \
    --overlap 0.3 \
    --nms_thresh 0.3 \
    --overwrite
  require_file "$metrics"
  require_file "$runtime"
  done_stage "$stage"
}

cpu_benchmark() {
  local stage="$1"
  local key="$2"
  local ckpt="${3:-}"
  local out_dir="$BENCH_DIR/$key"
  local summary="$out_dir/timing_median.csv"
  if is_done "$stage" && [ -f "$summary" ]; then
    log "SKIP $stage already complete"
    return
  fi
  start_stage "$stage"
  mkdir -p "$out_dir"
  local pruned_args=()
  if [ -n "$ckpt" ]; then
    pruned_args=(--pruned_model_path "$ckpt")
  fi
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export TORCH_NUM_THREADS=4
  export TORCH_NUM_INTEROP_THREADS=1
  for run_name in warmup1 measured1 measured2 measured3; do
    run_logged "$PYTHON" "$ADAPTER" \
      --config_file "$FCOS18_CONFIG" \
      --dataset "$QUICK_DATASET" \
      --guide_repo "$GUIDE_REPO" \
      --img_dir "$IMG_DIR" \
      --metrics_output "$out_dir/${run_name}_metrics.json" \
      --runtime_metadata_output "$out_dir/${run_name}_runtime.json" \
      --runtime_backend pytorch_eager \
      "${pruned_args[@]}" \
      --split test \
      --device cpu \
      --batch_size 1 \
      --num_workers 0 \
      --overlap 0.3 \
      --nms_thresh 0.3 \
      --overwrite \
      --profile_pipeline \
      --timing_output_json "$out_dir/${run_name}_timing.json" \
      --timing_output_csv "$out_dir/${run_name}_timing.csv"
  done
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/summarize_quick_benchmark_timing.py" \
    --timing_glob "$out_dir/measured*_timing.json" \
    --output_csv "$summary"
  require_file "$summary"
  done_stage "$stage"
}

start_stage preflight
if [ "$PWD" != "$PROJECT_ROOT" ]; then
  log "Run from repository root: $PROJECT_ROOT"
  (exit 2)
fi
require_file "$PYTHON"
require_file "$SUP_CONFIG"
require_file "$DISTILL_CONFIG"
require_file "$PRUNED60"
require_file "$TEACHER"
require_file "$QUICK_DATASET"
require_file "$FCOS18_CONFIG"
require_dir "$IMG_DIR"
require_dir "$GUIDE_REPO"
run_logged "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'
done_stage preflight

run_training_stage train_supervised "$SUP_CONFIG" "$SUP_RUN" "$SUP_FINAL"
run_training_stage train_distillation "$DISTILL_CONFIG" "$DISTILL_RUN" "$DISTILL_FINAL"

quality_eval quality_fcos18 fcos18 ""
quality_eval quality_pruned60_untuned pruned60_untuned "$PRUNED60"
quality_eval quality_supervised supervised "$SUP_FINAL"
quality_eval quality_distilled distilled "$DISTILL_FINAL"

cpu_benchmark benchmark_fcos18 fcos18 ""
cpu_benchmark benchmark_pruned60_untuned pruned60_untuned "$PRUNED60"
cpu_benchmark benchmark_supervised supervised "$SUP_FINAL"
cpu_benchmark benchmark_distilled distilled "$DISTILL_FINAL"

start_stage final_report
run_logged "$PYTHON" "$PROJECT_ROOT/scripts/generate_pruned60_recovery_report.py" \
  --output "$REPORT" \
  --quality-dir "$QUALITY_DIR" \
  --benchmark-dir "$BENCH_DIR"
require_file "$REPORT"
done_stage final_report

log "DONE pruned60 recovery pipeline"
