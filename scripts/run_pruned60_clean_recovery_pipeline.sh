#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
ROOT="$PROJECT_ROOT/experiments/distillation/pruned60_clean_recovery"
DATA_DIR="$ROOT/data"
LOG_DIR="$ROOT/logs"
STAGE_DIR="$ROOT/.stages"
PIPELINE_LOG="${PIPELINE_LOG:-$ROOT/pipeline.log}"

SUP_CONFIG="$PROJECT_ROOT/experiments/distillation/configs/fcos18_pruned60_supervised_clean.yaml"
DISTILL_CONFIG="$PROJECT_ROOT/experiments/distillation/configs/fcos_x101_to_fcos18_pruned60_distillation_clean.yaml"
SUP_RUN="$PROJECT_ROOT/experiments/distillation/runs/fcos18_pruned60_supervised_clean"
DISTILL_RUN="$PROJECT_ROOT/experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation_clean"

SOURCE_CSV="$PROJECT_ROOT/data/midogpp_guide_eval_xvalidation.csv"
QUICK_CSV="$PROJECT_ROOT/data/midogpp_guide_eval_xvalidation_quick.csv"
TRAIN_CSV="$DATA_DIR/train_seed42.csv"
VAL_CSV="$DATA_DIR/validation_seed42.csv"
CAL_CSV="$DATA_DIR/calibration_seed42.csv"
IMG_DIR="$PROJECT_ROOT/data/midogpp"
GUIDE_REPO="$PROJECT_ROOT/repos/MIDOG_2025_Guide"
ADAPTER="$PROJECT_ROOT/src/benchmark/midog_guide_adapter.py"
FCOS18_CONFIG="$PROJECT_ROOT/experiments/eval_guide/configs/FCOS_18_eval.yaml"

PRUNED60="$PROJECT_ROOT/experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt"
PRUNED50="$PROJECT_ROOT/experiments/pruning/models/FCOS_18_depgraph_fpn_head_50pct.pt"
PRUNED70="$PROJECT_ROOT/experiments/pruning/models/FCOS_18_depgraph_fpn_head_70pct.pt"
TEACHER="$PROJECT_ROOT/repos/FCOS_Inference_CLI/checkpoints/FCOS_x101.ckpt"

RECOVERY_BATCH_SIZE="${RECOVERY_BATCH_SIZE:-2}"
RECOVERY_ACCUM_STEPS="${RECOVERY_ACCUM_STEPS:-1}"
RECOVERY_NUM_WORKERS="${RECOVERY_NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-0}"

mkdir -p "$ROOT" "$LOG_DIR" "$STAGE_DIR"

timestamp() { date -Is; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$PIPELINE_LOG"; }
is_done() { [ -f "$STAGE_DIR/$1.done" ]; }
mark_done() { touch "$STAGE_DIR/$1.done"; log "DONE $1"; }
start_stage() { CURRENT_STAGE="$1"; log "START $CURRENT_STAGE"; }
fail_stage() { local rc="$1"; log "FAILED ${CURRENT_STAGE:-unknown} exit_code=$rc"; exit "$rc"; }
trap 'rc=$?; fail_stage "$rc"' ERR

run_logged() {
  log "CMD $*"
  set +e
  "$@" 2>&1 | tee -a "$PIPELINE_LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
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

run_stage_once() {
  local stage="$1"
  shift
  if is_done "$stage"; then
    log "SKIP $stage already complete"
    return
  fi
  start_stage "$stage"
  "$@"
  mark_done "$stage"
}

verify_repo_root() {
  if [ "$PWD" != "$PROJECT_ROOT" ]; then
    log "Run from repository root: $PROJECT_ROOT"
    return 2
  fi
}

preflight() {
  verify_repo_root
  require_file "$PYTHON"
  require_file "$SOURCE_CSV"
  require_file "$QUICK_CSV"
  require_file "$SUP_CONFIG"
  require_file "$DISTILL_CONFIG"
  require_file "$PRUNED60"
  require_file "$TEACHER"
  require_file "$FCOS18_CONFIG"
  require_dir "$IMG_DIR"
  require_dir "$GUIDE_REPO"
  run_logged "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'
}

create_splits() {
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/create_pruned60_clean_recovery_splits.py" \
    --source "$SOURCE_CSV" \
    --output-dir "$DATA_DIR" \
    --seed 42
}

verify_dataset_roles() {
  run_logged "$PYTHON" - "$TRAIN_CSV" "$VAL_CSV" "$CAL_CSV" "$SOURCE_CSV" <<'PY'
import sys
import pandas as pd
paths = sys.argv[1:]
train, val, cal, source = [pd.read_csv(p) for p in paths]
ids = {
    "train": set(train.filename.astype(str)),
    "validation": set(val.filename.astype(str)),
    "calibration": set(cal.filename.astype(str)),
}
source_test = source[source.split.astype(str) == "test"]
test_ids = set(source_test.filename.astype(str))
checks = {
    "train_validation": ids["train"] & ids["validation"],
    "train_calibration": ids["train"] & ids["calibration"],
    "validation_calibration": ids["validation"] & ids["calibration"],
    "train_test": ids["train"] & test_ids,
    "validation_test": ids["validation"] & test_ids,
    "calibration_test": ids["calibration"] & test_ids,
}
if source_test.filename.nunique() != 111:
    raise SystemExit(f"Expected 111 test images, got {source_test.filename.nunique()}")
bad = {k: len(v) for k, v in checks.items() if v}
if bad:
    raise SystemExit(f"Overlap check failed: {bad}")
print({k: len(v) for k, v in ids.items()} | {"test": len(test_ids), "overlap": 0})
PY
}

train_model() {
  local config="$1"
  local run_dir="$2"
  if [ -f "$run_dir/student_final.pt" ] && [ -f "$run_dir/student_epoch_20.pt" ]; then
    log "SKIP training output exists: $run_dir"
    return
  fi
  if [ -d "$run_dir" ] && [ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    log "Refusing to overwrite partial run directory: $run_dir"
    return 3
  fi
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/train_fcos_distillation.py" \
    --config "$config" \
    --overwrite \
    --batch-size "$RECOVERY_BATCH_SIZE" \
    --gradient-accumulation-steps "$RECOVERY_ACCUM_STEPS" \
    --device cuda
}

select_epoch() {
  local run_dir="$1"
  if [ -f "$run_dir/student_best_ap.pt" ] && [ -f "$run_dir/validation_results.csv" ]; then
    log "SKIP epoch selection output exists: $run_dir"
    return
  fi
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/select_pruned60_best_epoch.py" \
    --run-dir "$run_dir" \
    --dataset "$VAL_CSV" \
    --split val \
    --config-file "$FCOS18_CONFIG" \
    --guide-repo "$GUIDE_REPO" \
    --img-dir "$IMG_DIR" \
    --adapter "$ADAPTER" \
    --python "$PYTHON" \
    --device cuda \
    --batch-size 1 \
    --num-workers "$EVAL_NUM_WORKERS" \
    --det-thresh 0.584 \
    --nms-thresh 0.3 \
    --overlap 0.3
}

calibrate_thresholds() {
  local out="$ROOT/threshold_calibration"
  if [ -f "$out/selected_thresholds.json" ]; then
    log "SKIP threshold calibration output exists"
    return
  fi
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/calibrate_pruned60_thresholds.py" \
    --dataset "$CAL_CSV" \
    --output-dir "$out" \
    --pruned60-checkpoint "$PRUNED60" \
    --supervised-checkpoint "$SUP_RUN/student_best_ap.pt" \
    --distilled-checkpoint "$DISTILL_RUN/student_best_ap.pt" \
    --split calibration \
    --config-file "$FCOS18_CONFIG" \
    --guide-repo "$GUIDE_REPO" \
    --img-dir "$IMG_DIR" \
    --adapter "$ADAPTER" \
    --python "$PYTHON" \
    --device cuda \
    --batch-size 1 \
    --num-workers "$EVAL_NUM_WORKERS" \
    --nms-thresh 0.3 \
    --overlap 0.3
}

write_policy() {
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/write_pruned60_frozen_policy.py" \
    --output "$ROOT/frozen_evaluation_policy.json" \
    --supervised-run "$SUP_RUN" \
    --distilled-run "$DISTILL_RUN" \
    --thresholds "$ROOT/threshold_calibration/selected_thresholds.json" \
    --nms-thresh 0.3 \
    --patch-size 1024 \
    --overlap 0.3
}

policy_value() {
  "$PYTHON" - "$ROOT/frozen_evaluation_policy.json" "$1" "$2" <<'PY'
import json, sys
policy = json.load(open(sys.argv[1], encoding="utf-8"))
print(policy["models"][sys.argv[2]][sys.argv[3]])
PY
}

evaluate_fixed_model() {
  local key="$1"
  local out="$ROOT/final_test/$key"
  local metrics="$out/test_metrics.json"
  local runtime="$out/test_runtime.json"
  if [ -f "$metrics" ] && [ -f "$runtime" ]; then
    log "SKIP final test output exists: $key"
    return
  fi
  mkdir -p "$out"
  local threshold checkpoint
  threshold="$(policy_value "$key" selected_threshold)"
  checkpoint="$(policy_value "$key" checkpoint)"
  local pruned_args=()
  if [ "$checkpoint" != "None" ]; then
    pruned_args=(--pruned_model_path "$checkpoint")
  fi
  run_logged "$PYTHON" "$ADAPTER" \
    --config_file "$FCOS18_CONFIG" \
    --dataset "$SOURCE_CSV" \
    --guide_repo "$GUIDE_REPO" \
    --img_dir "$IMG_DIR" \
    --metrics_output "$metrics" \
    --runtime_metadata_output "$runtime" \
    --runtime_backend pytorch_eager \
    "${pruned_args[@]}" \
    --split test \
    --device cuda \
    --batch_size 1 \
    --num_workers "$EVAL_NUM_WORKERS" \
    --overlap 0.3 \
    --nms_thresh 0.3 \
    --det_thresh "$threshold" \
    --overwrite
}

run_final_test() {
  evaluate_fixed_model baseline
  evaluate_fixed_model pruned60
  evaluate_fixed_model supervised
  evaluate_fixed_model distilled
}

run_sensitivity() {
  if [ "$RUN_SENSITIVITY" != "1" ]; then
    log "SKIP sensitivity analysis disabled; set RUN_SENSITIVITY=1 to run"
    return
  fi
  require_file "$PRUNED50"
  require_file "$PRUNED70"
  for ratio in 50 60 70; do
    local ckpt="$PROJECT_ROOT/experiments/pruning/models/FCOS_18_depgraph_fpn_head_${ratio}pct.pt"
    local out="$ROOT/sensitivity_analysis/fpn_head_${ratio}pct"
    mkdir -p "$out"
    if [ ! -f "$out/selected_thresholds.json" ]; then
      run_logged "$PYTHON" "$PROJECT_ROOT/scripts/calibrate_pruned60_thresholds.py" \
        --dataset "$CAL_CSV" \
        --output-dir "$out" \
        --pruned60-checkpoint "$ckpt" \
        --supervised-checkpoint "$ckpt" \
        --distilled-checkpoint "$ckpt" \
        --split calibration \
        --config-file "$FCOS18_CONFIG" \
        --guide-repo "$GUIDE_REPO" \
        --img-dir "$IMG_DIR" \
        --adapter "$ADAPTER" \
        --python "$PYTHON" \
        --device cuda \
        --batch-size 1 \
        --num-workers "$EVAL_NUM_WORKERS" \
        --nms-thresh 0.3 \
        --overlap 0.3
    fi
    local threshold
    threshold="$("$PYTHON" - "$out/selected_thresholds.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["pruned60"]["selected_threshold"])
PY
)"
    if [ ! -f "$out/test_metrics.json" ]; then
      run_logged "$PYTHON" "$ADAPTER" \
        --config_file "$FCOS18_CONFIG" \
        --dataset "$SOURCE_CSV" \
        --guide_repo "$GUIDE_REPO" \
        --img_dir "$IMG_DIR" \
        --metrics_output "$out/test_metrics.json" \
        --runtime_metadata_output "$out/test_runtime.json" \
        --runtime_backend pytorch_eager \
        --pruned_model_path "$ckpt" \
        --split test \
        --device cuda \
        --batch_size 1 \
        --num_workers "$EVAL_NUM_WORKERS" \
        --overlap 0.3 \
        --nms_thresh 0.3 \
        --det_thresh "$threshold" \
        --overwrite
    fi
  done
}

cpu_timing_one() {
  local key="$1"
  local checkpoint="$2"
  local out="$ROOT/cpu_timing/$key"
  if [ -f "$out/measured3_timing.json" ]; then
    log "SKIP CPU timing output exists: $key"
    return
  fi
  mkdir -p "$out"
  local threshold
  threshold="$(policy_value "$key" selected_threshold)"
  local pruned_args=()
  if [ "$checkpoint" != "" ]; then
    pruned_args=(--pruned_model_path "$checkpoint")
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
      --dataset "$QUICK_CSV" \
      --guide_repo "$GUIDE_REPO" \
      --img_dir "$IMG_DIR" \
      --metrics_output "$out/${run_name}_metrics.json" \
      --runtime_metadata_output "$out/${run_name}_runtime.json" \
      --runtime_backend pytorch_eager \
      "${pruned_args[@]}" \
      --split test \
      --device cpu \
      --batch_size 1 \
      --num_workers 0 \
      --overlap 0.3 \
      --nms_thresh 0.3 \
      --det_thresh "$threshold" \
      --overwrite \
      --profile_pipeline \
      --timing_output_json "$out/${run_name}_timing.json" \
      --timing_output_csv "$out/${run_name}_timing.csv"
  done
}

run_cpu_timing() {
  cpu_timing_one baseline ""
  cpu_timing_one pruned60 "$PRUNED60"
  cpu_timing_one supervised "$SUP_RUN/student_best_ap.pt"
  cpu_timing_one distilled "$DISTILL_RUN/student_best_ap.pt"
}

generate_report() {
  run_logged "$PYTHON" "$PROJECT_ROOT/scripts/generate_pruned60_clean_recovery_report.py" \
    --root "$ROOT" \
    --output "$ROOT/final_comparison.md"
}

run_stage_once preflight preflight
run_stage_once create_splits create_splits
run_stage_once verify_dataset_roles verify_dataset_roles
run_stage_once train_supervised train_model "$SUP_CONFIG" "$SUP_RUN"
run_stage_once select_supervised_epoch select_epoch "$SUP_RUN"
run_stage_once train_distillation train_model "$DISTILL_CONFIG" "$DISTILL_RUN"
run_stage_once select_distilled_epoch select_epoch "$DISTILL_RUN"
run_stage_once threshold_calibration calibrate_thresholds
run_stage_once freeze_policy write_policy
run_stage_once final_test run_final_test
if [ "$RUN_SENSITIVITY" = "1" ]; then
  run_stage_once sensitivity_analysis run_sensitivity
else
  log "SKIP sensitivity_analysis disabled; set RUN_SENSITIVITY=1 to run"
fi
run_stage_once cpu_timing run_cpu_timing
run_stage_once final_report generate_report

log "DONE pruned60 clean recovery pipeline"
