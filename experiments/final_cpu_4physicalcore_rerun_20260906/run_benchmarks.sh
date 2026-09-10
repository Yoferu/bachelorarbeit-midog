#!/usr/bin/env bash
set -euo pipefail

# Reproducible final development-PC CPU rerun. Run from repository root.
# The audited Ryzen SMT topology maps 0,2,4,6 to four distinct physical cores.
ROOT="experiments/final_cpu_4physicalcore_rerun_20260906"
DATASET="data/midogpp_guide_eval_xvalidation_3img.csv"
MODE="diagnostic"
REPETITIONS=3

while (($#)); do
  case "$1" in
    --full) MODE="full"; DATASET="data/midogpp_guide_eval_xvalidation.csv"; REPETITIONS=1 ;;
    --diagnostic) MODE="diagnostic"; DATASET="data/midogpp_guide_eval_xvalidation_3img.csv"; REPETITIONS=3 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

export MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export TORCH_NUM_INTEROP_THREADS=1

COMMON=(
  .venv/bin/python src/benchmark/midog_guide_adapter.py
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml
  --dataset "$DATASET"
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp
  --split test --device cpu --batch_size 1 --num_workers 0
  --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584
  # An explicit ORT thread count is required: 0 lets ONNX Runtime create and
  # pin one worker per host core, overriding the inherited taskset affinity.
  --runtime_intra_op_threads 4 --runtime_inter_op_threads 1
  --openvino_inference_num_threads 4 --openvino_performance_hint LATENCY
  --openvino_num_streams 1 --warmup_iterations 5
  --profile_pipeline --overwrite
)

PRUNED="experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr3e-5_seed42/student_stable_selected.pt"
ORT_BASE="experiments/quantization/ptq_20260816T1815Z/baseline"
ORT_PRUNED="experiments/quantization/ptq_20260816T1815Z/pruned_recovered"
OV_BASE="experiments/quantization/openvino_ptq_20260819T0000Z/baseline"
OV_PRUNED="experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered"

run_one() {
  local name="$1" rep="$2"
  shift 2
  local out="$ROOT/$MODE/$name/rep$rep"
  mkdir -p "$out"
  if [[ -s "$out/timing.json" && -s "$out/metrics.json" ]]; then
    echo "Skipping completed $MODE/$name/rep$rep"
    return
  fi
  local cmd=(taskset -c 0,2,4,6 "${COMMON[@]}" "$@"
    --metrics_output "$out/metrics.json"
    --runtime_metadata_output "$out/runtime.json"
    --predictions_output "$out/predictions.json"
    --timing_output_json "$out/timing.json"
    --timing_output_csv "$out/timing.csv")
  printf '%q ' /usr/bin/time -v -o "$out/resource_usage.txt" "${cmd[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  echo "Starting $MODE/$name/rep$rep"
  /usr/bin/time -v -o "$out/resource_usage.txt" "${cmd[@]}" > "$out/stdout_stderr.log" 2>&1
  echo "Completed $MODE/$name/rep$rep"
}

for ((rep=1; rep<=REPETITIONS; rep++)); do
  run_one baseline_pytorch_fp32 "$rep" --runtime_backend pytorch_eager
  run_one pruned60_pytorch_fp32 "$rep" --runtime_backend pytorch_eager --pruned_model_path "$PRUNED"
  run_one baseline_ort_fp32 "$rep" --runtime_backend onnxruntime_cpu --export_dir "$ORT_BASE/export"
  run_one pruned60_ort_fp32 "$rep" --runtime_backend onnxruntime_cpu --pruned_model_path "$PRUNED" --export_dir "$ORT_PRUNED/export"
  run_one baseline_ort_int8 "$rep" --runtime_backend onnxruntime_int8 --int8_model_path "$ORT_BASE/int8/FCOS_18_qdq_int8.onnx"
  run_one pruned60_ort_int8 "$rep" --runtime_backend onnxruntime_int8 --pruned_model_path "$PRUNED" --int8_model_path "$ORT_PRUNED/int8/FCOS_18_pruned60_recovered_qdq_int8.onnx"
  run_one baseline_openvino_int8 "$rep" --runtime_backend openvino_int8 --int8_model_path "$OV_BASE/int8/FCOS_18_int8.xml" --openvino_inference_precision f32
  run_one pruned60_openvino_int8 "$rep" --runtime_backend openvino_int8 --pruned_model_path "$PRUNED" --int8_model_path "$OV_PRUNED/int8/FCOS_18_pruned60_recovered_int8.xml" --openvino_inference_precision f32
done
