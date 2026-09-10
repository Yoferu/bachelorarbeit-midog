#!/usr/bin/env bash
set -euo pipefail

# Runs only the missing Pruned60 OpenVINO strict-FP32 diagnostic or full row.
ROOT="experiments/final_cpu_4physicalcore_rerun_20260906"
MODE="${1:---diagnostic}"
case "$MODE" in
  --diagnostic) SUBDIR="diagnostic"; DATASET="data/midogpp_guide_eval_xvalidation_3img.csv"; REPS=3 ;;
  --full) SUBDIR="full"; DATASET="data/midogpp_guide_eval_xvalidation.csv"; REPS=1 ;;
  *) echo "usage: $0 [--diagnostic|--full]" >&2; exit 2 ;;
esac

export MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 TORCH_NUM_THREADS=4 TORCH_NUM_INTEROP_THREADS=1

PRUNED="experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr3e-5_seed42/student_stable_selected.pt"
EXPORT="experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/smoke_fp32/export"

for ((rep=1; rep<=REPS; rep++)); do
  out="$ROOT/$SUBDIR/pruned60_openvino_strict_fp32/rep$rep"
  mkdir -p "$out"
  cmd=(taskset -c 0,2,4,6 .venv/bin/python src/benchmark/midog_guide_adapter.py
    --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml
    --dataset "$DATASET" --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp
    --split test --device cpu --batch_size 1 --num_workers 0
    --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584
    --runtime_backend openvino_cpu --pruned_model_path "$PRUNED" --export_dir "$EXPORT"
    --openvino_compress_to_fp16 0 --openvino_performance_hint LATENCY
    --openvino_num_streams 1 --openvino_inference_num_threads 4
    --openvino_inference_precision f32 --runtime_intra_op_threads 4
    --runtime_inter_op_threads 1 --warmup_iterations 5 --profile_pipeline --overwrite
    --metrics_output "$out/metrics.json" --runtime_metadata_output "$out/runtime.json"
    --predictions_output "$out/predictions.json" --timing_output_json "$out/timing.json"
    --timing_output_csv "$out/timing.csv")
  printf '%q ' /usr/bin/time -v -o "$out/resource_usage.txt" "${cmd[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  echo "Starting $SUBDIR/pruned60_openvino_strict_fp32/rep$rep"
  /usr/bin/time -v -o "$out/resource_usage.txt" "${cmd[@]}" > "$out/stdout_stderr.log" 2>&1
  echo "Completed $SUBDIR/pruned60_openvino_strict_fp32/rep$rep"
done
