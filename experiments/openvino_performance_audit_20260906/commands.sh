#!/usr/bin/env bash
set -euo pipefail

# Run from /home/johan/bachelorarbeit-midog. CPUs 0,2,4,6 are four distinct
# physical cores on the audited AMD Ryzen 9 9950X3D host. CPUs 0-3 reproduce
# the historical mistake: four logical CPUs spanning only two physical cores.
COMMON=(
  .venv/bin/python src/benchmark/midog_guide_adapter.py
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml
  --dataset data/midogpp_guide_eval_xvalidation_3img.csv
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp
  --split test --device cpu --batch_size 1 --num_workers 0
  --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584
  --runtime_intra_op_threads 0 --runtime_inter_op_threads 1
  --openvino_inference_num_threads 4
  --warmup_iterations 5 --profile_pipeline --overwrite
)

# Historical/suspect topology reproduction (run three times with rep=1..3).
rep=1
taskset -c 0-3 "${COMMON[@]}" --runtime_backend openvino_cpu \
  --openvino_compress_to_fp16 0 --openvino_performance_hint LATENCY \
  --openvino_num_streams 1 \
  --export_dir experiments/quantization/openvino_ptq_20260819T0000Z/baseline/smoke_fp32/export \
  --metrics_output "experiments/openvino_performance_audit_20260906/diagnostic_old_openvino_rep${rep}_metrics.json" \
  --runtime_metadata_output "experiments/openvino_performance_audit_20260906/diagnostic_old_openvino_rep${rep}_runtime.json" \
  --timing_output_json "experiments/openvino_performance_audit_20260906/diagnostic_old_openvino_rep${rep}_timing.json" \
  --timing_output_csv "experiments/openvino_performance_audit_20260906/diagnostic_old_openvino_rep${rep}_timing.csv"

# Corrected controlled OpenVINO and equivalent ONNX Runtime diagnostics use
# affinity 0,2,4,6 and the exact same FP32 ONNX provenance.
# Repetitions 1..3 used the same adapter arguments above, with these changes:
#   OpenVINO BF16-hinted:
#     taskset -c 0,2,4,6 ... --runtime_backend openvino_cpu
#       --openvino_inference_num_threads 4 --openvino_inference_precision bf16
#       --openvino_performance_hint LATENCY --openvino_num_streams 1
#   OpenVINO strict FP32:
#     taskset -c 0,2,4,6 ... --runtime_backend openvino_cpu
#       --openvino_inference_num_threads 4 --openvino_inference_precision f32
#       --openvino_performance_hint LATENCY --openvino_num_streams 1
#   ONNX Runtime FP32:
#     taskset -c 0,2,4,6 ... --runtime_backend onnxruntime_cpu
#       --runtime_intra_op_threads 0 --runtime_inter_op_threads 1
# Exact expanded commands are recorded by /usr/bin/time in each *.log.

# Corrected full historical-policy run (FP32 storage, BF16 CPU execution).
taskset -c 0,2,4,6 .venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation.csv \
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp \
  --split test --device cpu --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584 \
  --runtime_backend openvino_cpu --openvino_compress_to_fp16 0 \
  --openvino_inference_num_threads 4 --openvino_inference_precision bf16 \
  --openvino_performance_hint LATENCY --openvino_num_streams 1 \
  --warmup_iterations 5 --profile_pipeline --overwrite \
  --export_dir experiments/quantization/openvino_ptq_20260819T0000Z/baseline/smoke_fp32/export \
  --metrics_output experiments/openvino_performance_audit_20260906/full_corrected_openvino_fp32_metrics.json \
  --runtime_metadata_output experiments/openvino_performance_audit_20260906/full_corrected_openvino_fp32_runtime.json \
  --predictions_output experiments/openvino_performance_audit_20260906/full_corrected_openvino_fp32_predictions.json \
  --timing_output_json experiments/openvino_performance_audit_20260906/full_corrected_openvino_fp32_timing.json \
  --timing_output_csv experiments/openvino_performance_audit_20260906/full_corrected_openvino_fp32_timing.csv

# Strict precision-equivalent full FP32 run. Output arguments are analogous and
# use the full_strict_f32_openvino_* prefix.
taskset -c 0,2,4,6 .venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation.csv \
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp \
  --split test --device cpu --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584 \
  --runtime_backend openvino_cpu --openvino_compress_to_fp16 0 \
  --openvino_inference_num_threads 4 --openvino_inference_precision f32 \
  --openvino_performance_hint LATENCY --openvino_num_streams 1 \
  --warmup_iterations 5 --profile_pipeline --overwrite \
  --export_dir experiments/quantization/openvino_ptq_20260819T0000Z/baseline/smoke_fp32/export \
  --metrics_output experiments/openvino_performance_audit_20260906/full_strict_f32_openvino_metrics.json \
  --runtime_metadata_output experiments/openvino_performance_audit_20260906/full_strict_f32_openvino_runtime.json \
  --predictions_output experiments/openvino_performance_audit_20260906/full_strict_f32_openvino_predictions.json \
  --timing_output_json experiments/openvino_performance_audit_20260906/full_strict_f32_openvino_timing.json \
  --timing_output_csv experiments/openvino_performance_audit_20260906/full_strict_f32_openvino_timing.csv
