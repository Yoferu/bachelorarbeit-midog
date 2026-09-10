#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. All inference commands use taskset -c 0-3,
# batch size 1, workers 0, overlap 0.3, and NMS 0.3.
RUN=experiments/quantization/openvino_ptq_20260819T0000Z
CAL=experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv
GUIDE=repos/MIDOG_2025_Guide

# Standard NNCF PTQ (repeat with the pruned FP32 IR/output paths for Q2).
taskset -c 0-3 .venv/bin/python scripts/quantize_openvino_fcos_ptq.py \
  --fp32-model "$RUN/baseline/fp32/FCOS_18_fp32.xml" \
  --output-model "$RUN/baseline/int8/FCOS_18_int8.xml" \
  --calibration-csv "$CAL" \
  --split-manifest experiments/distillation/pruned60_clean_recovery/data/split_manifest.json \
  --final-test-csv data/midogpp_guide_eval_xvalidation.csv \
  --image-dir data/midogpp --guide-repo "$GUIDE" \
  --patch-size 1024 --overlap 0.3 --subset-size 300 --seed 42 \
  --metadata-output "$RUN/baseline/int8/quantization_metadata.json"
# Exact PTQ configuration used by the script:
# nncf.quantize(model, dataset, subset_size=300,
#               preset=nncf.QuantizationPreset.PERFORMANCE,
#               target_device=nncf.TargetDevice.CPU)

# Representative full calibration inference command (substitute artifact,
# backend, checkpoint, and output directory for the other three candidates).
taskset -c 0-3 .venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset "$CAL" --guide_repo "$GUIDE" --img_dir data/midogpp \
  --split calibration --device cpu --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584 \
  --runtime_backend openvino_int8 \
  --int8_model_path "$RUN/baseline/int8/FCOS_18_int8.xml" \
  --metrics_output "$RUN/baseline/calibration/openvino_int8/metrics_fixed.json" \
  --runtime_metadata_output "$RUN/baseline/calibration/openvino_int8/runtime.json" \
  --predictions_output "$RUN/baseline/calibration/openvino_int8/predictions.json" \
  --profile_pipeline \
  --timing_output_json "$RUN/baseline/calibration/openvino_int8/timing.json" \
  --timing_output_csv "$RUN/baseline/calibration/openvino_int8/timing.csv" --overwrite

.venv/bin/python scripts/sweep_cached_midog_thresholds.py \
  --predictions "$RUN/baseline/calibration/openvino_int8/predictions.json" \
  --dataset "$CAL" --guide-repo "$GUIDE" --variant baseline \
  --model-artifact "$RUN/baseline/int8/FCOS_18_int8.xml" \
  --backend OpenVINO --precision INT8 --split calibration \
  --minimum 0.300 --maximum 0.800 --step 0.001 \
  --output-csv "$RUN/baseline/calibration/openvino_int8/threshold_sweep.csv" \
  --selection-output "$RUN/baseline/calibration/openvino_int8/threshold_selection.json"

# Frozen-threshold final scoring reuses the complete cached 111-image,
# post-NMS prediction set; this changes no inference outputs or timings.
.venv/bin/python scripts/score_cached_midog_predictions.py \
  --predictions "$RUN/baseline/final_test_fixed_int8/predictions.json" \
  --dataset data/midogpp_guide_eval_xvalidation.csv --guide-repo "$GUIDE" \
  --split test --threshold 0.600 \
  --output "$RUN/baseline/final_test_calibrated/openvino_int8/metrics.json"

.venv/bin/python scripts/build_final_quantization_report.py
