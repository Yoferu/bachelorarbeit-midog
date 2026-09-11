# Thesis experiment map

Run from the repository root with `.venv` active. Outputs listed here are
regenerated locally unless already tracked as compact summaries. The E numbers
refer to [the historical inventory](../reports/thesis_experiment_inventory.md),
not thesis chapter numbers. Prepared or failed configurations are not evidence
of completed experiments.

| Experiment | Main script | Configuration / inputs | Expected output |
| --- | --- | --- | --- |
| FCOS baseline (E01–E03) | `scripts/run_full_benchmark.sh` | `experiments/eval_guide/configs/FCOS_{18,x50,x101}_eval.yaml`, test CSV | `experiments/eval_guide/results/` metrics/runtime JSON |
| Corrected CPU four physical cores | `experiments/final_cpu_4physicalcore_rerun_20260906/run_benchmarks.sh` | FCOS_18 config, prepared FP32/INT8 artifacts, explicit threads/affinity | dated `diagnostic/` or `full/`, metrics/timing/runtime |
| OpenVINO strict FP32 CPU | `experiments/final_cpu_4physicalcore_rerun_20260906/run_pruned60_openvino_strict_fp32.sh` | Pruned60 selected checkpoint, explicit `f32` execution | strict-FP32 row; baseline reference in OpenVINO audit |
| Structural scope/ratio sweep (E08–E11) | `scripts/run_depgraph_structural_sweep.py` | `experiments/pruning/configs/fcos18_depgraph_*.yaml` | pruning artifacts and sweep summaries |
| Pruned60 creation/evaluation | `scripts/create_depgraph_structural_pruned_model.py`, benchmark wrappers | `fcos18_depgraph_fpn_head_60pct.yaml`; `PRUNED_MODEL_PATH` | `experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt`, metadata, evaluation metrics |
| Clean supervised/KD recovery (E14) | `scripts/train_fcos_distillation.py` | `experiments/distillation/configs/*clean*.yaml` | checkpoints and training/validation logs under configured output directory |
| Frozen supervised recovery (E15–E16) | `scripts/train_fcos_distillation.py` | `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_*.yaml` | per-LR/seed run directories; some runs fail/early-stop |
| Frozen recovery distillation | same trainer | same config directory, `distillation_*.yaml` | teacher/student training and validation artifacts |
| MobileNet distillation (E12) | same trainer; `scripts/run_distillation_pilot.py` for pilots | `experiments/distillation/configs/fcos_x101_to_mobilenetv3_{small,large}.yaml` | configured distillation runs; Large/pilot status differs from Small |
| ONNX Runtime FP32 (E18) | `src/benchmark/midog_guide_adapter.py` | `--runtime_backend onnxruntime_cpu` | ONNX export and validation, predictions, metrics, runtime |
| ONNX Runtime INT8 PTQ (E19) | `scripts/quantize_onnx_fcos_ptq.py`, adapter | FP32 ONNX, clean calibration CSV/manifest; `onnxruntime_int8` | QDQ INT8 ONNX, quantization metadata, evaluation |
| OpenVINO FP32/INT8 (E20–E21) | adapter; `scripts/quantize_openvino_fcos_ptq.py` | FP32 IR, clean calibration CSV/manifest; `openvino_cpu` / `openvino_int8` | XML/BIN models, validation and evaluation metadata |
| Threshold calibration (E17) | `scripts/sweep_cached_midog_thresholds.py`, `scripts/score_cached_midog_predictions.py` | calibration predictions then test predictions, frozen threshold | threshold sweep CSV, selection JSON, rescored metrics |
| Pruning threshold diagnostic | `scripts/run_fpn_head_threshold_sweep.py` | quick 32-image test subset | diagnostic sweep; not leakage-free calibration |
| Raspberry Pi 5 (E23) | `scripts/create_rpi4_fcos_benchmark_bundle.sh`; generated `run_all_benchmarks.sh` | raw Pruned60, FCOS_18, three-image CSV and prepared exports | six variants, one warm-up + three measured repetitions, bundle summaries |

## Baseline, pruning, and runtime backends

The README supplies dataset and checkpoint preparation. Create the three-image
CSV required by CPU diagnostics and the Pi bundle using its recorded image IDs:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
source = pd.read_csv('data/midogpp_guide_eval_xvalidation.csv')
ids = source.filename.map(lambda name: int(Path(name).stem))
subset = source[(source.split == 'test') & ids.isin([413, 366, 487])]
assert subset.filename.nunique() == 3
subset.to_csv('data/midogpp_guide_eval_xvalidation_3img.csv', index=False)
subset.to_csv('data/midogpp_guide_eval_xvalidation_smoke.csv', index=False)
PY
bash scripts/run_smoke_test.sh FCOS_18
PROFILE_PIPELINE=1 bash scripts/run_quick_benchmark.sh FCOS_18
bash scripts/run_full_benchmark.sh FCOS_18
python scripts/create_depgraph_structural_pruned_model.py \
  --config experiments/pruning/configs/fcos18_depgraph_fpn_head_60pct.yaml \
  --output experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt
PRUNED_MODEL_PATH="$PWD/experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt" \
  PROFILE_PIPELINE=1 bash scripts/run_quick_benchmark.sh FCOS_18
RUNTIME_BACKEND=onnxruntime_cpu PROFILE_PIPELINE=1 \
  bash scripts/run_quick_benchmark.sh FCOS_18
```

For a new full evaluation/export, use the adapter. Choose a fresh output directory;
the example defines a reusable Bash argument array for the next sections:

```bash
OUT=experiments/reproduction
mkdir -p "$OUT"
COMMON=(--config_file experiments/eval_guide/configs/FCOS_18_eval.yaml
  --dataset data/midogpp_guide_eval_xvalidation.csv
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp
  --split test --device cpu --batch_size 1 --num_workers 0
  --overlap 0.3 --nms_thresh 0.3 --det_thresh 0.584
  --runtime_intra_op_threads 4 --runtime_inter_op_threads 1)
python src/benchmark/midog_guide_adapter.py "${COMMON[@]}" \
  --runtime_backend onnxruntime_cpu --export_dir "$OUT/export" \
  --metrics_output "$OUT/ort_fp32_metrics.json" \
  --runtime_metadata_output "$OUT/ort_fp32_runtime.json"
python src/benchmark/midog_guide_adapter.py "${COMMON[@]}" \
  --runtime_backend openvino_cpu --export_dir "$OUT/export" \
  --openvino_compress_to_fp16 0 --openvino_inference_precision f32 \
  --openvino_inference_num_threads 4 --openvino_num_streams 1 \
  --openvino_performance_hint LATENCY \
  --metrics_output "$OUT/openvino_fp32_metrics.json" \
  --runtime_metadata_output "$OUT/openvino_fp32_runtime.json"
```

For Pruned60, add `--pruned_model_path` and use a separate output/export directory.
Do not substitute raw Pruned60 for a historically selected recovery checkpoint
without labelling the change and checking hashes. Adapter `--help` documents
prediction and timing output flags.

## Recovery and distillation

Generate the clean splits as in the README, create Pruned60, and obtain the
FCOS_x101 teacher for KD. Examples of the actual configurations:

```bash
python scripts/train_fcos_distillation.py \
  --config experiments/distillation/configs/fcos18_pruned60_supervised_clean_dense_lr1e-4_seed42.yaml
python scripts/train_fcos_distillation.py \
  --config experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr3e-5_seed42.yaml
python scripts/train_fcos_distillation.py \
  --config experiments/distillation/pruned60_frozen_backbone_recovery/configs/distillation_lr1e-5_seed42.yaml
python scripts/train_fcos_distillation.py \
  --config experiments/distillation/configs/fcos_x101_to_mobilenetv3_small.yaml
```

These are separate experiments, not consecutive stages of one model. The
configuration controls output location, initialization, seed, learning rate,
backbone freeze, validation, and checkpoint grid. Use `--smoke-test` for a
technical check and `--resume` only for a compatible existing run. Avoid
`--overwrite`/`--force` when preserving evidence.

`select_pruned60_best_epoch.py` implements epoch selection;
`select_stable_pruned60_checkpoint.py` implements later stable checkpoint
selection. Consult each `--help` and preserve the validation split and policy.
The older `run_pruned60_recovery_pipeline.sh` and
`run_pruned60_clean_recovery_pipeline.sh` are historical orchestration, not the
canonical final frozen-backbone LR sweep. Their planned downstream comparisons
were not all completed. Do not use a completed shell stage as evidence that a
checkpoint improved on step 0.

## Quantization and threshold calibration

Export FP32 models first. Export names include generated identifiers: inspect
`$OUT/export` and set `FP32_ONNX` and `FP32_XML` to the actual baseline files
reported by the preceding runs. Do not assume historical hash-based filenames
will recur. OpenVINO XML requires its adjacent BIN file.

```bash
CAL=experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv
MANIFEST=experiments/distillation/pruned60_clean_recovery/data/split_manifest.json
TEST=data/midogpp_guide_eval_xvalidation.csv
PTQ=(--calibration-csv "$CAL" --split-manifest "$MANIFEST"
  --final-test-csv "$TEST" --image-dir data/midogpp
  --guide-repo repos/MIDOG_2025_Guide)
# Set these to existing generated paths before running:
: "${FP32_ONNX:?Set FP32_ONNX to the exported ONNX file}"
: "${FP32_XML:?Set FP32_XML to the exported FP32 XML file}"
python scripts/quantize_onnx_fcos_ptq.py "${PTQ[@]}" \
  --fp32-model "$FP32_ONNX" --output-model "$OUT/fcos18_int8.onnx" \
  --metadata-output "$OUT/ort_quantization.json"
python scripts/quantize_openvino_fcos_ptq.py "${PTQ[@]}" \
  --fp32-model "$FP32_XML" --output-model "$OUT/fcos18_int8.xml" \
  --subset-size 300 --seed 42 --metadata-output "$OUT/openvino_quantization.json"
python src/benchmark/midog_guide_adapter.py "${COMMON[@]}" \
  --runtime_backend onnxruntime_int8 --int8_model_path "$OUT/fcos18_int8.onnx" \
  --predictions_output "$OUT/test_predictions.json" \
  --metrics_output "$OUT/int8_metrics.json"
python src/benchmark/midog_guide_adapter.py "${COMMON[@]}" \
  --dataset "$CAL" --split calibration \
  --runtime_backend onnxruntime_int8 --int8_model_path "$OUT/fcos18_int8.onnx" \
  --predictions_output "$OUT/calibration_predictions.json" \
  --metrics_output "$OUT/calibration_metrics.json"
python scripts/sweep_cached_midog_thresholds.py \
  --predictions "$OUT/calibration_predictions.json" --dataset "$CAL" \
  --guide-repo repos/MIDOG_2025_Guide --variant baseline \
  --model-artifact "$OUT/fcos18_int8.onnx" --backend ONNXRuntime --precision INT8 \
  --split calibration --minimum 0.300 --maximum 0.800 --step 0.001 \
  --output-csv "$OUT/threshold_sweep.csv" --selection-output "$OUT/threshold_selection.json"
```

Read the selected threshold from the JSON and pass that value unchanged to
`score_cached_midog_predictions.py --predictions "$OUT/test_predictions.json"
--dataset "$TEST" --guide-repo repos/MIDOG_2025_Guide --split test
--threshold VALUE --output "$OUT/calibrated_metrics.json"`.
For OpenVINO, use `openvino_int8` and the XML model, with the explicit OpenVINO
controls above; generate separate calibration/test predictions. Repeat with a
separate Pruned60 export/artifact directory for the pruned rows. Thresholds must
be selected on calibration, never on final-test predictions. The ONNX quantizer
has its own patch-selection implementation/defaults; consult generated metadata
rather than assuming OpenVINO's 300-sample setting applies to it.

## Corrected CPU and Raspberry Pi 5 benchmarks

The dated CPU runners require the historical selected checkpoint and INT8 exports
at the paths assigned near their top. Prepare those exact artifacts externally
or adapt a copy of the runners to your new outputs and record their hashes.
The first runner skips existing completed rows; the strict-FP32 runner overwrites
its row, so use a fresh output root for a new experiment.

```bash
lscpu -e=CPU,CORE,SOCKET
bash experiments/final_cpu_4physicalcore_rerun_20260906/run_benchmarks.sh --diagnostic
bash experiments/final_cpu_4physicalcore_rerun_20260906/run_benchmarks.sh --full
bash experiments/final_cpu_4physicalcore_rerun_20260906/run_pruned60_openvino_strict_fp32.sh --full
```

The audited affinity `0,2,4,6` is specific to the recorded Ryzen topology; adapt
it for another host. Baseline OpenVINO strict-FP32 is documented in the OpenVINO audit report and
`experiments/openvino_performance_audit_20260906/commands.sh`; that file mixes
executable historical diagnostics with command notes and is not a complete
strict-FP32 reproduction runner. Use the adapter example above for a new run. Historical `0-3`
launchers are retained as provenance, not as four-physical-core templates.

On the development machine, after preparing raw Pruned60 and the three-image CSV:

```bash
bash scripts/create_rpi4_fcos_benchmark_bundle.sh --baseline-onnx "$FP32_ONNX"
```

The builder invokes `prepare_rpi4_fcos_deployment_artifacts.py`, requires existing
source exports, and rebuilds its staging directory. Inspect that script's
`--help` for source-artifact overrides. The six-variant builder is canonical;
`create_rpi4_fcos18_bundle.sh` is the earlier baseline-only package. Copy the
resulting bundle to a Raspberry Pi 5 with a 64-bit OS, then inside it run:

```bash
bash install_rpi4.sh
bash run_smoke_test.sh
bash run_all_benchmarks.sh
```

Record OS, package versions, power/cooling, CPU settings, and temperature. Legacy
`rpi4` names are intentionally unchanged. Historical physical measurements are
indexed in `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/README_RPI4.md`;
regenerating a bundle does not recreate those results. TVM did not complete an
inference benchmark; Pi OpenVINO/PyTorch INT8 are not executed result families.
