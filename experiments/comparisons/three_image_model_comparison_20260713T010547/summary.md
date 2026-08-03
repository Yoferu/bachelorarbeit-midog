# Three-image matched FCOS comparison

Date: 2026-07-13

This is a diagnostic comparison, not a final benchmark.

## Matched policy

- source dataset: `data/midogpp_guide_eval_xvalidation_quick.csv`
- generated subset: `data/midogpp_guide_eval_xvalidation_3img.csv`
- selection: first three unique filenames in deterministic source CSV order
- images: `413.tiff`, `366.tiff`, `487.tiff`
- annotations in subset: 72
- processed patches per run: 228 for every model and repetition
- config: `experiments/eval_guide/configs/FCOS_18_eval.yaml`
- split: `test`
- image directory: `data/midogpp`
- patch size: 1024
- overlap: 0.3
- operating/confidence threshold: 0.584 from the config
- NMS threshold: 0.3
- CPU, PyTorch eager, batch size 1, `num_workers=0`
- one warm-up and three measured repetitions
- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`
- `torch.set_num_threads(4)`, `torch.set_num_interop_threads(1)`
- no CPU affinity: the prior matched quick benchmark did not use `taskset`

## Results

Times are the mean/median of the three newly executed measured repetitions. Throughput uses three images divided by mean end-to-end time. MACs are for a 1024 input.

| Model | Parameters | MACs | Artifact | Forward mean / median | E2E mean / median | Mean patch latency | Throughput | Detections | F1 | Precision | Recall | Forward ratio vs baseline | E2E ratio vs baseline | Loading method |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original FCOS_18 | 18.524M | 505.503G | 222.50 MB | 265.720 / 265.796 s | 267.842 / 267.937 s | 1.16954 s | 0.01120 img/s | 16 | 0.75862 | 0.68750 | 0.84615 | 1.000x | 1.000x | Guide `ModelFactory`, original FCOS_18 checkpoint |
| FCOS_18 FPN+Head 60% | 15.605M | 395.736G | 62.59 MB | 213.060 / 213.556 s | 215.282 / 215.829 s | 0.93889 s | 0.01394 img/s | 10 | 0.78261 | 0.90000 | 0.69231 | **1.247x faster** | **1.244x faster** | structural `model_object` checkpoint path |
| Distilled MobileNetV3-Small | 8.199M | 467.204G | 33.04 MB | 243.115 / 243.886 s | 245.877 / 246.606 s | 1.07217 s | 0.01220 img/s | 127 | 0.07143 | 0.03937 | 0.38462 | **1.093x faster** | **1.089x faster** | `registry_state_dict` (adapter label: `registry`) |

Measured totals by repetition:

| Model | Forward totals (s) | End-to-end totals (s) |
| --- | --- | --- |
| Original FCOS_18 | 265.796, 266.253, 265.110 | 267.937, 268.372, 267.216 |
| FCOS_18 FPN+Head 60% | 215.152, 213.556, 210.472 | 217.381, 215.829, 212.635 |
| Distilled MobileNetV3-Small | 243.886, 244.362, 241.095 | 246.606, 247.172, 243.852 |

Relative to baseline using mean totals:

- pruned forward speedup: 24.72%; end-to-end speedup: 24.41%
- distilled forward speedup: 9.30%; end-to-end speedup: 8.93%
- MobileNetV3-Small was faster than FCOS_18 in every measured repetition, not slower
- the 60% pruned model was faster than both other models in every measured repetition

## Loading verification

- Baseline: loaded as `FCOS_18`, FCOS detector with ResNet-18 backbone, from `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt` through the guide `ModelFactory`.
- Pruned: adapter accepted `experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt` and loaded its structurally pruned serialized model through the existing `model_object` fallback path.
- Distilled: adapter accepted `experiments/distillation/runs/fcos_x101_to_mobilenetv3_small/student_final.pt`, resolved `fcos_mobilenetv3_small_fpn`, and recreated/loaded it through `registry_state_dict`. Runtime metadata labels this path `registry`.
- All runtime JSON files record CPU, batch size 1, workers 0, overlap 0.3, NMS 0.3, and PyTorch eager.
- All timing JSON files record exactly 3 images and 228 patches.

## Quality interpretation

The evaluator supports metrics on this subset, but three images are too small for final quality conclusions. The baseline and pruned values are plausible for a diagnostic sample. The distilled model is clearly problematic at the matched FCOS_18 threshold: it emitted 127 scored detections, precision 0.0394, and F1 0.0714. No threshold tuning or sweep was performed.

## Command template

The following adapter invocation was executed for each model/run combination. The baseline omitted `--pruned_model_path`; the pruned and distilled paths used their checkpoints listed above.

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

.venv/bin/python -c "import runpy,sys,torch; torch.set_num_threads(4); torch.set_num_interop_threads(1); sys.argv=['src/benchmark/midog_guide_adapter.py']+sys.argv[1:]; runpy.run_path('src/benchmark/midog_guide_adapter.py',run_name='__main__')" \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation_3img.csv \
  --guide_repo repos/MIDOG_2025_Guide \
  --img_dir data/midogpp \
  --split test --device cpu --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 \
  --runtime_backend pytorch_eager \
  [--pruned_model_path CHECKPOINT] \
  --metrics_output "$OUT/${MODEL}_${RUN}_metrics.json" \
  --runtime_metadata_output "$OUT/${MODEL}_${RUN}_runtime.json" \
  --profile_pipeline \
  --timing_output_json "$OUT/${MODEL}_${RUN}_timing.json" \
  --timing_output_csv "$OUT/${MODEL}_${RUN}_timing.csv"
```

The exact execution order was baseline, pruned60, distilled_small; within each model: warmup1, measured1, measured2, measured3.

## Outputs and warnings

- `selection.json` records the subset source, method, IDs, row count, and split.
- Every run has a `.log`, metrics JSON, runtime JSON, timing JSON, and timing CSV in this directory.
- The only repeated warning was `Can't initialize NVML`; all runs were CPU-only and this did not affect execution.
- Runtime metadata contains no backend warnings or errors.
- No model, checkpoint, previous result, or external-guide file was modified.
- No training, fine-tuning, threshold sweep, 32-image evaluation, or held-out evaluation was run.

## Conclusion

- MobileNetV3-Small is **not consistently slower** than FCOS_18 here; it is consistently about 9% faster on this strictly matched three-image test.
- The 60% pruned model remains faster than FCOS_18 by about 24–25% and is the fastest of the three.
- All comparisons used identical images, 228 patches, runtime settings, repetition policy, and adapter command structure.
- No timing result appears invalid due to loading or configuration mismatch. The distilled quality result is valid for the matched threshold but is extremely poor and the subset is too small for final accuracy claims.
