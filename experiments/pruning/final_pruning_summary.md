# FCOS_18 Structural Pruning Summary (quick selection; later backend finals exist)

> **Scope/status correction (2026-09-03):** The quality and latency table below
> is a non-final 32-image/2,486-patch quick evaluation. The eager-PyTorch full-run
> commands in this document were not completed. Later full 111-image/8,500-patch
> ORT and OpenVINO evaluations did complete; see
> `reports/final_results_provenance.md`. Those later paths retain the historical
> key `pruned_recovered`, but the evaluated model is Pruned60 selected at optimizer
> step 0, before a meaningful recovery update.

## Status

Candidate selection is complete. The selected main candidate is
`experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt`. The 50% model
is the conservative backup and the 70% model is the boundary candidate. No
FPN+head candidate has been fine-tuned.

The full held-out evaluation and final four-core CPU benchmark have not been
executed in this environment because CUDA is unavailable and the full split has
111 test images. Exact non-overwriting commands are recorded below. Until those
commands finish, the final-result fields remain pending rather than being
inferred from the quick subset.

## Methodology

Dependency-aware magnitude pruning was applied with Torch-Pruning/DepGraph to
internal FPN and FCOS detection-head convolutions. The ResNet-18 backbone root
and fixed class, box, and centerness output layers were excluded. Models were
evaluated through the existing MIDOG guide adapter with split `test`, overlap
0.3, NMS threshold 0.3, batch size 1, and no data-loader workers. The operating
policy comes from `experiments/eval_guide/configs/FCOS_18_eval.yaml`.

Quick timing used PyTorch eager CPU execution. The final constrained benchmark
uses the same adapter and profiling output with `taskset -c 0-3`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, CPU, batch size 1, and zero workers.

## Candidate Selection

| Role | Artifact | Rationale |
| --- | --- | --- |
| Main | `FCOS_18_depgraph_fpn_head_60pct.pt` | Clear CPU speedup with moderate quick-set quality loss |
| Backup | `FCOS_18_depgraph_fpn_head_50pct.pt` | Essentially baseline quick F1, but just below the 10% speed target |
| Boundary | `FCOS_18_depgraph_fpn_head_70pct.pt` | Fastest measured variant, but substantially larger quality loss |

## Quick Results

These are existing measurements on
`data/midogpp_guide_eval_xvalidation_quick.csv`, not full-evaluation results.
Latency columns are totals over the 32-image quick split.

| Variant | F1 | Precision | Recall | Detections | Forward pass | End-to-end | Forward speedup | E2E speedup | Params reduction | MAC reduction | Artifact size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCOS_18 | 0.8018 | 0.7961 | 0.8076 | 559 | 1522.60 s | 1541.88 s | - | - | - | - | 212.19 MiB checkpoint |
| FPN+head 50% | 0.8019 | 0.8266 | 0.7786 | 519 | 1372.15 s | 1389.98 s | 9.88% | 9.85% | 13.40% | 18.01% | 61.35 MiB |
| FPN+head 60% | 0.7801 | 0.9104 | 0.6824 | 413 | 1244.71 s | 1263.55 s | 18.25% | 18.05% | 15.76% | 21.71% | 59.69 MiB |
| FPN+head 70% | 0.7152 | 0.9355 | 0.5789 | 341 | 1227.08 s | 1245.61 s | 19.41% | 19.21% | 17.27% | 24.29% | 58.62 MiB |

Artifact size is serialized file size and is not directly comparable with the
Lightning baseline checkpoint, which contains additional training state.

## Full Evaluation

Status: **not completed for eager PyTorch**. The commands below are retained as a
historical plan; do not execute them merely to update documentation. Later ORT
and OpenVINO final inference runs completed and are indexed separately. The
original instruction was to run the baseline and selected candidate in a CUDA-capable WSL
terminal. These paths are separate from all existing sweep outputs.

```bash
mkdir -p experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda

.venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation.csv \
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp \
  --split test --device cuda --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --runtime_backend pytorch_eager \
  --metrics_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_baseline_metrics.json \
  --runtime_metadata_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_baseline_runtime.json \
  --profile_pipeline \
  --timing_output_json experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_baseline_timing.json \
  --timing_output_csv experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_baseline_timing.csv

.venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation.csv \
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp \
  --split test --device cuda --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --runtime_backend pytorch_eager \
  --pruned_model_path experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt \
  --metrics_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_fpn_head_60pct_metrics.json \
  --runtime_metadata_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_fpn_head_60pct_runtime.json \
  --profile_pipeline \
  --timing_output_json experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_fpn_head_60pct_timing.json \
  --timing_output_csv experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/FCOS_18_fpn_head_60pct_timing.csv
```

Do not add `--overwrite`. Before running, verify that the two final output
directories do not already contain results; use a new suffixed directory for a
repeat measurement.

| Variant | F1 | Precision | Recall | Detections |
| --- | ---: | ---: | ---: | ---: |
| FCOS_18 | pending | pending | pending | pending |
| FPN+head 60% | pending | pending | pending | pending |

## Final CPU_4C_LIMITED Benchmark

Status: **not completed for eager PyTorch**. The commands remain for provenance.
Later four-core ORT/OpenVINO full runs are the completed final backend benchmarks.
The original plan was to run both commands under the same machine state. This benchmark
also evaluates the full split so its metrics provide a CPU consistency check.

```bash
mkdir -p experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 \
  .venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation.csv \
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp \
  --split test --device cpu --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --runtime_backend pytorch_eager \
  --metrics_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_baseline_metrics.json \
  --runtime_metadata_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_baseline_runtime.json \
  --profile_pipeline \
  --timing_output_json experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_baseline_timing.json \
  --timing_output_csv experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_baseline_timing.csv

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 \
  .venv/bin/python src/benchmark/midog_guide_adapter.py \
  --config_file experiments/eval_guide/configs/FCOS_18_eval.yaml \
  --dataset data/midogpp_guide_eval_xvalidation.csv \
  --guide_repo repos/MIDOG_2025_Guide --img_dir data/midogpp \
  --split test --device cpu --batch_size 1 --num_workers 0 \
  --overlap 0.3 --nms_thresh 0.3 --runtime_backend pytorch_eager \
  --pruned_model_path experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt \
  --metrics_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_fpn_head_60pct_metrics.json \
  --runtime_metadata_output experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_fpn_head_60pct_runtime.json \
  --profile_pipeline \
  --timing_output_json experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_fpn_head_60pct_timing.json \
  --timing_output_csv experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cpu4/FCOS_18_fpn_head_60pct_timing.csv
```

| Variant | Forward pass | End-to-end | Forward speedup | E2E speedup |
| --- | ---: | ---: | ---: | ---: |
| FCOS_18 | pending | pending | - | - |
| FPN+head 60% | pending | pending | pending | pending |

## Final Conclusion

The structural pruning search and candidate selection are complete. The 60%
FPN+head model is the only main candidate; no further sweep is planned. The
pruning phase becomes fully closed for thesis reporting after the full held-out
evaluation and matched `CPU_4C_LIMITED` baseline/candidate benchmark are
recorded here.

Fine-tuning is not currently justified by the quick result. It remains a
conditional recovery step only if the full evaluation shows unacceptable
degradation.

## Remaining Limitations

- Full held-out F1, precision, recall, and detection count are pending.
- Final four-core CPU latency and speedup are pending.
- Quick latency is a single measured run and not a repeated confidence interval.
- Serialized artifact sizes include format overhead and are not pure weight size.
- The selected structural model has not been recovery fine-tuned.
