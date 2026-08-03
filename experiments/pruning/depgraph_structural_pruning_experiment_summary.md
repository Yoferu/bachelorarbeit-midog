# FCOS_18 DepGraph Structural Pruning Experiment

Status: candidate selection complete after the extended FPN+head follow-up.
`FCOS_18_depgraph_fpn_head_60pct.pt` is the main structural candidate,
`FCOS_18_depgraph_fpn_head_50pct.pt` is the conservative backup, and
`FCOS_18_depgraph_fpn_head_70pct.pt` is the boundary candidate. The earlier
conclusion below describes the initial smoke sweep and is superseded by the
extended quick results in
`experiments/pruning/sweeps/fcos18_depgraph_fpn_head_extended_quick/summary.md`.
The selected 60% model is not fine-tuned; fine-tuning is conditional on its full
held-out evaluation.

## Purpose

The goal was to test whether physical structured pruning with Torch-Pruning /
DepGraph can reduce the `FCOS_18` architecture enough to justify recovery
fine-tuning. Unlike the earlier masked pruning baseline, these artifacts remove
channels and update dependent tensors.

All results below use the same smoke evaluation setup:

- model family: `FCOS_18`
- split: `test`
- dataset: `data/midogpp_guide_eval_xvalidation_smoke.csv`
- image directory: `data/midogpp`
- runtime: PyTorch eager on CPU
- batch size: `1`
- num workers: `0`
- overlap: `0.3`
- NMS threshold: `0.3`

## Decision

Do not fine-tune any current pruning variant.

Structural pruning technically works: tensors are physically smaller, parameter
counts decrease, MAC estimates decrease, and the pruned artifacts can run
through the existing MIDOG evaluation pipeline. The speed-accuracy tradeoff is
not useful, however:

- conservative variants that preserve detections provide only about 3-4%
  forward-pass speedup
- aggressive variants reach up to 7.5% forward-pass speedup, still below the
  fine-tuning threshold, and collapse thresholded detections
- head-only DepGraph pruning did not physically change the model in this setup

The next optimization branch should focus on OpenVINO CPU deployment and INT8
post-training quantization instead of recovery training for these pruning
artifacts.

## Normal Sweep

Decision rule: recommend fine-tuning only for variants with at least 10% forward
speedup or at least 8% end-to-end speedup while still producing plausible
detections.

| Variant | Params After | MAC Reduction | Changed Layers | Eval Detections | F1 | Precision | Recall | Forward Speedup | Selection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FCOS_18_baseline` | - | - | - | 26 | 0.6774 | 0.8077 | 0.5833 | - | baseline |
| `fcos18_depgraph_structural_20pct` | 13.10M | 11.00% | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | 4.88% | not selected |
| `fcos18_depgraph_structural_30pct` | 10.94M | 15.24% | 29 | 1 | 0.0541 | 1.0000 | 0.0278 | 5.61% | not selected |
| `fcos18_depgraph_head_only_20pct` | 18.52M | 0.00% | 0 | 26 | 0.6774 | 0.8077 | 0.5833 | 0.21% | not selected |
| `fcos18_depgraph_fpn_head_20pct` | 17.29M | 8.50% | 10 | 25 | 0.6885 | 0.8400 | 0.5833 | 3.05% | not selected |
| `fcos18_depgraph_backbone_fpn_head_20pct` | 13.10M | 11.00% | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | 5.63% | not selected |

Full normal sweep outputs:

- `experiments/pruning/sweeps/fcos18_depgraph_speed_sweep/summary.md`
- `experiments/pruning/sweeps/fcos18_depgraph_speed_sweep/summary.csv`
- `experiments/pruning/sweeps/fcos18_depgraph_speed_sweep/summary.json`

## Boundary Test

This was an explicit boundary test, not the main optimization path. It checked
whether much more aggressive pruning could produce speedups large enough to
justify recovery training.

Boundary decision rules:

- reject immediately if eval detections collapse to `0`
- reject if F1 `< 0.2`
- do not fine-tune if forward speedup `< 10%`
- recommend fine-tuning only if forward speedup is at least 10-15% and
  detections remain plausible

| Variant | Params After | MAC Reduction | Changed Layers | Representative Detections | Eval Detections | F1 | Precision | Recall | Forward Speedup | Selection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FCOS_18_baseline` | - | - | - | - | 26 | 0.6774 | 0.8077 | 0.5833 | - | baseline |
| `fcos18_depgraph_structural_40pct` | 9.15M | 19.33% | 29 | 114 | 0 | 0.0000 | 0.0000 | 0.0000 | 5.59% | not selected |
| `fcos18_depgraph_structural_50pct` | 7.67M | 22.77% | 29 | 180 | 0 | 0.0000 | 0.0000 | 0.0000 | 7.49% | not selected |
| `fcos18_depgraph_fpn_head_30pct` | 16.83M | 11.85% | 10 | 155 | 25 | 0.6885 | 0.8400 | 0.5833 | 2.97% | not selected |
| `fcos18_depgraph_fpn_head_40pct` | 16.42M | 15.02% | 10 | 21 | 25 | 0.6885 | 0.8400 | 0.5833 | 3.78% | not selected |

Full boundary test outputs:

- `experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary/summary.md`
- `experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary/summary.csv`
- `experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary/summary.json`

## FPN+Head Metric Tie Diagnostic

The FPN+head 20%, 30%, and 40% variants produced identical smoke F1, precision,
recall, and image-level TP/FP/FN counts. This was checked explicitly:

- the three model artifact hashes are different
- the runtime metadata loads the correct artifact path for each run
- metric/timing JSON and CSV files are unique per variant
- raw prediction file hashes are different
- raw prediction counts differ before thresholding and before/after NMS
- top-10 post-NMS boxes and scores differ by image

The identical smoke metrics are therefore genuine metric ties, not an artifact
loading or result-overwrite bug. The smoke split is too small to distinguish
these FPN+head variants at the aggregate metric level: after the MIDOG threshold
and patch-merge NMS, all three variants leave the same number of high-confidence
detections per image and these detections yield identical TP/FP/FN counts.

Diagnostic outputs:

- `experiments/pruning/diagnostics/fpn_head_prediction_comparison/report.md`
- `experiments/pruning/diagnostics/fpn_head_prediction_comparison/summary.json`

## Thesis Interpretation

The experiment demonstrates that dependency-aware structural pruning is feasible
for the torchvision FCOS architecture, including ResNet backbone, FPN, and FCOS
head dependencies. It also shows that physical channel removal alone does not
guarantee deployment-relevant CPU speedups. The measured latency improvement is
substantially smaller than the MAC reduction, which suggests that the runtime is
limited by implementation overheads, memory behavior, postprocessing, and CPU
kernel efficiency rather than only arithmetic count.

For this model and evaluation setup, pruning does not provide an attractive
speed-accuracy tradeoff. Conservative variants retain smoke accuracy but are too
slow to justify recovery training. Aggressive variants reduce parameter and MAC
counts more strongly, but their thresholded detection behavior fails before the
latency gain becomes large enough. Therefore, pruning should be reported as a
completed negative result and the optimization effort should move to deployment
backends.

## Next Optimization Branch

Prepare the next branch around:

- OpenVINO CPU export/runtime validation
- INT8 post-training quantization
- latency and accuracy comparison against PyTorch eager and OpenVINO FP32/FP16
- no dependence on pruning recovery training

The pruning artifacts should remain available for reproducibility, but no
current pruned artifact should be used as a fine-tuning starting point.
