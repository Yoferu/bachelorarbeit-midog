# FCOS quantization and combined-optimization report

> **Checkpoint identity and evaluation mode.** The historical directory/run key
> `pruned_recovered` is retained in artifact paths for provenance. In every final
> result below it means **Pruned60 (step-0 recovery checkpoint)**, selected from
> `experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/selected_recovery_checkpoint.json`
> with `global_optimizer_step = "0"`. It therefore represents the structurally
> pruned model before a meaningful recovery update, not a successfully recovered
> or fine-tuned model. The eight fixed rows are full inference evaluations. The
> four calibrated rows are cached-prediction threshold rescoring and reuse the
> corresponding fixed INT8 timing; they are not four additional inference runs.

## Method

OpenVINO 2026.1.0 and NNCF 3.2.0 were used. Standard static PTQ was
`nncf.quantize(model, dataset, subset_size=300, preset=PERFORMANCE,
target_device=CPU)` with default `model_type=None`, fast bias correction enabled,
and no advanced parameters. Inputs were float32 CHW tensors of shape
`3×1024×1024`, produced by the existing MIDOG ROI preprocessing at overlap 0.3.
The calibration manifest contains 39 images and 2,289 rows (968 positive
annotations), with 2,922 available patches. NNCF selected 300 samples; the model's
control-flow body caused 600 transform calls, covering all 39 images.

Both INT8 IRs contain 202 `FakeQuantize` nodes around a graph containing 83
convolutions. The serialized weights contain INT8 elements, and validation records
show OpenVINO device `CPU`, effective backend `openvino_int8`, with no ORT execution
provider. Postprocessing, NMS, and evaluation remain FP32/Python by design.

All complete final runs used 111 images, 5,383 annotation rows, 8,500 patches,
batch size 1, cores 0–3, patch size 1024, overlap 0.3, and NMS 0.3. Calibration and
test filenames have zero overlap.

## Fixed operating point (threshold 0.584)

| Model | Evaluation mode | Scope | Runtime | Precision | F1 | Precision | Recall | AP | Forward s | E2E s | RAM MiB | Size MiB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Full inference | Final test: 111 images/8,500 patches | ORT | FP32 | 0.825050 | 0.809598 | 0.841103 | 0.885915 | 3780.11 | 3913.47 | 2227.1 | 71.00 |
| Baseline | Full inference | Final test: 111 images/8,500 patches | ORT | INT8 | 0.829630 | 0.819620 | 0.839887 | 0.885769 | 2682.91 | 2785.57 | — | 18.57 |
| Baseline | Full inference | Final test: 111 images/8,500 patches | OpenVINO | FP32 | 0.823180 | 0.803785 | 0.843535 | 0.886192 | 7605.41 | 7675.31 | 1934.7 | 71.53 |
| Baseline | Full inference | Final test: 111 images/8,500 patches | OpenVINO | INT8 | 0.827195 | 0.825859 | 0.828537 | 0.876927 | 4058.16 | 4159.59 | 1476.3 | 19.05 |
| Pruned60 (step-0 recovery checkpoint) | Full inference | Final test: 111 images/8,500 patches | ORT | FP32 | 0.823016 | 0.844056 | 0.803000 | 0.878764 | 3436.42 | 3523.50 | 2232.6 | 59.82 |
| Pruned60 (step-0 recovery checkpoint) | Full inference | Final test: 111 images/8,500 patches | ORT | INT8 | 0.820491 | 0.851045 | 0.792055 | 0.878639 | 2465.51 | 2608.88 | 2205.3 | 15.70 |
| Pruned60 (step-0 recovery checkpoint) | Full inference | Final test: 111 images/8,500 patches | OpenVINO | FP32 | 0.824112 | 0.839714 | 0.809080 | 0.878863 | 5950.70 | 6021.39 | 1819.5 | 60.38 |
| Pruned60 (step-0 recovery checkpoint) | Full inference | Final test: 111 images/8,500 patches | OpenVINO | INT8 | 0.816456 | 0.851298 | 0.784353 | 0.870903 | 6065.25 | 6141.89 | 1461.8 | 16.24 |

The pruned OpenVINO INT8 timing is anomalous but valid: it was slower than its FP32
counterpart despite successful low-precision graph verification. It therefore must
not be presented as a speedup and is not Pareto-relevant.

## Calibration selections

The grid was 0.300–0.800 inclusive at 0.001 spacing; maximum aggregate calibration
F1 was selected, with exact ties resolved to the lowest threshold (the repository's
existing `np.argmax` policy).

| Model | Runtime | Threshold | Delta vs 0.584 | Calibration F1 | Precision | Recall | AP | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | ORT INT8 | 0.583 | -0.001 | 0.844626 | 0.833166 | 0.856405 | 0.899721 | 829 | 166 | 139 |
| Pruned60 (step-0 recovery checkpoint) | ORT INT8 | 0.546 | -0.038 | 0.841302 | 0.828657 | 0.854339 | 0.895446 | 827 | 171 | 141 |
| Baseline | OpenVINO INT8 | 0.600 | +0.016 | 0.847584 | 0.872131 | 0.824380 | 0.894818 | 798 | 117 | 170 |
| Pruned60 (step-0 recovery checkpoint) | OpenVINO INT8 | 0.565 | -0.019 | 0.840249 | 0.843750 | 0.836777 | 0.890948 | 810 | 150 | 158 |

## Frozen-threshold final results (cached-prediction rescoring)

These quality metrics were recomputed from the saved full-run INT8 predictions.
The model was not executed again. Consequently the runtime columns below are
references to the corresponding fixed-threshold full inference run, not timings
of rescoring and not new benchmark measurements.

| Model | Evaluation mode | Scope | Runtime | Threshold | F1 | Precision | Recall | AP | Referenced full-run forward s | Referenced full-run E2E s | E2E speedup vs baseline ORT FP32 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Cached prediction rescoring | Final test: 111 images/8,500 cached predictions | ORT INT8 | 0.583 | 0.828469 | 0.817357 | 0.839887 | 0.885769 | 2682.91 | 2785.57 | 1.405× |
| Pruned60 (step-0 recovery checkpoint) | Cached prediction rescoring | Final test: 111 images/8,500 cached predictions | ORT INT8 | 0.546 | 0.820709 | 0.815779 | 0.825699 | 0.878639 | 2465.51 | 2608.88 | 1.500× |
| Baseline | Cached prediction rescoring | Final test: 111 images/8,500 cached predictions | OpenVINO INT8 | 0.600 | 0.828031 | 0.850128 | 0.807053 | 0.876927 | 4058.16 | 4159.59 | 0.941× |
| Pruned60 (step-0 recovery checkpoint) | Cached prediction rescoring | Final test: 111 images/8,500 cached predictions | OpenVINO INT8 | 0.565 | 0.820259 | 0.833473 | 0.807458 | 0.870903 | 6065.25 | 6141.89 | 0.637× |

| Model | Runtime | Fixed F1/P/R | Calibrated F1/P/R | F1 delta |
|---|---|---:|---:|---:|
| Baseline | ORT INT8 | 0.829630 / 0.819620 / 0.839887 | 0.828469 / 0.817357 / 0.839887 | -0.001161 |
| Pruned60 (step-0 recovery checkpoint) | ORT INT8 | 0.820491 / 0.851045 / 0.792055 | 0.820709 / 0.815779 / 0.825699 | +0.000218 |
| Baseline | OpenVINO INT8 | 0.827195 / 0.825859 / 0.828537 | 0.828031 / 0.850128 / 0.807053 | +0.000835 |
| Pruned60 (step-0 recovery checkpoint) | OpenVINO INT8 | 0.816456 / 0.851298 / 0.784353 | 0.820259 / 0.833473 / 0.807458 | +0.003804 |

## Speedups and interpretation

- ORT FP32→INT8: baseline 1.409× forward/1.405× E2E; pruned 1.394×/1.351×.
- OpenVINO FP32→INT8: baseline 1.874×/1.845×; pruned 0.981×/0.980× (a measured slowdown).
- ORT→OpenVINO at INT8: baseline 0.661× forward/0.670× E2E and pruned
  0.406×/0.425×; values below 1 mean OpenVINO was slower.
- Pruning within ORT: FP32 1.100×/1.111× and INT8 1.088×/1.068×.
  Pruning within OpenVINO: FP32 1.278×/1.275×, but INT8 0.669×/0.677×.
- INT8 reduced serialized size by 73.8% for both ORT models and by 73.4%/73.1%
  for baseline/pruned OpenVINO. OpenVINO peak RAM fell 23.7%/19.7%; comparable
  baseline ORT INT8 peak RAM was not captured, and no value is inferred.
- ORT AP changed by only -0.00015/-0.00012 under INT8. OpenVINO AP changed by
  -0.00926/-0.00796, a larger loss. Threshold calibration cannot recover AP because
  AP reflects score ranking across thresholds.
- Calibration mainly moved precision/recall, most visibly for pruned ORT and both
  OpenVINO INT8 models. It recovered 0.00380 F1 for pruned OpenVINO, but did not
  change the preferred deployment candidate.

## Pareto conclusions

The measured F1/AP/E2E frontier contains pruned ORT INT8 calibrated (fastest),
baseline ORT INT8 calibrated (best speed/accuracy compromise and highest calibrated
INT8 F1), baseline ORT FP32 (slightly higher AP), and baseline OpenVINO FP32
(highest measured AP, but slowest). The last point is technically non-dominated only
because its AP is 0.000276 above ORT FP32; the difference is practically tiny.

Fastest and smallest is pruned ORT INT8: 2608.88 s E2E, 15.70 MiB, with a cost of
0.00776 F1 and 0.00713 AP versus baseline ORT INT8 after calibration. It remains
worthwhile only when the additional 6.8% E2E speed and 15.5% size reduction justify
that accuracy loss. Baseline ORT INT8 is the recommended general compromise.

OpenVINO INT8 did not beat ORT INT8 on this CPU: it was slower and lost more AP.
Accuracy-aware OpenVINO quantization is justified as a possible follow-up because
the OpenVINO AP loss is around 0.008–0.009, but it was not run. QAT is not justified
before investigating the pruned OpenVINO timing anomaly and trying the less invasive
accuracy-aware PTQ path; neither is launched here.
