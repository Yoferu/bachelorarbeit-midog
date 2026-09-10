# Reproducibility and Experiment Documentation Audit

Audit date: 2026-09-03  
Repository: `/home/johan/bachelorarbeit-midog`

## Audit method and evidence policy

This is a read-only reconstruction of surviving evidence. No inference, training,
calibration, or benchmark was run. The search covered tracked, untracked, and
ignored files visible in the repository, including `experiments/`, `results/`,
`reports/`, `docs/`, `scripts/`, `configs/`, `logs/`, `dist/`, and the vendored
repositories. Git history was searched for experiment/result paths. The ignored
`.venv` trees were excluded as software environments rather than experimental
records; ignored experiment and deployment-bundle directories were included.

Evidence grades used below are: **A**, complete machine-readable output and a
matching runtime/config/completion record; **B**, complete logs/reports or several
consistent artifacts; **C**, prose, partial output, or directory/name evidence;
**D**, insufficient to claim completion. A script or planned command is not a run.
Calibrated final-test metrics produced by rescoring cached predictions are counted
as evaluation results, but not as new inference runs.

## A. Executive Summary

The audit identifies **25 meaningful experiment families** (E01-E25 below),
containing hundreds of individual training checkpoints and evaluation repetitions.
For the current final test, there are **8 strongly verified full inference runs**
(two model variants × ORT/OpenVINO × FP32/INT8) and **4 strongly verified
calibrated INT8 rescoring results**, all on 111 images and 8,500 patches. Thus the
final-result table contains 12 operating-point rows but only eight underlying
inference executions. All 12 are evidence grade A.

Across the 25 families, **19 are complete for their stated scope** (including
smoke/quick/calibration scope), **4 are partly completed or failed** (E13, E15,
E16, E24), **1 is prepared-only** (E22), and **1 mixes completed packaging with
unexecuted variants** (E25). At least **two published/upstream baseline F1 claims**
survive only as copied documentation values, without the originating evaluation
artifacts in this repository. Several planned full eager-PyTorch pruning outputs
are absent, but the source document labels them pending; they must not be counted
as missing completed runs.

The most important inconsistencies are:

1. Final reports call the selected pruned model `pruned_recovered`, although
   checkpoint selection identifies optimizer step 0; no recovery update is
   demonstrated.
2. `experiments/pruning/final_pruning_summary.md` still says full evaluation is
   pending, while later ORT/OpenVINO 111-image evaluations exist. A standalone
   eager-PyTorch full evaluation remains absent.
3. Historical model-family metrics cover 105 images, current final metrics cover
   111; these are not interchangeable.
4. The physical-device bundle says Raspberry Pi 4, while TVM metadata targets
   Cortex-A76 and uses `rpi5` paths. The exact board is unresolved.
5. Smoke (3-image), quick (32-image), historical (105-image), current final
   (111-image), and Pi diagnostic (3-image/228-patch) results coexist and require
   explicit scope labels.

Primary overview evidence: `reports/thesis_experiment_inventory.md`; final
completion evidence:
`experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/run_status.json -> status = "complete"`,
`validation.all_fixed_runs_images = [111 × 8]`, and
`validation.all_fixed_runs_patches = [8500 × 8]`.

## B. Experiment Inventory

The rows deliberately preserve distinct experiment families and do not merge
similarly named runs. Exact subrun detail is in the cited directories.

| Run | Type | Model | Backend | Precision | Dataset | Threshold | Status | Evidence | Main artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01 historical family evaluation | validation/evaluation | FCOS_18, x50, x101 | PyTorch eager CPU | FP32 | historical test, 105 images | model-specific (FCOS_18 0.584) | complete, superseded split | A | `experiments/eval_guide/results/*_xvalidation_test_cpu_metrics.json` |
| E02 CPU_4C_LIMITED family | full historical benchmark | FCOS_18, x50, x101 | PyTorch eager CPU | FP32 | `data/midog_test`, 20 images | config-specific | complete | B | `logs/*_CPU_4C_LIMITED.log`; `results/raw_predictions/*/` |
| E03 eager smoke/quick | smoke + quick benchmark | FCOS_18 | PyTorch eager CPU/CUDA | FP32 | 3 / 32 images; quick 2,486 patches | 0.584 | complete for diagnostic scope | A | `experiments/eval_guide/results/*smoke*`; `*test_quick_cpu_measured*` |
| E04 compile quick | smoke + quick benchmark | FCOS_18 | torch.compile/Inductor CPU | FP32 | 3 / 32 images | 0.584 | complete for quick scope | A | `experiments/eval_guide/results/*pytorch_compile*` |
| E05 ORT FP32 family | smoke/quick/final benchmark | FCOS_18, Pruned60 | ORT CPUExecutionProvider | FP32 | 3, 32, and 111 images | 0.584 | complete | A | `experiments/quantization/ptq_20260816T1815Z/*/final_test_fp32_fixed/` |
| E06 OpenVINO FP32 family | smoke/quick/final benchmark | FCOS_18, Pruned60 | OpenVINO CPU | FP32 | 3, 32, and 111 images | 0.584 | complete | A | `experiments/quantization/openvino_ptq_20260819T0000Z/*/final_test_fixed_fp32/` |
| E07 TensorRT pilot | smoke + quick benchmark | FCOS_18 | TensorRT/CUDA | FP32 | 3 / 32 images | 0.584 | complete for pilot scope | A | `experiments/eval_guide/results/*cuda_tensorrt*` |
| E08 early pruning | development/smoke | FCOS_18 masked/structured head 10% | PyTorch eager | FP32 | smoke / quick | 0.584 | completed pilots, obsolete | A/B | `experiments/pruning/evaluation/fcos18_depgraph_structural_10pct/`; `docs/pruning.md` |
| E09 scope sweep | smoke test sweep | FCOS_18 structural variants | PyTorch eager CPU | FP32 | 3 images | 0.584 | complete | A | `experiments/pruning/sweeps/fcos18_depgraph_speed_sweep/summary.json` |
| E10 ratio/boundary sweep | smoke + quick evaluation | FCOS_18 structural 20–70% | PyTorch eager CPU | FP32 | 3 / 32 images | 0.584 | complete | A | `experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary/summary.json`; `experiments/pruning/sweeps/fcos18_depgraph_fpn_head_extended_quick/summary.json` |
| E11 candidate selection | quick/subset evaluation | Pruned60 | PyTorch eager CPU | FP32 | 32 images, 2,486 patches | 0.584 | complete selection | A | `experiments/pruning/sweeps/fcos18_depgraph_fpn_head_extended_quick/`; `experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt` |
| E12 MobileNet distillation | development/pilot/quick | MobileNetV3 Small/Large | PyTorch CUDA training, CPU eval | FP32 | pilot; Small quick/3-image | 0.584 diagnostic | Small full training complete; Large pilot only | A/B | `experiments/distillation/pilot_runs/`; `experiments/distillation/runs/fcos_x101_to_mobilenetv3_small/` |
| E13 early Pruned60 recovery | development/recovery | supervised and KD Pruned60 | PyTorch CUDA | FP32 | old training split; partial quick output | 0.584 | partly completed | C | `experiments/distillation/runs/*pruned60*recovery/`; `experiments/distillation/pruned60_recovery_comparison.md` |
| E14 clean dense recovery | validation/recovery | Pruned60 supervised/KD | PyTorch CUDA | FP32 | train 314; validation 39; calibration 39 | N/A | training/validation complete; downstream final comparison absent | A | `experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/` |
| E15 frozen recovery LR sweep | validation/recovery | Pruned60 supervised/KD | PyTorch CUDA | FP32 | train + 39-image validation | N/A | mixed complete, early-stop, failure, prepared-only | A–D by subrun | `experiments/distillation/pruned60_frozen_backbone_recovery/runs/`; `reports/current_recovery_results_summary.json` therein |
| E16 seed replication | reproducibility/seed replication | Pruned60 frozen supervised | PyTorch CUDA | FP32 | train + 39-image validation | N/A | seeds 42/44 complete; 43 failed | A for 42/44; D for completed-43 claim | `experiments/distillation/pruned60_frozen_backbone_recovery/reports/supervised_lr3e-5_seed_reproducibility.json` |
| E17 threshold work | threshold sweep + calibration | baseline and Pruned60 | PyTorch/ORT/OpenVINO | INT8 for final calibration | quick test 32; final calibration 39 | grids; selected 0.583/0.546/0.600/0.565 | complete; quick sweep diagnostic only | A | `experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/threshold_selection.json` |
| E18 final ORT FP32 | final test + full benchmark | baseline and step-0 Pruned60 | ORT CPUExecutionProvider, 4 cores | FP32 | 111 images, 8,500 patches | 0.584 | complete | A | `experiments/quantization/ptq_20260816T1815Z/*/final_test_fp32_fixed/` |
| E19 final ORT INT8 | calibration + final test + benchmark | baseline and step-0 Pruned60 | ORT CPUExecutionProvider, 4 cores | INT8 | calibration 39; test 111/8,500 | fixed 0.584; calibrated 0.583/0.546 | complete | A | `experiments/quantization/ptq_20260816T1815Z/*/final_test_fixed/`; later calibrated metrics |
| E20 final OpenVINO FP32 | final test + full benchmark | baseline and step-0 Pruned60 | OpenVINO CPU, 4 cores | FP32 | 111 images, 8,500 patches | 0.584 | complete | A | `experiments/quantization/openvino_ptq_20260819T0000Z/*/final_test_fixed_fp32/` |
| E21 final OpenVINO INT8 | calibration + final test + benchmark | baseline and step-0 Pruned60 | OpenVINO/NNCF CPU, 4 cores | INT8 | calibration 39; test 111/8,500 | fixed 0.584; calibrated 0.600/0.565 | complete | A | `experiments/quantization/openvino_ptq_20260819T0000Z/*/final_test_fixed_int8/` |
| E22 FP16 path | development/prepared | FCOS exports | prepared runtime paths | FP16 | N/A | N/A | not executed | D | code references only; no FP16 results found |
| E23 physical-device comparison | full repeated diagnostic benchmark | baseline and raw Pruned60 | eager/ORT on aarch64 | FP32/INT8 | 3 images, 228 patches | 0.584 | six variants complete | A | `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/results/` |
| E24 TVM feasibility | failed/incomplete run | FCOS_18; Pruned60 planned | TVM Relax/LLVM Cortex-A76 | FP32 | one-patch validation intended | N/A | import/compile partial; inference interrupted; benchmark not started | C/D | `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/results/fcos18/tvm_llvm_fp32/20260819T175827Z/` |
| E25 Pi packaging/unsupported | development/prepared | deployment variants | scripts/package | mixed | intended 3-image subset | 0.584 | package complete; extra variants unexecuted | B/D | `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/MANIFEST.txt`; launchers |

## C. Final-Test / Evaluation Inventory

Times are total seconds for the complete 8,500-patch run. Calibrated rows reuse
the fixed INT8 inference timings. Full precision is retained from
`experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/final_comparison.json -> fixed_rows/calibrated_rows`.

| Model | Variant | Backend | Precision | Images | Patches | F1 | AP | Forward | E2E | Artifact | Confidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| FCOS_18 | baseline fixed | ORT | FP32 | 111 | 8500 | 0.8250497017892643 | 0.8859153985977173 | 3780.1057181580213 | 3913.469515539 | `experiments/quantization/ptq_20260816T1815Z/baseline/final_test_fp32_fixed/metrics.json`; `timing.json` | A |
| FCOS_18 | baseline fixed | ORT | INT8 | 111 | 8500 | 0.8296296296296295 | 0.8857691884040833 | 2682.9147991340337 | 2785.5696128419995 | `experiments/quantization/ptq_20260816T1815Z/baseline/final_test_fixed/metrics.json`; `timing.json` | A |
| FCOS_18 | baseline calibrated 0.583 | ORT | INT8 | 111 | 8500 | 0.8284686125549778 | 0.8857691884040833 | 2682.9147991340337 | 2785.5696128419995 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_calibrated/onnxruntime_int8/metrics.json` | A (cached rescore) |
| FCOS_18 | baseline fixed | OpenVINO | FP32 | 111 | 8500 | 0.8231803797468353 | 0.886191725730896 | 7605.406250312074 | 7675.307430016999 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_fixed_fp32/metrics.json`; `timing.json` | A |
| FCOS_18 | baseline fixed | OpenVINO | INT8 | 111 | 8500 | 0.8271954674220962 | 0.8769268989562988 | 4058.158076837177 | 4159.587845473001 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_fixed_int8/metrics.json`; `timing.json` | A |
| FCOS_18 | baseline calibrated 0.600 | OpenVINO | INT8 | 111 | 8500 | 0.8280307756290287 | 0.8769268989562988 | 4058.158076837177 | 4159.587845473001 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_calibrated/openvino_int8/metrics.json` | A (cached rescore) |
| FCOS_18 | Pruned60 step-0 fixed | ORT | FP32 | 111 | 8500 | 0.8230162027420024 | 0.8787636160850525 | 3436.415442693009 | 3523.5016467470004 | `experiments/quantization/ptq_20260816T1815Z/pruned_recovered/final_test_fp32_fixed/metrics.json`; `timing.json` | A |
| FCOS_18 | Pruned60 step-0 fixed | ORT | INT8 | 111 | 8500 | 0.8204912870039889 | 0.878639280796051 | 2465.5140217480202 | 2608.876823035999 | `experiments/quantization/ptq_20260816T1815Z/pruned_recovered/final_test_fixed/metrics.json`; `timing.json` | A |
| FCOS_18 | Pruned60 step-0 calibrated 0.546 | ORT | INT8 | 111 | 8500 | 0.8207091055600321 | 0.878639280796051 | 2465.5140217480202 | 2608.876823035999 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_calibrated/onnxruntime_int8/metrics.json` | A (cached rescore) |
| FCOS_18 | Pruned60 step-0 fixed | OpenVINO | FP32 | 111 | 8500 | 0.8241123038810898 | 0.8788632750511169 | 5950.6957847679805 | 6021.392172322001 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_fixed_fp32/metrics.json`; `timing.json` | A |
| FCOS_18 | Pruned60 step-0 fixed | OpenVINO | INT8 | 111 | 8500 | 0.8164556962025314 | 0.8709028959274292 | 6065.249473927266 | 6141.891476417 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_fixed_int8/metrics.json`; `timing.json` | A |
| FCOS_18 | Pruned60 step-0 calibrated 0.565 | OpenVINO | INT8 | 111 | 8500 | 0.8202594193946879 | 0.8709028959274292 | 6065.249473927266 | 6141.891476417 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_calibrated/openvino_int8/metrics.json` | A (cached rescore) |

Sanity check: the baseline ORT FP32 raw file records
`metrics.json -> aggregates.true_positives = 2075`, `false_positives = 488`,
`false_negatives = 392`; these reproduce its precision/recall/F1. Its
`runtime.json -> execution_providers = ["CPUExecutionProvider"]`,
`det_thresh = 0.584`, `num_images = 111`; its `timing.json -> num_patches = 8500`.

## D. Documentation vs Repository

| Documented claim | Documentation source | Repository evidence | Assessment |
| --- | --- | --- | --- |
| Current final comparison comprises eight fixed runs on 111 images/8,500 patches | `reports/thesis_experiment_inventory.md`, E18-E21 | combined `run_status.json -> validation`; eight raw result directories | VERIFIED |
| Four calibrated INT8 thresholds are 0.583, 0.546, 0.600, 0.565 | combined `final_report.md` | combined `threshold_selection.json -> selections[*].threshold`; each has dataset/model/prediction SHA-256 | VERIFIED |
| Calibration is disjoint from test | inventory and recovery split audit | combined `run_status.json -> validation.calibration_test_overlap_count = 0`; `threshold_selection.json -> dataset_sha256` | VERIFIED |
| `pruned_recovered` is a recovered model | final comparison/report naming | `experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/selected_recovery_checkpoint.json` selects optimizer step 0 | CONFLICT |
| Final pruning full evaluation and four-core benchmark are pending | `experiments/pruning/final_pruning_summary.md:10-13,53-55,96-102` | later ORT/OpenVINO final directories complete; no eager `.pt` full run at the paths proposed in that document | VERIFIED WITH MINOR ISSUE (stale/generalized wording) |
| Selected Pruned60 was not recovery fine-tuned | `experiments/pruning/final_pruning_summary.md:155` | step-0 selection and final export provenance | VERIFIED |
| Historical FCOS family values are final/current-test values | upstream/copied result prose and old eval outputs | `experiments/eval_guide/results/*xvalidation_test_cpu_metrics.json` have 105 cases; current files have 111 | CONFLICT if presented without historical qualifier |
| MobileNetV3-Small result is still TBD | `experiments/distillation/distillation_experiment_summary.md:7` | `experiments/distillation/runs/fcos_x101_to_mobilenetv3_small/quick_evaluation_summary.md` and completed run files | DOCUMENTATION MISSING |
| MobileNetV3-Large full result exists/is expected | same summary line 8 | pilot artifacts exist; no complete full-run result found | INSUFFICIENT EVIDENCE |
| Early supervised/KD Pruned60 comparison is complete | `experiments/distillation/pruned60_recovery_comparison.md` | every comparison cell remains `pending`; training artifacts exist | PARTIALLY VERIFIED |
| OpenVINO INT8 unavailable | earlier `experiments/quantization/ptq_20260816T1815Z/run_status.json` | later OpenVINO status is complete and raw outputs exist | VERIFIED WITH MINOR ISSUE (historically true, now superseded) |
| Physical benchmark ran on Raspberry Pi 4 | `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/README_RPI4.md` and directory name | physical aarch64 outputs exist, but TVM target says Cortex-A76 and paths contain `rpi5` | PARTIALLY VERIFIED |
| Physical-device comparison is a final quality evaluation | any unqualified use of Pi metrics | bundle outputs are only 3 images/228 patches, one warm-up plus three measured runs | CONFLICT if described as full-test quality |
| FP16 was evaluated | benchmarking/pruning future-work references | no FP16 metrics/timing/result directory found | INSUFFICIENT EVIDENCE |
| TVM benchmark completed | TVM preparation/attempt material | TVM record says `benchmark_started=false`; head inference was interrupted | CONFLICT if claimed complete |
| FCOS_x101 F1 0.7528 and FCOS_18 F1 0.7369 | `experiments/distillation/distillation_experiment_summary.md:5-6` (“reported by upstream README”) | no originating raw evaluation artifacts for those exact upstream claims located locally | ARTIFACT MISSING |
| Pruned60 quick F1 0.7801 vs baseline 0.8018 | `docs/pruning.md:246-247`; `experiments/pruning/final_pruning_summary.md:43-48` | extended-quick summary JSON/metrics and 32-image timing files | VERIFIED |

## E. Undocumented Experiments

The following real executions are absent from, or stale in, the primary tracked
documentation (`README.md`, `docs/*.md`, and older experiment summaries). They are
described in the untracked 2026-08-27 inventory, but that file is itself not a
stable tracked source at audit time (`git status --short` reports `?? reports/`).

- **Final ORT/OpenVINO 111-image matrix and calibration.** Strong evidence is the
  eight complete fixed directories, calibrated metric files, hashes, commands,
  and `status = complete`. Older pruning documentation still says final work is
  pending. Evidence: `experiments/quantization/ptq_20260816T1815Z/`,
  `experiments/quantization/openvino_ptq_20260819T0000Z/`, and
  `experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/`.
- **Physical-device six-variant benchmark.** All eager/ORT FP32/INT8 outputs for
  baseline and Pruned60 survive in
  `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/results/`; it is not covered by
  tracked top-level documentation.
- **TVM partial failure.** Import/compilation evidence and interruption survive in
  `.../results/fcos18/tvm_llvm_fp32/20260819T175827Z/`; it should be documented as
  failed/incomplete, never as a benchmark value.
- **Completed MobileNetV3-Small full training/diagnostic evaluation.** The central
  distillation summary is still TBD although the run and quick summary exist.
- **Later frozen-backbone LR and seed runs.** Older recovery summaries omit
  supervised 3e-6/3e-5/5e-5 and seed 44; use the newer JSON reports and run status
  files in `experiments/distillation/pruned60_frozen_backbone_recovery/`.
- **Duplicate/legacy OpenVINO quick families.** Both `openvino_cpu` and
  `openvino_cpu_fp32` result names survive under `experiments/eval_guide/results/`.
  They require runtime metadata to distinguish them and should be catalogued as
  quick, not final.

## F. Missing or Unlocatable Artifacts

- The original raw outputs supporting the copied upstream F1 values 0.7528
  (FCOS_x101) and 0.7369 (FCOS_18) were not located. This means “reported by
  upstream README,” not “independently reproduced here.” Source of claim:
  `experiments/distillation/distillation_experiment_summary.md:5-6`.
- The planned eager-PyTorch full baseline/Pruned60 outputs under
  `experiments/pruning/evaluation/fcos18_fpn_head_60pct_full_cuda/` and
  `...full_cpu4/` do not exist. Their source labels them pending, so evidence does
  **not** support saying they were performed. Source:
  `experiments/pruning/final_pruning_summary.md:53-135`.
- The clean-recovery `final_comparison.md` has no completed calibration/final-test
  outputs belonging to that recovery experiment. Later deployment runs use the
  selected step-0 checkpoint; they do not fill the intended recovery comparison.
- Early recovery has student/training artifacts but lacks the expected completed
  comparison metrics/benchmarks; `experiments/distillation/pruned60_recovery_comparison.md`
  remains pending. Classify as probably/partly performed, original comparison
  output incomplete—not “never run.”
- No executed FP16 output was found. Code support is not evidence of execution.

## G. Conflicting Results and Provenance Hazards

1. **Model identity:** “Pruned/recovered” in the final JSON/report versus
   **Pruned60 selected at optimizer step 0** in
   `dense_checkpoint_lr_sweep/selected_recovery_checkpoint.json`. Rename the
   presentation label; do not claim recovery gain.
2. **Dataset size:** 105 cases in historical
   `experiments/eval_guide/results/*xvalidation_test_cpu_metrics.json` versus 111
   cases/8,500 patches in final runtime files. Values from these populations must
   not share an unqualified “test” column.
3. **Pending versus complete:** `experiments/pruning/final_pruning_summary.md`
   predates E18-E21. It is correct only for the specifically proposed eager `.pt`
   paths, not for all full evaluations.
4. **Threshold meaning:** 0.584 is the fixed baseline policy; 0.583/0.546/0.600/
   0.565 were selected on a separate calibration set. The older quick threshold
   sweep used a 32-image test subset and is diagnostic; it cannot justify final
   operating-point selection.
5. **Runtime comparability:** PC “CPU_4C_LIMITED” means affinity/threads on a
   development PC; physical-device runs are separate; early quick runs and Pi
   runs use different image/patch counts. Speedups are valid only within matched
   groups.
6. **Hardware identity:** `rpi4` naming conflicts with Cortex-A76/`rpi5` TVM
   metadata. Record the actual board model and CPU obtained from the device.
7. **OpenVINO chronology:** the earlier ORT PTQ status records missing NNCF;
   `openvino_ptq_20260819T0000Z/run_status.json` later records completion. Both are
   true only with dates/environment context.
8. **OpenVINO anomaly:** Pruned60 OpenVINO INT8 forward time
   6065.249473927266 s is slower than its FP32 5950.6957847679805 s. Preserve the
   observation; do not generalize an INT8 speedup.
9. **Calibrated timing provenance:** calibrated rows inherit cached INT8 inference
   time; they are not separate timed executions. The report should state this.
10. **Stale summaries:** MobileNet and recovery TBD/pending tables coexist with
    later artifacts and can be mistaken for current state.

No contradictory numeric values were found between the raw final metrics/timing
files and the consolidated `final_comparison.json`; the conflicts are chiefly
identity, scope, chronology, and labeling.

## H. Reproducibility Gaps

| Important experiment | Preserved | Missing or ambiguous |
| --- | --- | --- |
| Final ORT FP32/INT8 | exact outputs, dataset path, thresholds, opset, provider, model files, config, timing, commands; quantized-model hashes in calibration manifest | host CPU model and full package/OS snapshot are not consistently embedded in every raw run; baseline FP32 runtime JSON has `torch_version = null` and `precision = null`; seed is not relevant to deterministic cached scoring but is not explicitly stated |
| Final OpenVINO FP32/INT8 | metrics/predictions/timing, IR files, OpenVINO 2026.1.0, NNCF 3.2.0, device/preset, completion status | exact CPU/OS provenance should be centralized; verify hashes for FP32 IR and source checkpoint are recorded alongside final table |
| Threshold calibration | 39-image CSV, dataset SHA-256, zero overlap, 501-point grid, criterion, selected thresholds, prediction/model hashes | exact metric-code commit/hash and Python/dependency lock are not included in the selection rows |
| Pruned60 final model | `.pt`, ONNX/IR/INT8 artifacts and some hashes; pruning config and structural metadata | final report obscures step-0 identity; a single provenance manifest linking original `.pt` SHA-256 → selected checkpoint → ONNX → IR/QDQ is missing |
| Historical 105-image family evaluation | metrics and configs | exact dataset revision/split manifest, command, host/thread settings, and checkpoint hashes are incomplete; x50/x101 lack current 111-image reevaluation |
| Physical-device runs | per-run metrics/timing/runtime and logs; thread counts and subset | exact board model is contradictory; OS/kernel, thermal/throttling state, power mode, CPU governor and stable artifact hashes should be recorded; subset is only diagnostic |
| Recovery LR/seed experiments | configs, checkpoints, validation metrics, statuses for many subruns | no accepted post-update candidate; failed/early-stop semantics are scattered; seed 43 cannot support a three-seed reproducibility estimate |
| MobileNetV3 | configs, pilot/full-Small artifacts, logs | final threshold calibration and full 111-image quality/runtime absent; main summary remains stale |
| TVM | target/version/compile artifacts and failure context | no successful numerical validation, no inference timing, no completion marker; hardware identity unresolved |

The metric implementation is discoverable through
`src/benchmark/midog_guide_adapter.py` and the vendored MIDOG guide, but exact
reproduction should pin the Git commit of both repositories and the dependency
environment. Absolute paths embedded in manifests also reduce portability.

## I. Recommended Documentation Changes

### 1. Must fix before thesis submission

1. Replace every final-table `pruned_recovered` label with **Pruned60,
   recovery-selected step 0 (no effective recovery update)**, and cite the
   selection JSON.
2. Separate historical 105-image, quick 32-image, smoke/Pi 3-image, and current
   111-image/8,500-patch results in headings and table captions.
3. Update `experiments/pruning/final_pruning_summary.md` or mark it explicitly
   superseded: later full ORT/OpenVINO runs completed, but eager `.pt` runs did not.
4. State that calibrated INT8 metrics are frozen-threshold rescoring of cached
   predictions and reuse the fixed-run timing.
5. Resolve the physical board identity from device records or describe it only as
   “physical aarch64 Pi-class device”; do not assert Pi 4 while Cortex-A76 evidence
   remains.
6. Cite raw metrics and timing paths for every thesis result, not only generated
   summary Markdown. Track/preserve the ignored final artifact set or archive it
   with hashes; currently the key quantization and post-Pi directories are ignored.

### 2. Should fix

1. Mark stale MobileNet/recovery summaries as superseded and link the newest
   machine-readable reports.
2. Add a canonical manifest with run ID, UTC time, command, Git commits, dataset
   hash, source checkpoint hash, derived model hashes, runtime versions, CPU,
   affinity/thread settings, and raw output paths.
3. Label the quick test-subset threshold sweep “diagnostic, not calibration.”
4. Record failed/prepared variants explicitly: seed 43 numerical failure, LR 1e-4
   failure, unexecuted KD rates, FP16 unexecuted, and TVM benchmark not started.
5. Report the Pruned60 OpenVINO INT8 slowdown rather than implying INT8 always
   accelerates inference.

### 3. Nice to have

1. Generate checksums for every retained checkpoint/export/result and a portable
   relative-path manifest.
2. Archive environment lock files and `/proc/cpuinfo`/OS/governor/thermal metadata
   with deployment benchmarks.
3. Add a machine-generated index distinguishing inference executions from cached
   rescoring and warm-ups from measured repetitions.
4. Move legacy/smoke outputs under visibly named archival directories and add a
   one-line `SUPERSEDED_BY` marker to old reports.

## Provenance notes

- The consolidated final report is derived evidence; raw directories listed in
  Section C are the primary evidence.
- File modification times were used only to aid chronology and never as proof of
  completion.
- Git history confirms many pruning/quick artifacts were committed together in
  commit `5dec13f89f6cc77d8217faeb31091c6ffaca4a4f` (2026-08-03), but the final
  quantization/recovery/Pi archives are ignored/untracked in the current worktree.
- No evidence was found that missing eager, FP16, or TVM final benchmarks were
  silently completed elsewhere in the visible repository. The defensible wording
  is “execution cannot currently be independently verified” or “prepared but no
  completed artifact located,” not “never performed.”
