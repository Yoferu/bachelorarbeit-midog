# Canonical Thesis Experiment Inventory

Inventory date: 2026-08-27. This is an artifact/status inventory, not an interpretation of scientific results.

## Evidence policy and status rules

- Generated metrics, timing files, predictions, checkpoints, run-status files, and logs take precedence over plans and prose summaries.
- A script/config alone is **prepared but not executed**. Smoke, pilot, quick, or three-image output does not count as a final evaluation or final benchmark.
- **Final quality evaluation** means a completed evaluation on the repository's complete current 111-image test split. Older 105-image evaluations and the 20-image `midog_test` benchmark are retained as historical baselines, not current final tests.
- **Final runtime benchmark** means a completed deployment-oriented full-test or explicitly final repeated benchmark. Quick 32-image and diagnostic three-image timings are not final.
- Unless stated otherwise, MIDOG evaluation uses 1024-pixel patches, overlap 0.3, NMS 0.3, batch size 1, and detection threshold 0.584.
- `CPU_4C_LIMITED` denotes a development-PC approximation (`taskset -c 0-3`, four threads), not physical Raspberry Pi execution.

## Canonical findings that affect thesis structure

1. The current 111-image test split **was evaluated**. The strongest evidence is the complete ORT/OpenVINO quantization comparison: eight fixed-threshold runs each report 111 images and 8,500 patches, followed by calibrated-threshold scoring.
2. Threshold calibration used the separate 39-image `calibration_seed42.csv` derived from the original training split. Its overlap with the 111-image test split is zero. Test data was not used to select the four final INT8 thresholds.
3. The post-test Raspberry Pi bundle contains completed physical-device runs for six variants: FCOS_18 and Pruned60 under PyTorch eager FP32, ORT FP32, and ORT INT8. Each final Pi benchmark uses a three-image/228-patch diagnostic subset, one warm-up, and three measured repetitions.
4. The physical device is confirmed to be a **Raspberry Pi 5 with four Cortex-A76 CPU cores**. Bundle and path names beginning with `rpi4_*` are incorrect legacy naming retained only to preserve artifact provenance; they do not identify the measured hardware.
5. The artifact called `pruned_recovered` in final quantization reports resolves to the selected clean-recovery checkpoint at optimizer step 0. It is structurally pruned Pruned60 with no effective recovery update. The label is misleading and conflicts with the actual checkpoint-selection record.
6. OpenVINO INT8 has completed, including 111-image fixed and calibrated evaluations. The older ORT PTQ status saying OpenVINO was unavailable is historically correct for that earlier environment but is superseded by the later OpenVINO/NNCF run.
7. No executed FP16 experiment was found. FP16 appears only as supported/prepared code paths and documentation.

## Experiment inventory

### E01 — Original FCOS model-family baseline evaluation

- **Purpose:** establish quality and scale baselines for the available pretrained FCOS variants.
- **Models:** FCOS_18 (ResNet-18-FPN), FCOS_x50 (ResNeXt-50-32x4d-FPN), FCOS_x101 (ResNeXt-101-32x8d-FPN).
- **Optimization / runtime:** none; PyTorch eager CPU through the MIDOG guide adapter.
- **Data:** historical full `test` evaluation containing 105 images (not the later 111-image version).
- **Hardware:** development PC CPU; no four-core constraint is recorded by these metric files.
- **Main configuration:** model-specific thresholds (FCOS_18 0.584; x101 0.548; x50 config-specific), 1024 patches, overlap/NMS 0.3.
- **Status:** **completed**, but historical/superseded as the final test definition changed.
- **Final quality / runtime:** quality **yes for the historical 105-image split; no for current 111-image finality**. Final runtime **no**.
- **Outputs:** `experiments/eval_guide/results/FCOS_18_midogpp_xvalidation_test_cpu_metrics.json`, corresponding `FCOS_x50_*` and `FCOS_x101_*` files; configs in `experiments/eval_guide/configs/`.
- **Superseded by:** E18–E21 for current 111-image baseline quality/runtime; x50/x101 have no newer final evaluation.
- **Thesis placement:** **main thesis results** for architecture selection, with the 105-vs-111 split caveat; detailed per-tumor metrics in **appendix only**.

### E02 — Original `CPU_4C_LIMITED` model benchmarks

- **Purpose:** compare raw CPU deployment cost of FCOS_18, FCOS_x50, and FCOS_x101.
- **Models / optimization / runtime:** the three pretrained FCOS models; no optimization; original CLI PyTorch eager CPU.
- **Data:** `data/midog_test/images`, 20 images; this is not the 111-image final test split.
- **Hardware:** development PC constrained with `taskset -c 0-3`; batch 1, workers 0.
- **Main configuration:** 1024 patches, overlap 0.3, performance measurement; one run per model.
- **Status:** **completed** (all logs end with exit status 0 and 20 processed images).
- **Final quality / runtime:** quality **no** (raw detections only); final runtime **yes for the historical 20-image CPU_4C_LIMITED comparison**, not for the final pipeline.
- **Outputs:** `logs/FCOS_18_CPU_4C_LIMITED.log`, `logs/FCOS_x50_CPU_4C_LIMITED.log`, `logs/FCOS_x101_CPU_4C_LIMITED.log`, and `results/raw_predictions/*_CPU_4C_LIMITED/`.
- **Superseded by:** E05 and E18–E21 for matched adapter/full-test runtime comparisons.
- **Thesis placement:** **main thesis results** as initial model-selection runtime evidence; raw prediction details **appendix only**.

### E03 — FCOS_18 PyTorch eager CPU/CUDA benchmark family

- **Purpose:** establish the adapter-level runtime baseline and validate CPU/GPU execution.
- **Models / optimization / runtime:** FCOS_18; none; PyTorch eager on development-PC CPU and CUDA.
- **Data:** three-image smoke and 32-image quick subsets; repeated quick runs use one warm-up plus three measurements.
- **Hardware:** CPU runs use four threads; GPU runs use CUDA. Quick CPU runs are development-PC approximations, not Raspberry Pi.
- **Main configuration:** 32 images, 2,486 patches for quick; threshold 0.584.
- **Status:** **completed** for smoke and quick CPU/CUDA runs.
- **Final quality / runtime:** **no / no**; these are diagnostic/quick results.
- **Outputs:** `experiments/eval_guide/results/FCOS_18_midogpp_xvalidation_smoke_*` and `FCOS_18_midogpp_xvalidation_test_quick_cpu_measured*`; logs in `experiments/eval_guide/logs/`.
- **Superseded by:** E18–E21 and E22–E24.
- **Thesis placement:** **methodology only**; selected quick runtime comparison may be **appendix only**.

### E04 — `torch.compile` CPU experiment

- **Purpose:** test Inductor compilation as a drop-in FCOS_18 CPU optimization.
- **Models / optimization / runtime:** FCOS_18; graph compilation; PyTorch `torch.compile`/Inductor CPU.
- **Data:** smoke and 32-image quick test subset; warm-up plus three measured repetitions.
- **Hardware:** development PC, four CPU threads; not Raspberry Pi.
- **Main configuration:** batch 1, 2,486 quick-set patches, threshold 0.584.
- **Status:** **completed**. Effective backend metadata records `pytorch_compile`/`inductor`.
- **Final quality / runtime:** **no / no**; repeated quick benchmark only.
- **Outputs:** `experiments/eval_guide/results/*pytorch_compile*` and matching logs.
- **Superseded by:** later ORT/OpenVINO final runs (E18–E21) for deployment selection.
- **Thesis placement:** **main thesis results** as a runtime-optimization comparison, with detailed repetitions in **appendix only**.

### E05 — ONNX Runtime FP32 CPU experiment

- **Purpose:** evaluate ONNX export correctness and CPU inference speed.
- **Models / optimization / runtime:** FCOS_18 initially; later FCOS_18 and Pruned60; ONNX export, ORT CPUExecutionProvider FP32.
- **Data:** smoke/32-image quick set in the early family; 111-image test in E18; three-image physical-device set in E23.
- **Hardware:** early/final PC runs use four-core CPU constraints; Pi runs are separate.
- **Main configuration:** opset 18, fixed 3×1024×1024 patch input, batch 1.
- **Status:** **completed** across quick and final PC evaluations.
- **Final quality / runtime:** **yes / yes** through E18's 111-image runs.
- **Outputs:** `experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_*.onnx`, `experiments/eval_guide/results/*onnxruntime_cpu*`, and `experiments/quantization/ptq_20260816T1815Z/*/final_test_fp32_fixed/`.
- **Superseded by:** E18 supplies the canonical final PC results; E23 supplies physical-device results.
- **Thesis placement:** **main thesis results**.

### E06 — OpenVINO FP32 CPU experiment

- **Purpose:** test OpenVINO conversion and CPU execution against eager/ORT.
- **Models / optimization / runtime:** FCOS_18 and later Pruned60; OpenVINO IR FP32 on CPU.
- **Data:** early smoke/32-image quick; later complete 111-image test.
- **Hardware:** development PC, four-core constrained final runs.
- **Main configuration:** OpenVINO 2026.1.0, CPU device, 1024 patch, batch 1.
- **Status:** **completed**. Two early quick artifact names (`openvino_cpu` and explicit `openvino_cpu_fp32`) coexist and are potentially confusing but both contain effective-backend validation.
- **Final quality / runtime:** **yes / yes** through E20.
- **Outputs:** `experiments/eval_guide/results/*openvino_cpu*`, exported IR in `experiments/eval_guide/exported_models/`, and `experiments/quantization/openvino_ptq_20260819T0000Z/*/final_test_fixed_fp32/`.
- **Superseded by:** E20 for canonical final FP32 results.
- **Thesis placement:** **main thesis results**.

### E07 — CUDA TensorRT pilot

- **Purpose:** validate a GPU deployment backend as an ancillary runtime experiment.
- **Models / optimization / runtime:** FCOS_18; TensorRT compilation/inference on CUDA.
- **Data:** smoke and 32-image quick test subset, repeated measurements.
- **Hardware:** development GPU; unrelated to CPU/Raspberry Pi target.
- **Main configuration:** same adapter policy, fixed patch shape.
- **Status:** **completed** for smoke/quick only.
- **Final quality / runtime:** **no / no**.
- **Outputs:** `experiments/eval_guide/results/*cuda_tensorrt*` and matching logs.
- **Superseded by:** none; out of the final CPU deployment branch.
- **Thesis placement:** **appendix only** or **not included** if the thesis scope is CPU edge deployment.

### E08 — Early masked/structured-head pruning and fine-tuning pilots

- **Purpose:** explore pruning machinery before dependency-aware physical channel removal.
- **Models / optimization / runtime:** FCOS_18; structured head L2 pruning, masked/early fine-tuning variants; PyTorch eager CPU/CUDA.
- **Data:** smoke and 32-image quick subsets.
- **Hardware:** development PC CPU/CUDA.
- **Main configuration:** nominal 10% head pruning; several smoke and recovery checkpoints.
- **Status:** **completed pilots**, now **obsolete/superseded**; some repeated smoke attempts exist.
- **Final quality / runtime:** **no / no**.
- **Outputs:** `experiments/pruning/results/fcos18_structured_head_l2_10pct_*`, `experiments/eval_guide/results/*structured_head_l2_10pct*`.
- **Superseded by:** E09–E11 DepGraph structural pruning.
- **Thesis placement:** **methodology only** or **appendix only**.

### E09 — DepGraph pruning-scope sweep

- **Purpose:** determine which FCOS components can be physically pruned while preserving executable dependencies.
- **Models / optimization / runtime:** FCOS_18; Torch-Pruning/DepGraph magnitude pruning; PyTorch eager CPU.
- **Data:** three-image smoke test subset.
- **Hardware:** development PC CPU.
- **Main configuration:** 20% variants over backbone/full structural, head-only, FPN+head, and backbone+FPN+head scopes; fixed output layers excluded.
- **Status:** **completed**. Head-only made no physical change; aggressive full-scope variants collapsed detections.
- **Final quality / runtime:** **no / no**.
- **Outputs:** `experiments/pruning/sweeps/fcos18_depgraph_speed_sweep/summary.{md,csv,json}` and `artifact_logs/`.
- **Superseded by:** E10–E11, which retain FPN+head as the useful scope.
- **Thesis placement:** **main thesis results** for scope selection; full variant detail **appendix only**.

### E10 — DepGraph pruning-ratio and boundary sweeps

- **Purpose:** find the useful pruning-ratio region and speed/quality boundary.
- **Models / optimization / runtime:** FCOS_18 FPN+head and full-structural variants; dependency-aware magnitude pruning; PyTorch eager CPU.
- **Data:** smoke subset for boundary tests, then 32-image quick test subset for extended FPN+head ratios.
- **Hardware:** development PC CPU; quick runs are not final CPU_4C_LIMITED benchmarks.
- **Main configuration:** full structural 40/50%; FPN+head 20/30/40%, extended to 50/60/70%; fixed outputs/backbone excluded for FPN+head.
- **Status:** **completed**.
- **Final quality / runtime:** **no / no**; quick/sensitivity evidence only.
- **Outputs:** `experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary/`, `experiments/pruning/sweeps/fcos18_depgraph_fpn_head_extended_quick/`, and `experiments/pruning/diagnostics/fpn_head_prediction_comparison/`.
- **Superseded by:** E11 candidate selection.
- **Thesis placement:** **main thesis results**; diagnostic tie analysis **appendix only**.

### E11 — Selected Pruned60 model

- **Purpose:** select a structurally smaller deployment candidate.
- **Models / optimization / runtime:** `FCOS_18_depgraph_fpn_head_60pct.pt`; 60% channel pruning within eligible FPN/head layers; PyTorch eager and later ONNX/ORT/OpenVINO.
- **Data:** selection used the 32-image quick test subset; later final uses are listed in E18–E24.
- **Hardware:** candidate selection on development PC CPU.
- **Main configuration:** 15.604807M parameters, about 395.736G MACs, fixed output layers and ResNet-18 backbone excluded.
- **Status:** **completed selection**. The 50% model is backup; 70% is boundary. The old `final_pruning_summary.md` statement that full evaluation was pending is superseded by later 111-image backend evaluations, although those use the step-0 selected clean checkpoint.
- **Final quality / runtime:** **yes / yes indirectly** through later ORT/OpenVINO final evaluations; the original raw `.pt` has no standalone 111-image eager final benchmark.
- **Outputs:** `experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt`, `experiments/pruning/final_pruning_summary.md`, extended sweep outputs.
- **Superseded by:** deployment-format step-0 copy in E18–E21; no genuinely recovered checkpoint supersedes it.
- **Thesis placement:** **main thesis results**.

### E12 — MobileNetV3 distillation pilots and full Small run

- **Purpose:** replace the backbone/model with a smaller MobileNetV3 student trained from FCOS_x101.
- **Models / optimization / runtime:** FCOS_x101 teacher; MobileNetV3-Small-FPN and Large-FPN students; supervised plus output KD; training on CUDA, evaluation/benchmark with PyTorch eager CPU.
- **Data:** smoke/pilot training subsets for Small/Large; one full Small training run; quality/runtime only on quick/three-image diagnostics.
- **Hardware:** development GPU for training, development CPU (four threads, no affinity in matched comparison) for timing.
- **Main configuration:** seed 42; pilot/smoke configs plus `fcos_x101_to_mobilenetv3_small.yaml` and Large equivalents.
- **Status:** Small **completed** training with quick evaluation and repeated three-image benchmark. Large has **completed pilots/smoke only**; no full final Large result. The summary file containing TBD fields is stale.
- **Final quality / runtime:** **no / no**. Small quality at the fixed FCOS_18 threshold is diagnostic and no threshold sweep was run.
- **Outputs:** `experiments/distillation/pilot_runs/`, `experiments/distillation/runs/fcos_x101_to_mobilenetv3_small/`, and `experiments/comparisons/three_image_model_comparison_20260713T010547/`.
- **Superseded by:** Pruned60 path E11/E13–E21 for final deployment.
- **Thesis placement:** **main thesis results** as an alternative architecture branch; Large pilots and smoke retries **appendix only**.

### E13 — Early Pruned60 supervised and KD recovery

- **Purpose:** recover quality after pruning using ordinary fine-tuning or FCOS_x101 knowledge distillation.
- **Models / optimization / runtime:** Pruned60 student; supervised detection loss versus supervised+output KD; CUDA training, PyTorch evaluation.
- **Data:** original train split without later clean validation/calibration holdouts; quick evaluation artifacts are incomplete/pilot-level.
- **Hardware:** development GPU for training.
- **Main configuration:** nominal 20 epochs; unfrozen backbone; supervised and KD configs under `experiments/distillation/configs/`.
- **Status:** training artifacts exist, but the generated `pruned60_recovery_comparison.md` remains all `pending`; classify as **partially completed pilot**, not a completed comparison.
- **Final quality / runtime:** **no / no**.
- **Outputs:** `experiments/distillation/runs/fcos18_pruned60_supervised_recovery/`, `experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation/`, `experiments/distillation/pruned60_recovery_comparison.md`.
- **Superseded by:** E14 clean recovery and E15 frozen-backbone recovery.
- **Thesis placement:** **methodology only** or **appendix only**.

### E14 — Clean unfrozen Pruned60 recovery and dense-checkpoint LR sweep

- **Purpose:** rerun supervised/KD recovery with train-derived validation and calibration holdouts and dense checkpoint evaluation.
- **Models / optimization / runtime:** Pruned60; unfrozen supervised LR 1e-4, 3e-5, 1e-5 and unfrozen KD 1e-4; CUDA training and validation AP selection.
- **Data:** 314-image train, 39-image validation, 39-image calibration, all disjoint from the 111-image test split.
- **Hardware:** development GPU; no final CPU benchmark.
- **Main configuration:** seed 42; dense validation checkpoints; best AP with tolerance then earliest step.
- **Status:** **completed training/validation**, but downstream threshold calibration and final-test directories are absent; `final_comparison.md` remains pending. The selected checkpoint is LR 1e-4 at optimizer step 0.
- **Final quality / runtime:** **no / no** for this recovery experiment.
- **Outputs:** `experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/`, split audit under `data/`, and `final_comparison.md`.
- **Superseded by:** E15 for recovery methodology; its step-0 checkpoint is nevertheless reused/mislabeled as `pruned_recovered` in E18–E21.
- **Thesis placement:** **main thesis results** for the negative recovery finding; dense checkpoint details **appendix only**.

### E15 — Frozen-backbone supervised/KD recovery LR sweep

- **Purpose:** test whether freezing the unchanged backbone stabilizes recovery and identify a learning rate.
- **Models / optimization / runtime:** Pruned60; frozen-backbone supervised and KD training; CUDA training, PyTorch validation.
- **Data:** clean train and 39-image validation split; no test or calibration scoring was launched by this experiment family.
- **Hardware:** development GPU.
- **Main configuration:** supervised LR 1e-6, 3e-6, 1e-5, 3e-5, 5e-5, 1e-4; KD LR 1e-6 and 1e-5; seed 42; common optimizer-step grid.
- **Status:** **partially completed family**. Supervised 3e-6/1e-5/3e-5/5e-5 completed; 1e-6 early-stopped; 1e-4 failed numerical instability at step 345. KD 1e-5 completed and 1e-6 early-stopped. KD 3e-6/3e-5 were prepared but not executed. The 1e-3 supervised full run was not executed (smoke only).
- **Final quality / runtime:** **no / no**.
- **Outputs:** `experiments/distillation/pruned60_frozen_backbone_recovery/runs/`, `reports/supervised_learning_rate_comparison.*`, `reports/current_recovery_results_summary.*`.
- **Superseded by:** no completed final recovery candidate; E16 tests reproducibility of supervised 3e-5.
- **Thesis placement:** **main thesis results**; failed/prepared variants **appendix only**.

### E16 — Seed/reproducibility and determinism checks

- **Purpose:** check repeatability of the selected frozen-supervised LR 3e-5 and deterministic inference.
- **Models / optimization / runtime:** Pruned60 frozen supervised recovery; seeds 42, 43, 44; CUDA training. Separate repeated PyTorch inference determinism check.
- **Data:** same train/39-image validation split; determinism predictions use the validation setup.
- **Hardware:** development GPU for training, local inference environment for determinism.
- **Main configuration:** common validation grid; deterministic algorithms requested with `warn_only=True`.
- **Status:** **partially completed**. Seeds 42 and 44 completed; seed 43 failed numerical instability at step 397. The report correctly labels reproducibility incomplete. Two inference prediction files are byte-identical by SHA-256.
- **Final quality / runtime:** **no / no**.
- **Outputs:** `reports/supervised_lr3e-5_seed_reproducibility.*`, `runs/supervised_lr3e-5_seed{42,43,44}/`, `determinism_checks/`.
- **Superseded by:** none; three comparable completed training seeds were not obtained.
- **Thesis placement:** **main thesis results** for reproducibility status; detailed checkpoints **appendix only**.

### E17 — Threshold-sweep pilots and calibration protocol

- **Purpose:** inspect threshold sensitivity and establish a leakage-free operating-point selection protocol.
- **Models / optimization / runtime:** early Pruned60 variants under PyTorch plus final baseline/Pruned60 INT8 under ORT/OpenVINO; post-hoc confidence-threshold optimization.
- **Data:** early pruning sweep uses the 32-image quick **test subset** and is diagnostic only. Final calibration uses the separate 39-image `calibration_seed42.csv`, not test; 501 thresholds from 0.300 to 0.800.
- **Hardware:** development PC CPU; calibration inference uses the corresponding ORT/OpenVINO backend.
- **Main configuration:** maximize aggregate calibration F1, exact ties choose the lowest threshold. Selected thresholds: baseline ORT INT8 0.583; Pruned60 ORT INT8 0.546; baseline OpenVINO INT8 0.600; Pruned60 OpenVINO INT8 0.565.
- **Status:** pruning threshold sweep **completed diagnostic**; final backend calibration **completed**.
- **Final quality / runtime:** calibration itself **no / no**; frozen thresholds are evaluated on final test in E19/E21.
- **Outputs:** `experiments/pruning/sweeps/fcos18_depgraph_fpn_head_threshold_sweep_quick/`, `experiments/quantization/openvino_ptq_20260819T0000Z/*/calibration/`, and `experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/threshold_selection.json`.
- **Superseded by:** final leakage-free calibration supersedes test-subset threshold diagnostics for operating-point claims.
- **Thesis placement:** **methodology only** for protocol; selected thresholds and calibrated comparison in **main thesis results**.

### E18 — Final ORT FP32 evaluation and runtime benchmark

- **Purpose:** provide the unquantized ONNX baseline for final quantization/combined-optimization comparison.
- **Models / optimization / runtime:** FCOS_18 and step-0 Pruned60 (called `pruned_recovered`); ONNX FP32; ORT CPUExecutionProvider.
- **Data:** complete 111-image test set, 5,383 annotation rows, 8,500 patches.
- **Hardware:** development PC, cores 0–3 / four threads.
- **Main configuration:** threshold 0.584, batch 1, overlap/NMS 0.3.
- **Status:** **completed**.
- **Final quality / runtime:** **yes / yes**.
- **Outputs:** `experiments/quantization/ptq_20260816T1815Z/{baseline,pruned_recovered}/final_test_fp32_fixed/` and consolidated report in `experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/`.
- **Superseded by:** not superseded as FP32 reference; E19 is the optimized ORT candidate.
- **Thesis placement:** **main thesis results**.

### E19 — Final ORT INT8 PTQ evaluation, calibration, and runtime

- **Purpose:** measure static post-training INT8 quantization for baseline and Pruned60.
- **Models / optimization / runtime:** FCOS_18 and step-0 Pruned60; static QDQ, per-channel QInt8 weights/QUInt8 activations; ORT CPUExecutionProvider.
- **Data:** calibration uses 39 disjoint calibration images (300 selected patches for quantization; complete calibration predictions for threshold sweep); final fixed and calibrated scoring uses all 111 test images/8,500 patches.
- **Hardware:** development PC, four-core constrained CPU.
- **Main configuration:** opset 18; fixed threshold 0.584 plus frozen calibrated thresholds 0.583/0.546.
- **Status:** **completed**. PyTorch static INT8 and compiled INT8 were not implemented/supported and must not be presented as executed.
- **Final quality / runtime:** **yes / yes**.
- **Outputs:** `experiments/quantization/ptq_20260816T1815Z/{baseline,pruned_recovered}/{int8,final_test_fixed}/`, calibration/final scoring under `openvino_ptq_20260819T0000Z/*/calibration/onnxruntime_int8/` and `final_test_calibrated/onnxruntime_int8/`.
- **Superseded by:** none; baseline ORT INT8 is the final general deployment compromise in the generated report.
- **Thesis placement:** **main thesis results**.

### E20 — Final OpenVINO FP32 evaluation and runtime benchmark

- **Purpose:** establish matched OpenVINO FP32 references for both model structures.
- **Models / optimization / runtime:** FCOS_18 and step-0 Pruned60; OpenVINO IR FP32, CPU device.
- **Data:** complete 111-image test set, 8,500 patches.
- **Hardware:** development PC, cores 0–3 / four threads.
- **Main configuration:** OpenVINO 2026.1.0, threshold 0.584.
- **Status:** **completed**.
- **Final quality / runtime:** **yes / yes**.
- **Outputs:** `experiments/quantization/openvino_ptq_20260819T0000Z/{baseline,pruned_recovered}/final_test_fixed_fp32/`.
- **Superseded by:** not superseded as OpenVINO precision reference.
- **Thesis placement:** **main thesis results**.

### E21 — Final OpenVINO INT8 PTQ evaluation, calibration, and runtime

- **Purpose:** test NNCF/OpenVINO static INT8 PTQ and compare it with ORT INT8.
- **Models / optimization / runtime:** FCOS_18 and step-0 Pruned60; NNCF static PTQ, OpenVINO INT8 IR on CPU.
- **Data:** 39-image disjoint calibration split; complete 111-image fixed and calibrated final-test evaluations.
- **Hardware:** development PC, cores 0–3 / four threads.
- **Main configuration:** OpenVINO 2026.1.0, NNCF 3.2.0, 300 calibration samples, PERFORMANCE preset; 202 `FakeQuantize` nodes; fixed 0.584 and calibrated 0.600/0.565 thresholds.
- **Status:** **completed**. Accuracy-aware PTQ and QAT were discussed only and were **not executed**. Pruned OpenVINO INT8 completed but its measured runtime is anomalously slower than FP32.
- **Final quality / runtime:** **yes / yes**.
- **Outputs:** `experiments/quantization/openvino_ptq_20260819T0000Z/`, especially `run_status.json` and `reports/openvino_quantization_comparison.md`; consolidated `experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/final_report.md`.
- **Superseded by:** none.
- **Thesis placement:** **main thesis results**.

### E22 — FP16 quantization/runtime path

- **Purpose:** provide half-precision deployment support.
- **Models / optimization / runtime:** FCOS exports; FP16 conversion paths mentioned in adapter/backend/scripts.
- **Data / hardware / configuration:** no generated run artifacts establish a dataset or device.
- **Status:** **prepared but not executed**.
- **Final quality / runtime:** **no / no**.
- **Outputs:** implementation references in `src/benchmark/runtime_backends.py`, `src/benchmark/midog_guide_adapter.py`, and benchmarking scripts/docs; no FP16 result directory.
- **Superseded by:** completed FP32/INT8 comparisons E18–E21.
- **Thesis placement:** **methodology only** if supported formats are described; otherwise **not included**.

### E23 — Physical Raspberry Pi/Pi-class six-variant benchmark

- **Purpose:** measure actual edge-device inference for baseline versus Pruned60 and FP32 versus INT8.
- **Models / optimization / runtime:** FCOS_18 and raw Pruned60; PyTorch eager FP32, ORT FP32, ORT QDQ INT8.
- **Data:** three images (`413`, `366`, `487`), 72 annotation rows, 228 patches; diagnostic subset, not the 111-image final test.
- **Hardware:** **physical Raspberry Pi 5**, four Cortex-A76 cores, aarch64. Four threads and one inter-op thread; not the development-PC approximation. The `rpi4_*` bundle name is legacy/misleading and is retained only for provenance.
- **Main configuration:** one warm-up + three measured repetitions per variant, batch 1, threshold 0.584. Median E2E totals recorded: FCOS_18 eager about 3,114 s; ORT FP32 4,486 s; ORT INT8 854 s; Pruned60 eager 2,553 s; ORT FP32 3,491 s; ORT INT8 729 s.
- **Status:** **completed** for all six variants. Initial smoke test failed/aborted and was rerun successfully; the full log continues through all variants.
- **Final quality / runtime:** quality **no** (diagnostic three-image metrics only); runtime **yes for the physical-device diagnostic benchmark**, but not a full 111-image Pi benchmark.
- **Outputs:** canonical bundle `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/results/{fcos18,pruned60}/{pytorch_fp32,onnx_fp32,onnx_int8}/`, plus `results/rpi4_actual/full_benchmark.log` and smoke logs.
- **Superseded by:** no newer physical-device results found. Older `dist/rpi4_fcos*_benchmark_bundle` directories are pre-test/preparation bundles and are superseded by this post-test bundle.
- **Thesis placement:** **main thesis results**, with explicit board-identity and three-image-scope caveats.

### E24 — Physical-device TVM experiment

- **Purpose:** determine whether TVM can compile and benchmark FCOS on the Pi-class target.
- **Models / optimization / runtime:** FCOS_18 ONNX FP32; TVM Relax/LLVM targeting aarch64 Cortex-A76; Pruned60 planned behind a validation gate.
- **Data:** single patch intended for numerical validation; no image-set inference benchmark began.
- **Hardware:** same physical Pi-class environment; target declares four Cortex-A76 cores and NEON.
- **Main configuration:** TVM 0.25.0.post1, LLVM 22; full graph import attempted, then FCOS heads compiled separately.
- **Status:** **failed/partially completed**. Full graph import failed on dynamic Resize/ReduceMin/TopK patterns. Head graph imported and compiled (`compiled_heads.so`, 15.46 s), but one TVM head inference exceeded five minutes and was interrupted before numerical validation. `benchmark_started=false`; Pruned60 was not attempted.
- **Final quality / runtime:** **no / no**. Compilation success is not inference-benchmark success.
- **Outputs:** `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/results/fcos18/tvm_llvm_fp32/20260819T175827Z/`.
- **Superseded by:** none.
- **Thesis placement:** **appendix only** as a failed compiler/runtime feasibility experiment, or **methodology only** if space is limited.

### E25 — Raspberry Pi preparation-only and unsupported variants

- **Purpose:** package reproducible Pi launchers and models beyond what was eventually run.
- **Models / optimization / runtime:** six executed variants plus unsupported PyTorch INT8; no OpenVINO or `torch.compile` Pi result artifacts.
- **Data / hardware:** intended physical Pi execution on the three-image bundle subset.
- **Main configuration:** launcher scripts and manifests in the post-test bundle.
- **Status:** packaging **completed**; PyTorch INT8 intentionally unsupported; Pi OpenVINO/`torch.compile` **not prepared as executed result families**. Scripts are not evidence of a run.
- **Final quality / runtime:** **no / no** for unexecuted variants.
- **Outputs:** `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/run_*.sh`, `README_RPI4.md`, `MANIFEST.txt`, `models/deployment_artifacts_report.json`.
- **Superseded by:** E23 for variants with actual output.
- **Thesis placement:** **methodology only**; unsupported/unrun variants **not included**.

## Conflicts, stale files, and naming hazards

- `experiments/pruning/final_pruning_summary.md` says the full held-out and final four-core runs were pending. That was true when written, but E18–E21 later completed 111-image final runs. It remains true that no standalone eager `.pt` full benchmark was added to the pruning directory.
- `experiments/distillation/pruned60_clean_recovery/final_comparison.md` and `pruned60_recovery_comparison.md` remain pending and must not be treated as completed recovery comparisons.
- `experiments/distillation/distillation_experiment_summary.md` contains stale TBD entries despite later MobileNetV3-Small artifacts.
- `experiments/distillation/pruned60_frozen_backbone_recovery/reports/frozen_backbone_recovery_summary.md` is older than `current_recovery_results_summary.md` and the later LR/seed reports; it omits executed 3e-6/3e-5/5e-5 and seed runs.
- The seed reproducibility report says only two comparable seeds completed; seed 43's run-status file confirms numerical failure. A smoke status of `running` must not override the failed full run.
- The earlier ORT PTQ `run_status.json` says OpenVINO INT8 was unavailable because NNCF was absent. The later `openvino_ptq_20260819T0000Z/run_status.json` is authoritative and records complete OpenVINO fixed, calibration, and frozen-threshold phases.
- Final reports use the label `pruned_recovered`, but `selected_recovery_checkpoint.json` selects LR 1e-4 at optimizer step 0. Treat this model as **Pruned60 / recovery-selected step 0**, not as demonstrated recovered weights.
- The physical bundle's `rpi4_*` directory/README naming is legacy and incorrect. The confirmed hardware was Raspberry Pi 5 with four Cortex-A76 cores; preserve old paths only as provenance identifiers.
- The early FCOS family metric files contain 105 cases, whereas the current audited test set has 111 images. Do not merge these as if they used the same test definition.
- Two early OpenVINO quick naming families (`openvino_cpu` and `openvino_cpu_fp32`) exist. Their runtime validation files should be consulted; filenames alone are insufficient to infer precision.

## FINAL EXPERIMENT PIPELINE

- Baseline
  - FCOS_18 / FCOS_x50 / FCOS_x101 historical quality evaluation
  - 20-image development-PC `CPU_4C_LIMITED` comparison
  - FCOS_18 selected as lightweight baseline
- Runtime optimization
  - PyTorch eager reference
  - `torch.compile`/Inductor quick benchmark
  - ONNX Runtime FP32 quick benchmark
  - OpenVINO FP32 quick benchmark
- Structural pruning
  - Early masked/head-pruning pilots
  - DepGraph pruning-scope sweep
  - DepGraph pruning-ratio and boundary sweeps
  - FPN+head 60% candidate selection (Pruned60)
- Alternative compact architecture
  - MobileNetV3 Small/Large KD pilots
  - MobileNetV3-Small full training and diagnostic evaluation
- Recovery
  - Early unfrozen supervised/KD pilots
  - Clean unfrozen dense-checkpoint LR sweep
  - Frozen-backbone supervised/KD LR sweep
  - Seed/determinism checks
  - No recovery checkpoint accepted beyond step 0 for final deployment
- Quantization and calibration
  - ONNX Runtime FP32 export/reference
  - ONNX Runtime static QDQ INT8
  - OpenVINO FP32 conversion/reference
  - OpenVINO/NNCF static INT8 PTQ
  - Leakage-free 39-image calibration threshold sweeps
- Final CPU evaluation
  - Eight fixed-threshold 111-image evaluations
  - Four frozen calibrated-threshold INT8 evaluations
  - Matched four-core ORT/OpenVINO runtime benchmarks
- Raspberry Pi/Pi-class deployment
  - Six completed physical-device three-image repeated benchmarks
  - TVM compile/import feasibility attempt; inference validation failed, no benchmark

## RECOMMENDED RESULTS CHAPTER STRUCTURE

- Baseline Model Evaluation
  - FCOS_18, FCOS_x50, and FCOS_x101 Quality
  - Four-Core CPU Baseline Runtime
- Runtime Backend Optimization
  - PyTorch Eager and `torch.compile`
  - ONNX Runtime FP32
  - OpenVINO FP32
- Structured Pruning
  - Pruning-Scope Sweep
  - Pruning-Ratio Sweep
  - Pruned60 Candidate Selection
- Compact-Model and Recovery Experiments
  - MobileNetV3 Knowledge Distillation
  - Supervised and Distillation Recovery
  - Frozen-Backbone Learning-Rate Sweep
  - Seed and Reproducibility Checks
- Quantization and Threshold Calibration
  - ONNX Runtime INT8
  - OpenVINO INT8
  - Calibration-Set Threshold Selection
- Final Test-Set Evaluation
  - FP32 and INT8 Quality
  - Fixed and Calibrated Operating Points
- Final Runtime Evaluation
  - Four-Core CPU Benchmarks
  - Physical Raspberry Pi/Pi-Class Benchmarks
