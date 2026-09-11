# Submission cleanup audit — 2026-09-11

## Scope and decisions before editing

The initial worktree was clean, with 1,452 tracked files. Inspection covered the
tracked tree, ignored source/configuration files, documentation, launchers,
upstream clone revisions and changes, CLI parsers, imports, and regression tests.
The thesis manuscript is absent; experiment relevance is based on the existing
inventory and provenance reports, not a direct manuscript comparison.

Keep experiment implementations, configurations, regression checks, compact
metrics/timing/runtime records, and provenance. Masked-pruning pilots, structural
scope/ratio sweeps, recovery repairs, and diagnostic scripts explain experiments
or are used by tests; none was deleted merely for age. No byte-identical Python
source duplicates were found in scripts/src/tests. The two Pi bundle builders
serve different scopes (baseline-only versus six variants); the latter is canonical.

Clean stale milestone prose, duplicated runtime instructions, local path defaults,
incorrect generated Pi 4 prose, and ignore rules that hide recovery source while
allowing large artifacts. Add setup/dependency revisions and the experiment map.

Remove models, predictions, and logs from tracking while preserving local copies
and a checksum manifest. Keep datasets, clones, environments, training outputs,
exports, bundles, and raw benchmarks ignored. Preserve numerical results and
historical experiment statuses.

## Changes prepared

- Replaced README with setup, data/model preparation, repository navigation,
  artifact policy, and limits; consolidated benchmarking/pruning documentation.
- Added `docs/experiments.md`, covering all major experiment families with
  scripts, inputs, expected outputs, and source-derived command examples.
- Exposed 23 previously ignored recovery YAMLs and three report modules required
  by existing tests. No model, prediction, or result directory was bulk-added.
  Other local recovery orchestration/repair wrappers remain ignored; their
  underlying trainer, selectors, and selected report modules are public inputs.
- Added NNCF to optional runtime requirements. Requirements remain unpinned.
- Made smoke/quick/full checkout defaults portable and their `--help` safe.
- Reworked config preparation into an explicit CLI with missing-input errors and
  preflight of all models before writes. Default tracked evaluation YAMLs now use
  repository-relative checkpoint paths; preparation writes local absolute paths.
- Corrected Pi 5 prose in the older bundle generator, preserving `rpi4` identifiers.
  Added `--baseline-onnx` to deployment preparation and forwarding/help to the
  six-variant builder, removing its dependence on one export filename.
- Scoped pytest discovery to project tests. Updated one stale report-test fixture
  to supply stable-selected AP, matching the current selection policy; its numeric
  values are unchanged. Historical result reports were not regenerated.
- Clarified September CPU report precedence and the separate Pruned60 strict-FP32
  runner. Historical provenance counts remain explicitly scoped to August.

## Artifact removal completed

`removed_artifacts.csv` lists **455 files, 1,348,433,947 bytes**, with paths, sizes,
and SHA-256 hashes. They include 17 `.pt` models, 262 prediction/detection JSON
files, and 176 logs. All 455 files have been removed from the Git index with
`git rm --cached`; **every local copy remains intact**. Their sizes and hashes
were checked before and after removal, and all are covered by `.gitignore`.
The removals are staged; source and documentation changes still require staging
and review before committing.

Removing these files does not shrink existing Git history. Do not rewrite public
history as part of this submission cleanup. Checkpoints are thesis evidence:
retain local backups or provide a separate artifact location if exact historical
weights must be available to readers. Hashes alone do not reconstruct weights.

Remaining ignored categories include datasets/split CSVs, upstream clones,
virtual environments/caches, checkpoints, ONNX/OpenVINO exports, training runs,
TensorBoard/events, raw predictions, logs, diagnostic outputs, and Pi bundles.
Existing tracked compact metrics/runtime/summary files remain versioned even
where new output files are ignored. The three `*_eval.json` files next to the
baseline YAMLs are historical evaluation metrics, **not configurations**; they
were retained to preserve their existing provenance references.

Useful optional additions, after checking thesis table correspondence, are:

- Frozen recovery `reports/supervised_learning_rate_comparison.csv` and
  `reports/supervised_lr3e-5_seed_reproducibility.csv` under
  `experiments/distillation/pruned60_frozen_backbone_recovery/`.
- Exact checkpoint selection JSON and hashes used by the September runners.
- A compact six-variant Pi 5 median timing table plus package/hardware metadata
  extracted from the post-test bundle.

These are recommendations, not newly added result trees. Existing combined
quantization summaries and September CPU reports already provide useful table
verification.

## Reproducibility risks still requiring attention

1. Confirm that the final thesis cites September corrected CPU timings, separates
   historical 105-image results from the 111-image final workload, and distinguishes
   raw Pruned60, August step-0 selection, and later frozen selected artifacts.
   The old experiment inventory remains a dated historical account.
2. Exact upstream model distribution and hash-matched recovery checkpoints are
   external requirements. Record an artifact access location if required for the
   appendix. The current local guide `evaluate.py` has pruning additions, but the
   project adapter imports unchanged guide utilities rather than that script.
3. Environment requirements are not a historical lockfile. CPU/CUDA and ARM wheels,
   OpenSlide, exporter behavior, and hardware-specific settings need validation on
   the target system. Python 3.12.3 is an observed local version only.
4. CPU IDs are host-specific; four threads do not establish four physical cores.
   OpenVINO FP32 weights do not establish FP32 execution. Follow runtime metadata.
5. Some LR/seed runs failed and some prepared KD configurations were never run.
   Stable selection and test/calibration separation must not be inferred merely
   from filenames. No new model training or full inference was executed here.
6. Pi measurements cover three images/228 patches, not the full test set. The
   regenerated bundle is not guaranteed identical to the post-test bundle.
7. Historical command logs are partial provenance, often with fixed paths and
   output reuse. Use fresh output directories when reproducing, and record actual
   commands and artifact hashes. Some legacy pipelines skip existing outputs.

## Validation

- All 73 Python sources in scripts/src/tests parse; shell launchers pass `bash -n`.
- Fifteen important Python CLI help calls pass, including trainer, pruning,
  selectors, quantizers, and cached scoring. Modified benchmark/bundle help paths
  are checked without launching experiments.
- Project regression suite: **43 passed**. Initial broad pytest collection found
  an upstream test with its own working-directory assumption; `pytest.ini` now
  restricts collection to this project's `tests/`.
- A separate source-only checkout (without repos, data, model artifacts, or run
  outputs) also passes all **43 tests** using the installed environment.
- All 54 public YAML files parse, 144 public source/config paths pass syntax
  checks, documentation links resolve, and `git diff --check` passes.
- The documented annotation conversion was exercised under `/tmp`: it produces
  392 train / 111 test images and 5,383 test annotation rows.
- No experiment result numbers were edited and no training/benchmark was rerun.

## Final Git steps

The repository cleanup is complete. Before tagging, review/stage the source and
documentation changes, commit the submission state, and reconcile the thesis
references and artifact access requirements above. From the root:

```bash
# Artifact removals are already staged; local copies are preserved.
git diff --check
.venv/bin/python -m pytest -q
git add .gitignore README.md docs reports requirements-runtime.txt pytest.ini scripts tests \
  experiments/eval_guide/configs/*.yaml \
  experiments/final_cpu_4physicalcore_rerun_20260906/README.md \
  experiments/distillation/configs \
  experiments/distillation/pruned60_frozen_backbone_recovery/configs \
  experiments/distillation/pruned60_frozen_backbone_recovery/scripts
git diff --cached --stat
git diff --cached --name-status
git status --short
# Review the staged patch, then commit.
git commit -m "Prepare thesis submission code and reproduction guide"
# Only after the submission state is approved:
git tag -a thesis-v1.0 -m "Bachelor thesis submission snapshot"
git show --stat thesis-v1.0
git rev-parse thesis-v1.0^{commit}
```

No commit, tag, release, or push was performed by this cleanup.

## Changed and newly exposed files

- `.gitignore`
- `README.md`
- `docs/benchmarking.md`
- `docs/experiments.md`
- `docs/pruning.md`
- `docs/repository_structure.md`
- `experiments/distillation/configs/fcos18_pruned60_supervised_clean.yaml`
- `experiments/distillation/configs/fcos18_pruned60_supervised_clean_dense_lr1e-4_seed42.yaml`
- `experiments/distillation/configs/fcos18_pruned60_supervised_clean_dense_lr1e-5_seed42.yaml`
- `experiments/distillation/configs/fcos18_pruned60_supervised_clean_dense_lr1e-5_seed42_extend10.yaml`
- `experiments/distillation/configs/fcos18_pruned60_supervised_clean_dense_lr3e-5_seed42.yaml`
- `experiments/distillation/configs/fcos18_pruned60_supervised_clean_dense_lr3e-5_seed42_extend5.yaml`
- `experiments/distillation/configs/fcos18_pruned60_supervised_recovery.yaml`
- `experiments/distillation/configs/fcos_x101_to_fcos18_pruned60_distillation.yaml`
- `experiments/distillation/configs/fcos_x101_to_fcos18_pruned60_distillation_clean.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/distillation_lr1e-5_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/distillation_lr1e-6_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/distillation_lr3e-5_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/distillation_lr3e-6_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr1e-3_seed42_upper_boundary.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr1e-4_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr1e-5_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr1e-6_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr3e-5_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr3e-5_seed43.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr3e-5_seed44.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr3e-6_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr3e-6_seed42_extension.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/configs/supervised_lr5e-5_seed42.yaml`
- `experiments/distillation/pruned60_frozen_backbone_recovery/scripts/generate_comparison_report.py`
- `experiments/distillation/pruned60_frozen_backbone_recovery/scripts/generate_supervised_lr3e-5_reproducibility_report.py`
- `experiments/distillation/pruned60_frozen_backbone_recovery/scripts/generate_supervised_lr_comparison_report.py`
- `experiments/eval_guide/configs/FCOS_18_eval.yaml`
- `experiments/eval_guide/configs/FCOS_x101_eval.yaml`
- `experiments/eval_guide/configs/FCOS_x50_eval.yaml`
- `experiments/final_cpu_4physicalcore_rerun_20260906/README.md`
- `pytest.ini`
- `reports/final_results_provenance.md`
- `reports/removed_artifacts.csv`
- `reports/submission_cleanup.md`
- `requirements-runtime.txt`
- `scripts/create_rpi4_fcos18_bundle.sh`
- `scripts/create_rpi4_fcos_benchmark_bundle.sh`
- `scripts/prepare_fcos_configs_for_guide.py`
- `scripts/prepare_rpi4_fcos_deployment_artifacts.py`
- `scripts/run_full_benchmark.sh`
- `scripts/run_quick_benchmark.sh`
- `scripts/run_smoke_test.sh`
- `tests/test_frozen_backbone_report.py`
