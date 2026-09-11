# MIDOG FCOS thesis experiments

Code and configurations accompanying the bachelor thesis on efficient FCOS mitosis
detection on CPUs and Raspberry Pi 5. Experiments cover baseline evaluation,
structured pruning, supervised recovery, knowledge distillation, and FP32/INT8
inference with PyTorch, ONNX Runtime, and OpenVINO.

Start with the [experiment map and commands](docs/experiments.md).
The [result provenance index](reports/final_results_provenance.md) identifies
historical artifacts; the [corrected CPU report](experiments/final_cpu_4physicalcore_rerun_20260906/final_report.md)
supersedes the August CPU timings. The thesis manuscript is not included, so the
experiment map follows the repository's audited experiment inventory.

## Setup

Run commands from the repository root on Linux. Python 3.12.3 is installed in the
audited development environment. Create `.venv` and install a matching PyTorch /
torchvision pair for your CPU or CUDA environment, the upstream requirements,
and this project's additional requirements:

```bash
mkdir -p repos
git clone https://github.com/DeepMicroscopy/MIDOG_2025_Guide.git repos/MIDOG_2025_Guide
git -C repos/MIDOG_2025_Guide checkout 28ca3b940f0d04a87b4e517396d38c537291b19f
git clone https://github.com/jonas-amme/FCOS_Inference_CLI.git repos/FCOS_Inference_CLI
git -C repos/FCOS_Inference_CLI checkout ad28c685680cf922566ef929a47b9293401c872d
git clone https://github.com/DeepMicroscopy/MIDOGpp.git repos/MIDOGpp
git -C repos/MIDOGpp checkout c8f4a18d261e5c9d09c04b0925d40cd7935de618
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r repos/MIDOG_2025_Guide/requirements.txt
python -m pip install -r repos/FCOS_Inference_CLI/requirements.txt
python -m pip install -r requirements.txt
# Optional: ONNX Runtime, OpenVINO and NNCF quantization
python -m pip install -r requirements-runtime.txt
```

The guide supplies model factories, patch inference, and evaluation utilities;
FCOS_Inference_CLI supplies model configurations and externally obtained
checkpoints; MIDOGpp supplies annotations and the official split. The locally
present MIDOG25 reference/evaluation Docker repositories are not required by
these entry points. These revisions describe the inspected clones, not a verified
lock of every historical run. The adapter imports guide utilities, not its locally
modified `evaluate.py`.

OpenSlide requires a system library as well as `openslide_python` (on Debian/Ubuntu,
`libopenslide0`). Benchmarks also use Bash, `taskset`, and `/usr/bin/time`.
Package requirements are not fully pinned; consult recorded runtime metadata
before comparing results from a new environment.

## Data and pretrained models

Obtain the MIDOG++ TIFF images under the dataset's access terms, and place them in
`data/midogpp/` with their original filenames. Use the annotation JSON and split
from the checked-out MIDOGpp repository:

```bash
python scripts/create_guide_eval_csv_from_midogpp_split.py \
  --midog-json repos/MIDOGpp/databases/MIDOG++.json \
  --xvalidation repos/MIDOGpp/datasets_xvalidation.csv \
  --out data/midogpp_guide_eval_xvalidation.csv --bbox-format xyxy
python scripts/create_quick_benchmark_subset.py \
  --input_csv data/midogpp_guide_eval_xvalidation.csv \
  --output_csv data/midogpp_guide_eval_xvalidation_quick.csv \
  --n 32 --seed 42 --split test --stratify_auto
python scripts/create_pruned60_clean_recovery_splits.py --seed 42
```

The bbox interpretation must match the annotations; do not change it to `xywh`
without checking the source format. The final workload is 111 test images / 8,500
patches. Clean recovery uses 314 train, 39 validation (`val`), and 39 calibration
images. Check generated split counts and the overlap manifest before training.

Obtain `FCOS_18.ckpt`, `FCOS_x50.ckpt`, and `FCOS_x101.ckpt` using the upstream
FCOS repository's model instructions and place them in
`repos/FCOS_Inference_CLI/checkpoints/`. They are not distributed here. FCOS_x101
is also the distillation teacher. Backbone initialization can require torchvision
weights. Then configure local checkpoint paths:

```bash
python scripts/prepare_fcos_configs_for_guide.py
```

This writes absolute checkpoint paths for your checkout into the evaluation YAML
files; review those local changes before committing. Training and pruning configs
use paths relative to the repository root. Benchmark wrappers accept `PROJECT`,
`PYTHON`, `GUIDE_REPO`, and `IMG_DIR` overrides.

## Running experiments

```bash
PROFILE_PIPELINE=1 bash scripts/run_quick_benchmark.sh FCOS_18
bash scripts/run_full_benchmark.sh FCOS_18
python scripts/create_depgraph_structural_pruned_model.py \
  --config experiments/pruning/configs/fcos18_depgraph_fpn_head_60pct.yaml \
  --output experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt
```

See [experiments](docs/experiments.md) for recovery, distillation, quantization,
calibration, and Raspberry Pi commands; [benchmarking](docs/benchmarking.md) for
thread/precision controls; and [pruning](docs/pruning.md) for model identity.
Smoke and quick runs are diagnostics, not substitutes for full test evaluation.

## Layout and generated artifacts

- `scripts/`, `src/`: command-line entry points and shared implementation.
- `experiments/*/configs/`: experiment inputs; dated experiment directories also
  retain launchers and compact historical summaries.
- `docs/`: reproduction instructions; `reports/`: audits and provenance.
- `tests/`: regression checks (`python -m pytest -q`).
- `data/`, `repos/`, `.venv/`, `dist/`: local data, dependencies, environment, and
  generated deployment bundles, with a few tracked bundle documentation files.

Checkpoints, exports, cached predictions, logs, datasets, and raw run directories
remain ignored. Existing compact metrics and summaries are retained without
altering numbers. [Artifact-removal manifest](reports/removed_artifacts.csv)
records 455 binaries/predictions/logs removed from tracking during submission
cleanup. Their local copies are intact and excluded by `.gitignore`. Removing tracked files does not remove them from Git
history. Do not force-add result trees. Regenerate artifacts with the documented
entry points or obtain the exact hash-matched historical artifacts separately.

## Reproducibility limits and submission

The legacy key `pruned_recovered` in August results identifies a recovery-selected
**step-0 Pruned60**, not successful recovery. Later launchers name a frozen-recovery
selected checkpoint; establish its hash/selection record before equating models.
CPU IDs `0-3` did not mean four physical cores on the development host; corrected
runs use `0,2,4,6`. OpenVINO FP32 storage does not establish FP32 execution.
The Pi measurements used a **Raspberry Pi 5**, three images, and 228 patches;
`rpi4` filenames are preserved compatibility identifiers.

Training was not stable for every learning rate/seed. Exact historical package
resolution, checkpoints, hardware conditions, and all raw outputs are not
reconstructed by a fresh checkout. The [submission audit](reports/submission_cleanup.md)
records remaining checks. After review and validation, tag the exact submitted
commit `thesis-v1.0` and cite that tag/commit in the appendix. No release or tag is
created by this cleanup.
