#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$REPO_ROOT/dist"
BUNDLE_NAME="rpi4_fcos18_benchmark_bundle"
STAGING="$DIST_DIR/$BUNDLE_NAME"
ZIP_PATH="$DIST_DIR/${BUNDLE_NAME}.zip"
REPORT_PATH="$DIST_DIR/${BUNDLE_NAME}_report.md"
PYTHON_BIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

FCOS18_CHECKPOINT="$REPO_ROOT/repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt"
FCOS18_ONNX="$REPO_ROOT/experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx"
FCOS18_CONFIG="$REPO_ROOT/experiments/eval_guide/configs/FCOS_18_eval.yaml"
SUBSET_CSV="$REPO_ROOT/data/midogpp_guide_eval_xvalidation_3img.csv"
IMAGE_ROOT="$REPO_ROOT/data/midogpp"

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Required file is missing: $path" >&2
    exit 1
  fi
}

copy_file() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
}

require_file "$FCOS18_CHECKPOINT"
require_file "$FCOS18_ONNX"
require_file "$FCOS18_CONFIG"
require_file "$SUBSET_CSV"
for image in 413.tiff 366.tiff 487.tiff; do
  require_file "$IMAGE_ROOT/$image"
done

mkdir -p "$DIST_DIR"
rm -rf "$STAGING"
mkdir -p \
  "$STAGING/configs" \
  "$STAGING/data/images" \
  "$STAGING/data/annotations" \
  "$STAGING/models" \
  "$STAGING/results" \
  "$STAGING/scripts" \
  "$STAGING/src/benchmark" \
  "$STAGING/src/pruning" \
  "$STAGING/src/distillation" \
  "$STAGING/repos/MIDOG_2025_Guide/utils" \
  "$STAGING/experiments/eval_guide/exported_models"

copy_file "$FCOS18_CHECKPOINT" "$STAGING/models/fcos18_checkpoint.ckpt"
copy_file "$FCOS18_ONNX" "$STAGING/models/fcos18_fp32.onnx"
copy_file "$SUBSET_CSV" "$STAGING/data/rpi4_test_subset.csv"
copy_file "$SUBSET_CSV" "$STAGING/data/annotations/midogpp_guide_eval_xvalidation_3img.csv"
copy_file "$IMAGE_ROOT/413.tiff" "$STAGING/data/images/413.tiff"
copy_file "$IMAGE_ROOT/366.tiff" "$STAGING/data/images/366.tiff"
copy_file "$IMAGE_ROOT/487.tiff" "$STAGING/data/images/487.tiff"

while IFS= read -r file; do
  rel="${file#$REPO_ROOT/}"
  copy_file "$file" "$STAGING/$rel"
done < <(find "$REPO_ROOT/src" -type f -name '*.py' | sort)

for file in \
  "$REPO_ROOT/scripts/summarize_quick_benchmark_timing.py" \
  "$REPO_ROOT/scripts/run_quick_benchmark.sh" \
  "$REPO_ROOT/scripts/run_full_benchmark.sh"; do
  copy_file "$file" "$STAGING/scripts/$(basename "$file")"
done

for rel in \
  README.md \
  requirements.txt \
  utils/inference.py \
  utils/eval_utils.py \
  utils/factory.py \
  utils/litmodel.py \
  utils/model.py; do
  copy_file "$REPO_ROOT/repos/MIDOG_2025_Guide/$rel" "$STAGING/repos/MIDOG_2025_Guide/$rel"
done

"$PYTHON_BIN" - "$STAGING" <<'PY'
from pathlib import Path
import csv
import hashlib
import os
import stat
import sys

root = Path(sys.argv[1])

config = """backbone: resnet18
checkpoint: models/fcos18_checkpoint.ckpt
det_thresh: 0.584
detector: FCOS
extra_blocks: false
model_name: FCOS_18
num_classes: 2
patch_size: 1024
returned_layers:
- 1
- 2
- 3
- 4
weights: null
means: null
stds: null
"""
(root / "configs/FCOS_18_eval.yaml").write_text(config, encoding="utf-8")

smoke_rows = []
with (root / "data/rpi4_test_subset.csv").open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    for row in reader:
        if row["filename"] == "413.tiff":
            smoke_rows.append(row)
with (root / "data/rpi4_smoke_subset.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(smoke_rows)

runtime_path = root / "src/benchmark/runtime_backends.py"
runtime_text = runtime_path.read_text(encoding="utf-8")
needle = """    export_path = _onnx_export_path(
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
    )
    export_reused = export_path.exists()
"""
replacement = """    packaged_onnx_model = os.environ.get("PACKAGED_ONNX_MODEL")
    if packaged_onnx_model:
        export_path = Path(packaged_onnx_model)
        export_reused = export_path.exists()
    else:
        export_path = _onnx_export_path(
            export_dir=export_dir,
            model_name=model_name,
            patch_size=patch_size,
            onnx_opset=onnx_opset,
            config_file=config_file,
        )
        export_reused = export_path.exists()
"""
if needle not in runtime_text:
    raise SystemExit("Could not patch ONNX Runtime export path hook.")
runtime_text = runtime_text.replace(needle, replacement, 1)
if "import os\n" not in runtime_text:
    runtime_text = runtime_text.replace("import json\n", "import json\nimport os\n", 1)
runtime_path.write_text(runtime_text, encoding="utf-8")

requirements = """faster-coco-eval
evalutils
lightning
numpy
onnx
onnxruntime
opencv-python-headless
openslide-python
pandas
Pillow
PyYAML
torchmetrics
tqdm
"""
(root / "requirements-rpi4.txt").write_text(requirements, encoding="utf-8")

install = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "This bundle expects a 64-bit ARM OS. uname -m is: $(uname -m)" >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential libopenblas-dev libjpeg-dev libtiff-dev libopenjp2-7-dev libopenslide0 openslide-tools libglib2.0-0 libgl1
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rpi4.txt
python verify_bundle.py
"""
(root / "install_rpi4.sh").write_text(install, encoding="utf-8")

common = """#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$BUNDLE_ROOT"
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif [ -x "$BUNDLE_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$BUNDLE_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
export PYTHONPATH="$BUNDLE_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
export PACKAGED_ONNX_MODEL="$BUNDLE_ROOT/models/fcos18_fp32.onnx"
TASKSET=()
if command -v taskset >/dev/null 2>&1; then
  TASKSET=(taskset -c 0-3)
fi
run_adapter() {
  local backend="$1"
  local dataset="$2"
  local out_dir="$3"
  local run_name="$4"
  mkdir -p "$out_dir"
  "${TASKSET[@]}" /usr/bin/time -v "$PYTHON_BIN" "$BUNDLE_ROOT/src/benchmark/midog_guide_adapter.py" \\
    --config_file "$BUNDLE_ROOT/configs/FCOS_18_eval.yaml" \\
    --dataset "$dataset" \\
    --guide_repo "$BUNDLE_ROOT/repos/MIDOG_2025_Guide" \\
    --img_dir "$BUNDLE_ROOT/data/images" \\
    --metrics_output "$out_dir/${run_name}_metrics.json" \\
    --predictions_output "$out_dir/${run_name}_predictions.json" \\
    --runtime_metadata_output "$out_dir/${run_name}_runtime.json" \\
    --runtime_backend "$backend" \\
    --onnx_opset 18 \\
    --export_dir "$BUNDLE_ROOT/experiments/eval_guide/exported_models" \\
    --split test \\
    --device cpu \\
    --batch_size 1 \\
    --num_workers "$NUM_WORKERS" \\
    --overlap 0.3 \\
    --nms_thresh 0.3 \\
    --profile_pipeline \\
    --timing_output_json "$out_dir/${run_name}_timing.json" \\
    --timing_output_csv "$out_dir/${run_name}_timing.csv" \\
    --overwrite
}
timestamp_dir() {
  local base="$1"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local candidate="$base/$stamp"
  local n=1
  while [ -e "$candidate" ]; do
    candidate="$base/${stamp}_$n"
    n=$((n + 1))
  done
  printf '%s\\n' "$candidate"
}
"""

smoke = common + """OUT_DIR="$(timestamp_dir "$BUNDLE_ROOT/results/smoke")"
run_adapter pytorch_eager "$BUNDLE_ROOT/data/rpi4_smoke_subset.csv" "$OUT_DIR/pytorch_eager" smoke_pytorch_eager
run_adapter onnxruntime_cpu "$BUNDLE_ROOT/data/rpi4_smoke_subset.csv" "$OUT_DIR/onnxruntime_cpu" smoke_onnxruntime_cpu
test -s "$OUT_DIR/pytorch_eager/smoke_pytorch_eager_metrics.json"
test -s "$OUT_DIR/onnxruntime_cpu/smoke_onnxruntime_cpu_metrics.json"
echo "Smoke test outputs: $OUT_DIR"
"""
(root / "run_smoke_test.sh").write_text(smoke, encoding="utf-8")

pytorch = common + """BASE="$BUNDLE_ROOT/results/pytorch_eager"
for kind in warmup measured measured measured; do
  :
done
OUT_DIR="$(timestamp_dir "$BASE")"
run_adapter pytorch_eager "$BUNDLE_ROOT/data/rpi4_test_subset.csv" "$OUT_DIR" warmup1
for run in 1 2 3; do
  run_adapter pytorch_eager "$BUNDLE_ROOT/data/rpi4_test_subset.csv" "$OUT_DIR" "measured${run}"
done
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/summarize_quick_benchmark_timing.py" --timing_glob "$OUT_DIR/measured*_timing.json" --output_csv "$OUT_DIR/timing_median.csv"
echo "PyTorch eager benchmark outputs: $OUT_DIR"
"""
(root / "run_pytorch_benchmark.sh").write_text(pytorch, encoding="utf-8")

onnx = common + """BASE="$BUNDLE_ROOT/results/onnxruntime_cpu"
OUT_DIR="$(timestamp_dir "$BASE")"
run_adapter onnxruntime_cpu "$BUNDLE_ROOT/data/rpi4_test_subset.csv" "$OUT_DIR" warmup1
for run in 1 2 3; do
  run_adapter onnxruntime_cpu "$BUNDLE_ROOT/data/rpi4_test_subset.csv" "$OUT_DIR" "measured${run}"
done
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/summarize_quick_benchmark_timing.py" --timing_glob "$OUT_DIR/measured*_timing.json" --output_csv "$OUT_DIR/timing_median.csv"
echo "ONNX Runtime CPU benchmark outputs: $OUT_DIR"
"""
(root / "run_onnxruntime_benchmark.sh").write_text(onnx, encoding="utf-8")

all_bench = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./run_smoke_test.sh
./run_pytorch_benchmark.sh
./run_onnxruntime_benchmark.sh
"""
(root / "run_all_benchmarks.sh").write_text(all_bench, encoding="utf-8")

verify = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib
import os
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "configs/FCOS_18_eval.yaml",
    "models/fcos18_checkpoint.ckpt",
    "models/fcos18_fp32.onnx",
    "data/rpi4_test_subset.csv",
    "data/rpi4_smoke_subset.csv",
    "data/images/413.tiff",
    "data/images/366.tiff",
    "data/images/487.tiff",
    "src/benchmark/midog_guide_adapter.py",
    "src/benchmark/runtime_backends.py",
    "repos/MIDOG_2025_Guide/utils/inference.py",
    "repos/MIDOG_2025_Guide/utils/eval_utils.py",
    "repos/MIDOG_2025_Guide/utils/factory.py",
    "repos/MIDOG_2025_Guide/utils/model.py",
    "repos/MIDOG_2025_Guide/utils/litmodel.py",
]

def fail(message: str) -> None:
    raise SystemExit(f"verify_bundle.py: {message}")

for rel in REQUIRED:
    path = BUNDLE_ROOT / rel
    if not path.is_file():
        fail(f"missing required file: {rel}")
    if not os.access(path, os.R_OK):
        fail(f"required file is not readable: {rel}")

for path in BUNDLE_ROOT.rglob("*"):
    if path.is_symlink():
        target = path.resolve()
        try:
            target.relative_to(BUNDLE_ROOT)
        except ValueError:
            fail(f"symlink points outside bundle: {path.relative_to(BUNDLE_ROOT)} -> {target}")

with (BUNDLE_ROOT / "data/rpi4_test_subset.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    fail("data/rpi4_test_subset.csv contains no rows")
filenames = {row["filename"] for row in rows}
if filenames != {"413.tiff", "366.tiff", "487.tiff"}:
    fail(f"unexpected dataset filenames: {sorted(filenames)}")
for row in rows:
    filename = row["filename"]
    if Path(filename).is_absolute() or ".." in Path(filename).parts:
        fail(f"dataset contains non-portable filename: {filename}")
    if not (BUNDLE_ROOT / "data/images" / filename).is_file():
        fail(f"dataset references missing image: {filename}")

for rel in ["results", "experiments/eval_guide/exported_models"]:
    path = BUNDLE_ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

sys.path.insert(0, str(BUNDLE_ROOT))
sys.path.insert(0, str(BUNDLE_ROOT / "repos/MIDOG_2025_Guide"))
imports = [
    "numpy",
    "pandas",
    "PIL.Image",
    "yaml",
    "torch",
    "torchvision",
    "tqdm",
    "cv2",
    "openslide",
    "evalutils.scorers",
    "torchmetrics.detection",
    "src.benchmark.midog_guide_adapter",
    "src.benchmark.runtime_backends",
    "utils.inference",
    "utils.eval_utils",
    "utils.factory",
]
for name in imports:
    try:
        importlib.import_module(name)
    except Exception as exc:
        fail(f"import failed for {name}: {type(exc).__name__}: {exc}")

import torch
try:
    state = torch.load(BUNDLE_ROOT / "models/fcos18_checkpoint.ckpt", map_location="cpu", weights_only=False)
except Exception as exc:
    fail(f"PyTorch could not load FCOS_18 checkpoint on CPU: {type(exc).__name__}: {exc}")
if not isinstance(state, dict):
    fail(f"checkpoint loaded as unexpected type: {type(state).__name__}")
if torch.cuda.is_available():
    print("verify_bundle.py: CUDA is visible, but benchmark launchers force --device cpu.")

try:
    import onnxruntime as ort
    session = ort.InferenceSession(str(BUNDLE_ROOT / "models/fcos18_fp32.onnx"), providers=["CPUExecutionProvider"])
except Exception as exc:
    fail(f"ONNX Runtime could not load fcos18_fp32.onnx with CPUExecutionProvider: {type(exc).__name__}: {exc}")
providers = session.get_providers()
if "CPUExecutionProvider" not in providers:
    fail(f"ONNX Runtime CPUExecutionProvider unavailable; providers={providers}")
inputs = session.get_inputs()
outputs = session.get_outputs()
if len(inputs) != 1:
    fail(f"expected one ONNX model input, found {len(inputs)}")
roles = {"boxes": False, "scores": False, "labels": False}
for output in outputs:
    shape = list(output.shape)
    if len(shape) == 2 and shape[-1] == 4 and output.type == "tensor(float)":
        roles["boxes"] = True
    elif len(shape) == 1 and output.type == "tensor(float)":
        roles["scores"] = True
    elif len(shape) == 1 and output.type in {"tensor(int64)", "tensor(int32)"}:
        roles["labels"] = True
missing = [key for key, ok in roles.items() if not ok]
if missing:
    fail(f"ONNX outputs are not compatible with detection wrapper; missing roles: {missing}")

config_text = (BUNDLE_ROOT / "configs/FCOS_18_eval.yaml").read_text(encoding="utf-8")
for expected in ["checkpoint: models/fcos18_checkpoint.ckpt", "det_thresh: 0.584", "patch_size: 1024", "weights: null"]:
    if expected not in config_text:
        fail(f"config missing expected setting: {expected}")

print("verify_bundle.py: OK")
'''
(root / "verify_bundle.py").write_text(verify, encoding="utf-8")

readme = """# Raspberry Pi 5 FCOS_18 Benchmark Bundle

This bundle runs the existing MIDOG FCOS benchmark adapter with two CPU backends:

1. PyTorch eager inference
2. ONNX Runtime with `CPUExecutionProvider`

Expected platform: Raspberry Pi 5 with a 64-bit Raspberry Pi OS or other 64-bit Debian-based ARM OS. Check the architecture first:

```bash
uname -m
```

It must print `aarch64`.

## Included Model And Data

- PyTorch checkpoint: `models/fcos18_checkpoint.ckpt`, copied from `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt`
- ONNX model: `models/fcos18_fp32.onnx`, copied from the validated export `experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx`
- Config: `configs/FCOS_18_eval.yaml`
- Test subset: `data/rpi4_test_subset.csv`
- Images: `413.tiff`, `366.tiff`, `487.tiff`

The bundled config uses the same FCOS_18 architecture, checkpoint, detection threshold `0.584`, patch size `1024`, overlap `0.3`, and patch-merge NMS threshold `0.3`. The checkpoint path is made relative to the bundle, and `weights` is set to `null` so torchvision does not try to download ImageNet weights before loading the checkpoint.

## Install

```bash
unzip rpi4_fcos18_benchmark_bundle.zip
cd rpi4_fcos18_benchmark_bundle
chmod +x install_rpi4.sh run_*.sh
./install_rpi4.sh
```

`install_rpi4.sh` installs Debian runtime libraries, creates `.venv`, installs CPU PyTorch/Torchvision, installs `requirements-rpi4.txt`, and runs:

```bash
python verify_bundle.py
```

## Run

```bash
./run_smoke_test.sh
./run_pytorch_benchmark.sh
./run_onnxruntime_benchmark.sh
```

To run everything:

```bash
./run_all_benchmarks.sh
```

Results are written to timestamped directories under:

```text
results/smoke/
results/pytorch_eager/
results/onnxruntime_cpu/
```

The launchers set:

```bash
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
TORCH_NUM_THREADS=4
TORCH_NUM_INTEROP_THREADS=1
```

They use `taskset -c 0-3` when available.

## Monitoring

Temperature and throttling:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
vcgencmd measure_clock arm
```

Peak RAM and runtime:

```bash
/usr/bin/time -v ./run_pytorch_benchmark.sh
/usr/bin/time -v ./run_onnxruntime_benchmark.sh
```

## Regenerate

From the original repository root:

```bash
scripts/create_rpi4_fcos18_bundle.sh
```

The script rebuilds `dist/rpi4_fcos18_benchmark_bundle/`, verifies it, writes `MANIFEST.txt`, `SHA256SUMS`, a size report, and creates `dist/rpi4_fcos18_benchmark_bundle.zip`.

## Known Limitations

- ARM64 installation is documented but cannot be fully validated on non-ARM development hardware.
- Runtime depends on ARM64 wheels being available for the installed Python version.
- OpenSlide and OpenCV are imported by the existing guide inference module even when using ROI TIFF images rather than WSI mode.
- The ONNX model is reused from the included validated FP32 export; the Raspberry Pi should not export ONNX during benchmarking.
"""
(root / "README_RPI4.md").write_text(readme, encoding="utf-8")

for rel in [
    "install_rpi4.sh",
    "run_smoke_test.sh",
    "run_pytorch_benchmark.sh",
    "run_onnxruntime_benchmark.sh",
    "run_all_benchmarks.sh",
    "verify_bundle.py",
]:
    path = root / rel
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY

"$PYTHON_BIN" - "$STAGING" "$REPORT_PATH" "$ZIP_PATH" <<'PY'
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2]).resolve()
zip_path = Path(sys.argv[3]).resolve()

reasons = {
    "README_RPI4.md": "Raspberry Pi installation and benchmark documentation",
    "requirements-rpi4.txt": "Python runtime requirements",
    "install_rpi4.sh": "Raspberry Pi installer",
    "run_smoke_test.sh": "Smoke-test launcher for both backends",
    "run_pytorch_benchmark.sh": "PyTorch eager benchmark launcher",
    "run_onnxruntime_benchmark.sh": "ONNX Runtime benchmark launcher",
    "run_all_benchmarks.sh": "Combined benchmark launcher",
    "verify_bundle.py": "Bundle integrity and runtime verification",
    "configs/FCOS_18_eval.yaml": "Portable FCOS_18 architecture and threshold config",
    "models/fcos18_checkpoint.ckpt": "Exact original FCOS_18 PyTorch checkpoint",
    "models/fcos18_fp32.onnx": "Validated FCOS_18 FP32 ONNX model",
    "data/rpi4_test_subset.csv": "Three-image deterministic benchmark subset",
    "data/rpi4_smoke_subset.csv": "One-image smoke-test subset",
    "data/annotations/midogpp_guide_eval_xvalidation_3img.csv": "Annotation copy for included subset",
}

def reason_for(rel: str) -> str:
    if rel in reasons:
        return reasons[rel]
    if rel.startswith("src/"):
        return "Project benchmark, timing, backend, or import dependency source"
    if rel.startswith("scripts/"):
        return "Existing benchmark support script or original entrypoint reference"
    if rel.startswith("repos/MIDOG_2025_Guide/"):
        return "MIDOG guide inference/model/evaluation runtime dependency"
    if rel.startswith("data/images/"):
        return "Deterministic benchmark source image"
    if rel == "MANIFEST.txt":
        return "Generated file manifest"
    if rel == "SHA256SUMS":
        return "Generated checksums"
    return "Runtime support file"

files = sorted(p for p in root.rglob("*") if p.is_file())

large = [(p, p.stat().st_size) for p in files if p.stat().st_size > 100 * 1024 * 1024]
too_large = [(p, s) for p, s in large if s > 5 * 1024 * 1024 * 1024]
if too_large:
    raise SystemExit(f"Unexpected file larger than 5 GB: {too_large}")

bad_training = [
    p for p in files
    if p.name != "fcos18_checkpoint.ckpt"
    and any(token in str(p.relative_to(root)).lower() for token in ["training_state", "optimizer", "checkpoints/", "wandb", "runs/"])
]
if bad_training:
    raise SystemExit("Unexpected training-state or optimizer-like files included: " + ", ".join(str(p.relative_to(root)) for p in bad_training[:20]))

image_files = sorted((root / "data/images").glob("*.tif*"))
if [p.name for p in image_files] != ["366.tiff", "413.tiff", "487.tiff"]:
    raise SystemExit(f"Unexpected image set: {[p.name for p in image_files]}")

manifest_lines = []
for p in files:
    rel = p.relative_to(root).as_posix()
    manifest_lines.append(f"{rel}\t{p.stat().st_size}\t{reason_for(rel)}")
(root / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

checksum_targets = [
    p for p in sorted(root.rglob("*"))
    if p.is_file() and (
        p.suffix in {".py", ".sh", ".yaml", ".yml", ".csv", ".txt", ".md", ".onnx", ".ckpt"}
        or p.name in {"SHA256SUMS", "MANIFEST.txt"}
    )
]
sum_lines = []
for p in checksum_targets:
    rel = p.relative_to(root).as_posix()
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    sum_lines.append(f"{digest}  {rel}")
(root / "SHA256SUMS").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

files = sorted(p for p in root.rglob("*") if p.is_file())
total = sum(p.stat().st_size for p in files)
if total > 20 * 1024 * 1024 * 1024 and os.environ.get("ALLOW_RPI4_BUNDLE_OVER_20GB") != "1":
    raise SystemExit("Bundle is larger than 20 GB; set ALLOW_RPI4_BUNDLE_OVER_20GB=1 to override.")

def size(paths):
    return sum(p.stat().st_size for p in paths if p.is_file())

categories = {
    "source code": [p for p in files if p.relative_to(root).as_posix().startswith(("src/", "repos/", "scripts/"))],
    "models": [p for p in files if p.relative_to(root).as_posix().startswith(("models/", "experiments/eval_guide/exported_models/"))],
    "test images": [p for p in files if p.relative_to(root).as_posix().startswith("data/images/")],
    "annotations": [p for p in files if p.relative_to(root).as_posix().startswith("data/annotations/") or p.name.endswith("_subset.csv")],
    "Python environment files": [p for p in files if p.name.startswith("requirements") or p.name == "install_rpi4.sh"],
}
categorized = set().union(*[set(v) for v in categories.values()])
categories["other files"] = [p for p in files if p not in categorized]

for label, paths in categories.items():
    print(f"{label}: {size(paths) / (1024 ** 2):.2f} MiB")
print(f"total uncompressed size: {total / (1024 ** 2):.2f} MiB")
print("estimated ZIP size: measured after archive creation")
if total > 10 * 1024 * 1024 * 1024:
    print("WARNING: bundle exceeds 10 GB uncompressed")

if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in files:
        zf.write(p, Path(root.name) / p.relative_to(root))
zip_size = zip_path.stat().st_size

large_lines = "\n".join(
    f"- `{p.relative_to(root).as_posix()}`: {s / (1024 ** 2):.2f} MiB"
    for p, s in large
) or "- None"
included_lines = "\n".join(f"- `{p.relative_to(root).as_posix()}` ({p.stat().st_size} bytes)" for p in files)
report = f"""# Raspberry Pi 5 FCOS_18 Benchmark Bundle Report

## Included Files

{included_lines}

## Large Files Over 100 MB

{large_lines}

## Excluded Categories

- `.git/`, `.github/`, virtual environments, caches, notebook checkpoints, build outputs
- complete MIDOG++ dataset except `413.tiff`, `366.tiff`, `487.tiff`
- FCOS_x50 and FCOS_x101 checkpoints
- pruning, distillation, fine-tuning, optimizer, and training-state artifacts
- old benchmark logs, results, profiling traces, TensorBoard and W&B outputs
- OpenVINO IR files, TensorRT exports and engines, duplicate ONNX exports

## Size

- Total uncompressed size: {total / (1024 ** 2):.2f} MiB
- ZIP size: {zip_size / (1024 ** 2):.2f} MiB

## Exact Model Artifacts

- PyTorch checkpoint source: `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt`
- Bundle checkpoint: `models/fcos18_checkpoint.ckpt`
- ONNX source: `experiments/eval_guide/exported_models/FCOS_18_patch1024_opset18_d5892925a8.onnx`
- Bundle ONNX: `models/fcos18_fp32.onnx`

## Exact Test Images

- `data/images/413.tiff`
- `data/images/366.tiff`
- `data/images/487.tiff`

## Path Modifications

- `configs/FCOS_18_eval.yaml` uses `checkpoint: models/fcos18_checkpoint.ckpt`.
- `configs/FCOS_18_eval.yaml` uses `weights: null` to avoid an ImageNet weight download before checkpoint loading.
- The copied ONNX Runtime backend honors `PACKAGED_ONNX_MODEL`, set by the launchers to `models/fcos18_fp32.onnx`, so the Raspberry Pi reuses the validated export.

## Portability Problems

- Full ARM64 wheel compatibility and thermal behavior must be validated on Raspberry Pi 5 hardware.
- The existing guide inference module imports OpenSlide and OpenCV even for ROI TIFF input.

## Build And Verification Commands

- `scripts/create_rpi4_fcos18_bundle.sh`
- `python verify_bundle.py`
- ZIP creation: Python `zipfile` with deflate compression level 6
"""
report_path.write_text(report, encoding="utf-8")
print(f"ZIP path: {zip_path}")
print(f"ZIP size: {zip_size / (1024 ** 2):.2f} MiB")
print(f"Report path: {report_path}")
PY

(cd "$STAGING" && "$PYTHON_BIN" verify_bundle.py)

echo "Final archive: $ZIP_PATH"
du -h "$ZIP_PATH"
