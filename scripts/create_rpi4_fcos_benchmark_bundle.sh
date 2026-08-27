#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

DIST_DIR="$REPO_ROOT/dist"
BUNDLE_NAME="rpi4_fcos_benchmark_bundle"
STAGING="$DIST_DIR/$BUNDLE_NAME"
ZIP_PATH="$DIST_DIR/${BUNDLE_NAME}.zip"
REPORT_PATH="$DIST_DIR/${BUNDLE_NAME}_report.md"
ARTIFACT_DIR="$DIST_DIR/rpi4_fcos_artifacts"

mkdir -p "$DIST_DIR"

"$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_rpi4_fcos_deployment_artifacts.py"

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

for rel in \
  models/fcos18_checkpoint.ckpt \
  models/fcos18_fp32.onnx \
  models/fcos18_int8.onnx \
  models/fcos18_pruned60.pt \
  models/fcos18_pruned60_fp32.onnx \
  models/fcos18_pruned60_int8.onnx \
  results/deployment_artifacts_report.json; do
  require_file "$ARTIFACT_DIR/$rel"
done

for image in 413.tiff 366.tiff 487.tiff; do
  require_file "$REPO_ROOT/data/midogpp/$image"
done

rm -rf "$STAGING"
mkdir -p "$STAGING"/{configs,data/images,data/annotations,models,results,scripts,src/benchmark,src/pruning,src/distillation,repos/MIDOG_2025_Guide/utils}

for model in fcos18_checkpoint.ckpt fcos18_fp32.onnx fcos18_int8.onnx fcos18_pruned60.pt fcos18_pruned60_fp32.onnx fcos18_pruned60_int8.onnx; do
  copy_file "$ARTIFACT_DIR/models/$model" "$STAGING/models/$model"
done
copy_file "$ARTIFACT_DIR/results/deployment_artifacts_report.json" "$STAGING/models/deployment_artifacts_report.json"
copy_file "$REPO_ROOT/experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.meta.json" "$STAGING/models/fcos18_pruned60.meta.json"
copy_file "$REPO_ROOT/data/midogpp_guide_eval_xvalidation_3img.csv" "$STAGING/data/rpi4_test_subset.csv"
copy_file "$REPO_ROOT/data/midogpp_guide_eval_xvalidation_3img.csv" "$STAGING/data/annotations/midogpp_guide_eval_xvalidation_3img.csv"
for image in 413.tiff 366.tiff 487.tiff; do
  copy_file "$REPO_ROOT/data/midogpp/$image" "$STAGING/data/images/$image"
done

while IFS= read -r file; do
  rel="${file#$REPO_ROOT/}"
  copy_file "$file" "$STAGING/$rel"
done < <(find "$REPO_ROOT/src" -type f -name '*.py' | sort)

for file in \
  "$REPO_ROOT/scripts/summarize_quick_benchmark_timing.py" \
  "$REPO_ROOT/scripts/summarize_rpi4_benchmarks.py" \
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

"$PYTHON_BIN" - "$STAGING" "$REPORT_PATH" "$ZIP_PATH" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2]).resolve()
zip_path = Path(sys.argv[3]).resolve()

def write(path: str, text: str, executable: bool = False) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if executable:
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

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
write("configs/FCOS_18_eval.yaml", config)

pruned_config = config.replace("model_name: FCOS_18", "model_name: FCOS_18_pruned60")
write("configs/FCOS_18_pruned60_eval.yaml", pruned_config)

with (root / "data/rpi4_test_subset.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
fieldnames = list(rows[0].keys())
with (root / "data/rpi4_smoke_subset.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([row for row in rows if row["filename"] == "413.tiff"])

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
if needle not in runtime_text and "PACKAGED_ONNX_MODEL" not in runtime_text:
    raise SystemExit("Could not patch runtime_backends.py for packaged ONNX selection")
if needle in runtime_text:
    runtime_text = runtime_text.replace(needle, replacement, 1)
if "import os\n" not in runtime_text:
    runtime_text = runtime_text.replace("import json\n", "import json\nimport os\n", 1)
strict_validation = """    validation_status = str(validation["status"])
    if validation_status != "passed":
        raise RuntimeBackendUnavailable(
            f"ONNX Runtime validation failed for {export_path}. "
            f"Status: {validation_status}. Details: {validation}"
        )
"""
relaxed_validation = """    validation_status = str(validation["status"])
    if validation_status != "passed":
        if os.environ.get("ALLOW_QUANTIZED_ONNX_VALIDATION_MISMATCH") == "1":
            warnings.append(
                "ONNX Runtime validation did not exactly match PyTorch on the representative patch. "
                f"Status: {validation_status}. This is allowed only for packaged INT8 models; "
                "smoke-test evaluation still runs through the full preprocessing/postprocessing pipeline."
            )
            validation_status = "passed_with_int8_warnings"
        else:
            raise RuntimeBackendUnavailable(
                f"ONNX Runtime validation failed for {export_path}. "
                f"Status: {validation_status}. Details: {validation}"
            )
"""
if strict_validation in runtime_text:
    runtime_text = runtime_text.replace(strict_validation, relaxed_validation, 1)
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
write("requirements-rpi4.txt", requirements)

install = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "This installer expects a 64-bit ARM OS. uname -m is: $(uname -m)" >&2
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
write("install_rpi4.sh", install, executable=True)

run_benchmark = r'''#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$BUNDLE_ROOT"
MODEL=""
RUNTIME=""
PRECISION=""
SMOKE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$MODEL" ] || [ -z "$RUNTIME" ] || [ -z "$PRECISION" ]; then
  echo "Usage: ./run_benchmark.sh --model fcos18|pruned60 --runtime pytorch|onnx --precision fp32|int8 [--smoke]" >&2
  exit 2
fi
if [ "$RUNTIME" = "pytorch" ] && [ "$PRECISION" != "fp32" ]; then
  echo "Unsupported combination: PyTorch INT8 is not implemented in this bundle." >&2
  exit 2
fi
if [ "$RUNTIME" != "pytorch" ] && [ "$RUNTIME" != "onnx" ]; then
  echo "Unsupported runtime: $RUNTIME" >&2
  exit 2
fi
if [ "$MODEL" != "fcos18" ] && [ "$MODEL" != "pruned60" ]; then
  echo "Unsupported model: $MODEL" >&2
  exit 2
fi
if [ "$PRECISION" != "fp32" ] && [ "$PRECISION" != "int8" ]; then
  echo "Unsupported precision: $PRECISION" >&2
  exit 2
fi

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

DATASET="$BUNDLE_ROOT/data/rpi4_test_subset.csv"
RUNS=(warmup1 measured1 measured2 measured3)
if [ "$SMOKE" = "1" ]; then
  DATASET="$BUNDLE_ROOT/data/rpi4_smoke_subset.csv"
  RUNS=(smoke)
fi

CONFIG="$BUNDLE_ROOT/configs/FCOS_18_eval.yaml"
PRUNED_ARGS=()
MODEL_FILE="$BUNDLE_ROOT/models/fcos18_checkpoint.ckpt"
PARAM_COUNT=18524487
if [ "$MODEL" = "pruned60" ]; then
  CONFIG="$BUNDLE_ROOT/configs/FCOS_18_pruned60_eval.yaml"
  PRUNED_ARGS=(--pruned_model_path "$BUNDLE_ROOT/models/fcos18_pruned60.pt")
  MODEL_FILE="$BUNDLE_ROOT/models/fcos18_pruned60.pt"
  PARAM_COUNT=15604807
fi

BACKEND="pytorch_eager"
if [ "$RUNTIME" = "onnx" ]; then
  BACKEND="onnxruntime_cpu"
  if [ "$MODEL" = "fcos18" ] && [ "$PRECISION" = "fp32" ]; then
    MODEL_FILE="$BUNDLE_ROOT/models/fcos18_fp32.onnx"
  elif [ "$MODEL" = "fcos18" ] && [ "$PRECISION" = "int8" ]; then
    MODEL_FILE="$BUNDLE_ROOT/models/fcos18_int8.onnx"
  elif [ "$MODEL" = "pruned60" ] && [ "$PRECISION" = "fp32" ]; then
    MODEL_FILE="$BUNDLE_ROOT/models/fcos18_pruned60_fp32.onnx"
  elif [ "$MODEL" = "pruned60" ] && [ "$PRECISION" = "int8" ]; then
    MODEL_FILE="$BUNDLE_ROOT/models/fcos18_pruned60_int8.onnx"
  fi
  export PACKAGED_ONNX_MODEL="$MODEL_FILE"
  if [ "$PRECISION" = "int8" ]; then
    export ALLOW_QUANTIZED_ONNX_VALIDATION_MISMATCH=1
  else
    unset ALLOW_QUANTIZED_ONNX_VALIDATION_MISMATCH || true
  fi
fi

VARIANT="${RUNTIME}_${PRECISION}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$BUNDLE_ROOT/results/$MODEL/$VARIANT/$STAMP"
suffix=1
while [ -e "$OUT_DIR" ]; do
  OUT_DIR="$BUNDLE_ROOT/results/$MODEL/$VARIANT/${STAMP}_$suffix"
  suffix=$((suffix + 1))
done
mkdir -p "$OUT_DIR"

temp_before=""
temp_after=""
throttled_before=""
throttled_after=""
clock_before=""
clock_after=""
if command -v vcgencmd >/dev/null 2>&1; then
  temp_before="$(vcgencmd measure_temp || true)"
  throttled_before="$(vcgencmd get_throttled || true)"
  clock_before="$(vcgencmd measure_clock arm || true)"
fi

TASKSET=()
if command -v taskset >/dev/null 2>&1; then
  TASKSET=(taskset -c 0-3)
fi

for run_name in "${RUNS[@]}"; do
  "${TASKSET[@]}" /usr/bin/time -v -o "$OUT_DIR/${run_name}_time.txt" "$PYTHON_BIN" "$BUNDLE_ROOT/src/benchmark/midog_guide_adapter.py" \
    --config_file "$CONFIG" \
    --dataset "$DATASET" \
    --guide_repo "$BUNDLE_ROOT/repos/MIDOG_2025_Guide" \
    --img_dir "$BUNDLE_ROOT/data/images" \
    --metrics_output "$OUT_DIR/${run_name}_metrics.json" \
    --predictions_output "$OUT_DIR/${run_name}_predictions.json" \
    --runtime_metadata_output "$OUT_DIR/${run_name}_runtime.json" \
    --runtime_backend "$BACKEND" \
    "${PRUNED_ARGS[@]}" \
    --onnx_opset 18 \
    --export_dir "$BUNDLE_ROOT/results/exports_unused" \
    --split test \
    --device cpu \
    --batch_size 1 \
    --num_workers "$NUM_WORKERS" \
    --overlap 0.3 \
    --nms_thresh 0.3 \
    --profile_pipeline \
    --timing_output_json "$OUT_DIR/${run_name}_timing.json" \
    --timing_output_csv "$OUT_DIR/${run_name}_timing.csv" \
    --overwrite
done

if command -v vcgencmd >/dev/null 2>&1; then
  temp_after="$(vcgencmd measure_temp || true)"
  throttled_after="$(vcgencmd get_throttled || true)"
  clock_after="$(vcgencmd measure_clock arm || true)"
fi

"$PYTHON_BIN" - "$OUT_DIR" "$MODEL" "$RUNTIME" "$PRECISION" "$MODEL_FILE" "$PARAM_COUNT" "$temp_before" "$temp_after" "$throttled_before" "$throttled_after" "$clock_before" "$clock_after" <<'PYRUN'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
model_file = Path(sys.argv[5])
metadata = {
    "model": sys.argv[2],
    "runtime": sys.argv[3],
    "precision": sys.argv[4],
    "model_file": str(model_file),
    "model_size_bytes": model_file.stat().st_size,
    "parameter_count": int(sys.argv[6]),
    "temperature_before": sys.argv[7],
    "temperature_after": sys.argv[8],
    "throttling_before": sys.argv[9],
    "throttling_after": sys.argv[10],
    "arm_clock_before": sys.argv[11],
    "arm_clock_after": sys.argv[12],
}
rss = []
for path in out.glob("*_time.txt"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size" in line:
            rss.append(int(line.rsplit(":", 1)[1].strip()))
metadata["peak_rss_kb"] = max(rss) if rss else None
(out / "variant_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
PYRUN

if [ "$SMOKE" != "1" ]; then
  "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/summarize_quick_benchmark_timing.py" --timing_glob "$OUT_DIR/measured*_timing.json" --output_csv "$OUT_DIR/timing_median.csv"
fi
echo "Outputs: $OUT_DIR"
'''
write("run_benchmark.sh", run_benchmark, executable=True)

write("run_smoke_test.sh", """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for model in fcos18 pruned60; do
  ./run_benchmark.sh --model "$model" --runtime pytorch --precision fp32 --smoke
  ./run_benchmark.sh --model "$model" --runtime onnx --precision fp32 --smoke
  ./run_benchmark.sh --model "$model" --runtime onnx --precision int8 --smoke
done
""", executable=True)

write("run_pytorch_benchmark.sh", """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./run_benchmark.sh --model fcos18 --runtime pytorch --precision fp32
./run_benchmark.sh --model pruned60 --runtime pytorch --precision fp32
""", executable=True)

write("run_onnxruntime_benchmark.sh", """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./run_benchmark.sh --model fcos18 --runtime onnx --precision fp32
./run_benchmark.sh --model fcos18 --runtime onnx --precision int8
./run_benchmark.sh --model pruned60 --runtime onnx --precision fp32
./run_benchmark.sh --model pruned60 --runtime onnx --precision int8
""", executable=True)

write("run_all_benchmarks.sh", """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./run_smoke_test.sh
./run_pytorch_benchmark.sh
./run_onnxruntime_benchmark.sh
python_bin="${PYTHON:-.venv/bin/python}"
if [ ! -x "$python_bin" ]; then python_bin=python3; fi
"$python_bin" scripts/summarize_rpi4_benchmarks.py
""", executable=True)

verify = r'''#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib, json, os, sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "configs/FCOS_18_eval.yaml", "configs/FCOS_18_pruned60_eval.yaml",
    "models/fcos18_checkpoint.ckpt", "models/fcos18_fp32.onnx", "models/fcos18_int8.onnx",
    "models/fcos18_pruned60.pt", "models/fcos18_pruned60_fp32.onnx", "models/fcos18_pruned60_int8.onnx",
    "models/fcos18_pruned60.meta.json", "models/deployment_artifacts_report.json",
    "data/rpi4_test_subset.csv", "data/rpi4_smoke_subset.csv",
    "data/images/413.tiff", "data/images/366.tiff", "data/images/487.tiff",
    "run_benchmark.sh", "scripts/summarize_rpi4_benchmarks.py",
]
def fail(msg: str) -> None:
    raise SystemExit(f"verify_bundle.py: {msg}")
for rel in REQUIRED:
    path = BUNDLE_ROOT / rel
    if not path.is_file():
        fail(f"missing required file: {rel}")
for path in BUNDLE_ROOT.rglob("*"):
    if path.is_symlink():
        target = path.resolve()
        try:
            target.relative_to(BUNDLE_ROOT)
        except ValueError:
            fail(f"symlink points outside bundle: {path.relative_to(BUNDLE_ROOT)} -> {target}")
with (BUNDLE_ROOT / "data/rpi4_test_subset.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
filenames = {row["filename"] for row in rows}
if filenames != {"413.tiff", "366.tiff", "487.tiff"}:
    fail(f"unexpected dataset filenames: {sorted(filenames)}")
for row in rows:
    filename = row["filename"]
    if Path(filename).is_absolute() or ".." in Path(filename).parts:
        fail(f"non-portable filename in dataset: {filename}")
    if not (BUNDLE_ROOT / "data/images" / filename).is_file():
        fail(f"missing image: {filename}")

sys.path.insert(0, str(BUNDLE_ROOT))
sys.path.insert(0, str(BUNDLE_ROOT / "repos/MIDOG_2025_Guide"))
for module in ["numpy", "pandas", "PIL.Image", "yaml", "torch", "torchvision", "cv2", "openslide", "onnx", "onnxruntime", "utils.inference", "utils.factory", "src.benchmark.midog_guide_adapter"]:
    try:
        importlib.import_module(module)
    except Exception as exc:
        fail(f"import failed for {module}: {type(exc).__name__}: {exc}")
import torch, onnxruntime as ort
try:
    torch.load(BUNDLE_ROOT / "models/fcos18_checkpoint.ckpt", map_location="cpu", weights_only=False)
except Exception as exc:
    fail(f"original checkpoint failed to load: {type(exc).__name__}: {exc}")
try:
    artifact = torch.load(BUNDLE_ROOT / "models/fcos18_pruned60.pt", map_location="cpu", weights_only=False)
except Exception as exc:
    fail(f"Pruned60 artifact failed to load: {type(exc).__name__}: {exc}")
if not isinstance(artifact, dict) or "model_object" not in artifact:
    fail("Pruned60 artifact is not the expected custom model_object artifact")
report = json.loads((BUNDLE_ROOT / "models/deployment_artifacts_report.json").read_text(encoding="utf-8"))
for key, info in report["model_files"].items():
    rel = "models/" + Path(info["path"]).name
    if key == "checkpoint":
        rel = "models/fcos18_checkpoint.ckpt"
    path = BUNDLE_ROOT / rel
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"]:
            fail(f"hash mismatch for {rel}")
for rel in ["models/fcos18_fp32.onnx", "models/fcos18_int8.onnx", "models/fcos18_pruned60_fp32.onnx", "models/fcos18_pruned60_int8.onnx"]:
    try:
        session = ort.InferenceSession(str(BUNDLE_ROOT / rel), providers=["CPUExecutionProvider"])
    except Exception as exc:
        fail(f"ONNX Runtime could not load {rel}: {type(exc).__name__}: {exc}")
    if "CPUExecutionProvider" not in session.get_providers():
        fail(f"CPUExecutionProvider missing for {rel}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or list(inputs[0].shape) != [3, 1024, 1024]:
        fail(f"unexpected input shape for {rel}: {[i.shape for i in inputs]}")
    roles = {"boxes": False, "scores": False, "labels": False}
    for output in outputs:
        shape = list(output.shape)
        if len(shape) == 2 and shape[-1] == 4 and output.type == "tensor(float)":
            roles["boxes"] = True
        elif len(shape) == 1 and output.type == "tensor(float)":
            roles["scores"] = True
        elif len(shape) == 1 and output.type in {"tensor(int64)", "tensor(int32)"}:
            roles["labels"] = True
    if not all(roles.values()):
        fail(f"incompatible output roles for {rel}: {roles}")
for path in [BUNDLE_ROOT / "results"]:
    path.mkdir(exist_ok=True)
    probe = path / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
print("verify_bundle.py: OK")
'''
write("verify_bundle.py", verify, executable=True)

readme = """# Raspberry Pi FCOS Benchmark Bundle

This bundle benchmarks six CPU deployment variants on the unchanged three-image diagnostic set: FCOS_18 and FPN+Head 60% Pruned60, each as PyTorch FP32, ONNX Runtime FP32, and ONNX Runtime INT8.

Install on a 64-bit Raspberry Pi OS (`uname -m` must be `aarch64`):

```bash
unzip rpi4_fcos_benchmark_bundle.zip
cd rpi4_fcos_benchmark_bundle
chmod +x install_rpi4.sh run_*.sh
./install_rpi4.sh
```

Run all smoke tests and benchmarks:

```bash
./run_all_benchmarks.sh
```

Run one variant:

```bash
./run_benchmark.sh --model fcos18 --runtime pytorch --precision fp32
./run_benchmark.sh --model fcos18 --runtime onnx --precision fp32
./run_benchmark.sh --model fcos18 --runtime onnx --precision int8
./run_benchmark.sh --model pruned60 --runtime pytorch --precision fp32
./run_benchmark.sh --model pruned60 --runtime onnx --precision fp32
./run_benchmark.sh --model pruned60 --runtime onnx --precision int8
```

Results are timestamped under `results/<model>/<runtime>_<precision>/`. Generate the comparison table with:

```bash
python scripts/summarize_rpi4_benchmarks.py
```

The launchers use batch size 1, patch size 1024, overlap 0.3, NMS 0.3, FCOS detection threshold 0.584, four CPU threads, and one inter-op thread. PyTorch INT8 is intentionally unsupported.

Monitor Raspberry Pi thermals and throttling:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
vcgencmd measure_clock arm
/usr/bin/time -v ./run_benchmark.sh --model pruned60 --runtime onnx --precision int8
```

Model provenance and quantization metadata are in `models/deployment_artifacts_report.json`. Quantization used ONNX Runtime static QDQ INT8 with deterministic calibration patches from `experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv`; calibration images are not packaged.
"""
write("README_RPI4.md", readme)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def reason(rel: str) -> str:
    if rel.startswith("models/"):
        return "model artifact or model provenance metadata"
    if rel.startswith("data/images/"):
        return "three-image diagnostic benchmark source image"
    if rel.startswith("data/"):
        return "diagnostic subset annotations and dataset definition"
    if rel.startswith("src/") or rel.startswith("repos/MIDOG_2025_Guide/"):
        return "existing benchmark/model/evaluation runtime implementation"
    if rel.startswith("scripts/") or rel.endswith(".sh") or rel == "run_benchmark.sh":
        return "benchmark launcher, summarizer, or support script"
    if rel.startswith("configs/"):
        return "portable model configuration"
    return "bundle documentation or verification metadata"

files = sorted(p for p in root.rglob("*") if p.is_file())
if sorted(p.name for p in (root / "data/images").glob("*.tiff")) != ["366.tiff", "413.tiff", "487.tiff"]:
    raise SystemExit("Image subset is not exactly the expected three TIFF files")
bad = [p for p in files if "training_state" in str(p).lower() or "optimizer" in str(p).lower()]
if bad:
    raise SystemExit(f"Unexpected training state files: {bad[:5]}")
large = [(p, p.stat().st_size) for p in files if p.stat().st_size > 100 * 1024 * 1024]
too_large = [(p, s) for p, s in large if s > 5 * 1024 * 1024 * 1024]
if too_large:
    raise SystemExit(f"Unexpected individual file over 5 GB: {too_large}")
total = sum(p.stat().st_size for p in files)
if total > 20 * 1024 * 1024 * 1024 and os.environ.get("ALLOW_RPI4_BUNDLE_OVER_20GB") != "1":
    raise SystemExit("Bundle exceeds 20 GB; set ALLOW_RPI4_BUNDLE_OVER_20GB=1 to override")
if total > 10 * 1024 * 1024 * 1024:
    print("WARNING: bundle exceeds 10 GB uncompressed")

manifest = [f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}\t{reason(p.relative_to(root).as_posix())}" for p in files]
write("MANIFEST.txt", "\n".join(manifest) + "\n")
files = sorted(p for p in root.rglob("*") if p.is_file())
sum_lines = [f"{sha256(p)}  {p.relative_to(root).as_posix()}" for p in files if p.suffix in {'.py','.sh','.yaml','.csv','.txt','.md','.onnx','.ckpt','.pt','.json'} or p.name in {'MANIFEST.txt','SHA256SUMS'}]
write("SHA256SUMS", "\n".join(sum_lines) + "\n")
files = sorted(p for p in root.rglob("*") if p.is_file())

def cat_size(prefixes: tuple[str, ...]) -> int:
    return sum(p.stat().st_size for p in files if p.relative_to(root).as_posix().startswith(prefixes))
print(f"source code: {cat_size(('src/','repos/','scripts/')) / (1024**2):.2f} MiB")
print(f"models: {cat_size(('models/',)) / (1024**2):.2f} MiB")
print(f"test images: {cat_size(('data/images/',)) / (1024**2):.2f} MiB")
print(f"total uncompressed size: {sum(p.stat().st_size for p in files) / (1024**2):.2f} MiB")

if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in files:
        zf.write(p, Path(root.name) / p.relative_to(root))
zip_size = zip_path.stat().st_size
if zip_size > 2 * 1024 * 1024 * 1024:
    print("WARNING: ZIP exceeds 2 GiB")
artifact_report = json.loads((root / "models/deployment_artifacts_report.json").read_text(encoding="utf-8"))
large_lines = "\n".join(f"- `{p.relative_to(root).as_posix()}`: {s / (1024**2):.2f} MiB" for p, s in large) or "- None"
report = f"""# Raspberry Pi FCOS Benchmark Bundle Report

## Size

- Total uncompressed size: {sum(p.stat().st_size for p in files) / (1024**2):.2f} MiB
- ZIP size: {zip_size / (1024**2):.2f} MiB

## Model Artifacts

- `models/fcos18_checkpoint.ckpt`: original FCOS_18 PyTorch checkpoint
- `models/fcos18_fp32.onnx`: original validated FCOS_18 FP32 ONNX
- `models/fcos18_int8.onnx`: static QDQ INT8 quantized FCOS_18 ONNX
- `models/fcos18_pruned60.pt`: selected FPN+Head 60% structurally pruned artifact
- `models/fcos18_pruned60_fp32.onnx`: validated Pruned60 FP32 ONNX export
- `models/fcos18_pruned60_int8.onnx`: static QDQ INT8 quantized Pruned60 ONNX

## Large Files Over 100 MB

{large_lines}

## Dataset

- `413.tiff`
- `366.tiff`
- `487.tiff`

## Quantization

```json
{json.dumps(artifact_report.get('quantization', {}), indent=2, sort_keys=True)}
```

## Exclusions

Complete datasets, calibration images, calibration caches, training runs, optimizer states, fine-tuning checkpoints, LR sweeps, distillation outputs, OpenVINO IR files, TensorRT engines, virtual environments, git data, logs, and old results were excluded.

## Validation

Run `./verify_bundle.py` for file/import/model-load/ONNX-session checks. Run `./run_smoke_test.sh` for all six smoke-test executions.
"""
report_path.write_text(report, encoding="utf-8")
print(f"ZIP path: {zip_path}")
print(f"ZIP size: {zip_size / (1024**2):.2f} MiB")
print(f"Report path: {report_path}")
PY

(cd "$STAGING" && "$PYTHON_BIN" verify_bundle.py)
echo "Final archive: $ZIP_PATH"
du -h "$ZIP_PATH"
