# Structured Pruning

Structured pruning is evaluated separately from runtime/deployment backends. The
first experiment targets `FCOS_18` only and uses a conservative 10% structured
channel-L2 policy on internal FCOS detection-head convolutions.

## Masked vs Physical Pruning

The current implementation is a masked structured pruning baseline. It uses
`torch.nn.utils.prune.ln_structured` to zero full output-channel filters, then
removes the pruning reparameterization before saving the artifact. The saved
model is loadable as a normal PyTorch state dict, but tensor shapes are
unchanged, so this is not expected to produce large CPU speedups and no speedup
claim is made for this stage.

Physical structured pruning removes channels from tensors and updates dependent
layers so the graph performs less work. That is the preferred direction for real
speedups, but it is deferred until the FCOS/FPN/head dependencies can be handled
safely.

## First Experiment

Config:

```text
experiments/pruning/configs/fcos18_structured_10pct.yaml
```

Artifact and metadata:

```text
experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt
experiments/pruning/models/FCOS_18_structured_head_l2_10pct.meta.json
```

The policy excludes:

- the first input convolution
- backbone and FPN convolutions, because shape changes require dependency-aware
  pruning across skip/FPN connections
- FCOS classification, box regression, and centerness output prediction layers,
  because their output shapes are detection-head contract surfaces
- normalization layers, because the masked baseline only targets Conv2d weights

## Inspect Prunable Modules

```bash
"$HOME/bachelorarbeit-midog/.venv/bin/python" \
  "$HOME/bachelorarbeit-midog/scripts/inspect_prunable_modules.py" \
  --model FCOS_18
```

## Create The 10% Pruned Artifact

```bash
"$HOME/bachelorarbeit-midog/.venv/bin/python" \
  "$HOME/bachelorarbeit-midog/scripts/create_structured_pruned_model.py" \
  --config "$HOME/bachelorarbeit-midog/experiments/pruning/configs/fcos18_structured_10pct.yaml" \
  --output "$HOME/bachelorarbeit-midog/experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt"
```

## Smoke And Quick Benchmark

Use `PRUNED_MODEL_PATH` to load the pruned artifact without modifying the
baseline FCOS config or weights. Result names include the pruned artifact stem so
baseline runtime benchmark results are not overwritten.

```bash
NUM_WORKERS=0 \
PRUNED_MODEL_PATH="$HOME/bachelorarbeit-midog/experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt" \
RUNTIME_BACKEND=pytorch_eager \
"$HOME/bachelorarbeit-midog/scripts/run_smoke_test.sh" FCOS_18
```

```bash
NUM_WORKERS=0 \
PRUNED_MODEL_PATH="$HOME/bachelorarbeit-midog/experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt" \
RUNTIME_BACKEND=pytorch_eager \
PROFILE_PIPELINE=1 \
"$HOME/bachelorarbeit-midog/scripts/run_quick_benchmark.sh" FCOS_18
```

Final accuracy claims require the full benchmark. The quick benchmark is only
for iteration and regression detection.

The first masked 10% artifact passed smoke validation but caused a strong
quality drop:

```text
default FCOS_18 smoke: detections=26, F1=0.6774, recall=0.5833
masked 10% smoke:      detections=6,  F1=0.2857, recall=0.1667
```

Because this masked baseline does not reduce tensor shapes and degraded smoke
quality, no quick benchmark was run for it.

## Fine-Tuning Recovery

Fine-tuning is required because even conservative head-only channel masking
changed the FCOS detection behavior substantially. The first recovery run is
intentionally tiny and reproducible:

```text
config: experiments/pruning/configs/fcos18_structured_10pct_finetune_smoke.yaml
base checkpoint: repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt
pruned checkpoint: experiments/pruning/models/FCOS_18_structured_head_l2_10pct.pt
fine-tuned checkpoint: experiments/pruning/models/FCOS_18_structured_head_l2_10pct_finetuned_smoke.pt
training dataset: data/midogpp_guide_eval_xvalidation.csv
training split: train
optimizer: AdamW
learning rate: 1e-5
max steps: 2
seed: 42
requested device: auto
resolved device: cuda if available, otherwise cpu
```

The fine-tune wrapper keeps the zeroed channel filters zero after each optimizer
step, so the checkpoint remains a masked-pruned artifact rather than regrowing
the pruned weights.

Training is allowed to use GPU. Set `device: auto`, `device: cuda`, or
`device: cpu` in the fine-tune config. `auto` uses CUDA when
`torch.cuda.is_available()` is true and otherwise falls back to CPU. Runtime
Smoke/Quick/Full benchmarks remain CPU-based unless their benchmark environment
is explicitly changed.

`CPU_4C_LIMITED` is a deployment/runtime benchmarking constraint, not a
training constraint. Training device and evaluation device are intentionally
separate: use GPU for pruning recovery when available, then evaluate the
resulting checkpoint through the CPU benchmark scripts.

Check CUDA visibility before launching GPU recovery fine-tuning:

```bash
"$HOME/bachelorarbeit-midog/.venv/bin/python" \
  "$HOME/bachelorarbeit-midog/scripts/check_cuda_environment.py"
```

If the script reports `/dev/dxg` missing or `nvidia-smi` failing inside WSL, do
not install Linux NVIDIA display drivers. Update the Windows NVIDIA driver with
WSL CUDA support, then run:

```powershell
wsl --shutdown
wsl --update
```

Restart WSL and re-run the CUDA check. If CUDA works in a normal WSL terminal
but not in the Codex execution context, run GPU fine-tuning manually from that
normal WSL terminal.

Run the recovery pass:

```bash
"$HOME/bachelorarbeit-midog/.venv/bin/python" \
  "$HOME/bachelorarbeit-midog/scripts/finetune_pruned_model.py" \
  --config "$HOME/bachelorarbeit-midog/experiments/pruning/configs/fcos18_structured_10pct_finetune_smoke.yaml"
```

Then smoke-test it:

```bash
NUM_WORKERS=0 \
PRUNED_MODEL_PATH="$HOME/bachelorarbeit-midog/experiments/pruning/models/FCOS_18_structured_head_l2_10pct_finetuned_smoke.pt" \
RUNTIME_BACKEND=pytorch_eager \
"$HOME/bachelorarbeit-midog/scripts/run_smoke_test.sh" FCOS_18
```

The first tiny recovery run improved the un-finetuned masked baseline but did
not recover the original model quality:

```text
masked 10% smoke:             detections=6, F1=0.2857, recall=0.1667
masked 10% fine-tuned smoke:  detections=8, F1=0.3636, recall=0.2222
default FCOS_18 smoke:        detections=26, F1=0.6774, recall=0.5833
```

An additional `device: auto` smoke fine-tune was run in the current environment.
CUDA was not available, so it resolved to CPU and produced:

```text
masked 10% auto fine-tuned smoke: detections=7, F1=0.3256, recall=0.1944
```

Because CUDA was unavailable in the Codex execution context, longer GPU recovery
configs were added there but not run there:

```text
experiments/pruning/configs/fcos18_structured_10pct_finetune_recovery.yaml
experiments/pruning/configs/fcos18_structured_10pct_finetune_recovery_cuda.yaml
```

When CUDA is visible to PyTorch, run the 500-step CUDA recovery pass with:

```bash
"$HOME/bachelorarbeit-midog/.venv/bin/python" \
  "$HOME/bachelorarbeit-midog/scripts/finetune_pruned_model.py" \
  --config "$HOME/bachelorarbeit-midog/experiments/pruning/configs/fcos18_structured_10pct_finetune_recovery_cuda.yaml"
```

The 500-step CUDA recovery pass was run manually in a normal WSL2 terminal with
CUDA visible to PyTorch:

```text
checkpoint: experiments/pruning/models/FCOS_18_structured_head_l2_10pct_finetuned_recovery_cuda.pt
requested/resolved device: cuda/cuda
CUDA device: NVIDIA GeForce RTX 5090
training split: train
max steps: 500
CPU smoke detections: 27
CPU smoke F1: 0.7302
CPU smoke recall: 0.6389
CPU smoke AP: 0.8006
```

This is a meaningful recovery toward, and on this smoke subset slightly above,
the default FCOS_18 smoke F1/recall. The next validation step is a CPU quick
benchmark using the existing quick test CSV, not the training split.

Training logs are written under `experiments/pruning/logs/`, Lightning CSV logs
and training metadata are written under `experiments/pruning/results/`, and the
loadable checkpoint is written under `experiments/pruning/models/`.

Do not use this masked fine-tuned checkpoint for speedup claims. Runtime
benchmarking should wait for either stronger smoke recovery or a physical
structured-pruning implementation that actually removes compute.
