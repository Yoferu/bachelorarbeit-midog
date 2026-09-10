# FCOS INT8 PTQ Run

## Scope and status

ONNX Runtime static PTQ succeeded for canonical FCOS_18 and Pruned60. The
historical run key is `pruned_recovered`, and runtime metadata records the input
filename `student_stable_selected.pt`; however, the audited evaluated state is the
**optimizer-step-0 Pruned60 state**, before a meaningful recovery update. These
results must not be described as successful recovery/fine-tuning. Fixed-threshold
full inference and matched CPU_4C_LIMITED timing completed for both models on all
111 images and 8,500 patches.

## Inputs

- Baseline: `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt`, SHA-256 `143dd591013f06d1277b36fdf94707ca83acd3f5c867a10a59f2bb2e1b1f1343`.
- Recorded Pruned60 input path: `experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr3e-5_seed42/student_stable_selected.pt`, SHA-256 `523fd01e0502dcddac0b58a135e4a66fdfb3d739eb6762695ad75df0cb096f54`.
- Scientific identity: optimizer-step-0 Pruned60; no recovery benefit may be
  attributed to this result. The path and hash above are retained as execution
  provenance, not evidence of a meaningful recovery update.
- Calibration: `experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv`, 39 images, 2,289 annotations.
- Split audit: zero train/validation/final-test overlap; final test is 111 images and 5,383 annotations. One deterministic 1024x1024 patch from each calibration image was used.
- Fixed threshold: source policy `det_thresh=0.584`; overlap and NMS are both unchanged from the repository policy (`0.3`).

## Quantization

Both models use ONNX Runtime QDQ static PTQ, MinMax calibration, per-channel symmetric QInt8 weights and asymmetric QUInt8 activations for Conv/Gemm/MatMul. Each resulting graph contains 83 Conv nodes, 145 QuantizeLinear nodes, 267 DequantizeLinear nodes, and 223 INT8/UINT8 initializers. Execution is restricted to `CPUExecutionProvider`; surrounding patch extraction, merge/NMS, and MIDOG scoring remain Python/FP32.

PyTorch eager static INT8 was not used: on PyTorch 2.11 the appropriate forward path is torchao/PT2E, `torchao` is not installed, and legacy eager conversion cannot safely cover this torchvision detection graph. Dynamic Linear-only quantization would be misleading for this convolution-heavy detector. `INT8 + torch.compile` is consequently `not_supported`.

OpenVINO 2026.1 is installed, but NNCF is absent. OpenVINO INT8 is recorded as `unsupported_environment`; no dependency was silently installed. Existing OpenVINO FP32/FP16 measurements remain historical and were not relabelled as INT8.

## Smoke results

| Variant | F1 | AP | Forward s | E2E s | E2E speedup | ONNX MiB |
|---|---:|---:|---:|---:|---:|---:|
| FCOS_18 ONNX FP32 | 0.6774 | 0.7923 | 112.39 | 125.56 | 1.00x | 71.00 |
| FCOS_18 ONNX INT8 | 0.7097 | 0.8034 | 83.92 | 90.48 | 1.39x | 18.57 |
| Pruned60 step-0 ONNX FP32 | 0.7119 | 0.7824 | 97.14 | 106.53 | 1.18x | 59.82 |
| Pruned60 step-0 ONNX INT8 | 0.7018 | 0.7814 | 76.73 | 84.56 | 1.48x | 15.70 |

These three-image quality values cannot substitute for final-test accuracy. The complete fixed-threshold results are reported below. Threshold recalibration was not run because it is outside this reference experiment.

## Final-test fixed-threshold results

| Variant | F1 | Precision | Recall | AP | Forward s | E2E s | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCOS_18 ONNX INT8 | 0.8296 | 0.8196 | 0.8399 | 0.8858 | 2682.91 | 2785.57 | unavailable |
| Pruned60 step-0 ONNX INT8 | 0.8205 | 0.8510 | 0.7921 | 0.8786 | 2465.51 | 2608.88 | 2205 MiB |

Against baseline INT8, Pruned60 step-0 reduces forward time by 8.1% (1.088x
speedup) and end-to-end time by 6.3% (1.068x speedup). F1 changes by -0.00914
and AP by -0.00713 at the unchanged 0.584 threshold. Precision increases by
0.03143 while recall decreases by 0.04783. These differences are attributable to
the pruned/quantized artifact, not demonstrated recovery training.

## Combination matrix

| ID | Model optimization | Quantization | Runtime | Status |
|---|---|---|---|---|
| C0 | none | FP32 | PyTorch eager | historical quick result available |
| C1 | none | FP32 | OpenVINO CPU | historical quick leader: 550.79 s E2E over 32 images |
| C2 | none | INT8 | ONNX Runtime CPU | implemented; smoke measured |
| C3 | Pruned60 step-0 | FP32 | PyTorch eager | supported; final matched run pending |
| C4 | Pruned60 step-0 | FP32 | ONNX Runtime CPU | implemented; smoke measured |
| C5 | Pruned60 step-0 | INT8 | ONNX Runtime CPU | implemented; smoke measured |
| C6 | Pruned60 step-0 | INT8 | torch.compile | not_supported; compile does not wrap an ORT session |

The distilled model is excluded because existing records do not justify its training/inference complexity over supervised recovery. Threshold calibration is a deployment policy, not an architecture row.

## Additivity

On the matched smoke workload, pruning alone is 0.864x baseline forward runtime and baseline INT8 is 0.747x. Independence predicts 0.645x; actual pruned+INT8 is 0.683x, 5.8% slower than predicted. For E2E, independence predicts 0.611x and actual is 0.673x, 10.2% slower. The optimizations help together but are sub-multiplicative.

## Environment

AMD Ryzen 9 9950X3D; affinity `0-3`; CUDA unavailable; PyTorch 2.11.0+cu130; torchvision 0.26.0+cu130; ONNX 1.21.0; ONNX Runtime 1.26.0; OpenVINO 2026.1.0. Thread environment: OMP/MKL/OpenBLAS/NUMEXPR/TORCH = 4, PyTorch interop = 1, batch size 1, workers 0.

See `commands.sh`, per-model `quantization_metadata.json`, smoke validation JSON, `run_status.json`, and the comparison reports in this directory.

The complete FP32/INT8 ONNX Runtime reference matrix is in `onnx_fp32_int8_reference/onnx_fp32_int8_comparison.{csv,json,md}`.
