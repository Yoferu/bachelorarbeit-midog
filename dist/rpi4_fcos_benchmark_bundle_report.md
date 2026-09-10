# Raspberry Pi 5 FCOS Benchmark Bundle Report (legacy `rpi4_*` path)

The physical target was a **Raspberry Pi 5 with four Cortex-A76 CPU cores**.
Occurrences of `rpi4` below are historical artifact/path identifiers. This
pre-test bundle report is preparation material, not proof of a completed run.

## Size

- Total uncompressed size: 833.86 MiB
- ZIP size: 692.21 MiB

## Model Artifacts

- `models/fcos18_checkpoint.ckpt`: original FCOS_18 PyTorch checkpoint
- `models/fcos18_fp32.onnx`: original validated FCOS_18 FP32 ONNX
- `models/fcos18_int8.onnx`: static QDQ INT8 quantized FCOS_18 ONNX
- `models/fcos18_pruned60.pt`: selected FPN+Head 60% structurally pruned artifact
- `models/fcos18_pruned60_fp32.onnx`: validated Pruned60 FP32 ONNX export
- `models/fcos18_pruned60_int8.onnx`: static QDQ INT8 quantized Pruned60 ONNX

## Large Files Over 100 MB

- `data/images/366.tiff`: 148.95 MiB
- `data/images/413.tiff`: 123.77 MiB
- `data/images/487.tiff`: 123.77 MiB
- `models/fcos18_checkpoint.ckpt`: 212.19 MiB

## Dataset

- `413.tiff`
- `366.tiff`
- `487.tiff`

## Quantization

```json
{
  "activation_type": "QInt8",
  "calibration_csv": "experiments/distillation/pruned60_clean_recovery/data/calibration_seed42.csv",
  "format": "QDQ",
  "max_calibration_images": 8,
  "max_calibration_patches": 64,
  "method": "ONNX Runtime static post-training quantization",
  "op_types_to_quantize": [
    "Conv",
    "MatMul",
    "Gemm"
  ],
  "per_channel": true,
  "preprocessing": "existing MIDOG ROI_InferenceDataset patch extraction, 1024 patch size, 0.3 overlap, tensor scaling to [0,1]",
  "reports": {
    "fcos18_int8": {
      "float_operator_types_remaining": [
        "Conv"
      ],
      "onnx_path": "dist/rpi4_fcos_artifacts/models/fcos18_int8.onnx",
      "operator_counts": {
        "Add": 106,
        "And": 5,
        "Cast": 84,
        "Ceil": 2,
        "Concat": 103,
        "Constant": 699,
        "ConstantOfShape": 21,
        "Conv": 83,
        "DequantizeLinear": 267,
        "Div": 20,
        "Equal": 6,
        "Expand": 10,
        "Floor": 2,
        "Gather": 146,
        "GatherND": 5,
        "Greater": 5,
        "Identity": 64,
        "If": 1,
        "InstanceNormalization": 40,
        "Less": 10,
        "Max": 10,
        "MaxPool": 2,
        "Min": 11,
        "Mod": 10,
        "Mul": 85,
        "NonZero": 10,
        "Not": 5,
        "Pad": 1,
        "QuantizeLinear": 145,
        "Range": 10,
        "Reciprocal": 2,
        "ReduceMax": 4,
        "ReduceMin": 6,
        "ReduceProd": 1,
        "Relu": 62,
        "Reshape": 167,
        "Resize": 4,
        "Shape": 142,
        "Sigmoid": 10,
        "Slice": 41,
        "Split": 6,
        "Sqrt": 5,
        "Squeeze": 9,
        "Sub": 30,
        "TopK": 5,
        "Transpose": 26,
        "Unsqueeze": 188,
        "Where": 5,
        "Xor": 5
      },
      "quantized_operator_types": [
        "DequantizeLinear",
        "QuantizeLinear"
      ],
      "sha256": "f71952225e18edb01d15d3aa9609727944e0c633cce88af827969b03a3c7bc77",
      "size_bytes": 19476967
    },
    "pruned60_int8": {
      "float_operator_types_remaining": [
        "Conv"
      ],
      "onnx_path": "dist/rpi4_fcos_artifacts/models/fcos18_pruned60_int8.onnx",
      "operator_counts": {
        "Add": 106,
        "And": 5,
        "Cast": 84,
        "Ceil": 2,
        "Concat": 103,
        "Constant": 699,
        "ConstantOfShape": 21,
        "Conv": 83,
        "DequantizeLinear": 267,
        "Div": 20,
        "Equal": 6,
        "Expand": 10,
        "Floor": 2,
        "Gather": 146,
        "GatherND": 5,
        "Greater": 5,
        "Identity": 64,
        "If": 1,
        "InstanceNormalization": 40,
        "Less": 10,
        "Max": 10,
        "MaxPool": 2,
        "Min": 11,
        "Mod": 10,
        "Mul": 85,
        "NonZero": 10,
        "Not": 5,
        "Pad": 1,
        "QuantizeLinear": 145,
        "Range": 10,
        "Reciprocal": 2,
        "ReduceMax": 4,
        "ReduceMin": 6,
        "ReduceProd": 1,
        "Relu": 62,
        "Reshape": 167,
        "Resize": 4,
        "Shape": 142,
        "Sigmoid": 10,
        "Slice": 41,
        "Split": 6,
        "Sqrt": 5,
        "Squeeze": 9,
        "Sub": 30,
        "TopK": 5,
        "Transpose": 26,
        "Unsqueeze": 188,
        "Where": 5,
        "Xor": 5
      },
      "quantized_operator_types": [
        "DequantizeLinear",
        "QuantizeLinear"
      ],
      "sha256": "563e451c82d13488df4f7233ee7103b88fb4fb8a445dc50f6bc574e44efed271",
      "size_bytes": 16536693
    }
  },
  "weight_type": "QInt8"
}
```

## Exclusions

Complete datasets, calibration images, calibration caches, training runs, optimizer states, fine-tuning checkpoints, LR sweeps, distillation outputs, OpenVINO IR files, TensorRT engines, virtual environments, git data, logs, and old results were excluded.

## Validation

Run `./verify_bundle.py` for file/import/model-load/ONNX-session checks. Run `./run_smoke_test.sh` for all six smoke-test executions.
