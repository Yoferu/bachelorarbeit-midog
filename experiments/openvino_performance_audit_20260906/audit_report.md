# OpenVINO final-evaluation performance audit

Date: 2026-09-06

## A. Root cause

The severe historical OpenVINO slowdown was caused primarily by a CPU-topology
benchmark error. The historical command used `taskset -c 0-3` and described it
as four CPU cores. On this SMT host, logical CPUs 0/1 share physical core 0 and
2/3 share physical core 1. OpenVINO therefore compiled the suspect run with only
two effective inference threads, as preserved in
`runtime_openvino_validation.json`. Four distinct physical cores are
`0,2,4,6`; with that mask OpenVINO reports four inference threads.

There is a separate precision-labeling artifact. The historical "FP32"
OpenVINO IR stores FP32 weights, but OpenVINO 2026.1 defaults to a BF16 CPU
inference precision hint on this processor. It must be described as FP32 storage
/ BF16 execution, not strict FP32 compute. Explicit strict-FP32 execution is a
valid but genuinely slow result on this runtime/CPU combination.

Evidence:

- Historical full metadata: 2 inference threads, 1 stream, 7,605.41 s forward.
- Reproduced old-mask diagnostic: 201.72 s / 228 patches (884.75 ms/patch).
- Corrected-mask BF16 diagnostic median: 105.74 s (463.79 ms/patch), 1.91x faster.
- Corrected full BF16-hinted run: 3,929.31 s forward, 4,009.63 s E2E.
- Strict-FP32 diagnostic was stable at 208.79/210.23/208.85 s, and the full
  strict-FP32 run was 7,785.24 s forward. Thus the remaining strict-FP32
  slowdown is genuine for OpenVINO 2026.1 on this host, not compile/warm-up or
  Python pipeline overhead.
- Compilation was outside measured inference: backend setup was 3.10 s (BF16)
  and 3.54 s (F32). Five warm-up calls were also recorded in startup, outside
  the image-processing interval.
- Forward inference accounted for 98.24% (BF16) and 99.39% (F32) of measured
  inference time. Patch merge/NMS was only 0.65 s in each full run, ruling out
  postprocessing as the regression source.

## Configuration comparison

| Setting | Earlier OpenVINO quick | Suspect OpenVINO final | Corrected OpenVINO | ORT final | PyTorch quick |
|---|---|---|---|---|---|
| Model | FCOS-ResNet18 baseline | same | same | same ONNX weights | same checkpoint |
| Workload | 32 images / 2,486 patches | 111 / 8,500 | 111 / 8,500 | 111 / 8,500 | 32 / 2,486 |
| Affinity | unrestricted | `0-3` = 2 physical cores | `0,2,4,6` = 4 physical cores | historical `0-3` | unrestricted |
| Runtime threads | default; not recorded | OpenVINO reported 2 | OpenVINO reported 4 | ORT auto pool | Torch intra-op 4 |
| Hint / streams | default LATENCY / 1 | default LATENCY / 1 | LATENCY / 1 | sequential session | n/a |
| Storage / execution | FP32 IR / BF16 | FP32 IR / BF16 | FP32 IR / BF16 and strict F32 runs | FP32 ONNX / FP32 | FP32 |
| Batch / workers | 1 / 4 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 4 |
| Warm-up | separate process; compilation repeated | validation patch only | 5 calls in same process | diagnostic: 5 calls | separate process |
| Compile in forward time | no | no | no | no | n/a |
| Threshold / overlap / NMS | 0.584 / 0.3 / 0.3 | same | same | same | same |

The adapter creates each patch as a CPU Torch tensor. Both wrappers perform a
zero-copy-compatible `detach().cpu().numpy().astype(float32, copy=False)` inside
the patch loop and convert outputs back with `torch.from_numpy`. OpenVINO also
creates a result mapping for each call. These are real wrapper overheads, but
the stage timings show they are included in `forward_pass` and do not explain
the topology-dependent 1.94x change. No dynamic recompilation, device transfer,
or IR conversion occurs in the measured loop.

Patch generation, ordering, preprocessing, exported FCOS postprocessing,
thresholding, and MIDOG merge/NMS all use the same guide adapter. Batch size is
one. The full dataset contains 111 test filenames, 5,383 annotation rows, and
8,500 generated patches. The three-image diagnostic CSV generated 228 patches.

## Diagnostic comparison

| Backend | Precision | Patches | Forward time | ms/patch | E2E time | Notes |
|---|---|---:|---:|---:|---:|---|
| OpenVINO old mask | FP32 storage / BF16 execute | 228 | 201.72 s | 884.75 | 215.31 s | reproduction; 2 physical cores |
| OpenVINO corrected | FP32 storage / BF16 execute | 228 | 105.41, 105.74, 106.36 s | 463.79 median | 114.04, 114.51, 115.16 s | mean 105.84, SD 0.48, range 0.95 s |
| ORT controlled | FP32 | 228 | 99.04, 99.89, 100.22 s | 438.13 median | 107.80, 109.40, 110.56 s | mean 99.72, SD 0.61, range 1.17 s |
| OpenVINO strict | FP32 | 228 | 208.79, 208.85, 210.23 s | 915.99 median | 220.30, 220.90, 221.84 s | mean 209.29, SD 0.82, range 1.44 s |

For ORT, forcing `intra_op_num_threads=4` was tested and rejected: direct warmed
latency rose from approximately 0.423-0.426 s to 0.991 s/patch. The controlled
ORT run therefore uses its sequential auto-sized pool inside the identical
four-physical-core process affinity. This is recorded as `intra_op=0` (auto),
not falsely reported as four runtime-native threads.

## B. Old vs corrected OpenVINO result

| Measurement | Forward | E2E | ms/patch | Status |
|---|---:|---:|---:|---|
| Previous final result | 7,605.41 s | 7,675.31 s | 894.75 | suspect: only 2 physical cores; BF16 execution mislabeled FP32 |
| Reproduced old config (diagnostic) | 201.72 s | 215.31 s | 884.75 | reproduced |
| Corrected controlled config | 3,929.31 s | 4,009.63 s | 462.27 | valid FP32-storage/BF16-execution result |
| Corrected strict-FP32 config | 7,785.24 s | 7,844.51 s | 915.91 | valid precision-equivalent FP32 result |

## C. Final runtime comparison

Only genuinely equivalent full rows are compared below. The repository contains
no 111-image eager-PyTorch full result; its speedup cannot be calculated without
executing an additional full baseline and is therefore left unavailable.

| Backend | Precision | F1 | AP | Forward | E2E | Forward speedup vs PyTorch |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | FP32 | unavailable | unavailable | unavailable | unavailable | 1.00x reference unavailable |
| ONNX Runtime | FP32 | 0.825050 | 0.885915 | 3,780.11 s | 3,913.47 s | unavailable |
| OpenVINO | FP32 | 0.825050 | 0.885915 | 7,785.24 s | 7,844.51 s | unavailable |

At strict FP32, ORT is 2.060x faster in forward time and 2.004x faster E2E.
The corrected BF16-hinted OpenVINO run reaches 2.163 patches/s, but it is not a
precision-equivalent runtime-only comparison against ORT FP32. Its F1 is
0.823180 and AP 0.886192 because BF16 execution changes some exported detection
outputs near filtering/NMS boundaries.

Full strict-FP32 OpenVINO details: precision 0.809598, recall 0.841103,
F1 0.825050, AP 0.885915, 1.092 patches/s, peak RSS 2,146,692 KiB, and model size
75,009,719 bytes (XML + BIN). The corrected BF16-hinted run used peak RSS
1,907,776 KiB.

## D. Thesis interpretation

- Measured result: the historical OpenVINO final run used two physical cores,
  despite being documented as a four-core run; its timing is not a valid
  four-core performance result.
- Identified artifact: logical CPUs `0-3` map to two SMT cores on the benchmark
  host. Selecting `0,2,4,6` gives four distinct physical cores and reduced the
  BF16-hinted OpenVINO full forward time from 7,605 s to 3,929 s (1.94x).
- Identified artifact: the OpenVINO "FP32" artifact stored FP32 weights but
  defaulted to BF16 CPU execution. Storage precision and execution precision
  must be reported separately.
- Measured result: under strict FP32 execution and equivalent model provenance,
  OpenVINO required 7,785 s forward versus 3,780 s for ORT; this remaining
  2.06x difference is genuine for the tested OpenVINO 2026.1 / Ryzen 9 9950X3D
  environment.
- Measured result: compilation, warm-up, preprocessing, and Python NMS did not
  cause the slowdown; strict-FP32 forward execution accounted for 99.39% of the
  measured inference interval.
- Interpretation: the old OpenVINO number must not be presented as a valid
  four-core FP32 runtime result. Use either the corrected BF16 row with explicit
  mixed-precision labeling or the strict-FP32 row for a runtime-only FP32
  comparison.
