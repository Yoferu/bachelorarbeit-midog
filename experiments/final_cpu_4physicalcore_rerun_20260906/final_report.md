# Corrected final development-PC CPU benchmark (four physical cores)

Date: 2026-09-07  
Host: AMD Ryzen 9 9950X3D, 16 physical cores / 32 logical CPUs  
Controlled affinity: logical CPUs `0,2,4,6` = physical cores `0,1,2,3`  
Workload: MIDOG++ final test, 111 images, 5,383 annotations, 8,500 patches  
Policy: batch 1, workers 0, overlap 0.3, NMS 0.3, detection threshold 0.584, five warm-up patches, CPU only

## A. Runs executed

- Initial diagnostics: eight variants, three repetitions each. The PyTorch and
  OpenVINO diagnostics were controlled, but the four ORT variants used ORT's
  automatic intra-op pool (`0`) and are retained only as invalid diagnostics in
  `diagnostic_invalid_ort_auto_affinity/`.
- Replacement ORT diagnostics: four variants, three repetitions each, explicit
  intra-op 4 / inter-op 1 / sequential execution. All runtime threads reported
  an allowed CPU list of exactly `0,2,4,6`.
- Full PyTorch eager FP32: FCOS_18 and Pruned60.
- Full ORT: FCOS_18 and Pruned60, FP32 and INT8.
- Full OpenVINO INT8: FCOS_18 and Pruned60.
- Reused (not rerun): corrected baseline OpenVINO strict FP32 and OpenVINO
  FP32-storage/BF16-execution measurements from `../openvino_performance_audit_20260906/`.

One stale runner survived an interrupted tool session and began ORT with the old
automatic pool. It was stopped. Its output and every overlapping diagnostic are
preserved under `full_invalid_stale_ort_auto_affinity/` and
`diagnostic_invalid_concurrent_load/`; neither is used below.

## B. Diagnostic reliability

| Variant | Forward repetitions (s) | Mean (s) | Median (s) | Range (s) |
|---|---:|---:|---:|---:|
| ORT FCOS_18 FP32 | 233.390, 232.829, 232.166 | 232.795 | 232.829 | 1.224 |
| ORT Pruned60 FP32 | 196.878, 194.460, 195.283 | 195.541 | 195.283 | 2.418 |
| ORT FCOS_18 INT8 | 105.363, 108.916, 108.359 | 107.546 | 108.359 | 3.553 |
| ORT Pruned60 INT8 | 92.996, 93.102, 98.302 | 94.800 | 93.102 | 5.306 |
| OpenVINO Pruned60 strict FP32 | 173.731, 173.140, 174.016 | 173.629 | 173.731 | 0.875 |

The corresponding initial controlled PyTorch and OpenVINO diagnostic repetitions
are retained in `diagnostic/`. Prediction counts and metrics were identical within
each repeated variant.

### Quantization pilot table

| Model | Backend | Precision | Median forward (s) | Median E2E (s) | FP32 to INT8 forward speedup |
|---|---|---|---:|---:|---:|
| FCOS_18 | ORT | FP32 | 232.829 | 241.975 | 1.000x |
| FCOS_18 | ORT | INT8 | 108.359 | 113.838 | 2.149x |
| Pruned60 | ORT | FP32 | 195.283 | 203.142 | 1.000x |
| Pruned60 | ORT | INT8 | 93.102 | 97.999 | 2.098x |
| FCOS_18 | OpenVINO | strict FP32 | 210.072 | 220.901 | 1.000x |
| FCOS_18 | OpenVINO | INT8 | 58.926 | 62.211 | 3.565x |
| Pruned60 | OpenVINO | strict FP32 | 173.731 | 182.208 | 1.000x |
| Pruned60 | OpenVINO | INT8 | 48.900 | 51.960 | 3.553x |

The three-image quality values are functional sanity checks only and are not
used as definitive test-set accuracy estimates.

## C. Corrected final results

Times exclude model/session construction and five warm-up inferences. Model size
is the deployed artifact size (XML+BIN for OpenVINO); peak RAM is maximum RSS.

| Model | Backend | Precision | F1 | Precision | Recall | AP | Forward (s) | E2E (s) | ms/patch | patches/s | Peak RAM (GiB) | Model size (MB) | Forward speedup vs baseline PyTorch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FCOS_18 | PyTorch eager | FP32 | 0.82505 | 0.80960 | 0.84110 | 0.88592 | 9251.82 | 9261.67 | 1088.45 | 0.919 | 3.07 | 222.50 | 1.00x |
| Pruned60 | PyTorch eager | FP32 | 0.82319 | 0.84442 | 0.80300 | 0.87877 | 7555.64 | 7565.12 | 888.90 | 1.125 | 2.48 | 62.57 | 1.22x |
| FCOS_18 | ORT | FP32 | 0.82505 | 0.80960 | 0.84110 | 0.88592 | 8717.93 | 8729.30 | 1025.64 | 0.975 | 2.13 | 74.44 | 1.06x |
| Pruned60 | ORT | FP32 | 0.82302 | 0.84406 | 0.80300 | 0.87876 | 7258.12 | 7268.91 | 853.90 | 1.171 | 2.14 | 62.72 | 1.27x |
| FCOS_18 | ORT | INT8 | 0.82963 | 0.81962 | 0.83989 | 0.88577 | 5311.88 | 5320.51 | 624.93 | 1.600 | 2.24 | 19.48 | 1.74x |
| Pruned60 | ORT | INT8 | 0.82049 | 0.85105 | 0.79206 | 0.87864 | 4344.61 | 4353.19 | 511.13 | 1.956 | 2.13 | 16.47 | 2.13x |
| FCOS_18 | OpenVINO | strict FP32 | 0.82505 | 0.80960 | 0.84110 | 0.88592 | 7832.73 | 7844.51 | 921.50 | 1.085 | n/a | n/a | 1.18x |
| FCOS_18 | OpenVINO | FP32 storage / BF16 execution | 0.82318 | 0.80379 | 0.84353 | 0.88619 | 3999.54 | 4009.63 | 470.53 | 2.125 | n/a | n/a | 2.31x |
| Pruned60 | OpenVINO | strict FP32 | 0.82319 | 0.84442 | 0.80300 | 0.87876 | 6523.05 | 6534.54 | 767.42 | 1.303 | 1.88 | 63.31 | 1.42x |
| FCOS_18 | OpenVINO | INT8 | 0.82686 | 0.82519 | 0.82854 | 0.87689 | 2343.00 | 2348.50 | 275.65 | 3.628 | 1.42 | 19.98 | 3.95x |
| Pruned60 | OpenVINO | INT8 | 0.81621 | 0.85123 | 0.78395 | 0.87092 | 1944.55 | 1950.96 | 228.77 | 4.371 | 1.45 | 17.03 | 4.76x |

Every newly completed row used affinity `0,2,4,6` and four effective physical
cores. ORT used intra-op 4, inter-op 1, `ORT_SEQUENTIAL`, and CPUExecutionProvider.
PyTorch used intra-op 4 and inter-op 1 with OMP/MKL/OpenBLAS/NumExpr capped at 4.
OpenVINO used `LATENCY`, one stream, four inference threads, CPU pinning, and the
recorded precision hint. New ORT and OpenVINO full metadata show all live runtime
threads restricted to `0,2,4,6`.

## D. Speedups

| Effect | Forward speedup | E2E speedup |
|---|---:|---:|
| PyTorch pruning (FCOS_18 / Pruned60) | 1.224x | 1.224x |
| ORT FP32 pruning | 1.201x | 1.201x |
| ORT INT8 pruning | 1.223x | 1.222x |
| OpenVINO INT8 pruning | 1.205x | 1.204x |
| OpenVINO strict-FP32 pruning | 1.201x | 1.200x |
| ORT FCOS_18 quantization (FP32 / INT8) | 1.641x | 1.641x |
| ORT Pruned60 quantization | 1.671x | 1.670x |
| OpenVINO Pruned60 quantization | 3.355x | 3.349x |

Precision-equivalent backend comparisons: baseline ORT FP32 is 1.061x faster
than baseline PyTorch FP32; baseline OpenVINO strict FP32 is 1.181x faster than
PyTorch and 1.113x faster than ORT FP32. OpenVINO BF16 is a deployment-oriented
mixed-precision result, not a runtime-only FP32 comparison.

## E. Historical versus corrected

The historical ORT measurements are not valid 2C/4T measurements: ORT intra-op
was `0`, allowing its worker-affinity setup to escape the intended resource limit.
The ratio below is historical time / corrected time; values below 1 mean that the
invalid historical run was artificially faster. Historical OpenVINO INT8 used the
old `0-3` two-physical-core affinity; corrected OpenVINO is faster as expected.

| Variant | Historical forward (s) | Corrected 4C forward (s) | Historical / corrected |
|---|---:|---:|---:|
| ORT FCOS_18 FP32 | 3906.19 | 8717.93 | 0.448x |
| ORT Pruned60 FP32 | 3516.79 | 7258.12 | 0.485x |
| ORT FCOS_18 INT8 | 2780.14 | 5311.88 | 0.523x |
| ORT Pruned60 INT8 | 2601.53 | 4344.61 | 0.599x |
| OpenVINO FCOS_18 INT8 | 4152.49 | 2343.00 | 1.772x |
| OpenVINO Pruned60 INT8 | 6135.37 | 1944.55 | 3.155x |

Affinity alone should not make a genuinely restricted four-core ORT run slower
than a two-core run. The inversion, together with ORT's old automatic-thread
setting and the directly verified masks in the corrected runs, is evidence that
the historical ORT timing consumed resources beyond the intended cpuset.

## F. Accuracy sanity check

Baseline PyTorch, ORT FP32, and OpenVINO strict FP32 agree (F1 0.82505, AP
approximately 0.88592). Pruned PyTorch and ORT FP32 differ only at the expected
floating-point boundary level (F1 difference 0.00017; AP difference below
0.000002). Affinity changes did not cause material quality changes. INT8 backend
metrics differ from FP32 and from each other because they are separately
quantized deployment artifacts; their model/threshold provenance was retained.

## G. Reproducibility and provenance

- Commands: `run_benchmarks.sh`; each run also has an expanded `command.txt`.
- Outputs: each `diagnostic/<variant>/repN/` and `full/<variant>/rep1/` contains
  stdout/stderr, command, metrics, predictions, timing JSON/CSV, runtime metadata,
  validation metadata, and `/usr/bin/time -v` resource usage.
- Versions: PyTorch 2.11.0+cu130; ONNX Runtime 1.26.0; OpenVINO
  2026.1.0-21367-63e31528c62.
- Baseline checkpoint SHA-256:
  `143dd591013f06d1277b36fdf94707ca83acd3f5c867a10a59f2bb2e1b1f1343`.
- Pruned60 selected artifact SHA-256:
  `523fd01e0502dcddac0b58a135e4a66fdfb3d739eb6762695ad75df0cb096f54`.
- Baseline/pruned FP32 ONNX SHA-256: `bfb33fc437efcbd70127855177a5b864988eb37f3621096a2e928b7d8aa20a60`,
  `24c599e38a9772c9a51cc72fa603daf88dfab79b02372e9112fe7bfec60a7ab9`.
- Baseline/pruned INT8 ONNX SHA-256: `2275ae12466509ed5eaeea5fedaf3e702b84104e2e5210fea57718363a3aec2b`,
  `42cf1fc0784ec65b871c15d9974bee190ff3791182ef891dc26dcf0f4a3c645e`.
- Baseline/pruned OpenVINO INT8 XML SHA-256:
  `1ec0df449d507b582599f2a72f950634c382c146ab55184258065ba0c85f8ffc`,
  `731513a3182eec3db379c2a20c2576a55859a37eabd1c661a30585ec8bbc4d9c`.

No required final row remains missing. Pruned60 OpenVINO strict FP32 was added
after the initial matrix to complete the precision-matched OpenVINO pruning and
quantization comparisons. Pruned60 OpenVINO BF16 was not run because it is not
required for those comparisons.

## H. Thesis implications

- The earlier ORT timings must not be described as controlled 2C/4T or four-core
  results; automatic ORT worker affinity invalidated the intended CPU budget.
- Under the corrected four-physical-core limit, ORT FP32 improved baseline
  forward time by 1.061x relative to PyTorch eager FP32.
- ORT INT8 reduced forward time by 1.641x for FCOS_18 and 1.671x for Pruned60
  relative to the corresponding ORT FP32 model.
- Structured 60% pruning provided consistent forward speedups of 1.20x to 1.22x
  across PyTorch, ORT FP32/INT8, and OpenVINO INT8.
- OpenVINO strict FP32 was 1.113x faster than ORT FP32 for the same baseline
  network and precision semantics.
- OpenVINO BF16 execution was 2.31x and OpenVINO INT8 was 3.95x faster than the
  PyTorch FP32 baseline, but these are deployment comparisons across precision,
  not pure runtime-only FP32 comparisons.
- The corrected OpenVINO INT8 result reverses the historical apparent regression;
  baseline OpenVINO INT8 reached 3.628 patches/s and Pruned60 reached 4.371
  patches/s under the verified four-core budget.
