# CPU affinity audit for thesis benchmark results

Audit date: 2026-09-06. No inference or evaluation was executed for this audit.

## Classification rules

On the development PC (AMD Ryzen 9 9950X3D, SMT2), Linux logical CPUs map in
sibling pairs: 0/1 are physical core 0, 2/3 core 1, 4/5 core 2, and so on.
Consequently, `taskset -c 0-3` permits four logical CPUs but only **two physical
cores**. A mask such as `0,2,4,6` permits four distinct physical cores.

On the physical Raspberry Pi 5, CPUs 0-3 are four separate Cortex-A76 physical
cores; the same textual taskset mask is therefore correct there.

“Absolute 4P validity” below asks whether the result measures the intended four
physical cores. “Relative validity” asks the narrower question whether a stated
comparison remains useful because all compared candidates shared the same
resource policy. A “yes” in the latter column does not convert a two-core result
into a four-core result.

## Audited benchmark inventory

| Thesis benchmark/result | Scope | Recorded affinity/resource evidence | Effective physical cores | Backend | Absolute 4P validity | Relative comparison validity |
|---|---|---|---:|---|---|---|
| Original FCOS_18/x50/x101 historical evaluation (E01) | 105-image quality evaluation | No affinity recorded; metrics are not runtime claims | Unknown | PyTorch eager | N/A for quality; no valid 4P runtime | Quality comparisons remain usable on the historical split; no CPU-runtime claim |
| `FCOS_18_CPU_4C_LIMITED` (E02) | 20-image initial model-selection runtime | Explicit `taskset -c 0-3`; log CPU use 393% | **2** on PC | PyTorch eager | **No** | **Yes, conditional:** fair against x50/x101 under the same 2C/4T mask |
| `FCOS_x50_CPU_4C_LIMITED` (E02) | Same | Explicit `taskset -c 0-3`; 393% | **2** | PyTorch eager | **No** | **Yes, conditional:** same model-family comparison |
| `FCOS_x101_CPU_4C_LIMITED` (E02) | Same | Explicit `taskset -c 0-3`; 393% | **2** | PyTorch eager | **No** | **Yes, conditional:** same model-family comparison |
| FCOS_18 eager quick CPU (E03) | 32 images / 2,486 patches, repeated | No `taskset`; Torch configured to 4 intra-op threads, 4 data workers; log shows up to 1536% aggregate CPU | Not fixed; not demonstrably only 2 | PyTorch eager | **No: uncontrolled/unpinned**, not a strict 4P result | Repetitions within this row are comparable; cross-backend 4P claims are not controlled |
| FCOS_18 `torch.compile` quick (E04) | Same quick set | No `taskset`; Torch 4-thread setting; up to 1500% aggregate CPU | Not fixed | PyTorch/Inductor | **No: uncontrolled/unpinned** | Conditional versus eager because commands/workload match, but not an absolute 4P comparison |
| FCOS_18 ORT FP32 quick (early E05) | Same quick set | No `taskset`; ORT default pool; log up to 3165% aggregate CPU | Not fixed; potentially all host cores | ONNX Runtime FP32 | **No** | **No for a resource-matched backend claim**; usable only as host-default runtime behavior |
| FCOS_18 OpenVINO quick, compressed/default and explicit FP32-storage IR (early E06) | Same quick set | No `taskset`; OpenVINO defaults; log about 1300% CPU | Not fixed; potentially many host cores | OpenVINO CPU | **No** | **No for a four-core backend comparison**; internally useful for OpenVINO repeatability/storage-format checks |
| DepGraph pruning-scope smoke sweep (E09) | 3-image smoke | Commands have no `taskset`; adapter CPU thread default 4 | Not fixed; not demonstrably only 2 | PyTorch eager | **No: unpinned diagnostic** | **Yes, conditional:** variants share command/workload; smoke scope and scheduling noise remain |
| DepGraph pruning-ratio/boundary quick sweeps (E10/E11) | 3-image smoke and 32-image quick | Commands have no `taskset`; adapter CPU thread default 4 | Not fixed; not demonstrably only 2 | PyTorch eager | **No: unpinned diagnostic** | **Yes, conditional:** baseline and pruning variants share the same protocol; pruning ranking remains useful, not a strict 4P latency claim |
| Matched baseline/Pruned60/MobileNet three-image comparison (E12) | 228 patches, 1 warm-up + 3 repeats | Explicitly documented **no CPU affinity**; Torch intra-op 4/inter-op 1 | Not fixed; scheduler could use four different cores | PyTorch eager | **No: unpinned diagnostic** | **Yes:** strong within-run model comparison; cannot be labeled pinned four-physical-core performance |
| Final ORT FP32 baseline (E18) | 111 images / 8,500 patches | Explicit `taskset -c 0-3` | **2** | ONNX Runtime FP32 | **No** | **Yes versus Pruned60 ORT FP32**; conditional for other comparisons |
| Final ORT FP32 Pruned60 step-0 (E18) | Same | Explicit `taskset -c 0-3` | **2** | ONNX Runtime FP32 | **No** | **Yes versus baseline ORT FP32** under identical 2C/4T resources |
| Final ORT INT8 baseline (E19) | Same | Explicit `taskset -c 0-3` | **2** | ONNX Runtime QDQ INT8 | **No** | **Yes versus baseline ORT FP32 and Pruned60 ORT INT8**, as a 2C/4T ORT comparison |
| Final ORT INT8 Pruned60 step-0 (E19) | Same | Explicit `taskset -c 0-3` | **2** | ONNX Runtime QDQ INT8 | **No** | **Yes within ORT** under identical incorrect affinity |
| Final OpenVINO FP32-storage baseline (historical E20) | Same | Explicit `taskset -c 0-3`; validation records 2 inference threads | **2** | OpenVINO, FP32 storage/BF16 execution | **No** | **Yes versus matching OpenVINO candidates only as a 2C/4T/default-policy comparison** |
| Final OpenVINO FP32-storage Pruned60 step-0 (historical E20) | Same | Explicit `taskset -c 0-3`; 2 effective inference threads | **2** | OpenVINO, FP32 storage/BF16 execution | **No** | **Yes within historical OpenVINO policy**; precision label must still be corrected |
| Final OpenVINO INT8 baseline (E21) | Same | Explicit `taskset -c 0-3`; validation records 2 inference threads | **2** | OpenVINO/NNCF INT8 | **No** | **Yes within OpenVINO on the same 2C/4T envelope**; not transferable to 4P |
| Final OpenVINO INT8 Pruned60 step-0 (E21) | Same | Explicit `taskset -c 0-3`; 2 effective inference threads | **2** | OpenVINO/NNCF INT8 | **No** | **Yes, conditional:** anomaly is measured for this 2C/4T/default configuration only |
| Calibrated final ORT/OpenVINO INT8 rows | Cached rescoring of E19/E21 predictions | No new inference; inherit source-run timing and affinity | Inherit **2** | ORT/OpenVINO INT8 | **No new 4P timing** | Quality rescoring remains valid; referenced runtime has the same qualifications as E19/E21 |
| Final eager-PyTorch PC baseline/Pruned60 | Intended 111-image benchmark | Commands exist only as a plan; no completed artifacts | Not executed | PyTorch eager | **Not available** | No final PC eager comparison exists to validate or invalidate |
| Corrected OpenVINO BF16-hinted audit run | 111 images / 8,500 patches | Explicit `taskset -c 0,2,4,6`; runtime reports 4 threads | **4** | OpenVINO, FP32 storage/BF16 execution | **Yes for 4P**, but not strict FP32 | Valid absolute mixed-precision result; not precision-equivalent to ORT FP32 |
| Corrected OpenVINO strict-FP32 audit run | Same | Explicit `taskset -c 0,2,4,6`; runtime reports 4 threads | **4** | OpenVINO FP32 | **Yes** | Valid four-physical-core FP32 result; the old ORT final remains a 2-core result, so their full-run timing comparison is not resource-matched |
| Physical Pi 5 FCOS_18 PyTorch FP32 (E23) | 3 images / 228 patches, 3 repeats | Launcher uses `taskset -c 0-3`; Pi 5 has four Cortex-A76 cores, no SMT sibling pairing | **4** | PyTorch eager | **Yes** for physical-device diagnostic | **Yes** across all six Pi variants |
| Physical Pi 5 FCOS_18 ORT FP32 (E23) | Same | Same physical Pi mask | **4** | ONNX Runtime FP32 | **Yes** | **Yes** |
| Physical Pi 5 FCOS_18 ORT INT8 (E23) | Same | Same physical Pi mask | **4** | ONNX Runtime QDQ INT8 | **Yes** | **Yes** |
| Physical Pi 5 Pruned60 PyTorch FP32 (E23) | Same | Same physical Pi mask | **4** | PyTorch eager | **Yes** | **Yes** |
| Physical Pi 5 Pruned60 ORT FP32 (E23) | Same | Same physical Pi mask | **4** | ONNX Runtime FP32 | **Yes** | **Yes** |
| Physical Pi 5 Pruned60 ORT INT8 (E23) | Same | Same physical Pi mask | **4** | ONNX Runtime QDQ INT8 | **Yes** | **Yes** |
| Physical Pi TVM attempt (E24) | One patch intended | Four-core Cortex-A76 target, but inference was interrupted before validation/benchmark | 4 available; no completed run | TVM/LLVM | **N/A** | No benchmark result exists |

## Interpretation of shared incorrect affinity

### Comparisons that remain internally useful

1. The three original `CPU_4C_LIMITED` PyTorch model runs all used the same
   `0-3` mask. Their **relative model ranking on two physical cores/four SMT
   threads** remains usable. Their absolute times must not be described as
   four-physical-core times.
2. Baseline-versus-Pruned60 comparisons within ORT FP32, within ORT INT8, within
   OpenVINO FP32-storage/BF16 execution, and within OpenVINO INT8 used the same
   incorrect mask and pipeline. These relative model comparisons remain
   internally controlled for the 2C/4T configuration.
3. ORT FP32-versus-INT8 comparisons within the same model also share the same
   affinity and remain useful as quantization comparisons for that configuration.
   The same is true for OpenVINO's own format/quantization comparisons, subject
   to its separate precision-labeling issue.
4. Cached threshold rescoring changes no runtime. Its quality metrics are not
   invalidated by CPU affinity; only the timings referenced from the source
   inference runs inherit the 2C/4T caveat.

### Comparisons requiring stronger qualification

The historical ORT-versus-OpenVINO comparison gave both processes the same OS
cpuset, so it was procedurally symmetric as a **four-logical-CPU/default-runtime
comparison**. It was not a valid four-physical-core comparison. More
importantly, the runtimes interpreted the topology differently: OpenVINO chose
two inference threads, while ORT used its auto-sized pool within the same
cpuset. The corrected audit showed that OpenVINO is unusually sensitive to this
mask. Therefore the old cross-backend speed ratio should not be used to infer
the ranking under four physical cores, even though neither individual run was
fabricated or generally “invalid.”

The early eager/Inductor/ORT/OpenVINO quick comparison is weaker: it had no
affinity at all, and the logs show substantially different aggregate CPU usage
(roughly 13-15 logical CPUs for eager/OpenVINO and up to 31 for ORT). It remains
evidence of host-default behavior, not a controlled four-core backend result.

## Bottom line

- Every completed final development-PC ORT and historical OpenVINO run in
  E18-E21 used `taskset -c 0-3` and therefore only two physical cores.
- No completed final eager-PyTorch 111-image PC run exists.
- The original model-selection `CPU_4C_LIMITED` runs also used only two physical
  cores.
- The pruning and distillation diagnostics were generally unpinned, not pinned
  to the same two-core mistake. Their matched within-family comparisons remain
  useful but are not strict four-physical-core measurements.
- The physical Raspberry Pi 5 results are unaffected: CPUs 0-3 are four real
  physical cores on that device.
- A shared wrong mask preserves many within-backend/model relative comparisons,
  but it does not preserve the intended absolute four-core meaning and does not
  guarantee a transferable ORT-versus-OpenVINO comparison.
