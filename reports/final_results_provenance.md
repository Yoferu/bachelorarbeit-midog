# Final Results Provenance

Authoritative thesis-results index, 2026-09-03. Numerical values remain unchanged
in the cited machine-readable artifacts.

## Identity and counting rules

- The full final test is **111 images and 8,500 patches**.
- There are **eight full inference runs**: two models × two backends × two
  precisions.
- Four calibrated INT8 rows are **cached-prediction rescoring**, not additional
  inference or timing runs. Any timing shown for them belongs to the corresponding
  fixed-threshold run.
- The historical artifact key/path `pruned_recovered` means **Pruned60 (step-0
  recovery checkpoint)**. Its selection record has `global_optimizer_step = "0"`.
  It represents the structurally pruned model before a meaningful recovery update,
  not a successfully recovered/fine-tuned model.
- The physical target was a **Raspberry Pi 5 with four Cortex-A76 CPU cores**.
  `rpi4_*` artifact paths are incorrect legacy names retained for provenance.

## Final result index

| Result ID | Model | Checkpoint | Backend | Precision | Evaluation mode | Dataset | Threshold | Main result artifact | Model artifact/hash | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | FCOS_18 baseline | `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt` | ORT CPUExecutionProvider | FP32 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/ptq_20260816T1815Z/baseline/final_test_fp32_fixed/metrics.json` + `timing.json` | ONNX `bfb33fc437efcbd70127855177a5b864988eb37f3621096a2e928b7d8aa20a60` | Complete/A |
| F02 | FCOS_18 baseline | same as F01 | ORT CPUExecutionProvider | INT8 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/ptq_20260816T1815Z/baseline/final_test_fixed/metrics.json` + `timing.json` | QDQ ONNX `2275ae12466509ed5eaeea5fedaf3e702b84104e2e5210fea57718363a3aec2b` | Complete/A |
| F03 | FCOS_18 baseline | same as F01 | ORT CPUExecutionProvider | INT8 | Cached-prediction rescoring of F02 | final test; 111 images/8,500 cached patches | 0.583 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_calibrated/onnxruntime_int8/metrics.json` | same as F02 | Complete/A; no new timing |
| F04 | FCOS_18 baseline | same as F01 | OpenVINO CPU | FP32 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_fixed_fp32/metrics.json` + `timing.json` | XML `38f88baaa12a25049b7e54eec6d5355fa47fdbde63737af66696d8ba0709ad3b`; BIN `2d358d08dbdadcad1454f6d5af869436d038c64bef24e6c0de94ab1255180181` | Complete/A |
| F05 | FCOS_18 baseline | same as F01 | OpenVINO CPU | INT8 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_fixed_int8/metrics.json` + `timing.json` | XML `1ec0df449d507b582599f2a72f950634c382c146ab55184258065ba0c85f8ffc`; BIN `d3e110921075d83f91c047c90a3e4b646a0ce05ebfaf6901468a27dc9cb675f4` | Complete/A |
| F06 | FCOS_18 baseline | same as F01 | OpenVINO CPU | INT8 | Cached-prediction rescoring of F05 | final test; 111 images/8,500 cached patches | 0.600 | `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/final_test_calibrated/openvino_int8/metrics.json` | same as F05 | Complete/A; no new timing |
| F07 | Pruned60 (step-0 recovery checkpoint) | recorded input `experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr3e-5_seed42/student_stable_selected.pt` (`523fd01e…`); audited identity: step 0 | ORT CPUExecutionProvider | FP32 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/ptq_20260816T1815Z/pruned_recovered/final_test_fp32_fixed/metrics.json` + `timing.json` | ONNX `24c599e38a9772c9a51cc72fa603daf88dfab79b02372e9112fe7bfec60a7ab9` | Complete/A |
| F08 | Pruned60 (step-0 recovery checkpoint) | same as F07 | ORT CPUExecutionProvider | INT8 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/ptq_20260816T1815Z/pruned_recovered/final_test_fixed/metrics.json` + `timing.json` | QDQ ONNX `42cf1fc0784ec65b871c15d9974bee190ff3791182ef891dc26dcf0f4a3c645e` | Complete/A |
| F09 | Pruned60 (step-0 recovery checkpoint) | same as F07 | ORT CPUExecutionProvider | INT8 | Cached-prediction rescoring of F08 | final test; 111 images/8,500 cached patches | 0.546 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_calibrated/onnxruntime_int8/metrics.json` | same as F08 | Complete/A; no new timing |
| F10 | Pruned60 (step-0 recovery checkpoint) | same as F07 | OpenVINO CPU | FP32 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_fixed_fp32/metrics.json` + `timing.json` | XML `2c618e70f93b9ebdb0fb3312506fe8ece928d652b5ccf01a9f95d37ccb201fa1`; BIN `0c52fb052166c873360d8930cbbe67866dc955c3b2df93d70f219a33d495d8f9` | Complete/A |
| F11 | Pruned60 (step-0 recovery checkpoint) | same as F07 | OpenVINO CPU | INT8 | Full inference | final test; 111 images/8,500 patches | 0.584 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_fixed_int8/metrics.json` + `timing.json` | XML `731513a3182eec3db379c2a20c2576a55859a37eabd1c661a30585ec8bbc4d9c`; BIN `6bcb87613181a25c890b547bd153ece8ad95b6d4f30661981ce701dde419880d` | Complete/A |
| F12 | Pruned60 (step-0 recovery checkpoint) | same as F07 | OpenVINO CPU | INT8 | Cached-prediction rescoring of F11 | final test; 111 images/8,500 cached patches | 0.565 | `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/final_test_calibrated/openvino_int8/metrics.json` | same as F11 | Complete/A; no new timing |

Threshold provenance:
`experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/threshold_selection.json`
records 39 calibration images, dataset SHA-256
`54af94831a129f8573dc0f7e99b28378f1031cc64a6f6b72e224aca6c5c459a3`,
zero test overlap, and a 0.300–0.800 grid at 0.001 spacing.

## Large/external artifact manifest

These artifacts remain outside Git because they are large or live in intentionally
ignored external/experiment trees. Hashing did not execute the models.

| Artifact | Purpose | Bytes | SHA-256 | Required for exact reproduction | Git storage |
| --- | --- | ---: | --- | --- | --- |
| `repos/FCOS_Inference_CLI/checkpoints/FCOS_18.ckpt` | baseline source checkpoint | 222496424 | `143dd591013f06d1277b36fdf94707ca83acd3f5c867a10a59f2bb2e1b1f1343` | Yes | external/ignored |
| `experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt` | raw selected Pruned60 | 62585201 | `27d78f008ede925991339b5beb3f71f3394937e112c3addc65adbbe9440bbc6b` | Yes, lineage | already tracked |
| `experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr3e-5_seed42/student_stable_selected.pt` | input path recorded by final runtime metadata; audited as step-0-equivalent for result interpretation | 62572869 | `523fd01e0502dcddac0b58a135e4a66fdfb3d739eb6762695ad75df0cb096f54` | Yes | ignored |
| `experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/runs/fcos18_pruned60_supervised_clean_dense_lr1e-4_seed42/student_step_000000.pt` | explicit optimizer-step-0 selection evidence/lineage | 62571589 | `7bda9538d3651a99f620069d30501a799b521b0e2593cec223bbd74e12506cfa` | Provenance comparison | ignored |
| `experiments/quantization/ptq_20260816T1815Z/baseline/export/FCOS_18_patch1024_opset18_d5892925a8.onnx` | baseline ORT FP32 | 74444978 | `bfb33fc437efcbd70127855177a5b864988eb37f3621096a2e928b7d8aa20a60` | Yes | ignored |
| `experiments/quantization/ptq_20260816T1815Z/pruned_recovered/export/FCOS_18_patch1024_opset18_d5892925a8.onnx` | step-0 Pruned60 ORT FP32 | 62723586 | `24c599e38a9772c9a51cc72fa603daf88dfab79b02372e9112fe7bfec60a7ab9` | Yes | ignored; legacy path |
| `experiments/quantization/ptq_20260816T1815Z/baseline/int8/FCOS_18_qdq_int8.onnx` | baseline ORT INT8 | 19476233 | `2275ae12466509ed5eaeea5fedaf3e702b84104e2e5210fea57718363a3aec2b` | Yes | ignored |
| `experiments/quantization/ptq_20260816T1815Z/pruned_recovered/int8/FCOS_18_pruned60_recovered_qdq_int8.onnx` | step-0 Pruned60 ORT INT8 | 16466471 | `42cf1fc0784ec65b871c15d9974bee190ff3791182ef891dc26dcf0f4a3c645e` | Yes | ignored; legacy path/name |
| `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/fp32/FCOS_18_fp32.xml` + `.bin` | baseline OpenVINO FP32 | 929355 + 74080364 | see F04 | Yes | ignored |
| `experiments/quantization/openvino_ptq_20260819T0000Z/baseline/int8/FCOS_18_int8.xml` + `.bin` | baseline OpenVINO INT8 | 1382278 + 18597084 | see F05 | Yes | ignored |
| `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/fp32/FCOS_18_pruned60_recovered_fp32.xml` + `.bin` | step-0 Pruned60 OpenVINO FP32 | 913252 + 62401644 | see F10 | Yes | ignored; legacy path/name |
| `experiments/quantization/openvino_ptq_20260819T0000Z/pruned_recovered/int8/FCOS_18_pruned60_recovered_int8.xml` + `.bin` | step-0 Pruned60 OpenVINO INT8 | 1361395 + 15668444 | see F11 | Yes | ignored; legacy path/name |
| `dist/rpi4_fcos_benchmark_bundle_after_pi_tests/` | Raspberry Pi 5 3-image/228-patch results and bundle | directory | per-file checksums where present in bundle | Device verification | ignored; legacy path |

The compact final comparison JSON, CSV, Markdown, commands, environment, status,
and threshold-selection files in
`experiments/combined_optimization/final_quantization_comparison_20260819T0000Z/`
are explicitly unignored and should be committed with this index.

## Non-final and negative results

- Historical evaluation: 105 images; not the current final test.
- Quick evaluation: 32 images (normally 2,486 patches); non-final.
- Smoke and Raspberry Pi 5 device diagnostic: 3 images; the device run used 228
  patches and is not a final quality evaluation.
- FP16: **no completed FP16 benchmark is included in final results**.
- TVM: import/compilation was attempted on Raspberry Pi 5, but head inference was
  interrupted before numerical validation and `benchmark_started = false`. No
  completed TVM inference benchmark exists.
