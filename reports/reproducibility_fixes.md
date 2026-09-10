# Reproducibility Fix Status

Status date: 2026-09-03. Changes are based on
`reports/reproducibility_experiment_audit.md`; no experiment was rerun.

## Corrected issues

| Finding | Change made | Affected files | Resulting status |
| --- | --- | --- | --- |
| `pruned_recovered` implied successful recovery | Added step-0 identity, corrected thesis-facing labels, preserved legacy paths | provenance manifest; combined final report; pruning summary; thesis inventory; README | FIXED |
| Four calibrated rows looked like new inference | Defined cached rescoring, added mode/scope columns, marked timings as inherited | provenance manifest; combined final report; README; benchmarking docs | FIXED |
| 105/32/3-image workloads could be mixed with final 111-image test | Added canonical definitions and visible workload labels | README; benchmarking docs; provenance manifest; pruning summary | FIXED |
| Physical board identity was inconsistent | Recorded confirmed Raspberry Pi 5/four Cortex-A76 identity and labelled `rpi4_*` legacy | README; thesis inventory; bundle READMEs/reports; provenance manifest | FIXED |
| Compact final artifacts were broadly ignored | Narrowed `.gitignore` for the final combined summary package; hash-indexed large artifacts | `.gitignore`; provenance manifest | FIXED |
| Final result lineage was scattered | Added 12-row index separating eight inference runs and four rescores | provenance manifest | FIXED |
| FP16 support could imply a completed result | Explicitly states no completed FP16 benchmark | README; benchmarking docs; provenance manifest | FIXED |
| TVM compilation could imply a benchmark | Classified unsuccessful attempt; retained `benchmark_started=false` and no performance value | provenance manifest; thesis inventory | FIXED |
| Pruning summary said all full work was pending | Limited pending status to eager PyTorch and linked later ORT/OpenVINO finals | `experiments/pruning/final_pruning_summary.md` | FIXED |
| Negative results could disappear | Retained and classified failures, unsupported paths, and incomplete work | provenance manifest; thesis inventory; original audit | DOCUMENTED LIMITATION |
| MobileNet/recovery summaries remain stale | Canonical inventory identifies their status; old raw summaries were not made to imply missing runs completed | thesis inventory; original audit | PARTIALLY FIXED |

## Remaining limitations

- No genuinely updated recovered Pruned60 checkpoint has a final 111-image
  evaluation. Final pruned rows are optimizer step 0.
- Planned eager-PyTorch full baseline/Pruned60 outputs remain absent.
- Originating artifacts for copied upstream FCOS_x101 F1 0.7528 and FCOS_18 F1
  0.7369 remain unavailable locally.
- No completed FP16 or TVM inference benchmark exists.
- Historical 105-image runs lack some environment/checkpoint/dataset revision
  metadata; x50/x101 lack current 111-image evaluations.
- Some raw runtime files lack a complete CPU/OS/package snapshot.
- Prediction caches, weights, ONNX files, OpenVINO BIN files, recovery trees, and
  the Raspberry Pi 5 bundle remain outside Git because of size. They require
  durable external preservation in addition to this checksum index.
- Machine-readable historical artifacts retain `pruned_recovered` as a run key
  and `rpi4_*` as paths. These are intentional provenance identifiers, not current
  scientific labels or hardware identity.
