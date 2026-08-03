# FCOS_18 Structural Pruning Evaluation

| model | f1 | precision | recall | false_positives | false_negatives | total_detections | runtime_s | forward_pass_s | mean_patch_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCOS_18_baseline | 0.677419 | 0.807692 | 0.583333 | 5 | 15 | 26 | 168.944594 | 165.365403 | 0.631643 |
| FCOS_18_depgraph_structural_10pct | 0.689655 | 0.909091 | 0.555556 | 2 | 16 | 22 | 166.051862 | 162.892447 | 0.621290 |

Decision: fine-tuning needed

Reasons:
- recall dropped by 0.0278
- total detections shifted by 15.4%
