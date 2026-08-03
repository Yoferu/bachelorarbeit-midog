# FCOS_18 DepGraph Extreme Boundary Test

| variant | params_before | params_after | params_reduction_pct | macs_before | macs_after | macs_reduction_pct | changed_layers | smoke_original_detections | smoke_pruned_detections | f1 | precision | recall | false_positives | false_negatives | total_detections | runtime_s | forward_pass_s | mean_patch_latency_s | end_to_end_speedup_pct | forward_speedup_pct | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCOS_18_baseline |  |  |  |  |  |  |  |  |  | 0.6774 | 0.8077 | 0.5833 | 5 | 15 | 26 | 163.2286 | 159.5959 | 0.6102 |  |  | baseline |
| fcos18_depgraph_structural_40pct | 18524487 | 9150119 | 50.6053 | 505502520064.0000 | 407799239424.0000 | 19.3280 | 29 | 178 | 114 | 0.0000 | 0.0000 | 0.0000 | 0 | 36 | 0 | 153.7369 | 150.6807 | 0.5751 | 5.8150 | 5.5862 | reject: eval detections collapsed to 0 |
| fcos18_depgraph_structural_50pct | 18524487 | 7668551 | 58.6032 | 505502520064.0000 | 390383817472.0000 | 22.7731 | 29 | 178 | 180 | 0.0000 | 0.0000 | 0.0000 | 0 | 36 | 0 | 150.7720 | 147.6488 | 0.5639 | 7.6314 | 7.4859 | reject: eval detections collapsed to 0 |
| fcos18_depgraph_fpn_head_30pct | 18524487 | 16834247 | 9.1244 | 505502520064.0000 | 445605729024.0000 | 11.8490 | 10 | 178 | 155 | 0.6885 | 0.8400 | 0.5833 | 4 | 15 | 25 | 158.0242 | 154.8590 | 0.5914 | 3.1884 | 2.9681 | do not fine-tune: forward speedup below 10% |
| fcos18_depgraph_fpn_head_40pct | 18524487 | 16417031 | 11.3766 | 505502520064.0000 | 429591958272.0000 | 15.0169 | 10 | 178 | 21 | 0.6885 | 0.8400 | 0.5833 | 4 | 15 | 25 | 156.7363 | 153.5568 | 0.5866 | 3.9774 | 3.7840 | do not fine-tune: forward speedup below 10% |

Boundary decision rules: reject immediately if eval detections collapse to 0; reject if F1 < 0.2; do not fine-tune if forward speedup < 10%; recommend fine-tuning only if forward speedup is >=10-15% and detections remain plausible.
