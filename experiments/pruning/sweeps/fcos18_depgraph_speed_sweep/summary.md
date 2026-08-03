# FCOS_18 DepGraph Structural Pruning Speed Sweep

| variant | params_before | params_after | params_reduction_pct | macs_before | macs_after | macs_reduction_pct | changed_layers | smoke_original_detections | smoke_pruned_detections | f1 | precision | recall | false_positives | false_negatives | total_detections | runtime_s | forward_pass_s | mean_patch_latency_s | end_to_end_speedup_pct | forward_speedup_pct | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCOS_18_baseline |  |  |  |  |  |  |  |  |  | 0.6774 | 0.8077 | 0.5833 | 5 | 15 | 26 | 171.4274 | 167.6596 | 0.6408 |  |  | baseline |
| fcos18_depgraph_structural_20pct | 18524487 | 13098439 | 29.2912 | 505502520064.0000 | 449877431040.0000 | 11.0039 | 29 | 178 | 154 | 0.0000 | 0.0000 | 0.0000 | 0 | 36 | 0 | 162.6483 | 159.4847 | 0.6087 | 5.1212 | 4.8759 | reject: detections not plausible |
| fcos18_depgraph_structural_30pct | 18524487 | 10938295 | 40.9522 | 505502520064.0000 | 428453319424.0000 | 15.2421 | 29 | 178 | 176 | 0.0541 | 1.0000 | 0.0278 | 0 | 35 | 1 | 161.1975 | 158.2462 | 0.6034 | 5.9675 | 5.6146 | reject: detections not plausible |
| fcos18_depgraph_head_only_20pct | 18524487 | 18524487 | 0.0000 | 505502520064.0000 | 505502520064.0000 | 0.0000 | 0 | 178 | 178 | 0.6774 | 0.8077 | 0.5833 | 5 | 15 | 26 | 170.3257 | 167.3156 | 0.6379 | 0.6426 | 0.2052 | do not fine-tune yet: speedup below threshold |
| fcos18_depgraph_fpn_head_20pct | 18524487 | 17292935 | 6.6482 | 505502520064.0000 | 462521930496.0000 | 8.5025 | 10 | 178 | 11 | 0.6885 | 0.8400 | 0.5833 | 4 | 15 | 25 | 165.6651 | 162.5504 | 0.6202 | 3.3614 | 3.0474 | do not fine-tune yet: speedup below threshold |
| fcos18_depgraph_backbone_fpn_head_20pct | 18524487 | 13098439 | 29.2912 | 505502520064.0000 | 449877431040.0000 | 11.0039 | 29 | 178 | 154 | 0.0000 | 0.0000 | 0.0000 | 0 | 36 | 0 | 161.3810 | 158.2250 | 0.6040 | 5.8605 | 5.6273 | reject: detections not plausible |

Decision rule: recommend fine-tuning only for variants with >=10% forward-pass speedup or >=8% end-to-end speedup while still producing plausible detections.
