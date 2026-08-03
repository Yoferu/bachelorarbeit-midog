# FPN+Head Prediction Diagnostic

Pre-NMS counts are after torchvision FCOS model-internal filtering per patch and before patch-merge NMS. Post-NMS counts are after patch coordinate merge and class-wise NMS.

| Variant | Artifact SHA256 | Prediction SHA256 | Before Threshold, Pre-NMS | After Threshold, Pre-NMS | After NMS | After Threshold, Post-NMS | TP | FP | FN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fcos18_depgraph_fpn_head_20pct` | `effc7c44bac6` | `e797965acd04` | 14032 | 44 | 4419 | 25 | 21 | 4 | 15 |
| `fcos18_depgraph_fpn_head_30pct` | `bc81be70796a` | `381005782592` | 14418 | 44 | 4613 | 25 | 21 | 4 | 15 |
| `fcos18_depgraph_fpn_head_40pct` | `dbcf255a0f6f` | `98c8498fa247` | 15426 | 44 | 5294 | 25 | 21 | 4 | 15 |

## Per-Image Counts

| Variant | Image | Before Threshold, Pre-NMS | After Threshold, Pre-NMS | After NMS | After Threshold, Post-NMS | Score Min | Score Median | Score Max | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fcos18_depgraph_fpn_head_20pct` | `006.tiff` | 4893 | 14 | 1523 | 6 | 0.0665 | 0.0925 | 0.8809 | 4 | 2 | 2 |
| `fcos18_depgraph_fpn_head_20pct` | `007.tiff` | 5039 | 20 | 1519 | 14 | 0.0655 | 0.0869 | 0.9068 | 12 | 2 | 10 |
| `fcos18_depgraph_fpn_head_20pct` | `014.tiff` | 4100 | 10 | 1377 | 5 | 0.0689 | 0.1002 | 0.8757 | 5 | 0 | 3 |
| `fcos18_depgraph_fpn_head_30pct` | `006.tiff` | 5092 | 14 | 1631 | 6 | 0.0684 | 0.0963 | 0.8832 | 4 | 2 | 2 |
| `fcos18_depgraph_fpn_head_30pct` | `007.tiff` | 5137 | 20 | 1589 | 14 | 0.0679 | 0.0917 | 0.9089 | 12 | 2 | 10 |
| `fcos18_depgraph_fpn_head_30pct` | `014.tiff` | 4189 | 10 | 1393 | 5 | 0.0735 | 0.1065 | 0.8764 | 5 | 0 | 3 |
| `fcos18_depgraph_fpn_head_40pct` | `006.tiff` | 5517 | 14 | 1914 | 6 | 0.0715 | 0.0963 | 0.8775 | 4 | 2 | 2 |
| `fcos18_depgraph_fpn_head_40pct` | `007.tiff` | 5453 | 20 | 1842 | 14 | 0.0712 | 0.0924 | 0.9043 | 12 | 2 | 10 |
| `fcos18_depgraph_fpn_head_40pct` | `014.tiff` | 4456 | 10 | 1538 | 5 | 0.0766 | 0.1067 | 0.8725 | 5 | 0 | 3 |

Top-10 post-NMS boxes/scores for each image are stored in `summary.json` under each image's `top10` field.
