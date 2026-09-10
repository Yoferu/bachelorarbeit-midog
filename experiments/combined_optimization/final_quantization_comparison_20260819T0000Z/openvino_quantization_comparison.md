# OpenVINO fixed-threshold quantization comparison

All rows are full inference on the 111-image/8,500-patch final test at threshold
0.584. `Pruned60 (step-0)` retains a historical `pruned_recovered` artifact path;
it is not a successfully recovered model.

| Model | Precision | F1 | Precision | Recall | AP | Forward s | E2E s | RAM MiB | Size MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | FP32 | 0.823180 | 0.803785 | 0.843535 | 0.886192 | 7605.41 | 7675.31 | 1934.7 | 71.53 |
| Baseline | INT8 | 0.827195 | 0.825859 | 0.828537 | 0.876927 | 4058.16 | 4159.59 | 1476.3 | 19.05 |
| Pruned60 (step-0) | FP32 | 0.824112 | 0.839714 | 0.809080 | 0.878863 | 5950.70 | 6021.39 | 1819.5 | 60.38 |
| Pruned60 (step-0) | INT8 | 0.816456 | 0.851298 | 0.784353 | 0.870903 | 6065.25 | 6141.89 | 1461.8 | 16.24 |
