# Pareto frontier (F1, AP, E2E)

Calibrated INT8 quality rows are cached-prediction rescoring on the 111-image/
8,500-patch final test; E2E is inherited from the corresponding full inference
run. `Pruned60 (step-0)` is not a successfully recovered model.

| Model | Runtime | Precision | Threshold | F1 | AP | E2E s | Size MiB | RAM MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pruned60 (step-0) | ORT | INT8 | 0.546 | 0.820709 | 0.878639 | 2608.88 | 15.70 | 2205.3 |
| Baseline | ORT | INT8 | 0.583 | 0.828469 | 0.885769 | 2785.57 | 18.57 | — |
| Baseline | ORT | FP32 | 0.584 | 0.825050 | 0.885915 | 3913.47 | 71.00 | 2227.1 |
| Baseline | OpenVINO | FP32 | 0.584 | 0.823180 | 0.886192 | 7675.31 | 71.53 | 1934.7 |
