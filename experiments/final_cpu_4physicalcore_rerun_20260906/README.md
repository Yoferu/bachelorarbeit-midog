# Final CPU four-physical-core rerun (2026-09-06)

This directory preserves corrected development-PC benchmarks using logical CPUs
`0,2,4,6`, which map to four distinct physical cores on the audited Ryzen host.
Historical `0-3` (2-core/4-thread) artifacts remain untouched in their original
experiment directories.

Run diagnostics with `bash experiments/final_cpu_4physicalcore_rerun_20260906/run_benchmarks.sh --diagnostic`.
After validation, run the full matrix with the same script and `--full`.

Pruned60 OpenVINO strict FP32 is run separately with
`run_pruned60_openvino_strict_fp32.sh --diagnostic` or `--full`.
Corrected baseline OpenVINO strict-FP32 and BF16-execution results are reused
from the OpenVINO audit. See `final_report.md` for the combined comparison.

Both launchers expect prepared artifacts at their declared paths. The main
matrix skips completed rows; the strict-FP32 runner overwrites its output row.
Use a fresh output root when repeating and inspect CPU topology before changing
the host-specific affinity. Raw predictions/logs are local generated artifacts;
compact timing, metric, and runtime records remain as provenance.
