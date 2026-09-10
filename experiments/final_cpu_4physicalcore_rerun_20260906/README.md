# Final CPU four-physical-core rerun (2026-09-06)

This directory preserves corrected development-PC benchmarks using logical CPUs
`0,2,4,6`, which map to four distinct physical cores on the audited Ryzen host.
Historical `0-3` (2-core/4-thread) artifacts remain untouched in their original
experiment directories.

Run diagnostics with `bash experiments/final_cpu_4physicalcore_rerun_20260906/run_benchmarks.sh --diagnostic`.
After validation, run the full matrix with the same script and `--full`.

The matrix deliberately excludes Pruned60 OpenVINO FP32/BF16: those rows are not
required to replace an existing final comparison. Corrected baseline OpenVINO
strict-FP32 and BF16-execution results are reused from the OpenVINO audit.
