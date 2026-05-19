# Repository Structure

This repository is the tracked project layer for MIDOG FCOS benchmarking. Large
datasets and upstream clones stay local and gitignored.

```text
bachelorarbeit-midog/
  README.md
  docs/
    benchmarking.md
    repository_structure.md
  scripts/
    create_quick_benchmark_subset.py
    run_smoke_test.sh
    run_quick_benchmark.sh
    run_full_benchmark.sh
    summarize_quick_benchmark_timing.py
  src/
    benchmark/
      benchmark_config.py
      midog_guide_adapter.py
      pipeline_timing.py
      result_writer.py
  experiments/
    eval_guide/
      configs/
      logs/
      results/
      runs/
  data/
  repos/
```

`data/` contains local MIDOG/MIDOG++ files and generated CSV subsets. It is
gitignored.

`repos/` contains external repositories such as `MIDOG_2025_Guide`. It is
gitignored and should be treated as read-only by benchmark workflows.

`src/benchmark/midog_guide_adapter.py` is the tracked boundary to
`repos/MIDOG_2025_Guide`. It reuses the guide repository's model, inference, and
metric utilities at runtime, but optional timing instrumentation is applied
in memory only. No benchmark functionality requires uncommitted modifications
inside `repos/`.

No evaluator code was copied into `src/midog_eval/`. If that becomes necessary
later, add `src/midog_eval/README.md` with the source repository, source commit,
copied files, local modifications, and license notice.
