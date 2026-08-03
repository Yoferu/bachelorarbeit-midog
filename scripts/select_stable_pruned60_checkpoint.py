#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.stable_selection import write_stable_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a stability-supported checkpoint from dense validation results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--output-name", default="stable_checkpoint_selection.json")
    parser.add_argument("--artifact-name", default="student_stable_selected.pt")
    parser.add_argument("--plateau-tolerance", type=float, default=0.0005)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = write_stable_selection(
        run_dir=args.run_dir,
        results_csv=args.results_csv,
        output_name=args.output_name,
        artifact_name=args.artifact_name,
        plateau_tolerance=args.plateau_tolerance,
        dry_run=args.dry_run,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
