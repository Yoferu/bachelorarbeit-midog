#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning.inspect_model import collect_conv_modules, format_conv_table, load_guide_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Conv2d modules considered for pruning.")
    parser.add_argument("--model", default="FCOS_18")
    parser.add_argument("--guide_repo", type=Path, default=Path("repos/MIDOG_2025_Guide"))
    parser.add_argument("--eval_config", type=Path, default=None)
    args = parser.parse_args()

    model, config = load_guide_model(
        model_name=args.model,
        guide_repo=args.guide_repo,
        eval_config=args.eval_config,
    )
    rows = collect_conv_modules(model)
    print(f"model: {args.model}")
    print(f"model_class: {type(getattr(model, 'model', model)).__module__}.{type(getattr(model, 'model', model)).__name__}")
    print(f"backbone: {config.backbone}")
    print(f"checkpoint: {config.checkpoint}")
    print(f"conv2d_count: {len(rows)}")
    print(f"initial_prunable_count: {sum(1 for row in rows if row.prunable)}")
    print()
    print(format_conv_table(rows))


if __name__ == "__main__":
    main()
