#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning.inspect_model import DEFAULT_EVAL_CONFIG_DIR, load_guide_model
from src.pruning.pruning_config import PruningConfig
from src.pruning.save_pruned_model import save_pruned_artifact, write_pruning_metadata
from src.pruning.structured_pruning import (
    apply_masked_structured_pruning,
    count_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a conservative structured-pruned FCOS artifact.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pruning_config = PruningConfig.load(args.config)
    if pruning_config.pruning_mode != "masked":
        raise ValueError("The first implementation supports pruning_mode=masked only.")

    eval_config = pruning_config.eval_config or DEFAULT_EVAL_CONFIG_DIR / f"{pruning_config.model}_eval.yaml"
    model, model_config = load_guide_model(
        model_name=pruning_config.model,
        guide_repo=pruning_config.guide_repo,
        eval_config=eval_config,
    )
    model.eval()
    model.to(pruning_config.device)

    before_total = count_parameters(model, nonzero_only=False)
    before_nonzero = count_parameters(model, nonzero_only=True)
    pruning_result = apply_masked_structured_pruning(model=model, config=pruning_config)

    artifact_path = save_pruned_artifact(
        model=model,
        config=pruning_config,
        pruning_result=pruning_result,
        output=args.output,
        eval_config_path=eval_config,
    )
    metadata_path = write_pruning_metadata(
        model=model,
        config=pruning_config,
        pruning_result=pruning_result,
        output=args.output,
        eval_config_path=eval_config,
        base_checkpoint=model_config.checkpoint,
        parameters_before_total=before_total,
        parameters_before_nonzero=before_nonzero,
    )

    print(f"Saved pruned model artifact: {artifact_path}")
    print(f"Saved pruning metadata: {metadata_path}")
    print(f"Pruned modules: {len(pruning_result['target_modules'])}")
    print(f"Newly zeroed channels: {pruning_result['newly_zeroed_channels']}")
    print("Mode: masked structured pruning baseline; physical channels were not removed.")


if __name__ == "__main__":
    main()
