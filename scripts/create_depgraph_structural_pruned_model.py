#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pruning.depgraph_structural_pruning import (
    apply_depgraph_structural_pruning,
    build_example_input,
    collect_structural_pruning_plan,
    smoke_compare_detection_counts,
)
from src.pruning.inspect_model import DEFAULT_EVAL_CONFIG_DIR, load_guide_model
from src.pruning.pruning_config import PruningConfig
from src.pruning.save_pruned_model import current_git_commit


def save_structural_artifact(
    *,
    model,
    config: PruningConfig,
    pruning_result: dict,
    output: Path,
    eval_config_path: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_type": "depgraph_structural_pruned_detection_model",
        "model": config.model,
        "model_object": model,
        "state_dict": model.state_dict(),
        "pruning_config": config.as_dict(),
        "eval_config": str(eval_config_path),
        "pruning_result": pruning_result,
    }
    torch.save(artifact, output)
    return output


def write_structural_metadata(
    *,
    config: PruningConfig,
    pruning_result: dict,
    smoke_result: dict,
    output: Path,
    eval_config_path: Path,
    base_checkpoint: str,
) -> Path:
    metadata_path = output.with_suffix(".meta.json")
    metadata = {
        "artifact_path": str(output),
        "artifact_type": "depgraph_structural_pruned_detection_model",
        "base_model": config.model,
        "base_eval_config": str(eval_config_path),
        "base_checkpoint": base_checkpoint,
        "pruning_config": config.as_dict(),
        "pruning_result": pruning_result,
        "smoke_inference": smoke_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "notes": config.notes,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a DepGraph structurally-pruned FCOS artifact.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()

    pruning_config = PruningConfig.load(args.config)
    if pruning_config.pruning_mode != "depgraph_structural":
        raise ValueError("This script requires pruning_mode=depgraph_structural.")

    eval_config = pruning_config.eval_config or DEFAULT_EVAL_CONFIG_DIR / f"{pruning_config.model}_eval.yaml"
    model, model_config = load_guide_model(
        model_name=pruning_config.model,
        guide_repo=pruning_config.guide_repo,
        eval_config=eval_config,
    )
    model.eval()
    model.to(pruning_config.device)

    plan = collect_structural_pruning_plan(model, pruning_config)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    original_model = copy.deepcopy(model)
    example_input = build_example_input(pruning_config, model_config)
    pruning_result = apply_depgraph_structural_pruning(
        model=model,
        config=pruning_config,
        example_input=example_input,
    )
    smoke_result = smoke_compare_detection_counts(
        original_model=original_model,
        pruned_model=model,
        example_input=example_input,
    )

    artifact_path = save_structural_artifact(
        model=model,
        config=pruning_config,
        pruning_result=pruning_result,
        output=args.output,
        eval_config_path=eval_config,
    )
    metadata_path = write_structural_metadata(
        config=pruning_config,
        pruning_result=pruning_result,
        smoke_result=smoke_result,
        output=args.output,
        eval_config_path=eval_config,
        base_checkpoint=model_config.checkpoint,
    )

    print(f"Saved structurally pruned model artifact: {artifact_path}")
    print(f"Saved pruning metadata: {metadata_path}")
    print(f"Candidate modules: {len(pruning_result['target_modules'])}")
    print(f"Structurally changed layers: {len(pruning_result['pruned_layers'])}")
    print(
        "Parameters: "
        f"{pruning_result['parameters_before_total']} -> {pruning_result['parameters_after_total']} "
        f"({pruning_result['parameters_removed_ratio']:.2%} removed)"
    )
    if pruning_result["macs_before"] is not None and pruning_result["macs_after"] is not None:
        print(
            "MACs: "
            f"{pruning_result['macs_before']:.0f} -> {pruning_result['macs_after']:.0f} "
            f"({pruning_result['macs_removed_ratio']:.2%} removed)"
        )
    else:
        print(f"MAC counting unavailable: {pruning_result['macs_error_before'] or pruning_result['macs_error_after']}")
    print(
        "Smoke detections: "
        f"original={smoke_result['original_detection_count']}, "
        f"pruned={smoke_result['pruned_detection_count']}"
    )


if __name__ == "__main__":
    main()
