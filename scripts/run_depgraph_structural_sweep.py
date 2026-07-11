#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = PROJECT_ROOT / "experiments/pruning/sweeps/fcos18_depgraph_speed_sweep"
DEFAULT_BOUNDARY_DIR = PROJECT_ROOT / "experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "experiments/pruning/configs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "experiments/pruning/models"


@dataclass(frozen=True)
class Variant:
    name: str
    config: Path
    artifact: Path


def default_variants() -> list[Variant]:
    names = [
        "fcos18_depgraph_structural_20pct",
        "fcos18_depgraph_structural_30pct",
        "fcos18_depgraph_head_only_20pct",
        "fcos18_depgraph_fpn_head_20pct",
        "fcos18_depgraph_backbone_fpn_head_20pct",
    ]
    return [
        Variant(
            name=name,
            config=DEFAULT_CONFIG_DIR / f"{name}.yaml",
            artifact=DEFAULT_MODEL_DIR / f"FCOS_18_{name.removeprefix('fcos18_')}.pt",
        )
        for name in names
    ]


def boundary_variants() -> list[Variant]:
    names = [
        "fcos18_depgraph_structural_40pct",
        "fcos18_depgraph_structural_50pct",
        "fcos18_depgraph_fpn_head_30pct",
        "fcos18_depgraph_fpn_head_40pct",
    ]
    return [
        Variant(
            name=name,
            config=DEFAULT_CONFIG_DIR / f"{name}.yaml",
            artifact=DEFAULT_MODEL_DIR / f"FCOS_18_{name.removeprefix('fcos18_')}.pt",
        )
        for name in names
    ]


def fpn_head_extended_variants() -> list[Variant]:
    names = [
        "fcos18_depgraph_fpn_head_20pct",
        "fcos18_depgraph_fpn_head_30pct",
        "fcos18_depgraph_fpn_head_40pct",
        "fcos18_depgraph_fpn_head_50pct",
        "fcos18_depgraph_fpn_head_60pct",
        "fcos18_depgraph_fpn_head_70pct",
    ]
    return [
        Variant(
            name=name,
            config=DEFAULT_CONFIG_DIR / f"{name}.yaml",
            artifact=DEFAULT_MODEL_DIR / f"FCOS_18_{name.removeprefix('fcos18_')}.pt",
        )
        for name in names
    ]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FCOS_18 DepGraph structural pruning speed sweep.")
    parser.add_argument("--sweep_dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/midogpp_guide_eval_xvalidation_smoke.csv")
    parser.add_argument("--config_file", type=Path, default=PROJECT_ROOT / "experiments/eval_guide/configs/FCOS_18_eval.yaml")
    parser.add_argument("--guide_repo", type=Path, default=PROJECT_ROOT / "repos/MIDOG_2025_Guide")
    parser.add_argument("--img_dir", type=Path, default=PROJECT_ROOT / "data/midogpp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--runtime_backend", default="pytorch_eager")
    parser.add_argument(
        "--fpn_head_extended",
        action="store_true",
        help="Run the focused FPN+head-only 20-70%% follow-up with smoke and quick evaluation.",
    )
    parser.add_argument(
        "--smoke_dataset",
        type=Path,
        default=PROJECT_ROOT / "data/midogpp_guide_eval_xvalidation_smoke.csv",
        help="Smoke dataset used in --fpn_head_extended mode.",
    )
    parser.add_argument(
        "--boundary_test",
        action="store_true",
        help="Use only the extreme boundary-test variants and boundary decision rules.",
    )
    parser.add_argument("--overwrite_artifacts", action="store_true")
    parser.add_argument("--overwrite_eval", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variant names.")
    return parser.parse_args()


def run_command(command: list[str], *, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, text=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def create_artifact(variant: Variant, args: argparse.Namespace) -> None:
    if variant.artifact.exists() and not args.overwrite_artifacts:
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/create_depgraph_structural_pruned_model.py"),
        "--config",
        str(variant.config),
        "--output",
        str(variant.artifact),
    ]
    run_command(command, log_path=args.sweep_dir / "artifact_logs" / f"{variant.name}.log")


def eval_outputs_exist(args: argparse.Namespace, label: str) -> bool:
    return all(
        (args.sweep_dir / f"{label}_{suffix}").exists()
        for suffix in ["metrics.json", "timing.json", "runtime.json"]
    )


def run_evaluation(
    args: argparse.Namespace,
    label: str,
    artifact: Path | None = None,
    *,
    dataset: Path | None = None,
) -> None:
    if eval_outputs_exist(args, label) and not args.overwrite_eval:
        return
    eval_dataset = args.dataset if dataset is None else dataset
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src/benchmark/midog_guide_adapter.py"),
        "--config_file",
        str(args.config_file),
        "--dataset",
        str(eval_dataset),
        "--guide_repo",
        str(args.guide_repo),
        "--img_dir",
        str(args.img_dir),
        "--metrics_output",
        str(args.sweep_dir / f"{label}_metrics.json"),
        "--runtime_metadata_output",
        str(args.sweep_dir / f"{label}_runtime.json"),
        "--runtime_backend",
        args.runtime_backend,
        "--split",
        args.split,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--overlap",
        str(args.overlap),
        "--nms_thresh",
        str(args.nms_thresh),
        "--overwrite",
        "--profile_pipeline",
        "--timing_output_json",
        str(args.sweep_dir / f"{label}_timing.json"),
        "--timing_output_csv",
        str(args.sweep_dir / f"{label}_timing.csv"),
    ]
    if artifact is not None:
        command.extend(["--pruned_model_path", str(artifact)])
    run_command(command, log_path=args.sweep_dir / f"{label}.log")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_variant(args: argparse.Namespace, label: str, artifact: Path | None, baseline: dict[str, Any] | None) -> dict[str, Any]:
    metrics = load_json(args.sweep_dir / f"{label}_metrics.json")
    timing = load_json(args.sweep_dir / f"{label}_timing.json")
    aggregates = metrics["aggregates"]
    forward_pass_s = float(timing["stages"]["forward_pass"]["total_s"])
    runtime_s = float(timing["total_end_to_end_s"])
    row = {
        "variant": label,
        "params_before": None,
        "params_after": None,
        "params_reduction_pct": None,
        "macs_before": None,
        "macs_after": None,
        "macs_reduction_pct": None,
        "changed_layers": None,
        "smoke_original_detections": None,
        "smoke_pruned_detections": None,
        "f1": float(aggregates["f1_score"]),
        "precision": float(aggregates["precision"]),
        "recall": float(aggregates["recall"]),
        "false_positives": int(aggregates["false_positives"]),
        "false_negatives": int(aggregates["false_negatives"]),
        "total_detections": int(aggregates["total_detections"]),
        "runtime_s": runtime_s,
        "forward_pass_s": forward_pass_s,
        "mean_patch_latency_s": float(timing["mean_per_patch_s"]),
        "end_to_end_speedup_pct": None,
        "forward_speedup_pct": None,
        "recommendation": "baseline",
    }
    if artifact is not None:
        metadata = load_json(artifact.with_suffix(".meta.json"))
        pruning = metadata["pruning_result"]
        smoke = metadata["smoke_inference"]
        row.update(
            {
                "params_before": int(pruning["parameters_before_total"]),
                "params_after": int(pruning["parameters_after_total"]),
                "params_reduction_pct": float(pruning["parameters_removed_ratio"]) * 100.0,
                "macs_before": float(pruning["macs_before"]) if pruning["macs_before"] is not None else None,
                "macs_after": float(pruning["macs_after"]) if pruning["macs_after"] is not None else None,
                "macs_reduction_pct": None
                if pruning["macs_removed_ratio"] is None
                else float(pruning["macs_removed_ratio"]) * 100.0,
                "changed_layers": len(pruning["module_results"]),
                "smoke_original_detections": int(smoke["original_detection_count"]),
                "smoke_pruned_detections": int(smoke["pruned_detection_count"]),
            }
        )
    if baseline is not None:
        end_speedup = (baseline["runtime_s"] - runtime_s) / baseline["runtime_s"] * 100.0
        forward_speedup = (baseline["forward_pass_s"] - forward_pass_s) / baseline["forward_pass_s"] * 100.0
        eval_detection_shift = (
            abs(row["total_detections"] - baseline["total_detections"])
            / max(1, baseline["total_detections"])
        )
        plausible = (
            row["total_detections"] > 0
            and row["smoke_pruned_detections"] not in {None, 0}
            and eval_detection_shift <= 0.50
        )
        row["end_to_end_speedup_pct"] = end_speedup
        row["forward_speedup_pct"] = forward_speedup
        if row["total_detections"] == 0:
            row["recommendation"] = "reject: eval detections collapsed to 0"
        elif row["f1"] < 0.2:
            row["recommendation"] = "reject: F1 below 0.2"
        elif forward_speedup < 10.0:
            row["recommendation"] = "do not fine-tune: forward speedup below 10%"
        elif plausible and forward_speedup >= 10.0:
            row["recommendation"] = "fine-tuning candidate"
        else:
            row["recommendation"] = "reject: detections not plausible"
    return row


def summarize_fpn_head_extended(args: argparse.Namespace, variants: list[Variant]) -> list[dict[str, Any]]:
    smoke_baseline = summarize_variant(args, "smoke_FCOS_18_baseline", None, None)
    quick_baseline = summarize_variant(args, "quick_FCOS_18_baseline", None, None)
    rows = [
        {
            "variant": "FCOS_18_baseline",
            "artifact_path": None,
            "artifact_sha256": None,
            "params_after": None,
            "macs_reduction_pct": None,
            "changed_layers": None,
            "representative_detection_count": None,
            "smoke_eval_detections": smoke_baseline["total_detections"],
            "smoke_f1": smoke_baseline["f1"],
            "smoke_precision": smoke_baseline["precision"],
            "smoke_recall": smoke_baseline["recall"],
            "quick_eval_detections": quick_baseline["total_detections"],
            "quick_f1": quick_baseline["f1"],
            "quick_precision": quick_baseline["precision"],
            "quick_recall": quick_baseline["recall"],
            "quick_runtime_s": quick_baseline["runtime_s"],
            "quick_forward_pass_s": quick_baseline["forward_pass_s"],
            "quick_mean_patch_latency_s": quick_baseline["mean_patch_latency_s"],
            "quick_end_to_end_speedup_pct": None,
            "quick_forward_speedup_pct": None,
            "selection_decision": "baseline",
        }
    ]

    for variant in variants:
        smoke = summarize_variant(args, f"smoke_{variant.name}", variant.artifact, smoke_baseline)
        quick = summarize_variant(args, f"quick_{variant.name}", variant.artifact, quick_baseline)
        metadata = load_json(variant.artifact.with_suffix(".meta.json"))
        pruning = metadata["pruning_result"]
        smoke_inference = metadata["smoke_inference"]
        quick_detection_shift = abs(quick["total_detections"] - quick_baseline["total_detections"]) / max(
            1, quick_baseline["total_detections"]
        )
        if quick["total_detections"] == 0:
            decision = "reject: quick eval detections collapsed to 0"
        elif quick["f1"] < 0.2:
            decision = "reject: quick F1 below 0.2"
        elif quick["forward_speedup_pct"] < 10.0:
            decision = "do not fine-tune: quick forward speedup below 10%"
        elif quick_detection_shift > 0.50:
            decision = "reject: quick detection count shifted by more than 50%"
        elif quick["forward_speedup_pct"] < 15.0:
            decision = "fine-tuning candidate: quick forward speedup in 10-15% band"
        else:
            decision = "fine-tuning candidate: quick forward speedup >=15%"
        rows.append(
            {
                "variant": variant.name,
                "artifact_path": str(variant.artifact),
                "artifact_sha256": sha256_file(variant.artifact),
                "params_after": int(pruning["parameters_after_total"]),
                "macs_reduction_pct": None
                if pruning["macs_removed_ratio"] is None
                else float(pruning["macs_removed_ratio"]) * 100.0,
                "changed_layers": len(pruning["module_results"]),
                "representative_detection_count": int(smoke_inference["pruned_detection_count"]),
                "smoke_eval_detections": smoke["total_detections"],
                "smoke_f1": smoke["f1"],
                "smoke_precision": smoke["precision"],
                "smoke_recall": smoke["recall"],
                "quick_eval_detections": quick["total_detections"],
                "quick_f1": quick["f1"],
                "quick_precision": quick["precision"],
                "quick_recall": quick["recall"],
                "quick_runtime_s": quick["runtime_s"],
                "quick_forward_pass_s": quick["forward_pass_s"],
                "quick_mean_patch_latency_s": quick["mean_patch_latency_s"],
                "quick_end_to_end_speedup_pct": quick["end_to_end_speedup_pct"],
                "quick_forward_speedup_pct": quick["forward_speedup_pct"],
                "selection_decision": decision,
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.sweep_dir / "summary.csv"
    json_path = args.sweep_dir / "summary.json"
    md_path = args.sweep_dir / "summary.md"
    commands_path = args.sweep_dir / "commands.sh"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = ["| " + " | ".join(fmt(row[key]) for key in fieldnames) + " |" for row in rows]
    md_path.write_text(
        "\n".join(
            [
                (
                    "# FCOS_18 DepGraph FPN+Head Extended Quick Follow-up"
                    if args.fpn_head_extended
                    else "# FCOS_18 DepGraph Extreme Boundary Test"
                    if args.boundary_test
                    else "# FCOS_18 DepGraph Structural Pruning Speed Sweep"
                ),
                "",
                header,
                separator,
                *body,
                "",
                (
                    "FPN+head decision rules: reject if quick eval detections collapse to 0; "
                    "reject if quick F1 < 0.2; do not fine-tune if quick forward speedup < 10%; "
                    "recommend fine-tuning only if quick forward speedup is >=10-15% and detections remain plausible."
                    if args.fpn_head_extended
                    else "Boundary decision rules: reject immediately if eval detections collapse to 0; "
                    "reject if F1 < 0.2; do not fine-tune if forward speedup < 10%; "
                    "recommend fine-tuning only if forward speedup is >=10-15% and detections remain plausible."
                    if args.boundary_test
                    else "Decision rule: recommend fine-tuning only for variants with >=10% forward-pass speedup "
                    "or >=8% end-to-end speedup while still producing plausible detections."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"{sys.executable} {PROJECT_ROOT / 'scripts/run_depgraph_structural_sweep.py'} "
        f"{'--fpn_head_extended ' if args.fpn_head_extended else ''}"
        f"--sweep_dir {args.sweep_dir} --dataset {args.dataset} --device {args.device} "
        f"--batch_size {args.batch_size} --num_workers {args.num_workers}",
        "",
    ]
    commands_path.write_text("\n".join(commands), encoding="utf-8")


def main() -> None:
    args = get_args()
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    if args.boundary_test and args.sweep_dir == DEFAULT_SWEEP_DIR:
        args.sweep_dir = DEFAULT_BOUNDARY_DIR
    if args.fpn_head_extended:
        variants = fpn_head_extended_variants()
    else:
        variants = boundary_variants() if args.boundary_test else default_variants()
    if args.variants:
        selected = set(args.variants)
        variants = [variant for variant in variants if variant.name in selected]
        missing = selected - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    for variant in variants:
        create_artifact(variant, args)

    if not args.skip_eval and args.fpn_head_extended:
        run_evaluation(args, "smoke_FCOS_18_baseline", dataset=args.smoke_dataset)
        run_evaluation(args, "quick_FCOS_18_baseline", dataset=args.dataset)
        for variant in variants:
            run_evaluation(args, f"smoke_{variant.name}", variant.artifact, dataset=args.smoke_dataset)
            run_evaluation(args, f"quick_{variant.name}", variant.artifact, dataset=args.dataset)
        rows = summarize_fpn_head_extended(args, variants)
        write_summary(args, rows)
        print((args.sweep_dir / "summary.md").read_text(encoding="utf-8"))
        return

    if not args.skip_eval:
        run_evaluation(args, "FCOS_18_baseline")
        for variant in variants:
            run_evaluation(args, variant.name, variant.artifact)
    else:
        print(f"Created or reused {len(variants)} artifacts. Evaluation was skipped.")
        for variant in variants:
            print(f"{variant.name}: {variant.artifact}")
        return

    baseline = summarize_variant(args, "FCOS_18_baseline", None, None)
    rows = [baseline]
    rows.extend(summarize_variant(args, variant.name, variant.artifact, baseline) for variant in variants)
    write_summary(args, rows)
    print((args.sweep_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
