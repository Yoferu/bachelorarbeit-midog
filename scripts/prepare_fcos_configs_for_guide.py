from pathlib import Path
import yaml

project = Path.home() / "bachelorarbeit-midog"
fcos_repo = project / "repos" / "FCOS_Inference_CLI"
out_dir = project / "experiments" / "eval_guide" / "configs"
out_dir.mkdir(parents=True, exist_ok=True)

models = ["FCOS_18", "FCOS_x50", "FCOS_x101"]

for model in models:
    src = fcos_repo / "configs" / f"{model}.yaml"
    if not src.exists():
        raise FileNotFoundError(src)

    with src.open("r") as f:
        cfg = yaml.safe_load(f)

    ckpt = fcos_repo / cfg["checkpoint"]
    cfg["checkpoint"] = str(ckpt.resolve())

    cfg.setdefault("means", None)
    cfg.setdefault("stds", None)

    dst = out_dir / f"{model}_eval.yaml"
    with dst.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote {dst}")
    print(f"  checkpoint: {cfg['checkpoint']}")
