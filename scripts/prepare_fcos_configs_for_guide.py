"""Prepare guide evaluation configs for the local FCOS checkpoint directory."""
import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project.resolve()
    fcos_repo = project / 'repos' / 'FCOS_Inference_CLI'
    out_dir = project / 'experiments' / 'eval_guide' / 'configs'
    prepared = []
    for model in ('FCOS_18', 'FCOS_x50', 'FCOS_x101'):
        source = fcos_repo / 'configs' / f'{model}.yaml'
        if not source.is_file():
            parser.error(f'Missing upstream configuration: {source}')
        cfg = yaml.safe_load(source.read_text())
        checkpoint = (fcos_repo / cfg['checkpoint']).resolve()
        if not checkpoint.is_file():
            parser.error(f'Missing pretrained checkpoint: {checkpoint}')
        cfg['checkpoint'] = str(checkpoint)
        cfg.setdefault('means', None)
        cfg.setdefault('stds', None)
        prepared.append((out_dir / f'{model}_eval.yaml', cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    for destination, cfg in prepared:
        destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f'Wrote {destination}')


if __name__ == '__main__':
    main()
