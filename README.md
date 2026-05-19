# MIDOG Evaluation Artifacts

This repository contains the local scripts, configuration files, logs, and JSON/CSV evaluation outputs used for MIDOG-related FCOS experiments.

Large datasets and cloned upstream repositories are intentionally excluded from version control:

- `data/`
- `repos/`
- `.venv/`

## Contents

- `scripts/` - helper scripts for preparing MIDOG++ subsets, converting annotations, preparing FCOS configs, and collecting evaluation metrics.
- `logs/` - local FCOS inference logs.
- `results/raw_predictions/` - JSON detection outputs for the evaluated FCOS variants.
- `experiments/eval_guide/` - evaluation configs, logs, and summarized metrics.

## Python Dependencies

The scripts use Python 3 and the packages listed in `requirements.txt`.
