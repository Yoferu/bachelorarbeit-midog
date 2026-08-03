#!/usr/bin/env bash
set -euo pipefail
/home/johan/bachelorarbeit-midog/.venv/bin/python /home/johan/bachelorarbeit-midog/scripts/run_depgraph_structural_sweep.py --fpn_head_extended --sweep_dir experiments/pruning/sweeps/fcos18_depgraph_fpn_head_extended_quick --dataset data/midogpp_guide_eval_xvalidation_quick.csv --device cpu --batch_size 1 --num_workers 0
