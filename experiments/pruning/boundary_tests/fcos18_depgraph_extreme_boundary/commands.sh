#!/usr/bin/env bash
set -euo pipefail
/home/johan/bachelorarbeit-midog/.venv/bin/python /home/johan/bachelorarbeit-midog/scripts/run_depgraph_structural_sweep.py --sweep_dir /home/johan/bachelorarbeit-midog/experiments/pruning/boundary_tests/fcos18_depgraph_extreme_boundary --dataset /home/johan/bachelorarbeit-midog/data/midogpp_guide_eval_xvalidation_smoke.csv --device cpu --batch_size 1 --num_workers 0
