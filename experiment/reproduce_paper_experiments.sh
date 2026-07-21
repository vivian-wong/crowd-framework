#!/bin/bash
# ==========================================================================
# Reproduce the experiments of the ST-DIF paper (GCS dataset).
#
# Run from the experiment/ directory, inside the st_dif environment
# (see README "Installation Guide" and "Reproducing the ST-DIF paper"):
#
#     cd experiment
#     bash reproduce_paper_experiments.sh
#
# What it does, in order:
#   1. Table 2  - parameter counts for all 7 models (seconds, CPU is fine)
#   2. Table 3  - K-sensitivity sweep, DenseGCNGRU, GCS, T=20   (15 runs)
#   3. Table 4  - model-size sweep,   DenseGCNGRU, GCS, T=20   (15 runs)
#   4. Table 5  - 5 models x 4 horizons x 5 seeds, GCS        (100 runs)
#
# Every training run appends one row to a CSV in ../results_reproduction/.
# The sweeps are resume-aware: if the script is interrupted, re-running it
# skips already-completed (config, seed) cells. Aggregated tables are
# printed next to the paper's reference numbers after each sweep.
#
# Approximate wall time on a single consumer GPU: Table 3/4 about 1-2 h
# combined; Table 5 several hours (T=240 runs dominate). To reproduce only
# the T=20 column of Table 5 (the one cross-checked against Tables 3/4):
#
#     python reproduce_st_dif.py --table 5 --horizons 20
# ==========================================================================
set -e

python reproduce_st_dif.py --table 2
python reproduce_st_dif.py --table 3
python reproduce_st_dif.py --table 4
python reproduce_st_dif.py --table 5

# ==========================================================================
# Legacy sweep (thesis Chapters 3-4): DenseGCNGRU / GCNGRU / GRU on
# GCS / SEQ / STADIUM_2023 via main.py. Kept for reference; not required
# for the ST-DIF paper tables. Uncomment to run.
# ==========================================================================
# for FORECASTING_HORIZON in 20 60 120 240
# do
#     for MODEL in DenseGCNGRU GCNGRU GRU
#     do
#         for DATASET in GCS SEQ STADIUM_2023
#         do
#             python main.py \
#                 --DATASET $DATASET \
#                 --MODEL $MODEL \
#                 --forecasting_horizon $FORECASTING_HORIZON \
#                 --save_model True \
#                 --save_dir './checkpoints'
#         done
#     done
# done
