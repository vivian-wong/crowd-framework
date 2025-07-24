# sequential_tests.py
import matplotlib.pyplot as plt
from st_dif.data_utils import get_pyg_temporal_dataset
from new_vis import show_crowd_table

# ─────────────────────────── CONFIG ────────────────────────────
DATASET      = "Stadium"        # or "SEQ", "GCS", etc.
FORECAST_NS  = 20               # your forecasting_horizon
NODE_ORDER   = list(range(6))   # columns 1–6
T_START      = 0                # first timestep index
N_POINTS     = 10               # how many sequential tables

# ─────────────────────────── RUN ──────────────────────────────
_, g = get_pyg_temporal_dataset(DATASET, FORECAST_NS)

for i in range(N_POINTS):
    t0 = T_START + i
    print(f"Showing rows {t0}–{t0 + 4} for t_start={t0}")
    show_crowd_table(
        g,
        t_start=t0,
        n_rows=5,
        dataset_name=DATASET,
        node_order=NODE_ORDER,
    )
    plt.suptitle(f"{DATASET}: rows {t0}–{t0+4}", y=1.02, fontsize=14, weight="bold")
    plt.show()   # blocks until you close the window
