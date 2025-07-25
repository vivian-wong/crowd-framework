import matplotlib.pyplot as plt
import os
from st_dif.data_utils import get_pyg_temporal_dataset
from new_vis import show_crowd_table 

def test_show_crowd_table_sequentially():
    """
    Tests the sequential generation of crowd table visualizations by saving them as images.
    """
    # ─────────────────────────── CONFIG ────────────────────────────
    DATASET      = "Stadium"        # or "SEQ", "GCS", etc.
    FORECAST_NS  = 20               # your forecasting_horizon
    NODE_ORDER   = list(range(6))   # columns 1–6
    T_START      = 0                # first timestep index
    N_POINTS     = 5                # how many sequential tables to generate
    OUTPUT_DIR   = "test_outputs"   # Directory to save the plots

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─────────────────────────── RUN ──────────────────────────────
    _, g = get_pyg_temporal_dataset(DATASET, FORECAST_NS)
    
    # Add a simple assertion to make this a valid test
    assert g is not None, "Failed to load graph data."

    for i in range(N_POINTS):
        t0 = T_START + i
        print(f"Generating plot for t_start={t0}")

        # show_crowd_table should return the figure object
        fig = show_crowd_table(
            g,
            t_start=t0,
            n_rows=5,
            dataset_name=DATASET,
            node_order=NODE_ORDER,
        )
        
        # Set a title for the plot
        title = f"{DATASET}: Timesteps {t0}–{t0+4}"
        plt.suptitle(title, y=1.02, fontsize=14, weight="bold")
        
        # Save the figure to a file instead of showing it
        output_path = os.path.join(OUTPUT_DIR, f"crowd_table_t{t0}.png")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

        # Close the figure to free up memory
        plt.close()

    # You can add an assertion here to check if files were created
    assert len(os.listdir(OUTPUT_DIR)) >= N_POINTS

# # sequential_tests.py
# import matplotlib.pyplot as plt
# from st_dif.data_utils import get_pyg_temporal_dataset
# from new_vis import show_crowd_table

# # ─────────────────────────── CONFIG ────────────────────────────
# DATASET      = "Stadium"        # or "SEQ", "GCS", etc.
# FORECAST_NS  = 20               # your forecasting_horizon
# NODE_ORDER   = list(range(6))   # columns 1–6
# T_START      = 0                # first timestep index
# N_POINTS     = 10               # how many sequential tables

# # ─────────────────────────── RUN ──────────────────────────────
# _, g = get_pyg_temporal_dataset(DATASET, FORECAST_NS)

# for i in range(N_POINTS):
#     t0 = T_START + i
#     print(f"Showing rows {t0}–{t0 + 4} for t_start={t0}")
#     show_crowd_table(
#         g,
#         t_start=t0,
#         n_rows=5,
#         dataset_name=DATASET,
#         node_order=NODE_ORDER,
#     )
#     plt.suptitle(f"{DATASET}: rows {t0}–{t0+4}", y=1.02, fontsize=14, weight="bold")
#     plt.show()   # blocks until you close the window
