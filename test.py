from st_dif.data_utils import get_pyg_temporal_dataset
from new_vis import show_crowd_table

ds, g = get_pyg_temporal_dataset("Stadium", forecasting_horizon=20)

# render rows t=0 … t=4
show_crowd_table(g,
                 t_start=0,
                 n_rows=5,
                 dataset_name="Stadium",
                 node_order=np.arange(6))        # 1-6
plt.show()