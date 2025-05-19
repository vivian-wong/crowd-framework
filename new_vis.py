

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

__all__ = [
]


def show_crowd_heatmap_at_t(cmgraph, t: int, *, feature_idx: int = 0,
                            cmap: str = "YlOrRd", ax: plt.Axes | None = None):
    X = cmgraph.X.detach().cpu().numpy() if torch.is_tensor(cmgraph.X) else cmgraph.X
    if not 0 <= t < X.shape[2]:
        raise ValueError("t index out of range")

    vec = X[:, feature_idx, t]
    n = len(vec)
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    mat = np.full((rows, cols), np.nan)
    mat.flat[:n] = vec

    if ax is None:
        _, ax = plt.subplots(figsize=(cols * 0.55, rows * 0.55))

    sns.heatmap(mat, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="white",
                cbar=False, ax=ax)
    ax.set_title(f"Crowd counts at t={t}")
    plt.tight_layout()
    return ax

def _raw_matrix_and_meta(cmgraph, node_order: list[int], timestamps: list[float]):
    df = cmgraph.flow_df.copy().round({"timestamp": 1})
    # pivot
    pivot = df.pivot_table(index="timestamp", columns="PAR_id",
                           values="num_people", aggfunc="first")
    pivot = pivot.reindex(index=timestamps, columns=node_order)
    matrix = pivot.to_numpy().astype(float)
    # raw frame numbers if available
    if "raw_frame_number" in df.columns:
        raw_frames = (df.drop_duplicates("timestamp")
                        .set_index("timestamp")
                        .loc[timestamps, "raw_frame_number"]
                        .astype(float)
                        .to_numpy())
    else:
        raw_frames = None
    return matrix, raw_frames


def show_crowd_heatmap_at_t_1d(cmgraph, t: int, *, node_order: list[int] | None = None,
                               cmap: str = "coolwarm", font_size: int = 13,
                               ax: plt.Axes | None = None):
    # resolve timestamps
    ts_all = sorted(cmgraph.flow_df["timestamp"].round(1).unique())
    if not 0 <= t < len(ts_all):
        raise ValueError("t index out of range")
    timestamps = [ts_all[t]]

    # infer and correct node_order
    actual_ids = sorted(cmgraph.flow_df["PAR_id"].unique())
    if node_order is None:
        node_order = actual_ids
    elif all(isinstance(x, int) and x < len(actual_ids) for x in node_order):
        node_order = [actual_ids[i] for i in node_order]

    mat, raw_frames = _raw_matrix_and_meta(cmgraph, node_order, timestamps)
    has_rf = raw_frames is not None

    # offsets
    x_ts = -3.2 if has_rf else -1.6
    x_rf = -1.2 if has_rf else None

    if ax is None:
        fig_width = (len(node_order) + (1.5 if has_rf else 0.8)) * 0.60
        _, ax = plt.subplots(figsize=(fig_width, 1.8))

    sns.heatmap(mat, ax=ax, cmap=cmap, annot=mat.astype(int), fmt="d",
                vmin=np.nanmin(mat), vmax=np.nanmax(mat), cbar=False,
                linewidths=1, linecolor="white",
                xticklabels=[str(p) for p in node_order], yticklabels=[""])

    # left texts
    ax.text(x_ts, 0.5, f"{timestamps[0]:.0f}", ha="right", va="center",
            fontsize=font_size, fontweight="bold")
    if has_rf:
        ax.text(x_rf, 0.5, f"{raw_frames[0]:.0f}", ha="center", va="center",
                fontsize=font_size, fontweight="bold")

    # headers
    header_y = -0.6
    ax.text(x_ts, header_y, "timestamp", ha="right", va="center",
            fontsize=font_size, fontweight="bold")
    if has_rf:
        ax.text(x_rf, header_y, "raw_frame_number", ha="center", va="center",
                fontsize=font_size, fontweight="bold")

    ax.set_title("PAR", fontsize=font_size+2, weight="bold", pad=10)
    ax.tick_params(axis="x", rotation=0, labelsize=font_size)
    for spine in ("left","right","bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["top"].set_linewidth(2)
    plt.tight_layout()
    return ax


def show_crowd_table(cmgraph, *, t_start: int = 0, n_rows: int = 5,
                     node_order: list[int] | None = None,
                     dataset_name: str | None = None,
                     cmap: str = "coolwarm", font_size: int = 13,
                     ax: plt.Axes | None = None):
    # resolve timestamps slice
    ts_all = sorted(cmgraph.flow_df["timestamp"].round(1).unique())
    if t_start + n_rows > len(ts_all):
        raise ValueError("Requested rows exceed available timestamps")
    timestamps = ts_all[t_start:t_start+n_rows]

    # infer and correct node_order
    actual_ids = sorted(cmgraph.flow_df["PAR_id"].unique())
    if node_order is None:
        node_order = actual_ids
    elif all(isinstance(x,int) and x < len(actual_ids) for x in node_order):
        node_order = [actual_ids[i] for i in node_order]

    mat, raw_frames = _raw_matrix_and_meta(cmgraph, node_order, timestamps)
    has_rf = raw_frames is not None

    # offsets & margins
    x_ts = -3.2 if has_rf else -1.8
    x_rf = -1.2 if has_rf else None
    extra_left = 3.5 if has_rf else 1.3

    # figure
    if ax is None:
        fig_width = (len(node_order) + extra_left) * 0.70
        fig_height = n_rows * 0.65
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    sns.heatmap(mat, ax=ax, cmap=cmap, annot=mat.astype(int), fmt="d",
                vmin=np.nanmin(mat), vmax=np.nanmax(mat), cbar=False,
                linewidths=1, linecolor="white",
                xticklabels=[str(p) for p in node_order],
                yticklabels=[""]*n_rows)

    # alternating shading + left text
    for r in range(n_rows):
        y = r + 0.5
        if r % 2 == 0:
            ax.axhspan(r, r+1, color="#f5f6f6", zorder=0)
        ax.text(x_ts, y, f"{timestamps[r]:.0f}", ha="right", va="center",
                fontsize=font_size, fontweight="bold")
        if has_rf:
            ax.text(x_rf, y, f"{raw_frames[r]:.0f}", ha="center", va="center",
                    fontsize=font_size, fontweight="bold")

    # headers
    header_y = -0.6
    ax.text(x_ts, header_y, "timestamp", ha="right", va="center",
            fontsize=font_size, fontweight="bold")
    if has_rf:
        ax.text(x_rf, header_y, "raw_frame_number", ha="center", va="center",
                fontsize=font_size, fontweight="bold")

    # decorations
    ax.set_title("PAR", fontsize=font_size+2, weight="bold", pad=14)
    ax.tick_params(axis="x", rotation=0, labelsize=font_size)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("")

    if dataset_name is not None:
        fig.text(0.02, 0.5, dataset_name, ha="left", va="center",
                 fontsize=font_size*2.2, fontweight="bold")

    for spine in ("left","right","bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["top"].set_linewidth(2)
    plt.tight_layout(rect=(0.20,0,1,1))
    return ax
