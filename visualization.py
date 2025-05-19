"""
General-purpose heat‑map utilities for CMGraph‑formatted crowd datasets.

* `show_crowd_heatmap_at_t`  → square/rectangular grid view for **one** time‑step.

The numbers displayed are **raw counts** (not normalised) unless you pre‑process
`cmgraph.X` beforehand.
"""

from __future__ import annotations
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

__all__ = [
    "show_crowd_heatmap_at_t",
]


def _auto_grid(n_nodes: int) -> tuple[int, int]:
    """Return a near‑square grid for *n_nodes* cells."""
    rows = int(math.floor(math.sqrt(n_nodes)))
    cols = int(math.ceil(n_nodes / rows))
    return rows, cols


def show_crowd_heatmap_at_t(
    cmgraph,
    t: int,
    feature_idx: int = 0,
    *,
    grid_shape: tuple[int, int] | None = None,
    cmap: str = "YlOrRd",
    annotate: bool = True,
    ax: plt.Axes | None = None,
):
    """Render a 2‑D grid heat‑map for **one** time‑step."""
    X = cmgraph.X  # (N, D, T)
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    T = X.shape[2]
    if not 0 <= t < T:
        raise ValueError(f"time‑index {t} outside 0..{T - 1}")

    vec = X[:, feature_idx, t]
    n_nodes = vec.shape[0]

    if grid_shape is None:
        grid_shape = _auto_grid(n_nodes)
    rows, cols = grid_shape

    matrix = np.full((rows, cols), np.nan)
    matrix.flat[:n_nodes] = vec

    if ax is None:
        _, ax = plt.subplots(figsize=(cols * 0.55, rows * 0.55))

    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=annotate,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "# people"},
        ax=ax,
    )

    ax.set_title(f"Crowd counts at t = {t}")
    plt.tight_layout()
    return ax
