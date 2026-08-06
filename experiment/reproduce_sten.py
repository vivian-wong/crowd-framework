"""Reproduce the ST-DIF rows and ablations from the BuildSys paper."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

from st_dif.models import (  # noqa: E402
    DenseGCLSTM,
    DenseGCNGRU,
    DenseLRGCN,
    DenseMPNNLSTM,
    GCNGRU,
    GRU_only,
    RecurrentGCN,
)
from st_dif.train_test_utils import evaluate, train  # noqa: E402


PAPER_PARAMS = {
    "DenseGCNGRU": 97_300, "GCNGRU": 96_916, "GRU_only": 39_316,
    "RecurrentGCN": 122_260, "DenseGCLSTM": 51_476,
    "DenseLRGCN": 148, "DenseMPNNLSTM": 352_276,
}
PAPER_TABLE3 = {1: (0.0905, 0.2089), 2: (0.0894, 0.2080), 3: (0.0891, 0.2075)}
PAPER_TABLE4 = {(128, 64): (0.0891, 0.2075), (64, 32): (0.0949, 0.2126),
                (32, 16): (0.0987, 0.2156)}
PAPER_TABLE5 = {
    "RecurrentGCN": {20: (0.113, 0.229), 60: (0.192, 0.300),
                     120: (0.273, 0.364), 240: (0.354, 0.409)},
    "DenseGCLSTM": {20: (0.105, 0.223), 60: (0.103, 0.228),
                    120: (0.246, 0.340), 240: (0.351, 0.410)},
    "DenseLRGCN": {20: (0.121, 0.239), 60: (0.194, 0.301),
                   120: (0.270, 0.355), 240: (0.351, 0.407)},
    "DenseMPNNLSTM": {20: (0.147, 0.251), 60: (0.103, 0.223),
                      120: (0.148, 0.268), 240: (0.153, 0.276)},
    "DenseGCNGRU": {20: (0.089, 0.210), 60: (0.088, 0.215),
                    120: (0.086, 0.214), 240: (0.091, 0.218)},
}
FIELDS = [
    "table", "mode", "model", "horizon", "K", "D_GCN", "D_GRU", "seed",
    "lr", "weight_decay", "epochs", "paper_mse", "paper_mae", "status",
    "test_mse", "test_mae", "wall_seconds", "error",
]
KEY_FIELDS = FIELDS[:11]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_temporal_layers(model, seed):
    """Keep shared GCN and temporal initialization paired across K."""
    torch.manual_seed(seed + 10_000)
    model.gru.reset_parameters()
    model.fc.reset_parameters()


def load_gcs(horizon, device, seed, paired):
    from st_dif.data_utils import get_loaders, get_pyg_temporal_dataset

    dataset, _ = get_pyg_temporal_dataset("GCS", horizon)
    loaders = get_loaders(
        dataset, 32, 0.7, 0.0, 0.3, device,
        manual_seed=42, shuffle_seed=seed + 20_000 if paired else None,
    )
    edge_index = next(iter(dataset)).edge_index.to(device)
    return (*loaders, edge_index)


def build_model(config):
    common = dict(in_channels=2, periods=config["horizon"], batch_size=32)
    if config["model"] == "DenseGCNGRU":
        return DenseGCNGRU(
            **common, K=config["K"], D_GCN=config["D_GCN"],
            D_GRU=config["D_GRU"], gcn_mode=config["mode"],
        )
    if config["model"] == "GCNGRU":
        return GCNGRU(**common, gcn_mode=config["mode"])
    if config["model"] == "GRU_only":
        return GRU_only(**common)
    if config["model"] == "RecurrentGCN":
        return RecurrentGCN(**common, graph_mode=config["mode"])
    if config["model"] == "DenseGCLSTM":
        return DenseGCLSTM(
            in_channels=2, hidden_channels=64, periods=config["horizon"],
            graph_mode=config["mode"],
        )
    if config["model"] == "DenseLRGCN":
        return DenseLRGCN(**common, graph_mode=config["mode"])
    if config["model"] == "DenseMPNNLSTM":
        return DenseMPNNLSTM(
            in_channels=2, periods=config["horizon"], graph_mode=config["mode"],
        )
    raise ValueError(config["model"])


def configs_for(table, modes, horizons):
    if table == "3":
        return [dict(table="3", mode=m, model="DenseGCNGRU", horizon=20,
                     K=k, D_GCN=128, D_GRU=64,
                     paper_mse=PAPER_TABLE3[k][0], paper_mae=PAPER_TABLE3[k][1])
                for m in modes for k in (1, 2, 3)]
    if table == "4":
        sizes = [(128, 64), (64, 32), (32, 16)]
        return [dict(table="4", mode=m, model="DenseGCNGRU", horizon=20,
                     K=3, D_GCN=g, D_GRU=r,
                     paper_mse=PAPER_TABLE4[(g, r)][0],
                     paper_mae=PAPER_TABLE4[(g, r)][1])
                for m in modes for g, r in sizes]
    if table == "5":
        models = ["DenseGCNGRU", "RecurrentGCN", "DenseGCLSTM",
                  "DenseLRGCN", "DenseMPNNLSTM"]
        return [dict(
                    table="5", mode=m, model=model, horizon=h,
                    K=3 if model == "DenseGCNGRU" else "",
                    D_GCN=128 if model == "DenseGCNGRU" else "",
                    D_GRU=64 if model == "DenseGCNGRU" else "",
                    paper_mse=PAPER_TABLE5[model][h][0],
                    paper_mae=PAPER_TABLE5[model][h][1],
                ) for m in modes for h in horizons for model in models]
    raise ValueError(table)


def key(row):
    return tuple(str(row.get(name, "")) for name in KEY_FIELDS)


def read_completed(path):
    if not path.exists():
        return set()
    rows = pd.read_csv(path, keep_default_na=False).to_dict("records")
    return {
        key(row) for row in rows
        if row["status"] == "ok"
        and math.isfinite(float(row["test_mse"]))
        and math.isfinite(float(row["test_mae"]))
    }


def append_row(path, row):
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_config(config, seed, args, device):
    seed_everything(seed)
    loaders = load_gcs(config["horizon"], device, seed, args.paired_k)
    model = build_model(config)
    if args.paired_k and hasattr(model, "gru"):
        reset_temporal_layers(model, seed)
    model = model.to(device)
    train_loader, val_loader, test_loader, edge_index = loaders
    model, checkpoint = train(
        model, train_loader, val_loader, edge_index,
        num_epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
    )
    _, checkpoint = evaluate(model, test_loader, edge_index, checkpoint)
    return checkpoint["test_mse"], checkpoint["test_mae"]


def print_params():
    for name, expected in PAPER_PARAMS.items():
        config = dict(model=name, horizon=20, K=3, D_GCN=128, D_GRU=64,
                      mode="legacy")
        count = sum(p.numel() for p in build_model(config).parameters())
        print(f"{name:14s} {count:>8,d}  {'ok' if count == expected else 'mismatch'}")


def summarize(path):
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[(data.status == "ok") & data.test_mse.notna() & data.test_mae.notna()]
    if data.empty:
        return
    group = ["table", "mode", "model", "horizon", "K", "D_GCN", "D_GRU"]
    result = data.groupby(group, dropna=False).agg(
        paper_mse=("paper_mse", "first"), paper_mae=("paper_mae", "first"),
        mse_mean=("test_mse", "mean"), mse_min=("test_mse", "min"),
        mse_max=("test_mse", "max"), mae_mean=("test_mae", "mean"),
        mae_min=("test_mae", "min"), mae_max=("test_mae", "max"),
        runs=("seed", "count"),
    ).reset_index()
    result["mse_half_range"] = (result.mse_max - result.mse_min) / 2
    result["mae_half_range"] = (result.mae_max - result.mae_min) / 2
    result["mse_delta"] = result.mse_mean - result.paper_mse
    result["mae_delta"] = result.mae_mean - result.paper_mae
    print(result[[*group, "paper_mse", "mse_mean", "mse_half_range", "mse_delta",
                  "paper_mae", "mae_mean", "mae_half_range", "mae_delta",
                  "runs"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=["2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--modes", nargs="+", choices=["legacy", "corrected"],
                        default=["corrected"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[20, 60, 120, 240])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--paired-k", action="store_true",
                        help="Pair shuffle order and shared-layer initialization across K.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results_reproduction")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "sten_results.csv"
    import platform
    import subprocess
    import torch_geometric
    import torch_geometric_temporal

    metadata = {
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_geometric": torch_geometric.__version__,
        "torch_geometric_temporal": torch_geometric_temporal.__version__,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "seeds": args.seeds, "split_seed": 42, "batch_size": 32,
        "paired_k": args.paired_k,
        "train_val_test": [0.7, 0.0, 0.3],
        "aggregation": "mean and half range over successful seeds",
        "gcs_csv_sha256": hashlib.sha256(
            (REPO_ROOT / "data/gcs/gcs-processed/flow_df_gcs.csv").read_bytes()
        ).hexdigest(),
    }
    (args.results_dir / "sten_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    device = torch.device(metadata["device"])
    torch.use_deterministic_algorithms(True, warn_only=True)

    if args.table in ("2", "all"):
        print_params()
    completed = read_completed(results_path)
    tables = ("3", "4", "5") if args.table == "all" else (args.table,)
    for table in tables:
        if table == "2":
            continue
        for config in configs_for(table, args.modes, args.horizons):
            for seed in args.seeds:
                base = {**config, "seed": seed, "lr": args.lr,
                        "weight_decay": args.weight_decay, "epochs": args.epochs}
                if key(base) in completed:
                    continue
                started = time.time()
                try:
                    mse, mae = run_config(config, seed, args, device)
                    row = {**base, "status": "ok", "test_mse": mse,
                           "test_mae": mae, "error": ""}
                except Exception as exc:
                    row = {**base, "status": "failed", "test_mse": "",
                           "test_mae": "", "error": f"{type(exc).__name__}: {exc}"}
                row["wall_seconds"] = round(time.time() - started, 1)
                append_row(results_path, row)
                print(row)
    summarize(results_path)


if __name__ == "__main__":
    main()
