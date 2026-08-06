"""Reproduce the ST-DIF rows and ablations from the BuildSys paper."""

import argparse
import csv
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

from st_dif.models import DenseGCNGRU, GCNGRU, GRU_only  # noqa: E402
from st_dif.train_test_utils import evaluate, train  # noqa: E402


PAPER_PARAMS = {"DenseGCNGRU": 97_300, "GCNGRU": 96_916, "GRU_only": 39_316}
FIELDS = [
    "table", "mode", "model", "horizon", "K", "D_GCN", "D_GRU", "seed",
    "lr", "weight_decay", "epochs", "status", "test_mse", "test_mae",
    "wall_seconds", "error",
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


def load_gcs(horizon, device, seed):
    from st_dif.data_utils import get_loaders, get_pyg_temporal_dataset

    dataset, _ = get_pyg_temporal_dataset("GCS", horizon)
    loaders = get_loaders(
        dataset, 32, 0.7, 0.0, 0.3, device,
        manual_seed=42, shuffle_seed=seed + 20_000,
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
    raise ValueError(config["model"])


def configs_for(table, modes, horizons):
    if table == "3":
        return [dict(table="3", mode=m, model="DenseGCNGRU", horizon=20,
                     K=k, D_GCN=128, D_GRU=64)
                for m in modes for k in (1, 2, 3)]
    if table == "4":
        sizes = [(128, 64), (64, 32), (32, 16)]
        return [dict(table="4", mode=m, model="DenseGCNGRU", horizon=20,
                     K=3, D_GCN=g, D_GRU=r)
                for m in modes for g, r in sizes]
    if table == "5":
        return [dict(table="5", mode=m, model="DenseGCNGRU", horizon=h,
                     K=3, D_GCN=128, D_GRU=64)
                for m in modes for h in horizons]
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
    loaders = load_gcs(config["horizon"], device, seed)
    model = build_model(config)
    if hasattr(model, "gru"):
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
        mse_mean=("test_mse", "mean"), mse_min=("test_mse", "min"),
        mse_max=("test_mse", "max"), mae_mean=("test_mae", "mean"),
        mae_min=("test_mae", "min"), mae_max=("test_mae", "max"),
        runs=("seed", "count"),
    ).reset_index()
    result["mse_half_range"] = (result.mse_max - result.mse_min) / 2
    result["mae_half_range"] = (result.mae_max - result.mae_min) / 2
    print(result[[*group, "mse_mean", "mse_half_range", "mae_mean",
                  "mae_half_range", "runs"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=["2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--modes", nargs="+", choices=["legacy", "corrected"],
                        default=["legacy", "corrected"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[20, 60, 120, 240])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results_reproduction")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "sten_results.csv"
    metadata = {
        "torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "seeds": args.seeds,
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
