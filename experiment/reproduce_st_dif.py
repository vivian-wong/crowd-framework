"""Reproduce Tables 2-5 of the ST-DIF paper on the GCS dataset.

This script is a CLI packaging of the sweeps in
``examples/reproduction_report.ipynb`` (independent reproduction by Ao He).
All hyperparameters follow paper Section 5.2:

    train/val/test = 0.7 / 0.0 / 0.3, batch_size = 32, lr = 1e-3,
    epochs = 40, Adam, seeds = run_idx in {0..4}.

Usage (from the ``experiment/`` directory, inside the st_dif environment):

    python reproduce_st_dif.py --table 2                 # parameter counts (seconds, CPU is fine)
    python reproduce_st_dif.py --table 3                 # K-sensitivity sweep      (30 runs)
    python reproduce_st_dif.py --table 4                 # model-size sweep         (30 runs)
    python reproduce_st_dif.py --table 5                 # full horizon grid        (100 runs)
    python reproduce_st_dif.py --table 5 --horizons 20   # only the T=20 column     (25 runs)
    python reproduce_st_dif.py --table all

Each training run appends one row to a CSV under ``--results_dir``
(default ``../results_reproduction``). Re-running the script skips
already-completed (config, seed) cells, so interrupted sweeps resume
where they left off. Aggregated ``mean +/- (max-min)/2`` tables are
printed next to the paper's reference numbers at the end of each sweep.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import DeepGCNLayer

# Make the repo importable when running from experiment/
REPO_ROOT = Path(__file__).resolve().parent.parent
# Insert at the FRONT of sys.path so the repo's src/st_dif (with the updated
# sten.py) takes priority over any pip-installed st_dif in site-packages.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))

# st_dif.data_utils resolves the GCS config as './data/...' relative to the
# current working directory, so run everything from the repo root regardless
# of where this script was launched from.
os.chdir(REPO_ROOT)

from st_dif.data_utils import get_pyg_temporal_dataset, get_loaders          # noqa: E402
from st_dif.train_test_utils import train, evaluate                          # noqa: E402
from st_dif.models.sten import (                                             # noqa: E402
    DenseGCNGRU,
    GCNGRU,
    GRU_only,
    RecurrentGCN,
    DenseGCLSTM,
    DenseLRGCN,
    DenseMPNNLSTM,
)

# --------------------------------------------------------------------------
# Paper Section 5.2 configuration
# --------------------------------------------------------------------------
DATASET      = 'GCS'
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.0
TEST_RATIO   = 0.3
BATCH_SIZE   = 32
LR           = 0.001
EPOCHS       = 40
NUM_RUNS     = 5
T20          = 20

# Hidden size for DenseGCLSTM (not specified in paper Section 5.2;
# 64 chosen to match D_GRU — same choice as in the reproduction notebook).
GCLSTM_HIDDEN = 64

HORIZONS_ALL  = [20, 60, 120, 240]
TABLE5_MODELS = ['DenseGCNGRU', 'RecurrentGCN', 'DenseGCLSTM',
                 'DenseLRGCN', 'DenseMPNNLSTM']

# --------------------------------------------------------------------------
# Paper reference numbers (for side-by-side comparison in the printout)
# --------------------------------------------------------------------------
PAPER_TABLE2_PARAMS = {
    'DenseGCNGRU'  : 97_300,
    'GCNGRU'       : 96_916,
    'GRU_only'     : 39_316,
    'RecurrentGCN' : 122_260,   # DCRNN
    'DenseGCLSTM'  : 51_476,    # GC-LSTM
    'DenseLRGCN'   : 148,       # LRGCN
    'DenseMPNNLSTM': 352_276,   # MPNN-LSTM
}

PAPER_TABLE3 = pd.DataFrame([
    {'K': 1, 'paper_mse': '0.0905 ± 0.0009', 'paper_mae': '0.2089 ± 0.0012'},
    {'K': 2, 'paper_mse': '0.0894 ± 0.0010', 'paper_mae': '0.2080 ± 0.0010'},
    {'K': 3, 'paper_mse': '0.0891 ± 0.0009', 'paper_mae': '0.2075 ± 0.0013'},
])

PAPER_TABLE4 = pd.DataFrame([
    {'size_name': 'Standard', 'paper_mse': '0.0891 ± 0.0009', 'paper_mae': '0.2075 ± 0.0013'},
    {'size_name': 'Medium',   'paper_mse': '0.0949 ± 0.0013', 'paper_mae': '0.2126 ± 0.0032'},
    {'size_name': 'Tiny',     'paper_mse': '0.0987 ± 0.0004', 'paper_mae': '0.2156 ± 0.0025'},
])

# Published Table 5 (BuildSys '26, GCS dataset), all four horizons.
# Format: mean ± (max-min)/2 over 5 runs. Keyed by (model, horizon).
PAPER_TABLE5 = pd.DataFrame([
    # RecurrentGCN (DCRNN)
    {'model': 'RecurrentGCN',  'horizon':  20, 'paper_mse': '0.113 ± 0.006', 'paper_mae': '0.229 ± 0.006'},
    {'model': 'RecurrentGCN',  'horizon':  60, 'paper_mse': '0.192 ± 0.004', 'paper_mae': '0.300 ± 0.006'},
    {'model': 'RecurrentGCN',  'horizon': 120, 'paper_mse': '0.273 ± 0.006', 'paper_mae': '0.364 ± 0.008'},
    {'model': 'RecurrentGCN',  'horizon': 240, 'paper_mse': '0.354 ± 0.011', 'paper_mae': '0.409 ± 0.009'},
    # GC-LSTM
    {'model': 'DenseGCLSTM',   'horizon':  20, 'paper_mse': '0.105 ± 0.003', 'paper_mae': '0.223 ± 0.006'},
    {'model': 'DenseGCLSTM',   'horizon':  60, 'paper_mse': '0.103 ± 0.016', 'paper_mae': '0.228 ± 0.013'},
    {'model': 'DenseGCLSTM',   'horizon': 120, 'paper_mse': '0.246 ± 0.031', 'paper_mae': '0.340 ± 0.031'},
    {'model': 'DenseGCLSTM',   'horizon': 240, 'paper_mse': '0.351 ± 0.011', 'paper_mae': '0.410 ± 0.014'},
    # LRGCN
    {'model': 'DenseLRGCN',    'horizon':  20, 'paper_mse': '0.121 ± 0.008', 'paper_mae': '0.239 ± 0.008'},
    {'model': 'DenseLRGCN',    'horizon':  60, 'paper_mse': '0.194 ± 0.003', 'paper_mae': '0.301 ± 0.004'},
    {'model': 'DenseLRGCN',    'horizon': 120, 'paper_mse': '0.270 ± 0.005', 'paper_mae': '0.355 ± 0.004'},
    {'model': 'DenseLRGCN',    'horizon': 240, 'paper_mse': '0.351 ± 0.015', 'paper_mae': '0.407 ± 0.006'},
    # MPNN-LSTM
    {'model': 'DenseMPNNLSTM', 'horizon':  20, 'paper_mse': '0.147 ± 0.096', 'paper_mae': '0.251 ± 0.062'},
    {'model': 'DenseMPNNLSTM', 'horizon':  60, 'paper_mse': '0.103 ± 0.019', 'paper_mae': '0.223 ± 0.017'},
    {'model': 'DenseMPNNLSTM', 'horizon': 120, 'paper_mse': '0.148 ± 0.027', 'paper_mae': '0.268 ± 0.022'},
    {'model': 'DenseMPNNLSTM', 'horizon': 240, 'paper_mse': '0.153 ± 0.013', 'paper_mae': '0.276 ± 0.010'},
    # ST-DIF (Dense-GCN-GRU)
    {'model': 'DenseGCNGRU',   'horizon':  20, 'paper_mse': '0.089 ± 0.004', 'paper_mae': '0.210 ± 0.008'},
    {'model': 'DenseGCNGRU',   'horizon':  60, 'paper_mse': '0.088 ± 0.008', 'paper_mae': '0.215 ± 0.007'},
    {'model': 'DenseGCNGRU',   'horizon': 120, 'paper_mse': '0.086 ± 0.004', 'paper_mae': '0.214 ± 0.007'},
    {'model': 'DenseGCNGRU',   'horizon': 240, 'paper_mse': '0.091 ± 0.001', 'paper_mae': '0.218 ± 0.002'},
])


# --------------------------------------------------------------------------
# Patched DenseGCNGRU used for the Table 3 / Table 4 sweeps
# (identical to Section 4 of examples/reproduction_report.ipynb)
# --------------------------------------------------------------------------
class KLayerGCNConvPatched(torch.nn.Module):
    """Patched copy of ``st_dif.models.sten.KLayerGCNConv``.

    Adds ``fix_forward_bug``:
      - False (default): replicate the original forward, which returns after
        the first GCN layer regardless of K ("paper-faithful").
      - True: apply all K GCN layers in sequence.

    Module construction is otherwise identical to the original, so under the
    same RNG state the parameter tensors match bit-for-bit.
    """

    def __init__(self, K, in_channels, out_channels, node_dim,
                 improved=True, cached=False, add_self_loops=True,
                 fix_forward_bug=False):
        super().__init__()
        self.fix_forward_bug = fix_forward_bug
        convs = []
        for k in range(K):
            convs.append(GCNConv(
                in_channels=in_channels if k == 0 else out_channels,
                out_channels=out_channels,
                node_dim=node_dim,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
            ))
        self.convs = torch.nn.Sequential(*convs)

    def forward(self, x, edge_index, edge_weight):
        for k in range(len(self.convs)):
            x = self.convs[k].forward(x, edge_index, edge_weight)
            if not self.fix_forward_bug:
                return x          # paper-faithful: return after first layer
        return x                  # fixed: applied all K layers


class DenseGCNGRUPatched(torch.nn.Module):
    """Patched copy of ``st_dif.models.sten.DenseGCNGRU``.

    Exposes ``K``, ``D_GCN``, ``D_GRU``, ``num_gru_layers``, and
    ``fix_forward_bug``. Defaults reproduce the hardcoded values in the
    original (K=3, D_GCN=128, D_GRU=64, num_gru_layers=2,
    fix_forward_bug=False).
    """

    def __init__(self, in_channels, periods, batch_size,
                 K=3, D_GCN=128, D_GRU=64, num_gru_layers=2,
                 fix_forward_bug=False, improved=False, cached=False,
                 add_self_loops=True):
        super().__init__()
        self.in_channels     = in_channels
        self.periods         = periods
        self.batch_size      = batch_size
        self.K               = K
        self.D_GCN           = D_GCN
        self.D_GRU           = D_GRU
        self.num_gru_layers  = num_gru_layers
        self.fix_forward_bug = fix_forward_bug
        self.improved        = improved
        self.cached          = cached
        self.add_self_loops  = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self.densegcn = DeepGCNLayer(
            conv=KLayerGCNConvPatched(
                K=self.K,
                in_channels=self.in_channels,
                out_channels=self.D_GCN,
                node_dim=1,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
                fix_forward_bug=self.fix_forward_bug,
            ),
            norm=None,
            act=None,
            dropout=0,
            block='dense',
        )
        gru_input = self.D_GCN + self.in_channels   # dense-concat with original features
        self.gru = torch.nn.GRU(gru_input, self.D_GRU, self.num_gru_layers,
                                batch_first=True)
        self.fc  = torch.nn.Linear(self.D_GRU, self.periods)

    def forward(self, X, edge_index, edge_weight=None):
        gru_in = torch.zeros(
            X.shape[0], X.shape[1], X.shape[3],
            self.D_GCN + self.in_channels,
        ).to(X.device)
        for t in range(X.shape[3]):
            gcn_out = self.densegcn(X[:, :, :, t], edge_index, edge_weight)
            gru_in[:, :, t, :] = gcn_out
        gru_in = gru_in.flatten(start_dim=0, end_dim=1)
        gru_out, _ = self.gru(gru_in)
        out = self.fc(gru_out[:, -1, :])
        out = out.view(X.shape[0], X.shape[1], self.periods, -1)
        return out.squeeze(dim=3)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_model(name, horizon, batch_size):
    """Per-model constructor signatures (matching the reproduction notebook)."""
    if name == 'DenseGCNGRU':
        return DenseGCNGRU(in_channels=2, periods=horizon, batch_size=batch_size)
    if name == 'GCNGRU':
        return GCNGRU(in_channels=2, periods=horizon, batch_size=batch_size)
    if name == 'GRU_only':
        return GRU_only(in_channels=2, periods=horizon, batch_size=batch_size)
    if name == 'RecurrentGCN':
        return RecurrentGCN(in_channels=2, periods=horizon, batch_size=batch_size)
    if name == 'DenseGCLSTM':
        return DenseGCLSTM(in_channels=2, hidden_channels=GCLSTM_HIDDEN, periods=horizon)
    if name == 'DenseLRGCN':
        return DenseLRGCN(in_channels=2, periods=horizon, batch_size=batch_size)
    if name == 'DenseMPNNLSTM':
        return DenseMPNNLSTM(in_channels=2, periods=horizon)
    raise ValueError(f'Unknown model: {name}')


def load_gcs(horizon, device):
    dataset, _ = get_pyg_temporal_dataset(DATASET, horizon)
    train_loader, val_loader, test_loader = get_loaders(
        dataset, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, device,
    )
    for snapshot in dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break
    return train_loader, val_loader, test_loader, static_edge_index


def half_range(s):
    return (s.max() - s.min()) / 2


def run_one(model, loaders, num_epochs, device):
    """Train + evaluate a freshly built model; returns (mse, mae)."""
    train_loader, val_loader, test_loader, static_edge_index = loaders
    model = model.to(device)
    model, ckpt = train(model, train_loader, val_loader, static_edge_index,
                        num_epochs=num_epochs, lr=LR)
    model, ckpt = evaluate(model, test_loader, static_edge_index,
                           checkpoint_dict=ckpt)
    return float(ckpt['test_mse']), float(ckpt['test_mae'])


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Table 2: parameter counts (no training required)
# --------------------------------------------------------------------------
def reproduce_table2(device):
    print('\n=== Table 2: parameter counts (built at paper Section 5.2 defaults, T=20) ===\n')
    print(f'{"model":15s} {"paper":>10s} {"ours":>10s}   match')
    all_ok = True
    for name, paper_n in PAPER_TABLE2_PARAMS.items():
        torch.manual_seed(0)
        model = build_model(name, T20, BATCH_SIZE)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ok = (n == paper_n)
        all_ok &= ok
        print(f'{name:15s} {paper_n:>10,d} {n:>10,d}   {"✓" if ok else "✗ MISMATCH"}')
        model = None
        cleanup()
    print('\nAll parameter counts match the paper.' if all_ok
          else '\nWARNING: at least one parameter count deviates from the paper.')


# --------------------------------------------------------------------------
# Table 3: sensitivity to GCN depth K (DenseGCNGRU, GCS, T=20)
# --------------------------------------------------------------------------
def reproduce_table3(device, results_dir, modes):
    csv_path = results_dir / 'table3_K_sensitivity.csv'
    k_values = [1, 2, 3]
    n_total  = len(modes) * len(k_values) * NUM_RUNS

    done = set()
    if csv_path.exists():
        for _, r in pd.read_csv(csv_path).iterrows():
            done.add((str(r['mode']), int(r['K']), int(r['run_idx'])))
        print(f'[Table 3] found {len(done)} completed runs in {csv_path.name}; skipping them.')
    else:
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['mode', 'K', 'D_GCN', 'D_GRU', 'run_idx',
                                    'test_mse', 'test_mae', 'wall_seconds'])

    loaders = load_gcs(T20, device)
    ctr = 0
    for mode in modes:
        fix_flag = (mode == 'fixed')
        for K in k_values:
            for run_idx in range(NUM_RUNS):
                ctr += 1
                if (mode, K, run_idx) in done:
                    continue
                torch.manual_seed(run_idx)
                np.random.seed(run_idx)
                t0 = time.time()
                try:
                    model = DenseGCNGRUPatched(
                        in_channels=2, periods=T20, batch_size=BATCH_SIZE,
                        K=K, D_GCN=128, D_GRU=64, num_gru_layers=2,
                        fix_forward_bug=fix_flag,
                    )
                    mse, mae = run_one(model, loaders, EPOCHS, device)
                except Exception as e:   # noqa: BLE001 - record failure, keep sweeping
                    mse, mae = float('nan'), float('nan')
                    print(f'  !!! mode={mode} K={K} run={run_idx} failed: '
                          f'{type(e).__name__}: {e}')
                finally:
                    model = None
                    cleanup()
                with csv_path.open('a', newline='') as f:
                    csv.writer(f).writerow([mode, K, 128, 64, run_idx,
                                            mse, mae, f'{time.time()-t0:.1f}'])
                print(f'[T3 {ctr:2d}/{n_total}] mode={mode:14s} K={K} '
                      f'run={run_idx}  MSE={mse:.4f}  MAE={mae:.4f}')

    # ---- aggregate ----
    df = pd.read_csv(csv_path).dropna(subset=['test_mse', 'test_mae'])
    agg = (df.groupby(['mode', 'K'])
             .agg(mse_mean=('test_mse', 'mean'), mse_hr=('test_mse', half_range),
                  mae_mean=('test_mae', 'mean'), mae_hr=('test_mae', half_range),
                  n_runs=('test_mse', 'count'))
             .reset_index())
    print('\n=== Table 3 reproduction (paper Section 5.5.1) ===\n')
    for mode in modes:
        sub = agg[agg['mode'] == mode]
        if len(sub) == 0:
            continue
        print(f'--- mode = {mode} ---')
        fmt = sub.merge(PAPER_TABLE3, on='K')
        fmt['my_mse'] = fmt.apply(lambda r: f"{r['mse_mean']:.4f} ± {r['mse_hr']:.4f}", axis=1)
        fmt['my_mae'] = fmt.apply(lambda r: f"{r['mae_mean']:.4f} ± {r['mae_hr']:.4f}", axis=1)
        print(fmt[['K', 'paper_mse', 'my_mse', 'paper_mae', 'my_mae', 'n_runs']]
              .to_string(index=False), '\n')


# --------------------------------------------------------------------------
# Table 4: sensitivity to model size (DenseGCNGRU, GCS, T=20)
# --------------------------------------------------------------------------
def reproduce_table4(device, results_dir, modes):
    csv_path = results_dir / 'table4_size_sensitivity.csv'
    size_configs = [
        {'name': 'Standard', 'D_GCN': 128, 'D_GRU': 64},
        {'name': 'Medium',   'D_GCN':  64, 'D_GRU': 32},
        {'name': 'Tiny',     'D_GCN':  32, 'D_GRU': 16},
    ]
    n_total = len(modes) * len(size_configs) * NUM_RUNS

    done = set()
    if csv_path.exists():
        for _, r in pd.read_csv(csv_path).iterrows():
            done.add((str(r['mode']), str(r['size_name']), int(r['run_idx'])))
        print(f'[Table 4] found {len(done)} completed runs in {csv_path.name}; skipping them.')
    else:
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['mode', 'size_name', 'D_GCN', 'D_GRU', 'K',
                                    'run_idx', 'test_mse', 'test_mae', 'wall_seconds'])

    loaders = load_gcs(T20, device)
    ctr = 0
    for mode in modes:
        fix_flag = (mode == 'fixed')
        for cfg in size_configs:
            for run_idx in range(NUM_RUNS):
                ctr += 1
                if (mode, cfg['name'], run_idx) in done:
                    continue
                torch.manual_seed(run_idx)
                np.random.seed(run_idx)
                t0 = time.time()
                try:
                    model = DenseGCNGRUPatched(
                        in_channels=2, periods=T20, batch_size=BATCH_SIZE,
                        K=3, D_GCN=cfg['D_GCN'], D_GRU=cfg['D_GRU'],
                        num_gru_layers=2, fix_forward_bug=fix_flag,
                    )
                    mse, mae = run_one(model, loaders, EPOCHS, device)
                except Exception as e:   # noqa: BLE001
                    mse, mae = float('nan'), float('nan')
                    print(f'  !!! mode={mode} {cfg["name"]} run={run_idx} failed: '
                          f'{type(e).__name__}: {e}')
                finally:
                    model = None
                    cleanup()
                with csv_path.open('a', newline='') as f:
                    csv.writer(f).writerow([mode, cfg['name'], cfg['D_GCN'],
                                            cfg['D_GRU'], 3, run_idx,
                                            mse, mae, f'{time.time()-t0:.1f}'])
                print(f'[T4 {ctr:2d}/{n_total}] mode={mode:14s} {cfg["name"]:9s} '
                      f'run={run_idx}  MSE={mse:.4f}  MAE={mae:.4f}')

    # ---- aggregate ----
    df = pd.read_csv(csv_path).dropna(subset=['test_mse', 'test_mae'])
    agg = (df.groupby(['mode', 'size_name', 'D_GCN', 'D_GRU'])
             .agg(mse_mean=('test_mse', 'mean'), mse_hr=('test_mse', half_range),
                  mae_mean=('test_mae', 'mean'), mae_hr=('test_mae', half_range),
                  n_runs=('test_mse', 'count'))
             .reset_index())
    order = {'Standard': 0, 'Medium': 1, 'Tiny': 2}
    print('\n=== Table 4 reproduction (paper Section 5.5.2) ===\n')
    for mode in modes:
        sub = agg[agg['mode'] == mode]
        if len(sub) == 0:
            continue
        print(f'--- mode = {mode} ---')
        fmt = sub.merge(PAPER_TABLE4, on='size_name')
        fmt = (fmt.assign(_o=fmt['size_name'].map(order))
                  .sort_values('_o').drop(columns='_o'))
        fmt['my_mse'] = fmt.apply(lambda r: f"{r['mse_mean']:.4f} ± {r['mse_hr']:.4f}", axis=1)
        fmt['my_mae'] = fmt.apply(lambda r: f"{r['mae_mean']:.4f} ± {r['mae_hr']:.4f}", axis=1)
        print(fmt[['size_name', 'D_GCN', 'D_GRU', 'paper_mse', 'my_mse',
                   'paper_mae', 'my_mae', 'n_runs']].to_string(index=False), '\n')


# --------------------------------------------------------------------------
# Table 5: cross-domain baselines over forecasting horizons (GCS)
# --------------------------------------------------------------------------
def reproduce_table5(device, results_dir, horizons):
    csv_path = results_dir / 'table5_forecasting_horizons.csv'
    n_total  = len(horizons) * len(TABLE5_MODELS) * NUM_RUNS

    done = set()
    if csv_path.exists():
        for _, r in pd.read_csv(csv_path).iterrows():
            done.add((int(r['horizon']), str(r['model']), int(r['run_idx'])))
        print(f'[Table 5] found {len(done)} completed runs in {csv_path.name}; skipping them.')
    else:
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(['horizon', 'model', 'run_idx',
                                    'test_mse', 'test_mae', 'wall_seconds'])

    ctr = 0
    for horizon in horizons:
        loaders = load_gcs(horizon, device)   # reload once per horizon
        for model_name in TABLE5_MODELS:
            for run_idx in range(NUM_RUNS):
                ctr += 1
                if (horizon, model_name, run_idx) in done:
                    continue
                torch.manual_seed(run_idx)
                np.random.seed(run_idx)
                t0 = time.time()
                try:
                    model = build_model(model_name, horizon, BATCH_SIZE)
                    mse, mae = run_one(model, loaders, EPOCHS, device)
                except Exception as e:   # noqa: BLE001
                    mse, mae = float('nan'), float('nan')
                    print(f'  !!! T={horizon} {model_name} run={run_idx} failed: '
                          f'{type(e).__name__}: {e}')
                finally:
                    model = None
                    cleanup()
                with csv_path.open('a', newline='') as f:
                    csv.writer(f).writerow([horizon, model_name, run_idx,
                                            mse, mae, f'{time.time()-t0:.1f}'])
                print(f'[T5 {ctr:3d}/{n_total}] T={horizon:3d} {model_name:14s} '
                      f'run={run_idx}  MSE={mse:.4f}  MAE={mae:.4f}')

    # ---- aggregate ----
    df = pd.read_csv(csv_path).dropna(subset=['test_mse', 'test_mae'])
    agg = (df.groupby(['horizon', 'model'])
             .agg(mse_mean=('test_mse', 'mean'), mse_hr=('test_mse', half_range),
                  mae_mean=('test_mae', 'mean'), mae_hr=('test_mae', half_range),
                  n_runs=('test_mse', 'count'))
             .reset_index())
    agg['my_mse'] = agg.apply(lambda r: f"{r['mse_mean']:.4f} ± {r['mse_hr']:.4f}", axis=1)
    agg['my_mae'] = agg.apply(lambda r: f"{r['mae_mean']:.4f} ± {r['mae_hr']:.4f}", axis=1)

    print('\n=== Table 5 reproduction (GCS, vs published BuildSys 26 numbers) ===\n')
    for horizon in sorted(agg['horizon'].unique()):
        sub = agg[agg['horizon'] == horizon]
        ref = PAPER_TABLE5[PAPER_TABLE5['horizon'] == horizon][['model', 'paper_mse', 'paper_mae']]
        fmt = sub.merge(ref, on='model', how='left')
        cols = ['model', 'paper_mse', 'my_mse', 'paper_mae', 'my_mae', 'n_runs']
        print(f'--- T = {horizon} ---')
        print(fmt[cols].to_string(index=False), '\n')
    print('Note: MPNN-LSTM has high run-to-run variance at T=20 (352k parameters, '
          'dropout=0.0 on a small dataset); the wide +/- in both the published '
          'value (0.147 +/- 0.096) and a fresh reproduction reflects this. See '
          'examples/reproduction_report.ipynb for the per-seed analysis.')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Reproduce Tables 2-5 of the ST-DIF paper (GCS dataset).')
    parser.add_argument('--table', type=str, default='all',
                        choices=['2', '3', '4', '5', 'all'])
    parser.add_argument('--results_dir', type=str,
                        default=str(REPO_ROOT / 'results_reproduction'),
                        help='Where per-run CSVs are written (resume-aware).')
    parser.add_argument('--horizons', type=int, nargs='+', default=HORIZONS_ALL,
                        help='Table 5 horizons to run (default: 20 60 120 240).')
    parser.add_argument('--modes', type=str, nargs='+', default=['paper_faithful'],
                        choices=['paper_faithful', 'fixed'],
                        help='Table 3/4 forward modes. "paper_faithful" reproduces '
                             'the paper; "fixed" additionally applies all K GCN layers.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cpu':
        print('WARNING: no GPU detected. Table 3/4/5 sweeps involve 30-100 '
              'training runs and will be very slow on CPU. Table 2 is fine.')

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.table in ('2', 'all'):
        reproduce_table2(device)
    if args.table in ('3', 'all'):
        reproduce_table3(device, results_dir, args.modes)
    if args.table in ('4', 'all'):
        reproduce_table4(device, results_dir, args.modes)
    if args.table in ('5', 'all'):
        reproduce_table5(device, results_dir, args.horizons)


if __name__ == '__main__':
    main()
