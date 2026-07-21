# st_dif

A Python library for crowd flow prediction and spatiotemporal modeling. Built on PyTorch, PyTorch Geometric, and PyTorch Geometric Temporal.

This is the Python library developed for Chapter 3 and 4 of the thesis: 
```
V. W. H. Wong, Spatio-temporal Representation Learning: Applications to Manufacturing Planning and Pedestrian Crowd Analysis, Ph.D. Thesis, Department of Civil and Environmental Engineering, Stanford University, Stanford, CA, June 2024.
```

---

# Installation Guide

Follow these steps carefully to set up your environment.

---

# Option 1: Install from PYPI
## 1. Create a New Virtual Environment

Using **conda**:

```bash
conda create -n st_dif_env python=3.10
conda activate st_dif_env
```

Or using **venv**:

```bash
python -m venv st_dif_env
st_dif_env\Scripts\activate   # Windows
source st_dif_env/bin/activate # macOS/Linux
```

---

## 2. Install PyTorch

Install PyTorch matching your hardware (example for CUDA 12.6):

```bash
pip install torch torchvision torchaudio
```

(Refer to the [PyTorch Get Started](https://pytorch.org/get-started/locally/) guide if you need specific instructions.)

---

## 3. Install PyTorch Geometric Libraries

Because `st_dif` depends on CUDA-based libraries, it is necessary to install them manually:

```bash
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric-temporal
```

Important: Make sure you install from the special `https://data.pyg.org` wheels to ensure CUDA compatibility.

---

## 4. Install `st_dif`

Install the `st_dif` library:

```bash
pip install st_dif
```

---

## Summary:
- Follow the installation steps carefully to match CUDA and PyTorch versions.
- Manually install `torch-scatter`, `torch-sparse`, and `torch-geometric` from the correct source.
- Then install `st_dif`.

### Example Full Setup Commands

```bash
conda create -n st_dif_env python=3.10
conda activate st_dif_env

pip install torch torchvision torchaudio

pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric-temporal

pip install st_dif
```

At this point, you are ready to use `st_dif`.

---

### Notes

- If you encounter `[WinError 127] The specified procedure could not be found`, it usually indicates that torch-scatter or torch-sparse were installed incorrectly. Reinstall them following the instructions above.
- Ensure that your PyTorch version matches your CUDA version.
- If using a CPU-only version of PyTorch, install CPU-compatible versions of the dependencies.

---

# Option 2: Install from Source (developer)
```bash
git clone https://github.com/vivian-wong/crowd-framework/
cd crowd-framework
pip install -e .[dev]

``` 

---

## Run Installation Test
Assuming pytest has been installed: 
```python
python -m pytest
```

---

# Example Usage

```python
from st_dif.data_utils import get_pyg_temporal_dataset, get_loaders
from st_dif.models.sten import STEN
```
Check the examples/ directory for simplified demo notebooks.

---

# Reproducing paper experiments 
To run all experiments as detailed in the thesis, run 
```
bash reproduce_paper_experiments.sh
```
and generate plots with the jupyter notebook experiments/plot_results.ipynb

---

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

---

# License

MIT License

---

---

# Reproducing the ST-DIF paper (Tables 2-5, GCS dataset)

This fork adds an independent reproduction of the following paper:

> Vivian W.H. Wong, Dongdong Wang, and Ao He. 2026. ST-DIF: A
> Spatio-Temporal Framework Integrating Adjacency Information in Crowd
> Forecasting. In *The 13th ACM International Conference on Systems for
> Energy-Efficient Buildings, Cities, and Transportation (BuildSys '26)*,
> June 22-25, 2026, Banff, AB, Canada. ACM, New York, NY, USA, 10 pages.
> https://doi.org/10.1145/3744256.3812585

The reproduction covers the paper's quantitative results (Tables 2-5)
plus the code needed to re-run them end to end.

## What changed in this fork

| File | Change |
|---|---|
| `src/st_dif/models/sten.py` | Replaced with the updated version containing the four cross-domain baselines used in Table 5 (`RecurrentGCN`/DCRNN, `DenseGCLSTM`/GC-LSTM, `DenseLRGCN`/LRGCN, `DenseMPNNLSTM`/MPNN-LSTM) in addition to `DenseGCNGRU`, `GCNGRU`, `GRU_only`. One import was made portable (`glorot`, `GCNConv_Fixed_W` now come from `torch_geometric_temporal` instead of a local module). |
| `src/st_dif/models/__init__.py` | Exports all seven model classes. |
| `experiment/reproduce_st_dif.py` | New CLI script that reproduces Tables 2, 3, 4, and 5. Resume-aware CSV logging, fixed seeds, aggregated printout next to the paper's reference numbers. |
| `experiment/reproduce_paper_experiments.sh` | Updated to run the four tables in sequence (the legacy thesis sweep is preserved, commented out, at the bottom). |
| `examples/reproduction_report.ipynb` | The original reproduction notebook, including the recorded outputs of the completed sweeps and the diagnostic analysis (patched `DenseGCNGRU`, K-forward behavior, seed-sensitivity of MPNN-LSTM). Paths were made repo-relative. |
| `examples/script_forecast_crowd_flow.py` | Standalone forecasting sweep script (jupytext format), directory-independent. |

## Setup

Follow the Installation Guide above (Option 2, editable install, is
recommended so that the updated `sten.py` in `src/` is the one that gets
imported). Verify the environment sees all baselines:

```bash
python -c "from st_dif.models import RecurrentGCN, DenseGCLSTM, DenseLRGCN, DenseMPNNLSTM; print('ok')"
```

If you installed `st_dif` from PyPI instead (Option 1), the installed
package still contains the old `sten.py`; either switch to the editable
install or overwrite `<site-packages>/st_dif/models/sten.py` with
`src/st_dif/models/sten.py` from this fork.

The GCS data ships with the repo (`data/gcs/gcs-processed/`); no download
step is needed.

## Run

```bash
cd experiment
bash reproduce_paper_experiments.sh          # everything: Tables 2, 3, 4, 5
```

or table by table:

```bash
python reproduce_st_dif.py --table 2                # parameter counts, seconds, CPU ok
python reproduce_st_dif.py --table 3                # K in {1,2,3}, 15 runs
python reproduce_st_dif.py --table 4                # Standard/Medium/Tiny, 15 runs
python reproduce_st_dif.py --table 5 --horizons 20  # T=20 column only, 25 runs
python reproduce_st_dif.py --table 5                # full grid, 100 runs
```

Per-run results are appended to CSVs in `results_reproduction/` at the
repo root. Interrupted sweeps resume automatically on re-run. Reported
statistic matches the paper: `mean ± (max − min)/2` over 5 runs with
seeds `torch.manual_seed(run_idx)` for `run_idx` in `{0..4}`.

Hyperparameters follow paper Section 5.2 throughout: train/test =
0.7/0.3, batch size 32, lr 1e-3, 40 epochs, Adam. The hidden size of
`DenseGCLSTM` is not specified in the paper; 64 is used (matching
`D_GRU`), consistent with the reproduction notebook.

## What to expect

From the completed reproduction recorded in
`examples/reproduction_report.ipynb`:

- **Table 2**: all parameter counts match the paper exactly (Δ = 0).
- **Table 3** (K-sensitivity): all three rows within 3% of the paper;
  the monotonic trend in K matches.
- **Table 4** (model size): all three rows within 1% of the paper.
- **Table 5** (cross-domain baselines over horizons T = 20/60/120/240):
  reproduces the published numbers within seed noise across all four
  horizons. MPNN-LSTM has high run-to-run variance at T=20 (352k
  parameters, dropout 0.0 on a small dataset); this shows up as a wide
  `±` in both the published value (0.147 ± 0.096) and a fresh
  reproduction, driven mainly by seed `run_idx=1` landing around MSE
  0.19–0.21. The `reproduce_st_dif.py` printout compares every cell
  against the published Table 5 side by side.

The notebook also documents a forward-pass detail of `KLayerGCNConv`
(it returns after the first GCN layer regardless of K). The sweeps in
`reproduce_st_dif.py` default to the paper-faithful behavior; pass
`--modes paper_faithful fixed` to additionally run the variant that
applies all K layers.
