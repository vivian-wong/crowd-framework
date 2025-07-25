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