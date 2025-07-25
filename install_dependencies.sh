echo "››› Installing base dependencies from requirements.txt..."
pip install -r requirements.txt

echo "››› Installing PyTorch Geometric dependencies..."
# Determine the correct PyG wheel URL based on the installed PyTorch and CUDA versions
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu')")

echo "INFO: Detected PyTorch version: $TORCH_VERSION"
echo "INFO: Detected CUDA version for PyG URL: $CUDA_VERSION"

pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric==2.3
pip install torch-geometric-temporal
pip install pandas==2.2.3
