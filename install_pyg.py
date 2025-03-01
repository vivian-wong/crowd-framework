import subprocess
import torch
import sys

def install_with_subprocess(package, options=""):
    """Install a package using pip with subprocess."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", package] + options.split()
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        sys.exit(1)

# Detect Torch version and CUDA availability
TORCH = torch.__version__.split("+")[0]
if not torch.cuda.is_available():
    print(f"No CUDA found for torch=={TORCH}. Using CPU.")
    CUDA = "cpu"
else:
    CUDA = "cu" + torch.version.cuda.replace(".", "")
print(f"Using Torch version: {TORCH}, CUDA: {CUDA}")

# Install PyTorch Geometric dependencies
pyg_url = f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html"
print(f"Installing PyTorch Geometric dependencies from {pyg_url}")

install_with_subprocess("torch-scatter", f"-f {pyg_url}")
install_with_subprocess("torch-sparse", f"-f {pyg_url}")

# Install torch-geometric and torch-geometric-temporal
install_with_subprocess("torch-geometric==2.3")
install_with_subprocess("torch-geometric-temporal")

# Ensure pandas compatibility
try:
    import pandas
    if pandas.__version__ != "1.3.5":
        print("Installing pandas==1.3.5 for compatibility with torch-geometric-temporal.")
        install_with_subprocess("pandas==1.3.5 --force-reinstall")
except ImportError:
    print("Installing pandas==1.3.5 as it is missing.")
    install_with_subprocess("pandas==1.3.5")
