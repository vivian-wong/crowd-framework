import torch
from st_dif.data_utils import get_pyg_temporal_dataset, get_loaders

# run 'python -m pytest tests/test_package.py' if doing single test
def test_data_loading_and_loaders():
    print("Testing st_dif package loaders...")

    class Args:
        def __init__(self):
            self.DATASET = "GCS"
            self.forecasting_horizon = 20
            self.train_ratio = 0.7
            self.val_ratio = 0.1
            self.test_ratio = 0.2
            self.batch_size = 32

    args = Args()

    dataset, cmgraph = get_pyg_temporal_dataset(args.DATASET, args.forecasting_horizon)
    print("Dataset loaded successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        args.batch_size,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        device=device
    )
    print("Data loaders created successfully.")

    first_batch = next(iter(train_loader))
    print("First batch:", first_batch)

    print("Test script completed without errors.")
