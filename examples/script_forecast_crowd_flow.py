# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: test
#     language: python
#     name: python3
# ---

# %% [markdown]
# # STEN Forecasting demo
#
# In this notebook, we will:
# 1. **Load data**: Load pytorch geometric temporal data and split into torch train test val dataloaders.
# 2. **Model Building**: We will build and train a STEN model - the DenseGCNGRU model using the processed data.
# 3. **Model Evaluation**: We will print and plot model outputs to look at its performance. 

# %%
# import all the necessary libraries
import os
import torch

# for relative imports
os.chdir('..') 
print(os.getcwd()) # should print /your_local_dir/crowd-framework

# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %%
class Args:
    def __init__(self):
        self.DATASET = 'GCS'
        self.forecasting_horizon = 60 # 20 60 120 240
        self.train_ratio = 0.7
        self.test_ratio = 0.3
        self.val_ratio = 0.0
        self.batch_size = 64
        self.lr = 0.001
        self.epochs = 30
        self.save_model = False
        self.save_dir = './checkpoints'
args = Args()

# %% [markdown]
# ## 1. Load data. this step is the same as one shown in demo_dataset.ipynb. 

# %%
from st_dif.data_utils import get_pyg_temporal_dataset, get_loaders
import st_dif
print(st_dif.__file__)


# get pytorch dataloaders
dataset, _ = get_pyg_temporal_dataset(args.DATASET, args.forecasting_horizon)
train_loader, val_loader, test_loader = get_loaders(dataset, 
                                                    args.batch_size, 
                                                    args.train_ratio, 
                                                    args.val_ratio, 
                                                    args.test_ratio, 
                                                    device)



# %% [markdown]
# In order to run a graph neural network, the inputs are $A$, the adjacency matrix, and $X$, the node feature matrix. Since our $A$ is defined by inter-PAR connections and is assumed to never change, we can reduce computational time by only loading $A$ once. In PyTorch Geometric, $A$ is represented as the edge_index object.

# %%
# get static edge index (i.e. adjacency matrix). Only need to do this one since edge index doesn't change for each CMGraph. 
for snapshot in dataset:
    static_edge_index = snapshot.edge_index.to(device)
    break;
# Edge indices (represents adjacency matrix/PAR connections) of the CMGraphs. 
print(static_edge_index)

# %% [markdown]
# ## 2. Set up and Train a STEN Model

# %% [markdown]
# STEN stands for spatio-temporal encoder network. It is a framework for crowd flow forecasting that involves spatially connected pedestrian activity regions (PARs). In this repo we have provided two STEN models for easy plug in and play. We'll use the best performing mode, Dense-GCN-GRU here. The model can simply be called from our STEN model zoo.

# %%


#print(model)

#from st_dif.models import DenseGCNGRU_FC
#model = DenseGCNGRU_FC(in_channels=2,
#                    periods=args.forecasting_horizon,
#                    batch_size=args.batch_size).to(device)
#print(model)


#from st_dif.models import DenseGCNGRU_GRU
#model = DenseGCNGRU_GRU(in_channels=2,
#                    periods=args.forecasting_horizon,
#                    batch_size=args.batch_size).to(device)
#print(model)





## Dongdong Wang 01-11-2026
#
#from st_dif.models import DenseGCNTransformer
#
#import torch
#
#x_seq = []
#y_seq = []
#
#print(dataset)
#
## Iterate over time steps (snapshots)
#for t in range(dataset.snapshot_count):
#    x = torch.tensor(dataset.features[t], dtype=torch.float)  # [num_nodes, in_channels]
#    y = torch.tensor(dataset.targets[t], dtype=torch.float)   # [num_nodes, out_features]
#
#    # If your original code expects [seq_len, num_nodes, in_channels],
#    # we need to add a "time dimension". Here each time step is 1-length sequence.
#    # x = x.unsqueeze(0)  # [1, num_nodes, in_channels]
#    
#    x_seq.append(x)
#    y_seq.append(y)
#
## Convert to tensors
## x_seq: [seq_len, num_nodes, in_channels]
#x_seq = torch.stack(x_seq, dim=0)
## Add batch dimension: [batch=1, seq_len, num_nodes, in_channels]
#x_seq = x_seq.unsqueeze(0)
##x_seq = x_seq.permute(2,0,1,3)
#x_seq = x_seq[:, :, :, 0, :]  # [1, 5173, 9, 20]
#
## y_seq: [seq_len, num_nodes, out_features]
#y_seq = torch.stack(y_seq, dim=0)
## Add batch dimension: [batch=1, seq_len, num_nodes, out_features]
#y_seq = y_seq.unsqueeze(0)
#
#
#
#
#print('x_seq shape:', x_seq.shape)
#print('y_seq shape:', y_seq.shape)
#
#
#
#num_features = dataset.features[0].shape[1]
#seq_len = dataset.snapshot_count
#
#edge_index = dataset.edge_index
#edge_index = torch.tensor(edge_index, dtype=torch.long)  # [2, num_edges]
#
##model = DenseGCNTransformer(node_features=num_features, hidden_dim=32, out_features=y_seq.shape[-1], heads=4, seq_len=seq_len)
##model = DenseGCNTransformer(node_features=num_features, hidden_dim=32, out_features=2, heads=4, seq_len=seq_len)
#model = DenseGCNTransformer(node_features=num_features, hidden_dim=32, out_features=2, heads=4)
#print(edge_index)
#out = model(x_seq, edge_index)
#print(out.shape)  # [batch, num_nodes, out_features]
#exit()









# %% [markdown]
# To train the model
# with the data, we can use the train function from campuscrowd.train_test_utils. This function trains the model and return the model checkpoint. Inside train(), there is a training loop that essentially computes loss and does backprop based on the model-generated prediction vector y_hat. y_hat is computed in the following code snippet: 
# ```python
#  for encoder_inputs, labels in train_loader:
#     y_hat = model(encoder_inputs, static_edge_index)
#     # torch.tensor storing model predictions. Full training loop omitted for conciseness.
# ``` 

## %%
#from st_dif.train_test_utils import train, save_or_update_checkpoint, evaluate
#
#mse_array = []
#mae_array = []
#
#for i in range(10):
## train model
#model, checkpoint_dict = train( model, 
#                                train_loader, 
#                                val_loader, 
#                                static_edge_index, 
#                                num_epochs=args.epochs, lr=args.lr
#                                )
#
#if args.save_model:
#    filename = model.__class__.__name__+'_'+args.DATASET+'_'+'{}_steps'.format(args.forecasting_horizon)+'.pt'
#    path = os.path.join(args.save_dir,
#                        filename)
#    save_or_update_checkpoint(checkpoint_dict, path)
#
#model, checkpoint_dict, mse, mae = evaluate(model, test_loader, static_edge_index, checkpoint_dict=checkpoint_dict)
#
#mse_array.append(mse)
#mae_array.append(mae)




#####################################################################################################
### Dongdong Wang 01-21-2026
#import os
#import numpy as np
#from st_dif.train_test_utils import train, save_or_update_checkpoint, evaluate
#from st_dif.models import DenseGCNGRU, DenseGCNGRU_GRU, DenseGCNGRU_FC
#
#
#mse_list = []
#mae_list = []
#
#num_runs = 5
#
#for run in range(num_runs):
#    print("============> RUN " + str(run))
##    model = DenseGCNGRU(in_channels=2, 
##                    periods=args.forecasting_horizon, 
##                    batch_size=args.batch_size).to(device) 
##                                    
##    model = DenseGCNGRU_FC(in_channels=2,
##                        periods=args.forecasting_horizon,
##                        batch_size=args.batch_size).to(device)
#
#    model = DenseGCNGRU_GRU(in_channels=2,
#                        periods=args.forecasting_horizon,
#                        batch_size=args.batch_size).to(device)
#
#    # Train model
#    model, checkpoint_dict = train(
#        model,
#        train_loader,
#        val_loader,
#        static_edge_index,
#        num_epochs=args.epochs,
#        lr=args.lr,
#    )
#
#    # Save checkpoint (optional)
#    if args.save_model:
#        filename = (
#            f"{model.__class__.__name__}_"
#            f"{args.DATASET}_{args.forecasting_horizon}_steps_run{run}.pt"
#        )
#        path = os.path.join(args.save_dir, filename)
#        save_or_update_checkpoint(checkpoint_dict, path)
#
#    # Evaluate model
#    model, checkpoint_dict, mse, mae = evaluate(
#        model,
#        test_loader,
#        static_edge_index,
#        checkpoint_dict=checkpoint_dict,
#    )
#
#    mse_list.append(mse)
#    mae_list.append(mae)
#
## Convert to numpy arrays
#mse_array = np.array(mse_list)
#mae_array = np.array(mae_list)
#
## Final statistics (mean and range)
#mse_mean = mse_array.mean()
#mse_range = (mse_array.max() - mse_array.min())/2
#
#mae_mean = mae_array.mean()
#mae_range = (mae_array.max() - mae_array.min())/2
#
#print(f"MSE: mean = {mse_mean:.4f}, range = {mse_range:.4f}")
#print(f"MAE: mean = {mae_mean:.4f}, range = {mae_range:.4f}")



####################################################################################################
## Dongdong Wang 01-23-2026
import os
import numpy as np
from st_dif.train_test_utils import train, save_or_update_checkpoint, evaluate
from st_dif.models import DenseGCNGRU, DenseGCNGRU_GRU, DenseGCNGRU_FC, RecurrentGCN, A3TGCNWrapper, A3TGCNCompatible, DenseGCLSTM, DenseLRGCN, DenseMPNNLSTM


#model = RecurrentGCN(in_channels=2, 
#                    periods=args.forecasting_horizon, 
#                    batch_size=args.batch_size).to(device) 
                    
                    
                    
                    
#mse_list = []
#mae_list = []
#
#num_runs = 5
#
#for run in range(num_runs):
#    print("============> RUN " + str(run))
##    model = DenseGCNGRU(in_channels=2, 
##                    periods=args.forecasting_horizon, 
##                    batch_size=args.batch_size).to(device) 
##                                    
##    model = DenseGCNGRU_FC(in_channels=2,
##                        periods=args.forecasting_horizon,
##                        batch_size=args.batch_size).to(device)
#
##    model = DenseGCNGRU_GRU(in_channels=2,
##                        periods=args.forecasting_horizon,
##                        batch_size=args.batch_size).to(device)
#
#
#    model = RecurrentGCN(in_channels=2,
#                        periods=args.forecasting_horizon,
#                        batch_size=args.batch_size).to(device)
#                        
#                        
##    model = DenseGCLSTM(in_channels=2,
##                        hidden_channels=128,
##                        periods=args.forecasting_horizon).to(device)
#    
##    model = DenseLRGCN(in_channels=2,
##                        periods=args.forecasting_horizon,
##                        batch_size=args.batch_size).to(device)
#                        
##    model = DenseMPNNLSTM(in_channels=2,
##                        periods=args.forecasting_horizon).to(device)
#                        
#                        
#                        
##    model = A3TGCNWrapper(in_channels=2,
##                        periods=args.forecasting_horizon).to(device)
#                        
#
##    model = A3TGCNCompatible(in_channels=2,
##                        periods=args.forecasting_horizon).to(device)  
#
#    # Train model
#    model, checkpoint_dict = train(
#        model,
#        train_loader,
#        val_loader,
#        static_edge_index,
#        num_epochs=args.epochs,
#        lr=args.lr,
#    )
#
#    # Save checkpoint (optional)
#    if args.save_model:
#        filename = (
#            f"{model.__class__.__name__}_"
#            f"{args.DATASET}_{args.forecasting_horizon}_steps_run{run}.pt"
#        )
#        path = os.path.join(args.save_dir, filename)
#        save_or_update_checkpoint(checkpoint_dict, path)
#
#    # Evaluate model
#    model, checkpoint_dict, mse, mae = evaluate(
#        model,
#        test_loader,
#        static_edge_index,
#        checkpoint_dict=checkpoint_dict,
#    )
#
#    mse_list.append(mse)
#    mae_list.append(mae)
#
## Convert to numpy arrays
#mse_array = np.array(mse_list)
#mae_array = np.array(mae_list)
#
## Final statistics (mean and range)
#mse_mean = mse_array.mean()
#mse_range = (mse_array.max() - mse_array.min())/2
#
#mae_mean = mae_array.mean()
#mae_range = (mae_array.max() - mae_array.min())/2
#
#print(f"MSE: mean = {mse_mean:.4f}, range = {mse_range:.4f}")
#print(f"MAE: mean = {mae_mean:.4f}, range = {mae_range:.4f}")










forecasting_horizons = [120]   # adjust as needed
num_runs = 5

def build_model(model_name, horizon):
    if model_name == "DenseGCNGRU":
        return DenseGCNGRU(
            in_channels=2,
            periods=horizon,
            batch_size=args.batch_size
        )

    elif model_name == "RecurrentGCN":
        return RecurrentGCN(
            in_channels=2,
            periods=horizon,
            batch_size=args.batch_size
        )

    elif model_name == "DenseGCLSTM":
        return DenseGCLSTM(
            in_channels=2,
            hidden_channels=128,
            periods=horizon
        )

    elif model_name == "DenseLRGCN":
        return DenseLRGCN(
            in_channels=2,
            periods=horizon,
            batch_size=args.batch_size
        )

    elif model_name == "DenseMPNNLSTM":
        return DenseMPNNLSTM(
            in_channels=2,
            periods=horizon
        )


    else:
        raise ValueError(f"Unknown model: {model_name}")




model_names = [
    "DenseGCNGRU",
    "RecurrentGCN",
    "DenseGCLSTM",
    "DenseLRGCN",
    "DenseMPNNLSTM",
]



results = {}

# -----------------------------
# Main experiment loop
# -----------------------------
for horizon in forecasting_horizons:
    print(f"\n================ HORIZON {horizon} =================")
    args.forecasting_horizon = horizon
    results[horizon] = {}
    
    
    dataset, _ = get_pyg_temporal_dataset(args.DATASET, args.forecasting_horizon)
    train_loader, val_loader, test_loader = get_loaders(dataset, 
                                                        args.batch_size, 
                                                        args.train_ratio, 
                                                        args.val_ratio, 
                                                        args.test_ratio, 
                                                        device)
    
    
    
    for snapshot in dataset:
        static_edge_index = snapshot.edge_index.to(device)
        break;
    
    
    

    for model_name in model_names:
        print(f"\n---- Model: {model_name} ----")
        mse_list, mae_list = [], []

        for run in range(num_runs):
            print(f"============> RUN {run}")

            model = build_model(model_name, horizon).to(device)

            # Train
            model, checkpoint_dict = train(
                model,
                train_loader,
                val_loader,
                static_edge_index,
                num_epochs=args.epochs,
                lr=args.lr,
            )

            # Save checkpoint
            if args.save_model:
                filename = (
                    f"{model_name}_{args.DATASET}_"
                    f"{horizon}_steps_run{run}.pt"
                )
                path = os.path.join(args.save_dir, filename)
                save_or_update_checkpoint(checkpoint_dict, path)

            # Evaluate
            model, checkpoint_dict, mse, mae = evaluate(
                model,
                test_loader,
                static_edge_index,
                checkpoint_dict=checkpoint_dict,
            )

            mse_list.append(mse)
            mae_list.append(mae)

        # Aggregate
        mse_array = np.array(mse_list)
        mae_array = np.array(mae_list)

        results[horizon][model_name] = {
            "mse_mean": mse_array.mean(),
            "mse_range": (mse_array.max() - mse_array.min()) / 2,
            "mae_mean": mae_array.mean(),
            "mae_range": (mae_array.max() - mae_array.min()) / 2,
        }

        print(
            f"MSE: mean={results[horizon][model_name]['mse_mean']:.4f}, "
            f"range={results[horizon][model_name]['mse_range']:.4f}"
        )
        print(
            f"MAE: mean={results[horizon][model_name]['mae_mean']:.4f}, "
            f"range={results[horizon][model_name]['mae_range']:.4f}"
        )

# -----------------------------
# Export to CSV
# -----------------------------
csv_path = "./forecasting_horizon_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "horizon",
        "model",
        "mse_mean",
        "mse_range",
        "mae_mean",
        "mae_range",
    ])

    for horizon, model_dict in results.items():
        for model_name, metrics in model_dict.items():
            writer.writerow([
                horizon,
                model_name,
                metrics["mse_mean"],
                metrics["mse_range"],
                metrics["mae_mean"],
                metrics["mae_range"],
            ])

print(f"\n✅ Results exported to: {csv_path}")