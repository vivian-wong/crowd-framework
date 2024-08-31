# CMGraph-Crowd-Forecasting

This is the repository for the paper 
```
Modeling Crowd Data and Spatial Connectivity as Graphs for Effective Crowd Flow Forecasting in Public Urban Space
```

Instructions to run: 
1. Create a conda environment pedestrian-test with environments.yml: 
```
conda env create -f environment.yml
```
2. Navigate to main.ipynb. You should be able to run all codes in the notebook to reproduce the results in the paper. Note that the performance will not be exact because there is some stochasticity during training, but the test MSE/MAE should not deviate that much from those reported in the paper.  
