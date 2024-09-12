import sys
sys.path.append("/home/sait2024/Project/uncertainty")

import pandas as pd
import scipy.stats as stats
import numpy as np
import torch

from model.GP import GPRegressionModel
from data.AtomDataset import AtomDataset

import gpytorch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch.nn as nn


ood_ratio = 0.1
model_name = "gpr_RQKernel_lr_0.1_patience_1000"
file_path = "/home/sait2024/Project/data/"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load Test Dataset
dataset = AtomDataset(f"{file_path}Test.xyz", device=device)
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
test_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1) 

# Load Train Dataset (for model initialization)
dataset = AtomDataset(f"{file_path}Train_1400.xyz", device=device)
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
train_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1) 
train_y = target_energy 

# Load Model
checkpoint = torch.load(f"../weights/{model_name}/best_val_loss.pth")

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

model = GPRegressionModel(train_x, train_y, likelihood).to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

likelihood.eval()
model.eval()

# Perform prediction on training data to get the covariance matrix
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    
    output = model(train_x)
    
    cov_matrix = output.covariance_matrix
    # cov_matrix -= (cov_matrix.min())
    
    for i in range(1400):
    
        cov_value = cov_matrix[...,i]
        cov_value_exclude_self = torch.cat((cov_value[:i], cov_value[i+1:]))
        
        # cov_value_exclude_self = (cov_value_exclude_self - cov_value_exclude_self.min()) / (cov_value_exclude_self.max() - cov_value_exclude_self.min())
        
        cov_value_exclude_self_normalized = cov_value_exclude_self / cov_value_exclude_self.sum()
                
        energy_exclude_self = torch.cat((train_y[:i], train_y[i+1:]))
        
        predicted_energy_from_cov = ( cov_value_exclude_self_normalized * energy_exclude_self ).sum()
        
        
        print(f"Energy Prediction Based on Cov Matrix: {predicted_energy_from_cov}, GT: {train_y[i]}, Difference: {np.abs(predicted_energy_from_cov.item() - train_y[i].item())}")
    