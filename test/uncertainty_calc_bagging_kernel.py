import sys
sys.path.append("/home/sait2024/Project/uncertainty")

import pandas as pd
import scipy.stats as stats
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel, SpectralDeltaKernel, RQKernel, PolynomialKernelGrad, PolynomialKernel, LinearKernel, PiecewisePolynomialKernel, ArcKernel, RBFKernelGrad

from model.GP import GPRegressionModel
from data.AtomDataset import AtomDataset

import gpytorch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# kernels = [RBFKernel(), MaternKernel(), RQKernel(), PeriodicKernel()]
kernels = [RBFKernel(), RQKernel()]
model_names = [
    "gpr_rbf_kernel_lr_0.1_patience_1000",
    # "gpr_MaternKernel_kernel_lr_0.1_patience_1000",
    "gpr_RQKernel_lr_0.1_patience_1000",
    # "gpr_PeriodicKernel_lr_0.1_patience_1000"
]

file_path = "/home/sait2024/Project/data/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

all_means = []
all_uncertainty = []

for i in range(len(model_names)):
    # Load Model
    checkpoint = torch.load(f"../weights/{model_names[i]}/best_val_loss.pth")

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

    model = GPRegressionModel(kernels[i], train_x, train_y, likelihood).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    likelihood.eval()
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean.cpu()
        variance = observed_pred.variance.cpu()
        
        uncertainty = (variance - variance.min()) / (variance.max() - variance.min())
        
        all_means.append(mean)
        all_uncertainty.append(uncertainty)

# Bagging
mean_ensemble = torch.mean(torch.stack(all_means), dim=0)
variance_ensemble = torch.mean(torch.stack(all_uncertainty), dim=0)

pd.set_option("mode.chained_assignment", None)
df = pd.read_csv(f"{file_path}sample_submission.csv")

df["energy"] = mean_ensemble
df["energy_uncertainty"] = variance_ensemble
df["forces"] = torch.zeros(len(df), 96, 3).tolist()

save_name = "model_ensemble_[RBFKernel(), RQKernel()]"
df.to_csv(f"../csv/{save_name}.csv", index=False)
