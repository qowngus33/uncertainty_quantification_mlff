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


model_name = "gpr_RBF_ae_96_lr_0.1_patience_1000"
file_path = "/home/sait2024/Project/"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = AtomDataset(f"{file_path}data/Train_1400.xyz", device=device)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
train_y = target_energy.to(device)  # torch.Size([1400])
train_x = torch.load("/home/sait2024/Project/uncertainty/data/train_1400_latent_vectors_64.pt").to(device)
    
test_dataset = AtomDataset(f"{file_path}data/Test.xyz", device=device)
dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
test_y = target_energy.to(device)  # torch.Size([1400])
test_x = torch.load("/home/sait2024/Project/uncertainty/data/test_latent_vectors_64.pt").to(device)

# Load Model
checkpoint = torch.load(f"../weights/{model_name}/best_val_loss.pth")

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

model = GPRegressionModel(100, train_x, train_y, likelihood).to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

likelihood.eval()
model.eval()

pd.set_option("mode.chained_assignment", None)
df = pd.read_csv(f"{file_path}data/sample_submission.csv")

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean.cpu()
    variance = observed_pred.variance.cpu()
    
    uncertainty = (variance - variance.min()) / (variance.max() - variance.min())

    df["energy"] = mean
    df["energy_uncertainty"] = uncertainty
    df["forces"] = torch.zeros(len(df), 96, 3).tolist()

df.to_csv(f"../csv/{model_name}.csv", index=False)
