import sys
sys.path.append("/home/jhbae/Project/uncertainty/")

import pandas as pd
import torch

from model.GP import GPRegressionModel
from data.AtomDataset import AtomDataset

import gpytorch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

model_names = [
    "gpr_PeriodicKernel_lr_0.1_patience_1000",
    "gpr_PeriodicKernel_lr_0.1_patience_1000_use_atom_mass",
]

file_path = "/home/jhbae/Project/uncertainty/dataset/"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Load Dataset
test_dataset = AtomDataset(f"{file_path}Test.xyz", device=device)
dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
test_x = torch.cat((positions.view(len(test_dataset), -1), atomic_numbers), dim=1) 

train_dataset = AtomDataset(f"{file_path}Train_1400.xyz", device=device)
dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
train_x = torch.cat((positions.view(len(train_dataset), -1), atomic_numbers), dim=1) 
train_y = target_energy 

pd.set_option("mode.chained_assignment", None)
df = pd.read_csv(f"{file_path}sample_submission.csv")

df["energy"] = torch.zeros(len(df), dtype=torch.float)
df["energy_uncertainty"] = torch.zeros(len(df), dtype=torch.float)
df["forces"] = torch.zeros(len(df), 96, 3).tolist()

N = len(model_names)

# Load Model
for model_name in model_names:
    checkpoint = torch.load(f"../weights/{model_name}/best_val_loss.pth")
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

    model = GPRegressionModel(100, train_x, train_y, likelihood).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    likelihood.eval()
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean.cpu()
        variance = observed_pred.variance.cpu()
        uncertainty = (variance-variance.min()) / (variance.max()-variance.min())

        df["energy"] = (torch.tensor(df["energy"]) + mean / N).numpy()
        df["energy_uncertainty"] = (torch.tensor(df["energy_uncertainty"]) + (uncertainty / N)).numpy()
        df["forces"] = torch.zeros(len(df), 96, 3).tolist()

    df.to_csv(f"../csv/{model_names}.csv", index=False)
