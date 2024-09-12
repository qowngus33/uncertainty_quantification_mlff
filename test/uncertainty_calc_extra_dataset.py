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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


batch_size = 3000
model_name = "gpr_RQ_alpha_100_lr_0.1_patience_1000"

file_path = "/home/sait2024/Project/uncertainty/data/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load test dataset
test_dataset = AtomDataset(f"{file_path}Testset.xyz", device=device)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(test_dataloader))
test_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1) 
print("Loaded Test Dataset ... ")

# load OOD dataset (for model initialization)
ood_dataset = AtomDataset(f"{file_path}OOD.xyz", device=device)
dataloader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)
positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
ood_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1) 
print("Loaded OOD Dataset ... ")

# load train dataset (for model initialization)
dataset = AtomDataset(f"{file_path}Train_1400.xyz", device=device)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
train_x = torch.cat((positions.view(len(dataset), -1), atomic_numbers), dim=1)  # torch.Size([16, 384])
train_y = target_energy 
print("Loaded Train Dataset ... ")

# Load Model
checkpoint = torch.load(f"../weights/{model_name}/best_val_loss.pth")
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
model = GPRegressionModel(100, train_x, train_y, likelihood).to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
print("Loaded Model State Dict ... ")

likelihood.eval()
model.eval()

df = pd.read_csv(f"{file_path}sample_submission.csv")
df = pd.concat([df, df], ignore_index=True)

test_x = torch.concat([test_x, ood_x])
gt = np.array([0] * 3000 + [1] * 3000)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean.cpu()
    variance = observed_pred.variance.cpu()
    uncertainty = (variance-variance.min()) / (variance.max()-variance.min())

    df["energy"] = mean
    df["energy_uncertainty"] = uncertainty
    df["forces"] = torch.zeros(len(df), 96, 3).tolist()

    fpr, tpr, thresholds = roc_curve(gt, uncertainty)

df.to_csv(f"../csv/extra_dataset.csv", index=False)

# AUC 계산
roc_auc = auc(fpr, tpr)

# ROC 커브 시각화
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig("aucroc.png")
