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
import torch.nn.functional as F


ood_ratio = 0.1
file_path = "/home/sait2024/Project/uncertainty/data/"

model_name = "0826_use_model_last_feature_best_val_loss.pth"
checkpoint = torch.load(f"../weights/{model_name}")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Load Train, Test
train_data = torch.load(file_path+'dataset.pth')
train_data_tensors, train_ground_truths = train_data['data_tensors'], train_data['ground_truths']
train_data_tensors = torch.cat([t.reshape(1, -1) for t in train_data_tensors], dim=0) # torch.Size([1500, 233472])
train_data_tensors = F.adaptive_avg_pool1d(train_data_tensors, 76)
train_ground_truths = torch.tensor(train_ground_truths)
train_x, train_y = train_data_tensors[:1350].to(device), train_ground_truths[:1350].to(device)

test_data = torch.load(file_path+'test_dataset.pth')
test_data_tensors, test_ground_truths = test_data['data_tensors'], test_data['ground_truths']
test_data_tensors = torch.cat([t.reshape(1, -1) for t in test_data_tensors], dim=0) # torch.Size([3000, 233472])
test_data_tensors = F.adaptive_avg_pool1d(test_data_tensors, 76)
test_ground_truths = torch.tensor(test_ground_truths)
test_x, test_y = test_data_tensors.to(device), test_ground_truths.to(device)


# Load Model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

model = GPRegressionModel(train_x, train_y, likelihood).to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

likelihood.eval()
model.eval()

pd.set_option("mode.chained_assignment", None)
df = pd.read_csv(f"{file_path}sample_submission.csv")

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean.cpu()
    variance = observed_pred.variance.cpu()
    lower, upper = observed_pred.confidence_region()
    target_energy = test_y.cpu().numpy()

    df["energy"] = mean
    df["energy_uncertainty"] = variance
    df["forces"] = torch.zeros(len(df), 96, 3).tolist()

uncertainty = df["energy_uncertainty"]

energy_uncertainty_mean = uncertainty.mean()
energy_uncertainty_std = np.sqrt(uncertainty.var())

min_energy_uncertainty = uncertainty.min()
max_energy_uncertainty = uncertainty.max()


lower = stats.norm.ppf(
    ood_ratio, loc=energy_uncertainty_mean, scale=energy_uncertainty_std
)
upper = stats.norm.ppf(
    1 - ood_ratio, loc=energy_uncertainty_mean, scale=energy_uncertainty_std
)

print(f"Min Energy Uncertainty: {min_energy_uncertainty}")
print(f"Max Energy Uncertainty: {max_energy_uncertainty}")

print(f"Energy Uncertainty Average: {energy_uncertainty_mean}")
print(f"Energy Uncertainty Std: {energy_uncertainty_std}")

print(f"lower bound: {lower}")
print(f"upper bound: {upper}")


sigmoid = nn.Sigmoid()

for i in tqdm(range(len(df["energy_uncertainty"]))):
    if df["energy_uncertainty"][i] < lower or df["energy_uncertainty"][i] > upper:
        # df['energy_uncertainty'][i] = 0
        # df["energy"][i] = -917.0
        ...
    # else:
    #     df['energy_uncertainty'][i] = 1
    df["energy_uncertainty"][i] = (
        df["energy_uncertainty"][i] - min_energy_uncertainty
    ) / (max_energy_uncertainty - min_energy_uncertainty)
    
    # normalized_uncertainty = torch.tensor(
    #     ( df["energy_uncertainty"][i] - energy_uncertainty_mean ) / energy_uncertainty_std
    # )
    # df["energy_uncertainty"][i] = sigmoid(normalized_uncertainty).item()
    

# df['energy_uncertainty'] = df['energy_uncertainty'].astype(int)
df.to_csv(f"../csv/{model_name}.csv", index=False)
