import sys
sys.path.append("/home/sait2024/Project/uncertainty")

from data.AtomDataset import AtomDataset
from model.GP import GPRegressionModel, StudentTProcessModel, ApproximateGPModel

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
import gpytorch
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.optim.lr_scheduler import StepLR 
import torch.nn.functional as F


def evaluate(test_x, test_y):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    mean = observed_pred.mean.cpu()
    variance = observed_pred.variance.cpu()
    lower, upper = observed_pred.confidence_region()

    val_loss = np.sqrt(mean_squared_error(mean.numpy(), test_y.cpu()))
    print(val_loss)
    
    return val_loss


file_path = "/home/sait2024/Project/uncertainty/data/"
model_name = "0826_use_model_last_feature"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data = torch.load(file_path+'dataset.pth')
data_tensors = data['data_tensors']
ground_truths = data['ground_truths']

import pdb; pdb.set_trace()

data_tensors_precessed = [t.reshape(1, -1) for t in data_tensors]
# data_tensors_precessed = [t[...,2304:].reshape(1, -1) for t in data_tensors]

data_tensors = torch.cat(data_tensors_precessed, dim=0) # torch.Size([1500, 233472])
data_tensors = F.adaptive_avg_pool1d(data_tensors, 76)
ground_truths = torch.tensor(ground_truths)

train_x = data_tensors[:1350].to(device)
valid_x = data_tensors[1350:].to(device)

train_y = ground_truths[:1350].to(device)
valid_y = ground_truths[1350:].to(device)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPRegressionModel(train_x, train_y, likelihood).to(device)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=2400, gamma=0.5)

training_iterations = 100000
best_val_loss = np.inf
for i in range(training_iterations):
    model.train()
    likelihood.train()

    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % 100 == 0:
        print(f"Iteration {i + 1}/{training_iterations} - Loss: {loss.item()} - Learning Rate: {scheduler.get_lr()}")
        torch.save(
        {
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"../weights/{model_name}_checkpoint.pth",
        )
        val_loss = evaluate(valid_x, valid_y)
        best_val_loss = val_loss
        if val_loss < best_val_loss:
            torch.save(
            {
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"../weights/{model_name}_best_val_loss.pth",
        )
        

# 모델 및 likelihood 불러오기
checkpoint = torch.load(f"weights/{model_name}_final_model.pth")
model = GPRegressionModel(train_x, train_y, likelihood).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
