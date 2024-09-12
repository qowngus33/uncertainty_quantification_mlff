import sys
sys.path.append("/home/sait2024/Project/uncertainty")

from data.AtomDataset import AtomDataset
from model.spectral_mixture_gpr import SpectralMixtureGPModel

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

import gpytorch
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.optim.lr_scheduler import StepLR 
import os
import math


def evaluate(model, likelihood, file_path, device):
    model.eval()
    likelihood.eval()

    eval_dataset = AtomDataset(f"{file_path}data/Validset.xyz", device=device)
    batch_size = len(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(
        iter(eval_dataloader)
    )
    test_x = torch.cat(
        (positions.view(batch_size, -1), atomic_numbers), dim=1
    )  # torch.Size([16, 384])

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    mean = observed_pred.mean.cpu()
    target_energy = target_energy.cpu().numpy()

    val_loss = np.sqrt(mean_squared_error(mean.numpy(), target_energy))
    print(val_loss)

    return val_loss


def train(file_path, model_name, device):
    
    try:
        os.mkdir(f"{file_path}weights/{model_name}")
    except FileExistsError:
        print("File already exists.")

    # prepare train data
    dataset = AtomDataset(f"{file_path}data/Train_1400.xyz", device=device)
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
    train_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1).unsqueeze(0).to(device)  # torch.Size([1400, 384])
    train_y = target_energy.to(device)  # torch.Size([1400])
    
    
    # train_x = torch.linspace(0, 1, 15)
    # train_y = torch.sin(train_x * (2 * math.pi))
    
    # import pdb; pdb.set_trace()

    # setup model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SpectralMixtureGPModel(train_x, train_y, likelihood, device).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # setup train settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=2400, gamma=0.5)

    training_iterations = 100000
    best_val_loss = np.inf

    # start training loop
    for i in range(training_iterations):
        model.train()
        likelihood.train()

        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        
        import pdb; pdb.set_trace()

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 10 == 0:
            print(f"Iteration {i + 1}/{training_iterations} - Loss: {loss.item()} - Learning Rate: {scheduler.get_lr()}")

        if i % 100 == 0 and i > 0:
            torch.save(
            {
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"../weights/{model_name}/model_checkpoint_{i}.pth",
            )
            val_loss = evaluate(model, likelihood, file_path, device)
            if val_loss < best_val_loss:
                torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "likelihood_state_dict": likelihood.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                f"../weights/{model_name}/best_val_loss.pth",
            )
            
    

if __name__ == "__main__":
    
    file_path = "/home/sait2024/Project/uncertainty/"
    model_name = "sm_gpr_step_size_2400"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    train(file_path, model_name, device)
