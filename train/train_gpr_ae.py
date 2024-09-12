import os
import sys
sys.path.append("/home/sait2024/Project/uncertainty")

from data.AtomDataset import AtomDataset
from model.GP import GPRegressionModel

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import gpytorch

from sklearn.metrics import mean_squared_error
import numpy as np



def evaluate(model, likelihood, data):
    model.eval()
    likelihood.eval()
    
    test_x, test_y = data

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    mean = observed_pred.mean.cpu()
    test_y = test_y.cpu().numpy()

    val_loss = np.sqrt(mean_squared_error(mean.numpy(), test_y))

    return val_loss


def train(alpha, file_path, model_name, device):
    
    try:
        os.mkdir(f"{file_path}weights/{model_name}")
    except FileExistsError:
        print("File already exists.")
        
    writer = SummaryWriter(f"{file_path}logs/{model_name}")

    # prepare train data
    dataset = AtomDataset(f"{file_path}data/Train_1400.xyz", device=device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
    train_y = target_energy.to(device)  # torch.Size([1400])
    train_x = torch.load("/home/sait2024/Project/uncertainty/data/train_1400_latent_vectors_64.pt")
    
    test_dataset = AtomDataset(f"{file_path}data/Val_100.xyz", device=device)
    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
    test_y = target_energy.to(device)  # torch.Size([1400])
    test_x = torch.load("/home/sait2024/Project/uncertainty/data/val_100_latent_vectors_64.pt")
    
    # setup model    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(alpha, train_x, train_y, likelihood).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # setup train settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = None
    # scheduler = StepLR(optimizer, step_size=700, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1000, factor=0.8)

    training_iterations = 100000
    best_val_loss = 100000000000000

    # start training loop
    for i in range(training_iterations):
        model.train()
        likelihood.train()

        optimizer.zero_grad()
        output = model(train_x)
                
        loss = -mll(output, train_y)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss) # ReduceLROnPlateau
        # scheduler.step()
        
        writer.add_scalar('Loss/train', loss.item(), i)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], i)
        
        if i % 10 == 0:
            print(f"Iteration {i + 1}/{training_iterations} - Loss: {loss.item()} - Learning Rate: {scheduler.get_last_lr()}")

        if i % 100 == 0 and i > 0:
            val_loss = evaluate(model, likelihood, (test_x, test_y))
            
            print(f"Best Val Loss: {best_val_loss}, Val loss: {val_loss}")
            
            writer.add_scalar('Loss/val', val_loss, i)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
                
    writer.close()  
            
    

if __name__ == "__main__":
    
    alpha = 100000
    
    file_path = "/home/sait2024/Project/uncertainty/"
    model_name = f"gpr_RBF_ae_96_lr_0.1_patience_1000"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    train(alpha, file_path, model_name, device)

    