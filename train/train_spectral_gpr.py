import os
import sys
sys.path.append("/home/sait2024/Project/uncertainty")

from data.AtomDataset import AtomDataset
from model.spectral_delta_gp import SpectralDeltaGP
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import gpytorch

from sklearn.metrics import mean_squared_error
import numpy as np



def evaluate(model, likelihood, file_path, device):
    model.eval()
    likelihood.eval()

    eval_dataset = AtomDataset(f"{file_path}data/Val_100.xyz", device=device)
    batch_size = len(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    positions, atomic_numbers, _, (target_energy, _) = next(
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

    return val_loss


def train(file_path, model_name, device):
    
    try:
        os.mkdir(f"{file_path}weights/{model_name}")
    except FileExistsError:
        print("File already exists.")
        
    writer = SummaryWriter(f"{file_path}logs/{model_name}")

    # prepare train data
    dataset = AtomDataset(f"{file_path}data/Train_1400.xyz", device=device)
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
    train_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1).to(device)  # torch.Size([1400, 384])
    train_y = target_energy.to(device)  # torch.Size([1400])
    
    # setup model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SpectralDeltaGP(train_x, train_y, num_deltas=1500).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # setup train settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = None
    # scheduler = StepLR(optimizer, step_size=700, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1000, factor=0.8)

    training_iterations = 100000
    best_val_loss = 100000000000000
    
    # train_x = train_x.unsqueeze(0)

    # start training loop
    for i in range(training_iterations):
        model.train()
        model.likelihood.train()

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
            val_loss = evaluate(model, model.likelihood, file_path, device)
            
            print(f"Best Val Loss: {best_val_loss}, Val loss: {val_loss}")
            
            writer.add_scalar('Loss/val', val_loss, i)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "likelihood_state_dict": model.likelihood.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                f"../weights/{model_name}/best_val_loss.pth",
            )
                
    writer.close()  
            
    

if __name__ == "__main__":
    
    file_path = "/home/sait2024/Project/uncertainty/"
    model_name = "gpr_spectralgpr_lr_0.1_patience_1000"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    train(file_path, model_name, device)

    