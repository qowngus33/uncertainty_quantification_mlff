from data.AtomDataset import AtomDataset
from model.GP import GPRegressionModel, StudentTProcessModel, ApproximateGPModel
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
import gpytorch
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


file_path = "/home/sait2024/Project/sait/data/"
device = torch.device("cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


dataset = AtomDataset(f"{file_path}Train_1400.xyz", device=device)
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(dataloader))
train_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1)  # torch.Size([16, 384])
train_y = target_energy  # torch.Size([16])

kernel = DotProduct() + WhiteKernel()
model = GaussianProcessRegressor(kernel=kernel)

model.fit(train_x, train_y)

eval_dataset = AtomDataset(f"{file_path}Val_100.xyz", device=device)
batch_size = len(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

positions, atomic_numbers, lattice_vector, (target_energy, target_forces) = next(iter(eval_dataloader))
test_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1)  # torch.Size([16, 384])

predictions = model.predict(test_x)
print(predictions)
