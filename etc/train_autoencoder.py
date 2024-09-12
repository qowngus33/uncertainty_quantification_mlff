import sys
sys.path.append("/home/sait2024/Project/uncertainty")

from data.AtomDataset import AtomDataset

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder385(nn.Module):
    def __init__(self):
        super(Autoencoder385, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Latent vector of size 96
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 384),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# prepare train data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_path = "/home/sait2024/Project/uncertainty/"
dataset = AtomDataset(f"{file_path}data/Train_1400.xyz", device=device)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
# x_train = positions.view(len(dataset), -1).to(device)
x_train = torch.cat((positions.view(len(dataset), -1), atomic_numbers), dim=1) 
y_train = target_energy.to(device)  # torch.Size([1400])
train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=len(dataset), shuffle=True)

# prepare valid data
valid_dataset = AtomDataset(f"{file_path}data/Val_100.xyz", device=device)
valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
positions, atomic_numbers, _, (target_energy, _) = next(iter(valid_dataloader))
# valid_x = positions.view(len(eval_dataset), -1).to(device)
valid_x = torch.cat((positions.view(len(valid_dataset), -1), atomic_numbers), dim=1) 
valid_loader = DataLoader(TensorDataset(valid_x, valid_x), batch_size=len(valid_dataset), shuffle=True)

# prepare test data
test_dataset = AtomDataset(f"{file_path}data/Test.xyz", device=device)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
positions, atomic_numbers, _, (target_energy, _) = next(iter(test_dataloader))
# test_x = positions.view(len(eval_dataset), -1).to(device)
test_x = torch.cat((positions.view(len(test_dataset), -1), atomic_numbers), dim=1) 
test_loader = DataLoader(TensorDataset(test_x, test_x), batch_size=len(test_dataset), shuffle=True)

# initialize model
model = Autoencoder385().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)


# Train model
num_epochs = 30000
for epoch in range(num_epochs):
    
    model.train()
    
    train_loss = 0
    
    for data in train_loader:
        inputs, _ = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    model.eval()
    
    if epoch % 100 == 0:
        with torch.no_grad():
            val_loss = 0
            for data in test_loader:
                inputs, _ = data
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
            val_loss /= len(train_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, lr: {scheduler.get_last_lr()}")
    
    
# test_loader = DataLoader(TensorDataset(x_test, x_test), batch_size=64, shuffle=False)

model.eval()
all_latent_vectors = []
with torch.no_grad():
    for data in train_loader:
        inputs, _ = data
        latent_vectors = model.encoder(inputs)
        all_latent_vectors.append(latent_vectors)

all_latent_vectors = torch.cat(all_latent_vectors, dim=0)
torch.save(all_latent_vectors, 'train_1400_latent_vectors_64.pt')
print("Latent vectors saved to 'train_1400_latent_vectors_64.pt'")

all_latent_vectors = []
with torch.no_grad():
    for data in valid_loader:
        inputs, _ = data
        latent_vectors = model.encoder(inputs)
        all_latent_vectors.append(latent_vectors)

all_latent_vectors = torch.cat(all_latent_vectors, dim=0)
torch.save(all_latent_vectors, 'val_100_latent_vectors_64.pt')
print("Latent vectors saved to 'val_100_latent_vectors_64.pt'")

all_latent_vectors = []
with torch.no_grad():
    for data in test_loader:
        inputs, _ = data
        latent_vectors = model.encoder(inputs)
        all_latent_vectors.append(latent_vectors)

all_latent_vectors = torch.cat(all_latent_vectors, dim=0)
torch.save(all_latent_vectors, 'test_latent_vectors_64.pt')
print("Latent vectors saved to 'test_latent_vectors_64.pt'")