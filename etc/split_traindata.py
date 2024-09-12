import sys
sys.path.append("/home/sait2024/Project/uncertainty")

from data.AtomDataset import AtomDataset
from torch_geometric.data import DataLoader
import torch

file_path = "/home/sait2024/Project/uncertainty/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# prepare train data
dataset = AtomDataset(f"{file_path}data/Train_1400.xyz", device=device)
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

positions, atomic_numbers, _, (target_energy, _) = next(iter(dataloader))
train_x = torch.cat((positions.view(batch_size, -1), atomic_numbers), dim=1).to(device)  # torch.Size([1400, 384])
train_y = target_energy.to(device)

end_idx = 200
train_x_subset = train_x[end_idx:,...]
train_y_subset = train_y[end_idx:,...]

torch.save(train_x_subset,f"../split_data/train_1200_x_{0}.pt")
torch.save(train_y_subset,f"../split_data/train_1200_y_{0}.pt")


for i in range(1,6):
    start_idx = i * 200
    end_idx = (i+1) * 200
    train_x_subset = torch.cat((train_x[0:start_idx,...], train_x[end_idx:,...]),dim=0)
    train_y_subset = torch.cat((train_y[0:start_idx,...], train_y[end_idx:,...]),dim=0)
    print(start_idx, end_idx)
    torch.save(train_x_subset,f"../split_data/train_1200_x_{i}.pt")
    torch.save(train_y_subset,f"../split_data/train_1200_y_{i}.pt")

start_idx = 6 * 200
end_idx = (6+1) * 200
train_x_subset = train_x[6:start_idx,...]
train_y_subset = train_y[6:start_idx,...]

torch.save(train_x_subset,f"../split_data/train_1200_x_{6}.pt")
torch.save(train_y_subset,f"../split_data/train_1200_y_{6}.pt")

# cmd + D : 같은 단어 선택 