import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.data import DataLoader
from collections import OrderedDict
import numpy as np
from ase.io import read
import pandas as pd


class AtomDataset(torch_geometric.data.Dataset):
    def __init__(self, data_path, device):
        self.atom_list = read(data_path, format="extxyz", index=":")
        self.device = device
        # self.force_mean = 5.379083224013658
        # self.force_standard = 3.2435461525378497
        super(AtomDataset, self).__init__()

    def len(self):
        return len(self.atom_list)

    def get(self, idx):
        atoms = self.atom_list[idx]
        atomic_numbers = atoms.get_atomic_numbers()  # (96,)
        lattice_vector = atoms.get_cell()

        lattice_vector = (
            torch.from_numpy(np.array(lattice_vector))
            .reshape(9)
            .to(self.device)
            .float()
        )  # (3)
        atomic_numbers = torch.from_numpy(atomic_numbers).to(self.device)
        positions = (
            torch.from_numpy(atoms.get_positions()).to(self.device).float()
        )  # (96, 3)

        target_energy = torch.tensor(
            atoms.get_potential_energy(), dtype=torch.float32
        ).to(self.device)
        target_forces = torch.tensor(atoms.get_forces(), dtype=torch.float32).to(
            self.device
        )
        
        '''
        산소 원자 반지름: 약 60 pm
        하프늄 원자 반지름: 약 159 pm
        
        산소 전기 음성도 (Pauling 척도): 3.44
        하프늄 전기 음성도: 1.3
        
        산소의 첫 번째 이온화 에너지: 13.618 eV
        하프늄의 첫 번째 이온화 에너지: 6.825 eV
        
        산소 원자 질량: 15.999 g/mol
        하프늄 원자 질량: 178.49 g/mol
        '''

        new_tensor = atomic_numbers.clone()  
        # atom radius
        new_tensor[atomic_numbers == 8] = torch.tensor(15.999) # 산소
        new_tensor[atomic_numbers == 72] = torch.tensor(178.49) # 하프늄
        
        # atom mass
        # new_tensor[atomic_numbers == 8] = torch.tensor(15.999) # 산소
        # new_tensor[atomic_numbers == 72] = torch.tensor(178.49) # 하프늄
        
        
        # return positions, atomic_numbers, lattice_vector, (target_energy, target_forces)
        return positions, new_tensor, lattice_vector, (target_energy, target_forces)
