import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from data.AtomDataset import AtomDataset
from torch_geometric.loader import DataLoader
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm


file_path = "/home/sait2024/Project/uncertainty/data/"
device = torch.device("cpu")

dataset = AtomDataset(f"{file_path}Train_1400.xyz", device=device)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

positions, atomic_numbers, lattice_vectors, (target_energy, target_forces) = next(iter(dataloader))

def draw_lattice_box(origin, a1, a2, a3, ax):
    # 각 모서리 점을 정의
    vertices = np.array([
        origin,                 # 0 
        origin + a1,            # 1
        origin + a2,            # 2
        origin + a1 + a2,       # 3
        origin + a3,            # 4
        origin + a1 + a3,       # 5
        origin + a2 + a3,       # 6
        origin + a1 + a2 + a3   # 7
    ])
    
    # 각 면을 정의
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]], 
        [vertices[0], vertices[2], vertices[6], vertices[4]],  
        [vertices[0], vertices[1], vertices[3], vertices[2]],
        
        [vertices[4], vertices[5], vertices[7], vertices[6]],  
        [vertices[2], vertices[3], vertices[7], vertices[6]],  
        [vertices[1], vertices[3], vertices[7], vertices[5]],  
    ]
    
    # 면 그리기
    poly3d = Poly3DCollection(faces, alpha=0.2, linewidths=1, edgecolors='k', facecolors='gray')
    ax.add_collection3d(poly3d)

for i in tqdm(range(len(dataset))):
    # 격자 벡터
    a1 = np.array(lattice_vectors[i, ...][0:3])
    a2 = np.array(lattice_vectors[i, ...][3:6])
    a3 = np.array(lattice_vectors[i, ...][6:9])
        
    atom_positions = positions[i,...]
    force_vectors = target_forces[i,...]
    atom_types = atomic_numbers[i,...]
    energy = target_energy[i].tolist()

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(221, projection='3d')

    # Hf 원자 위치 플롯 (파란색)
    hf_positions = atom_positions[atom_types == 72]
    ax.scatter(hf_positions[:, 0], hf_positions[:, 1], hf_positions[:, 2], c='b', marker='o', s=50, label='Hf')

    # O 원자 위치 플롯 (빨간색)
    o_positions = atom_positions[atom_types == 8]
    ax.scatter(o_positions[:, 0], o_positions[:, 1], o_positions[:, 2], c='r', marker='o', s=50, label='O')

    # 힘 벡터 플롯
    ax.quiver(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2],
            force_vectors[:, 0], force_vectors[:, 1], force_vectors[:, 2],
            color='g', length=1.0, normalize=True)
    
    # 격자 벡터 플롯 (각 축에 대해 시각화)
    origin = np.zeros(3)
    draw_lattice_box(origin, a1, a2, a3, ax)

    # 축 라벨링 및 범례
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'System Energy {round(energy,3)}')
    ax.legend()
    
    # x-y 평면 투영
    ax2 = fig.add_subplot(222)
    ax2.scatter(atom_positions[:, 0], atom_positions[:, 1], c=atom_types, marker='o', s=50, label='Atoms')
    ax2.quiver(atom_positions[:, 0], atom_positions[:, 1],
                force_vectors[:, 0], force_vectors[:, 1],
                color='g', scale=5, width=0.004, scale_units='xy', angles='xy')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Projection onto XY Plane')
    ax2.legend()

    # x-z 평면 투영
    ax3 = fig.add_subplot(223)
    ax3.scatter(atom_positions[:, 0], atom_positions[:, 2], c=atom_types, marker='o', s=50, label='Atoms')
    ax3.quiver(atom_positions[:, 0], atom_positions[:, 2],
                force_vectors[:, 0], force_vectors[:, 2],
                color='g', scale=5, width=0.004, scale_units='xy', angles='xy')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Projection onto XZ Plane')
    ax3.legend()

    # y-z 평면 투영
    ax4 = fig.add_subplot(224)
    ax4.scatter(atom_positions[:, 1], atom_positions[:, 2], c=atom_types, marker='o', s=50, label='Atoms')
    ax4.quiver(atom_positions[:, 1], atom_positions[:, 2],
            force_vectors[:, 1], force_vectors[:, 2],
            color='g', scale=5, scale_units='xy', width=0.004, angles='xy')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('Projection onto YZ Plane')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"visualize_molecule_2D/molecule_{i}.png")
