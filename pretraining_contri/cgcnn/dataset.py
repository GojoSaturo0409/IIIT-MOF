import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pymatgen.core import Structure

def gaussian_distance(distances, centers, width):
    distances = np.asarray(distances)
    return np.exp(-((distances[..., None] - centers) ** 2) / (width ** 2))

class MOFCIFDataset(Dataset):
    def __init__(self, csv_file, cif_dir, target_col="working_capacity",
                 cutoff=8.0, max_num_nbr=12, num_gaussians=50):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.cif_dir = cif_dir
        self.target_col = target_col
        self.cutoff = cutoff
        self.max_num_nbr = max_num_nbr
        self.num_gaussians = num_gaussians

        self.centers = np.linspace(0, cutoff, num_gaussians)
        self.width = (cutoff / num_gaussians) if num_gaussians > 0 else 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cif_path = os.path.join(self.cif_dir, row["cif_file"])
        y = float(row[self.target_col])

        structure = Structure.from_file(cif_path)

        # Skip disordered structures; CGCNN assumes ordered crystals.
        if not structure.is_ordered:
            raise ValueError(f"Disordered structure found: {cif_path}")

        atom_z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
        n_atoms = len(structure)

        all_neighbors = structure.get_all_neighbors(self.cutoff)

        nbr_idx = np.zeros((n_atoms, self.max_num_nbr), dtype=np.int64)
        nbr_dist = np.zeros((n_atoms, self.max_num_nbr), dtype=np.float32)

        for i, nbrs in enumerate(all_neighbors):
            nbrs = sorted(nbrs, key=lambda x: x.nn_distance)

            if len(nbrs) == 0:
                # Fallback padding
                nbr_idx[i, :] = i
                nbr_dist[i, :] = self.cutoff
                continue

            if len(nbrs) < self.max_num_nbr:
                nbrs = nbrs + [nbrs[-1]] * (self.max_num_nbr - len(nbrs))
            else:
                nbrs = nbrs[:self.max_num_nbr]

            for j, nbr in enumerate(nbrs):
                nbr_idx[i, j] = nbr.index
                nbr_dist[i, j] = min(nbr.nn_distance, self.cutoff)

        nbr_fea = gaussian_distance(nbr_dist, self.centers, self.width)
        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)
        nbr_idx = torch.tensor(nbr_idx, dtype=torch.long)
        y = torch.tensor([y], dtype=torch.float32)

        return atom_z, nbr_fea, nbr_idx, y

def collate_pool(batch):
    atom_z_all = []
    nbr_fea_all = []
    nbr_idx_all = []
    crystal_atom_idx = []
    targets = []

    base = 0
    for crystal_id, (atom_z, nbr_fea, nbr_idx, y) in enumerate(batch):
        n_atoms = atom_z.shape[0]

        atom_z_all.append(atom_z)
        nbr_fea_all.append(nbr_fea)
        nbr_idx_all.append(nbr_idx + base)
        crystal_atom_idx.append(torch.full((n_atoms,), crystal_id, dtype=torch.long))
        targets.append(y)

        base += n_atoms

    atom_z_all = torch.cat(atom_z_all, dim=0)
    nbr_fea_all = torch.cat(nbr_fea_all, dim=0)
    nbr_idx_all = torch.cat(nbr_idx_all, dim=0)
    crystal_atom_idx = torch.cat(crystal_atom_idx, dim=0)
    targets = torch.cat(targets, dim=0)

    return atom_z_all, nbr_fea_all, nbr_idx_all, crystal_atom_idx, targets
