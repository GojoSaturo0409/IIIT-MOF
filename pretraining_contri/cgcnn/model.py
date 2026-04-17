import torch
import torch.nn as nn
import torch.nn.functional as F

class CGCNNConv(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.bn = nn.BatchNorm1d(atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_idx):
        """
        atom_fea: [N_atoms, atom_fea_len]
        nbr_fea : [N_atoms, M, nbr_fea_len]
        nbr_idx : [N_atoms, M]
        """
        N, M = nbr_idx.shape
        nbr_atom_fea = atom_fea[nbr_idx]                      # [N, M, atom_fea_len]
        self_fea = atom_fea.unsqueeze(1).expand(-1, M, -1)    # [N, M, atom_fea_len]

        total_fea = torch.cat([self_fea, nbr_atom_fea, nbr_fea], dim=2)
        total_fea = self.fc_full(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=2)
        nbr_msg = torch.sigmoid(filter_fea) * F.softplus(core_fea)
        nbr_msg = nbr_msg.sum(dim=1)

        out = self.bn(atom_fea + nbr_msg)
        return F.softplus(out)

class CGCNNRegressor(nn.Module):
    def __init__(self,
                 atom_fea_len=64,
                 nbr_fea_len=50,
                 n_conv=3,
                 h_fea_len=128,
                 out_dim=1,
                 dropout=0.1,
                 max_z=100):
        super().__init__()
        self.atom_emb = nn.Embedding(max_z + 1, atom_fea_len, padding_idx=0)
        self.convs = nn.ModuleList([
            CGCNNConv(atom_fea_len, nbr_fea_len) for _ in range(n_conv)
        ])

        self.fc1 = nn.Linear(atom_fea_len, h_fea_len)
        self.fc2 = nn.Linear(h_fea_len, h_fea_len // 2)
        self.out = nn.Linear(h_fea_len // 2, out_dim)
        self.dropout = dropout

    def pool_mean(self, atom_fea, crystal_atom_idx):
        n_crystals = int(crystal_atom_idx.max().item()) + 1
        device = atom_fea.device
        feat_dim = atom_fea.size(1)

        pooled = torch.zeros(n_crystals, feat_dim, device=device)
        pooled.index_add_(0, crystal_atom_idx, atom_fea)

        counts = torch.bincount(crystal_atom_idx, minlength=n_crystals).float().to(device)
        counts = counts.unsqueeze(1).clamp(min=1.0)

        return pooled / counts

    def forward(self, atom_z, nbr_fea, nbr_idx, crystal_atom_idx):
        atom_fea = self.atom_emb(atom_z)

        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)

        crystal_fea = self.pool_mean(atom_fea, crystal_atom_idx)

        x = F.softplus(self.fc1(crystal_fea))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.softplus(self.fc2(x))
        x = self.out(x)
        return x
