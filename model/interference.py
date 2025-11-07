import numpy as np
import copy
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GCNWithAttentionOneHead(nn.Module):
    def __init__(self, input_dim, hidden_dim,b):
        super(GCNWithAttentionOneHead, self).__init__()
        # Two-layer MLP for the attention mechanism
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Learnable scalar parameter b
        self.b = nn.Parameter(torch.tensor(b))  # e.g. start at 0.1

    def forward(self, x, nbrs_idx, t, e_hat):
        device = x.device
        n = x.size(0)
        Y_pred = torch.zeros(len(nbrs_idx), device=device)
        pairwise_w_ij = torch.zeros(n, n, device=device)

        for i, neighbors in enumerate(nbrs_idx):
            # 'neighbors' is a list/tensor of neighbor indices (including the node itself)
            current = int(neighbors[0].item() if torch.is_tensor(neighbors[0]) else neighbors[0])
            chosen = neighbors  # Use all neighbors

            # Build the MLP input by concatenating node i's features with its neighbors'
            z_i = x[current]
            z_concat = torch.cat([z_i.unsqueeze(0).expand(len(chosen), -1), x[chosen]], dim=-1)

            mlp_outputs = self.attention_mlp(z_concat).squeeze(dim=1)
            scores = torch.softmax(self.b * abs(mlp_outputs), dim=0)
            prop_res = t[chosen] - e_hat[chosen]
            pairwise = mlp_outputs * scores

            pairwise_w_ij[current, chosen] = pairwise
            Y_pred[i] = torch.sum(prop_res * pairwise)

        return Y_pred, pairwise_w_ij
    def predict(self, x, nbrs_idx, t):
      device = x.device
      n = x.size(0)
      # We'll store pairwise weights for all i,j in the neighborhood
      pairwise_w_ij = torch.zeros(n, n, device=device)
      pairwise_w_ij_raw = torch.zeros(n, n, device=device)
      # t is expected to be a tensor on the right device
      for i, neighbors in enumerate(nbrs_idx):
        current = int(neighbors[0].item() if torch.is_tensor(neighbors[0]) else neighbors[0])
        # Use ALL neighbors (no sampling)
        num_neighbors = len(neighbors)

        # Build MLP input for each neighbor j
        z_i = x[current]                               # shape (d,)
        z_i_repeated = z_i.unsqueeze(0).repeat(num_neighbors, 1)  # (num_neighbors, d)
        z_concat = torch.cat([z_i_repeated, x[neighbors]], dim=-1) # (num_neighbors, 2d)

        # MLP output
        mlp_outputs = self.attention_mlp(z_concat).squeeze(dim=1)  # shape (num_neighbors,)
        scores = torch.softmax(self.b*abs(mlp_outputs), dim=0)
        # If in your "forward" logic you used `mlp_outputs / num_neighbors`,
        # replicate that here so the definition of w_ij is consistent
        pairwise = mlp_outputs * scores
        # pairwise = mlp_outputs
        # Fill in the matrix
        pairwise_w_ij[current, neighbors] = pairwise
        pairwise_w_ij_raw[current, neighbors]=mlp_outputs
      return pairwise_w_ij.cpu().numpy(),pairwise_w_ij_raw.cpu().numpy()

class TensorDataset(Dataset):
    def __init__(self, Y):
        self.Y=Y
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        return self.Y[idx],idx

# (Legacy single-head and sign variants removed; kept TwoHead and OneHead for clarity.)



class GCNWithAttentionTwoHead(nn.Module):
    """
    Two-head variant:
      - Neighbor head f(X_i, X_j): excludes self from softmax normalization.
      - Self head g(X_i): produces a scalar weight for the self treatment term.

    Prediction for node i:
      Y_i = g(X_i) * (t_i - e_hat_i) + sum_{j in N(i)} f(X_i, X_j) * (t_j - e_hat_j)
    """
    def __init__(self, input_dim, hidden_dim, b):
        super(GCNWithAttentionTwoHead, self).__init__()
        # Neighbor attention MLP (over pairs [X_i; X_j])
        self.neigh_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Self treatment MLP (over X_i)
        self.self_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Temperature for neighbor softmax
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float32))

    def forward(self, x, nbrs_idx, t, e_hat):
        device = x.device
        n = x.size(0)
        Y_pred = torch.zeros(len(nbrs_idx), device=device)
        pairwise_w_ij = torch.zeros(n, n, device=device)
        self_w_i = torch.zeros(n, device=device)

        for i, neighbors in enumerate(nbrs_idx):
            if len(neighbors) < 1:
                continue
            current = int(neighbors[0].item() if torch.is_tensor(neighbors[0]) else neighbors[0])
            z_i = x[current]

            # Self contribution
            g_i = self.self_mlp(z_i).squeeze()
            self_w_i[current] = g_i
            self_contrib = g_i * (t[current] - e_hat[current])

            # Neighbor contribution (exclude self)
            neigh_contrib = 0.0
            if len(neighbors) > 1:
                chosen = neighbors[1:]
                z_concat = torch.cat([z_i.unsqueeze(0).expand(len(chosen), -1), x[chosen]], dim=-1)
                mlp_neigh = self.neigh_mlp(z_concat).squeeze(dim=1)
                scores = torch.softmax(self.b * abs(mlp_neigh), dim=0)
                prop_res = t[chosen] - e_hat[chosen]
                neigh_contrib = torch.sum(prop_res * mlp_neigh * scores)
                pairwise_w_ij[current, chosen] = mlp_neigh * scores

            # zero diagonal since neighbor head excludes self
            pairwise_w_ij[current, current] = 0.0
            Y_pred[i] = self_contrib + neigh_contrib

        return Y_pred, pairwise_w_ij, self_w_i

    def predict(self, x, nbrs_idx, t):
        device = x.device
        n = x.size(0)
        pairwise_w_ij = torch.zeros(n, n, device=device)
        pairwise_w_ij_raw = torch.zeros(n, n, device=device)
        self_w_i = torch.zeros(n, device=device)

        for i, neighbors in enumerate(nbrs_idx):
            if len(neighbors) < 1:
                continue
            current = int(neighbors[0].item() if torch.is_tensor(neighbors[0]) else neighbors[0])
            z_i = x[current]
            # Self scalar
            self_w_i[current] = self.self_mlp(z_i).squeeze()
            # Neighbor weights (exclude self)
            if len(neighbors) > 1:
                chosen = neighbors[1:]
                z_concat = torch.cat([z_i.unsqueeze(0).expand(len(chosen), -1), x[chosen]], dim=-1)
                mlp_neigh = self.neigh_mlp(z_concat).squeeze(dim=1)
                scores = torch.softmax(self.b * abs(mlp_neigh), dim=0)
                pairwise = mlp_neigh * scores
                pairwise_w_ij[current, chosen] = pairwise
                pairwise_w_ij_raw[current, chosen] = mlp_neigh

        return pairwise_w_ij.cpu().numpy(), pairwise_w_ij_raw.cpu().numpy(), self_w_i.cpu().numpy()

