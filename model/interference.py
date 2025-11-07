import numpy as np
import copy
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GCNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim,b):
        super(GCNWithAttention, self).__init__()
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

def run_gcn_experiment(
    X, mu_hat, y, nbrs_idx, t, e_hat,
    true_attention, A, attn_temperature, hyperparams,
    printing=True, device="cpu",patience=12
):
    y_resid = torch.tensor(y, dtype=torch.float32, device=device) \
                 - torch.tensor(mu_hat, dtype=torch.float32, device=device)
    x = torch.tensor(X, dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    e_hat = torch.tensor(e_hat, dtype=torch.float32, device=device)

    # Dimensions
    input_dim = x.shape[1]
    n = x.shape[0]

    # Instantiate model
    model = GCNWithAttention(
        input_dim,
        hyperparams["hidden_dim"],
        b=float(attn_temperature)
    ).to(device)

    loss_fn = nn.MSELoss()

    # Create DataLoader
    dataset = TensorDataset(y_resid)
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    start_time = time.time()
    best_spillover_loss = float('inf')
    best_predicted_attention = None
    best_model = None

    # Early stopping parameters
    no_improvement_count = 0
    patience = patience  # Stop if no improvement for 5 epochs in a row

    for epoch in range(hyperparams["epochs"]):
        model.train()
        train_loss = 0
        pairwise_w_ij_train = torch.zeros(n, n, device=device)
        Y_pred_all = torch.zeros(n, device=device)

        # Training step
        for batch_Y, idx in dataloader:
            batch_Y = batch_Y.to(device)
            idx_cpu = idx.tolist()
            batch_one_hop_neighbor = [nbrs_idx[i] for i in idx_cpu]

            optimizer.zero_grad()
            Y_pred, batch_pairwise_w_ij = model(x, batch_one_hop_neighbor, t, e_hat)
            loss = loss_fn(Y_pred, batch_Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            Y_pred_all[idx] = Y_pred.detach()
            pairwise_w_ij_train += batch_pairwise_w_ij.detach()

        # Evaluate spillover loss on the full dataset
        model.eval()
        with torch.no_grad():
            predicted_attention, _ = model.predict(x, nbrs_idx, t)
        spillover_loss = np.sum((predicted_attention - true_attention) ** 2)

        # Check improvement
        if spillover_loss < best_spillover_loss:
            best_spillover_loss = spillover_loss
            best_predicted_attention = predicted_attention
            best_model = copy.deepcopy(model)
            no_improvement_count = 0  # reset counter
            if printing:
                print(f"Epoch {epoch+1:03d}, New best Spillover diff: {spillover_loss:.4f}")
        else:
            no_improvement_count += 1
            if printing:
                print(f"No improve. Epoch {epoch+1:03d}, Spillover diff: {spillover_loss:.4f}")

            # Early stopping condition
            if no_improvement_count >= patience:
                if printing:
                    print(f"Stopping early at epoch {epoch+1} due to no improvement "
                          f"for {patience} consecutive epochs.")
                break

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return best_model, best_spillover_loss, best_predicted_attention

class GCNWithAttention_sign(nn.Module):
    def __init__(self, input_dim, hidden_dim, b):
        super(GCNWithAttention_sign, self).__init__()
        # Two-layer MLP for the attention mechanism.
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Learnable scalar parameter b.
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x, nbrs_idx, t, e_hat):
        device = x.device
        n = x.size(0)
        Y_pred = torch.zeros(len(nbrs_idx), device=device)
        pairwise_w_ij = torch.zeros(n, n, device=device)

        for i, neighbors in enumerate(nbrs_idx):
            # Ensure there is at least one index (self index should be present).
            if len(neighbors) < 1:
                continue

            # Get the current node index (the first element is the self index).
            current = int(neighbors[0].item() if torch.is_tensor(neighbors[0]) else neighbors[0])
            z_i = x[current]

            # --- Neighbor Contribution Only ---
            neighbor_contrib = 0.0
            if len(neighbors) > 1:
                # Exclude self index; use only neighbors j ≠ i.
                chosen = neighbors[1:]
                # Build MLP input for each neighbor j: concat [X_i; X_j].
                z_concat = torch.cat(
                    [z_i.unsqueeze(0).expand(len(chosen), -1), x[chosen]], dim=-1
                )
                mlp_neigh = self.attention_mlp(z_concat).squeeze(dim=1)  # shape: (num_neighbors,)
                # Compute softmax over scaled MLP outputs.
                scores = torch.softmax(self.b * abs(mlp_neigh), dim=0)
                # Compute treatment difference for neighbors.
                prop_res = t[chosen] - e_hat[chosen]
                # Aggregate neighbor contribution.
                neighbor_contrib = torch.sum(prop_res * mlp_neigh * scores)
                # Record the pairwise weight (weighted MLP output).
                pairwise_w_ij[current, chosen] = mlp_neigh * scores

            # Set self weight to 0 (we do not consider self-term).
            pairwise_w_ij[current, current] = 0.0
            Y_pred[i] = neighbor_contrib

        return Y_pred, pairwise_w_ij

    def predict(self, x, nbrs_idx, t):
        device = x.device
        n = x.size(0)
        pairwise_w_ij = torch.zeros(n, n, device=device)
        pairwise_w_ij_raw = torch.zeros(n, n, device=device)

        # Convert treatment and ehat to tensors.
        # t is expected to be a tensor on the right device
        for i, neighbors in enumerate(nbrs_idx):
            if len(neighbors) < 1:
                continue

            current = int(neighbors[0].item() if torch.is_tensor(neighbors[0]) else neighbors[0])
            z_i = x[current]
            if len(neighbors) > 1:
                chosen = neighbors[1:]
                z_concat = torch.cat(
                    [z_i.unsqueeze(0).expand(len(chosen), -1), x[chosen]], dim=-1
                )
                mlp_neigh = self.attention_mlp(z_concat).squeeze(dim=1)
                scores = torch.softmax(self.b * abs(mlp_neigh), dim=0)
                pairwise = mlp_neigh * scores
                pairwise_w_ij[current, chosen] = pairwise
                pairwise_w_ij_raw[current, chosen] = mlp_neigh

        return pairwise_w_ij.cpu().numpy(), pairwise_w_ij_raw.cpu().numpy()

def run_gcn_experiment_sign(
    X, mu_hat, y, nbrs_idx, t, e_hat,
    true_attention, A, attn_temperature,hyperparams,
    printing=True, device="cpu",patience=12
):
    y_resid = torch.tensor(y, dtype=torch.float32, device=device) \
                 - torch.tensor(mu_hat, dtype=torch.float32, device=device)
    x = torch.tensor(X, dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    e_hat = torch.tensor(e_hat, dtype=torch.float32, device=device)

    # Dimensions
    input_dim = x.shape[1]
    n = x.shape[0]

    # Instantiate model
    model = GCNWithAttention_sign(
        input_dim,
        hyperparams["hidden_dim"],
        b=float(attn_temperature),
    ).to(device)

    loss_fn = nn.MSELoss()

    # Create DataLoader
    dataset = TensorDataset(y_resid)
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    start_time = time.time()
    best_spillover_loss = float('inf')
    best_predicted_attention = None
    best_model = None

    # Early stopping parameters
    no_improvement_count = 0
    patience = patience  # Stop if no improvement for 5 epochs in a row

    for epoch in range(hyperparams["epochs"]):
        model.train()
        train_loss = 0
        pairwise_w_ij_train = torch.zeros(n, n, device=device)
        Y_pred_all = torch.zeros(n, device=device)

        # Training step
        for batch_Y, idx in dataloader:
            batch_Y = batch_Y.to(device)
            idx_cpu = idx.tolist()
            batch_one_hop_neighbor = [nbrs_idx[i] for i in idx_cpu]

            optimizer.zero_grad()
            Y_pred, batch_pairwise_w_ij = model(x, batch_one_hop_neighbor, t, e_hat)
            loss = loss_fn(Y_pred, batch_Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            Y_pred_all[idx] = Y_pred.detach()
            pairwise_w_ij_train += batch_pairwise_w_ij.detach()

        # Evaluate spillover loss on the full dataset
        model.eval()
        with torch.no_grad():
            predicted_attention, _ = model.predict(x, nbrs_idx, t)
        spillover_loss = np.sum((predicted_attention - true_attention) ** 2)

        # Check improvement
        if spillover_loss < best_spillover_loss:
            best_spillover_loss = spillover_loss
            best_predicted_attention = predicted_attention
            best_model = copy.deepcopy(model)
            no_improvement_count = 0  # reset counter
            if printing:
                print(f"Epoch {epoch+1:03d}, New best Spillover diff: {spillover_loss:.4f}")
        else:
            no_improvement_count += 1
            if printing:
                print(f"No improve. Epoch {epoch+1:03d}, Spillover diff: {spillover_loss:.4f}")

            # Early stopping condition
            if no_improvement_count >= patience:
                if printing:
                    print(f"Stopping early at epoch {epoch+1} due to no improvement "
                          f"for {patience} consecutive epochs.")
                break

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return best_model, best_spillover_loss, best_predicted_attention


