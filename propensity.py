import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from utils import to_torch

class PropensityPredictor(nn.Module):
  def __init__(self, in_features, hidden_dim, out_features):
    super().__init__()
    self.hidden_layer = nn.Sequential(
      nn.Linear(2 * in_features, hidden_dim),
      nn.ReLU()
    )
    self.out_layer = nn.Linear(hidden_dim, out_features)

  def compute_hidden(self, X, A_norm):
    Z = torch.mm(A_norm, X)
    Input = torch.cat([X, Z], dim=1)
    H = self.hidden_layer(Input)
    return H

  def forward(self, X, A_norm):
    H = self.compute_hidden(X, A_norm)
    output = self.out_layer(H)
    return output

  def get_embedding(self, X, A_norm):
    return self.compute_hidden(X, A_norm)

def fit_propensity(X, adj_matrix, treat_binary, y, m_star, partitions, hyperparams, aug=True, print_logs=False, device="cpu"):
  X_torch, A_norm, treat_torch, y_torch, d_var = to_torch(X, adj_matrix, treat_binary, y, device=device)
  if aug:
    X_torch_aug = torch.cat([X_torch, d_var], dim=1)
  else:
    X_torch_aug = X_torch
  in_features = X_torch_aug.shape[1]
  n = X_torch_aug.shape[0]

  ehat = np.zeros(n, dtype=np.float32)
  best_models = {}

  for fold in np.unique(partitions):
    model_propensity = PropensityPredictor(in_features=in_features,
                                           hidden_dim=hyperparams["hidden_dim"],
                                           out_features=1).to(device)
    optimizer = optim.Adam(model_propensity.parameters(), lr=hyperparams["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_idx = np.where(partitions != fold)[0]
    test_idx  = np.where(partitions == fold)[0]

    best_test_loss = float('inf')
    best_test_logits = None
    best_model = None

    for epoch in range(hyperparams["epochs"]):
      model_propensity.train()
      optimizer.zero_grad()
      logits = model_propensity(X_torch_aug, A_norm).squeeze(dim=1)
      train_logits = logits[train_idx]
      train_labels = treat_torch[train_idx].float()
      train_loss = loss_fn(train_logits, train_labels)
      train_loss.backward()
      optimizer.step()

      model_propensity.eval()
      with torch.no_grad():
        logits = model_propensity(X_torch_aug, A_norm).squeeze(dim=1)
        test_logits = logits[test_idx]
        test_labels = treat_torch[test_idx].float()
        test_loss = loss_fn(test_logits, test_labels)

      if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_test_logits = test_logits.detach().clone()
        best_model = copy.deepcopy(model_propensity)

      if print_logs and (epoch + 1) % 50 == 0:
        print(f"Partition: {fold}, epoch: {epoch+1:03d} - New best Testing Loss: {best_test_loss:.4f}")

    model_propensity.eval()
    with torch.no_grad():
      final_probs = torch.sigmoid(best_test_logits)
      ehat[test_idx] = final_probs.cpu().numpy()

    best_models[fold] = best_model

  return ehat, best_models

# Backwards compatibility
run_propensity_experiment = fit_propensity


