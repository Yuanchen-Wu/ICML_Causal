import torch
import torch.nn as nn

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
