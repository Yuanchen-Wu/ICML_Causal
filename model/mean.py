import torch
import torch.nn as nn
import torch.nn.functional as F
 

class MeanPredictor(nn.Module):
  def __init__(
    self,
    in_features,
    hidden_features,
    out_features,
    agg_method: str = 'mean',
    use_flexible_attention: bool = False,
    low_dimension: bool = False,
  ):
    super().__init__()
    self.agg_method = agg_method
    self.use_flexible_attention = use_flexible_attention
    self.low_dimension = low_dimension

    # Optional learnable projection used only when low_dimension=True.
    # This keeps base effects high-dimensional while allowing a separate
    # low-dimensional component to be fed into the mean model.
    self.proj = None  # shared projection for both X and Z
    proj_dim = 0
    if self.low_dimension:
      self.proj = nn.Linear(in_features, 1, bias=False)
      # We will concatenate proj(X) and proj(Z), so +2 dims in total
      proj_dim = 2

    # Input is [X, Z] or [X, Z, proj(X)] depending on low_dimension.
    self.fc1 = nn.Linear(2 * in_features + proj_dim, hidden_features)
    self.fc3 = nn.Linear(hidden_features, out_features)

    if self.agg_method == 'gat':
      self.gat_lin = nn.Linear(in_features, in_features, bias=False)
      if self.use_flexible_attention:
        self.a_scale = nn.Parameter(torch.tensor(1.0))
        self.leaky_relu = nn.LeakyReLU(0.2)
      else:
        self.a_left = nn.Parameter(torch.Tensor(in_features, 1))
        self.a_right = nn.Parameter(torch.Tensor(in_features, 1))
        nn.init.xavier_uniform_(self.a_left.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_right.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.2)

  def compute_hidden(self, X, A_norm):
    if self.agg_method == 'mean':
      Z = A_norm @ X
    elif self.agg_method == 'gat':
      H_trans = self.gat_lin(X)
      N = H_trans.size(0)
      if self.use_flexible_attention:
        e = torch.matmul(H_trans, H_trans.T)
        e = self.leaky_relu(e * self.a_scale)
      else:
        f1 = torch.matmul(H_trans, self.a_left)
        f2 = torch.matmul(H_trans, self.a_right)
        e = self.leaky_relu(f1 + f2.T)
      mask = (A_norm > 0)
      e = e.masked_fill(~mask, float('-inf'))
      alpha = torch.softmax(e, dim=1)
      Z = torch.matmul(alpha, H_trans)
    else:
      raise ValueError("Unknown aggregation method: choose 'mean' or 'gat'.")
    if self.low_dimension and self.proj is not None:
      proj_X = self.proj(X)      # shape (n, 1)
      proj_Z = self.proj(Z)      # shape (n, 1)
      Input = torch.cat([X, Z, proj_X, proj_Z], dim=1)
    else:
      Input = torch.cat([X, Z], dim=1)
    H = F.relu(self.fc1(Input))
    return H

  def forward(self, X, A_norm):
    H = self.compute_hidden(X, A_norm)
    out = self.fc3(H)
    return out

  def get_embedding(self, X, A_norm):
    return self.compute_hidden(X, A_norm)
