import torch
import torch.nn as nn
import torch.nn.functional as F
 

class MeanPredictor(nn.Module):
  def __init__(self, in_features, hidden_features, out_features, agg_method='mean', use_flexible_attention=False):
    super().__init__()
    self.fc1 = nn.Linear(2 * in_features, hidden_features)
    self.fc3 = nn.Linear(hidden_features, out_features)
    self.agg_method = agg_method
    self.use_flexible_attention = use_flexible_attention
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
    Input = torch.cat([X, Z], dim=1)
    H = F.relu(self.fc1(Input))
    return H

  def forward(self, X, A_norm):
    H = self.compute_hidden(X, A_norm)
    out = self.fc3(H)
    return out

  def get_embedding(self, X, A_norm):
    return self.compute_hidden(X, A_norm)
