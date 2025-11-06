import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import to_torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

def run_bagging(predictor, response, m_star, max_depth=15, n_estimators=500):
  estimator = DecisionTreeRegressor(max_depth=15, min_samples_split=2, random_state=42)
  bagging_regressor = BaggingRegressor(estimator=estimator, n_estimators=n_estimators, random_state=42)
  bagging_regressor.fit(predictor, response)
  response_bagging = bagging_regressor.predict(predictor)
  return response_bagging

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

def fit_mean(X, adj_matrix, treat_binary, y, m_star, ehat, partitions, hyperparams, 
             bagging=True, aug=True, print_logs=False, device="cpu", agg_method='mean', flexible=False):
  X_torch, A_norm, treat_torch, y_torch, d_var = to_torch(X, adj_matrix, treat_binary, y, device=device)
  m_star_torch = torch.from_numpy(m_star).view(-1, 1).to(torch.float32).to(device)
  t_var = torch.from_numpy(ehat).view(-1, 1).to(torch.float32).to(device)

  if aug:
    X_torch_aug = torch.cat([X_torch, t_var, d_var], dim=1)
  else:
    X_torch_aug = X_torch

  num_features = X_torch_aug.shape[1]
  n = X_torch_aug.shape[0]

  mhat = torch.zeros(n)
  best_models = {}
  criterion = nn.MSELoss()

  for fold in np.unique(partitions):
    model_mean = MeanPredictor(in_features=num_features, hidden_features=hyperparams["hidden_dim"],
                               out_features=1, agg_method=agg_method, use_flexible_attention=flexible).to(device)
    optimizer = optim.Adam(model_mean.parameters(), lr=hyperparams["lr"])

    train_idx = np.where(partitions != fold)[0]
    test_idx  = np.where(partitions == fold)[0]

    best_test_loss = float('inf')
    best_y_pred_test = None
    best_model = None

    for epoch in range(hyperparams["epochs"]):
      model_mean.train()
      optimizer.zero_grad()
      y_pred = model_mean(X_torch_aug, A_norm).squeeze(dim=1)
      train_loss = criterion(y_pred[train_idx], y_torch[train_idx].squeeze())
      train_loss.backward()
      optimizer.step()

      model_mean.eval()
      with torch.no_grad():
        y_pred = model_mean(X_torch_aug, A_norm).squeeze(dim=1)
        test_loss = criterion(y_pred[test_idx], m_star_torch[test_idx].squeeze())

      if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_y_pred_test = y_pred[test_idx].detach().clone()
        best_model = copy.deepcopy(model_mean)

      if print_logs and (epoch + 1) % 50 == 0:
        print(f"Partition: {fold}, epoch: {epoch+1:03d} - New best Testing Loss: {best_test_loss:.4f}")

    mhat[test_idx] = best_y_pred_test.squeeze().to('cpu')
    best_models[fold] = best_model

  mhat = mhat.detach().numpy()
  if bagging:
    predictor = X_torch_aug.cpu().detach().numpy()
    response_bagging = run_bagging(predictor, mhat, m_star)
    if np.sum((m_star - response_bagging) ** 2) < np.sum((m_star - mhat) ** 2):
      if print_logs:
        print("Improved by bagging")
      mhat = response_bagging
  return mhat, best_models

# Backwards compatibility
run_mean_experiment = fit_mean


