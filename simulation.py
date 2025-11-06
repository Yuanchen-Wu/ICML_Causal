import numpy as np
import torch
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_one_hop_neighbors, get_treat_neighbors, partition_with_metis

def simulate_treatment_linear_avg(X, A, beta=1.0, alpha=0.0, w=None, v=None, random_state=41):
  np.random.seed(random_state)
  n, d = X.shape

  # If not provided, randomly initialize weights
  if w is None:
    w = np.random.randn(d)
  if v is None:
    v = np.random.randn(d)

  # Compute degree (including self-loop)
  deg = A.sum(axis=1, keepdims=True)  # shape (n, 1)
  # Avoid division by zero
  deg[deg == 0] = 1.0

  # Compute neighbor average for each node
  neighbor_avg = (A @ X) / deg

  # Linear predictor z_i
  z = beta * (alpha + X.dot(w) + neighbor_avg.dot(v))
  p = 1.0 / (1.0 + np.exp(-z))  # logistic
  T = (np.random.rand(n) < p).astype(int)
  return T, p

def row_sparse_softmax(matrix, b=1):
  result = np.zeros_like(matrix, dtype=float)

  for i, row in enumerate(matrix):
    nonzero_indices = row != 0
    nonzero_values = b * abs(row[nonzero_indices])

    if len(nonzero_values) == 0:
      continue

    # Compute softmax for non-zero values only
    exp_values = np.exp(nonzero_values - np.max(nonzero_values))
    softmax_values = exp_values / exp_values.sum()

    # Assign softmax values back to the non-zero positions
    result[i, nonzero_indices] = softmax_values

  return result * matrix

def get_base(X, sigma=0.1, scale=5, seed=42):
  np.random.seed(seed)
  beta_true = np.random.uniform(low=-1, high=1, size=X.shape[1]) / scale
  U_0 = X.dot(beta_true)
  noise = np.random.normal(loc=0, scale=sigma, size=X.shape[0])
  return U_0, noise

def heter_nonlinear(X, coefs=(0.1, 0.3, -0.2, 0.2, 0.2, 0.3)):
  alpha, b1, b2, b3, b4, b5 = coefs
  x1 = X[:, 0]
  x2 = X[:, 1]
  x3 = X[:, 2]
  tau = (
    alpha
    + b1 * x1
    + b2 * (x2 ** 2)
    + b3 * (x3 ** 3)
    + b4 * (x1 * x2)
    + b5 * (x2 * x3)
  )
  return tau

def get_true_attention(X, adj_matrix, name, sparse_param):
  if name == "rbf":
    func = np.exp(-0.5 * (cdist(X, X, metric='euclidean') ** 2))
  if name == "cosine":
    func = cosine_similarity(X)
  if name == "homo":
    func = np.full((X.shape[0], X.shape[0]), 0.9)
    np.fill_diagonal(func, 1)
  if name == "heter":
    beta = np.array([1 / X.shape[1]] * X.shape[1])
    func = np.tile(X.dot(beta), (X.shape[0], 1))
  if name == "heter_nonlinear":
    func = np.tile(heter_nonlinear(X), (X.shape[0], 1))
  true_attention = row_sparse_softmax(func * adj_matrix, b=sparse_param)
  return true_attention

def compute_outcome(X, adj_matrix, treat_matrix, e_star, n, sigma, scale, sparse_param, name):
  true_attention = get_true_attention(X, adj_matrix, name=name, sparse_param=sparse_param)
  spillover = np.sum(treat_matrix * true_attention, axis=1)
  U_0, noise = get_base(X, sigma=sigma, scale=scale)
  y = spillover + U_0 + noise
  m_star = U_0 + np.sum(np.tile(e_star, (n, 1)) * adj_matrix * true_attention, axis=1)
  return true_attention, spillover, U_0, noise, y, m_star

def simulate_treatment(X, adj_matrix, mode="train", beta=1.0, alpha=0.0, random_state=41, num_partitions=5):
  adj_matrix = adj_matrix - np.eye(adj_matrix.shape[0])
  one_hop_neighbors = get_one_hop_neighbors(adj_matrix)
  one_hop_neighbors = [
    torch.cat((torch.tensor([i], dtype=torch.long), tensor.to(dtype=torch.long)))
    for i, tensor in enumerate(one_hop_neighbors)
  ]
  G = nx.from_numpy_array(adj_matrix)
  adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

  n, d = X.shape
  partitions = None
  if mode == "train":
    partitions = np.array(partition_with_metis(G, num_partitions=num_partitions))
  treat_binary, e_star = simulate_treatment_linear_avg(X, adj_matrix, beta=beta, alpha=alpha, random_state=random_state)

  treat_neighbor = get_treat_neighbors(one_hop_neighbors, treat_binary)
  treat_matrix = np.tile(treat_binary, (n, 1)) * adj_matrix
  return one_hop_neighbors, n, d, partitions, treat_binary, e_star, treat_neighbor, treat_matrix


