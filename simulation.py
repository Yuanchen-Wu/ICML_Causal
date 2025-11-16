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
  x2 = X[:, 1] if X.shape[1] > 1 else 0.0
  x3 = X[:, 2] if X.shape[1] > 2 else 0.0
  tau = (
    alpha
    + b1 * x1
    + b2 * (x2 ** 2)
    + b3 * (x3 ** 3)
    + b4 * (x1 * x2)
    + b5 * (x2 * x3)
  )
  return tau

def get_true_attention(X, A, name, attn_temperature, low_dimension: bool = False):
  X_use = X[:, :1] if low_dimension else X
  if name == "rbf":
    func = np.exp(-0.5 * (cdist(X_use, X_use, metric='euclidean') ** 2))
  if name == "cosine":
    func = cosine_similarity(X_use)
  if name == "homo":
    func = np.full((X_use.shape[0], X_use.shape[0]), 0.9)
    np.fill_diagonal(func, 1)
  if name == "heter":
    beta = np.array([1 / X_use.shape[1]] * X_use.shape[1])*5
    func = np.tile(X_use.dot(beta), (X_use.shape[0], 1))
  if name == "heter_nonlinear":
    func = np.tile(heter_nonlinear(X_use), (X_use.shape[0], 1))
  true_attention = row_sparse_softmax(func * A, b=attn_temperature)
  return true_attention

def get_true_attention_no_self(X, A, name, attn_temperature, low_dimension: bool = False):
  """Neighbor attention excluding self (diagonal zero) with row-wise softmax on non-zeros."""
  A_no_self = A.copy()
  np.fill_diagonal(A_no_self, 0.0)
  return get_true_attention(X, A_no_self, name=name, attn_temperature=attn_temperature, low_dimension=low_dimension)

def get_self_treatment_effect(X, name, seed=42, low_dimension: bool = False, w_self: float = 1.0):
  """Generate per-node scalar self-treatment effect g(X_i) for given name.
  - homo: constant 1.0
  - heter: linear X @ beta
  - cosine: cosine similarity to a fixed random anchor u
  - rbf: exp(-0.5 * ||X - u||^2)
  """
  rng = np.random.RandomState(seed)
  X_use = X[:, :1] if low_dimension else X
  n, d = X_use.shape
  if name == "homo":
    return w_self * np.ones(n, dtype=float)
  if name == "heter":
    beta = rng.uniform(low=-1, high=1, size=d) / max(1, d)
    g_base = X_use.dot(beta * 5)
    return w_self * g_base
  # anchor-based versions for single-argument variants
  u = rng.normal(size=d)
  if name == "cosine":
    # cosine(X_i, u)
    x_norm = np.linalg.norm(X_use, axis=1) + 1e-8
    u_norm = np.linalg.norm(u) + 1e-8
    g_base = (X_use.dot(u)) / (x_norm * u_norm)
    return w_self * g_base
  if name == "rbf":
    d2 = np.sum((X_use - u) ** 2, axis=1)
    g_base = np.exp(-0.5 * d2)
    return w_self * g_base
  # default
  return w_self * np.ones(n, dtype=float)

def compute_outcome(
  X, A, treat_matrix, e_star, n, sigma, scale, attn_temperature, name,
  outcome_mode: str = "with_self", self_name: str = None, t: np.ndarray = None, low_dimension: bool = False,
  w_self: float = 1.0, **kwargs
):
  """Compute outcome under two settings controlled by outcome_mode.

  with_self (default):
    - Attention includes self; y = spillover + U_0 + noise
    - Returns (..., g_true=None)

  separate_self:
    - Attention excludes self; y = spillover + g(X_i)*t_i + U_0 + noise
    - Requires t; returns g_true as last element

  Returns (true_attention, spillover, U_0, noise, y, m_star, g_true)
  """
  if outcome_mode == "with_self":
    true_attention = get_true_attention(X, A, name=name, attn_temperature=attn_temperature, low_dimension=low_dimension)
    spillover = np.sum(treat_matrix * true_attention, axis=1)
    U_0, noise = get_base(X, sigma=sigma, scale=scale)
    y = spillover + U_0 + noise
    m_star = U_0 + np.sum(np.tile(e_star, (n, 1)) * A * true_attention, axis=1)
    g_true = None
    return true_attention, spillover, U_0, noise, y, m_star, g_true

  # separate_self
  if t is None:
    raise ValueError("t is required when outcome_mode='separate_self'")
  true_attention = get_true_attention_no_self(X, A, name=name, attn_temperature=attn_temperature, low_dimension=low_dimension)
  A_no_self = A.copy(); np.fill_diagonal(A_no_self, 0.0)
  spillover = np.sum((treat_matrix * (1 - np.eye(n))) * true_attention, axis=1)
  U_0, noise = get_base(X, sigma=sigma, scale=scale)
  g_true = get_self_treatment_effect(X, name=self_name or name, low_dimension=low_dimension, w_self=w_self)
  y = spillover + g_true * t + U_0 + noise
  m_star = U_0 + np.sum(np.tile(e_star, (n, 1)) * A_no_self * true_attention, axis=1) + g_true * e_star
  return true_attention, spillover, U_0, noise, y, m_star, g_true

def simulate_treatment(X, A, mode="train", beta=1.0, alpha=0.0, random_state=41, num_partitions=5, include_self_loop=True):
  A = A - np.eye(A.shape[0])
  nbrs_idx = get_one_hop_neighbors(A)
  nbrs_idx = [
    torch.cat((torch.tensor([i], dtype=torch.long), tensor.to(dtype=torch.long)))
    for i, tensor in enumerate(nbrs_idx)
  ]
  G = nx.from_numpy_array(A)
  if include_self_loop:
    A = A + np.eye(A.shape[0])

  n, d = X.shape
  partitions = None
  if mode == "train":
    partitions = np.array(partition_with_metis(G, num_partitions=num_partitions))
  t, e_star = simulate_treatment_linear_avg(X, A, beta=beta, alpha=alpha, random_state=random_state)

  treat_neighbor = get_treat_neighbors(nbrs_idx, t)
  treat_matrix = np.tile(t, (n, 1)) * A
  return nbrs_idx, n, d, partitions, t, e_star, treat_neighbor, treat_matrix


