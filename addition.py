import random
from typing import List, Tuple

import numpy as np
import networkx as nx


def generate_sbm_graph(
    n: int,
    num_communities: int,
    p: float = 0.005,
    q: float = 0.0005,
    seed: int = 41,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a stochastic block model (SBM) graph with equal-sized communities.

    Returns
    -------
    A_full : np.ndarray
        Adjacency matrix of shape [n, n] with entries in {0, 1}.
    ground_truth : np.ndarray
        Community labels of shape [n], with values in {0, ..., num_communities-1}.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Equal community sizes with remainder distributed to the first few
    sizes: List[int] = [n // num_communities] * num_communities
    remainder = n % num_communities
    for i in range(remainder):
        sizes[i] += 1

    # Intra- and inter-community connection probabilities
    probs = [
        [p if i == j else q for j in range(num_communities)]
        for i in range(num_communities)
    ]

    # Generate SBM graph
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    # Ground-truth community labels (aligned with node ordering)
    ground_truth: List[int] = []
    for community_idx, size in enumerate(sizes):
        ground_truth.extend([community_idx] * size)

    # Adjacency matrix
    A_full = nx.to_numpy_array(G, dtype=float)

    return A_full, np.array(ground_truth, dtype=int)


def generate_covariates(
    n: int,
    d: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate i.i.d. Gaussian covariates X ~ N(0, I_d) of shape [n, d].
    """
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, d)).astype(np.float32)


def train_test_split_indices(
    n: int,
    train_fraction: float = 0.7,
    seed: int = 41,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly split {0, ..., n-1} into train and test indices.

    Returns
    -------
    train_idx, test_idx : np.ndarray
        Disjoint index arrays whose union is {0, ..., n-1}.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    n_train = int(round(train_fraction * n))
    n_train = max(1, min(n - 1, n_train))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return train_idx, test_idx


