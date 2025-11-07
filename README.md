# ICML_Causal: Self vs Neighbor (Spillover) Effects on Graphs

This repository contains a lightweight framework to study and estimate individual causal effects on graphs, decomposed into:
- Individual Main Effect (IME): direct/self effect g(X_i) on node i
- Individual Spillover Effect (ISE): neighbor-induced effect via attention weights w_ij = f(X_i, X_j)
- Individual Total Effect (ITE): IME + ISE

The code supports two outcome specifications and a low-dimensional variant for attention:
- Outcome mode
  - with_self: attention normalization includes the node itself
  - separate_self: excludes self from neighbor normalization and models self-treatment with a separate head g(X_i)
- Low-dimensional attention
  - low_dimension=True restricts both f and g to use only the first feature of X (all other components still use full X)

## Install

```bash
git clone <your-repo-url>
cd ICML_Causal
pip install -r requirements.txt
```

PyTorch is listed as a CPU build by default. If you have CUDA, install the appropriate wheel from the PyTorch website.

## Quick start (notebooks)

Open one of the notebooks in Jupyter:
- experiment_separate_self_neighbor.ipynb
  - Full-dimensional attention (low_dimension=False)
  - Toggle outcome_mode between "with_self" and "separate_self"
- experiment_lowdim_separate_self_neighbor.ipynb
  - Low-dimensional attention (low_dimension=True)

Each notebook:
1) Loads dataset splits from dataset/*.npz
2) Simulates treatment and outcomes according to outcome_mode
3) Fits nuisance models (propensity, mean outcome) on full X
4) Fits attention model matching outcome_mode (one-head vs two-head) and low_dimension
5) Reports IME/ISE/ITE and evaluation metrics (AME, PEHE)

## Key configuration knobs

- outcome_mode: "with_self" or "separate_self"
- low_dimension: True or False (attention f and g use X[:, 0] when True)
- attn_temperature: softmax temperature for attention
- Training (epochs, lr, batch_size, patience)

## Project structure

- model/
  - interference.py: attention models
    - GCNWithAttentionOneHead (with_self)
    - GCNWithAttentionTwoHead (separate_self: neighbor + self heads)
- simulation.py: data generation, attention ground truth, outcome computation
- train.py: training loops and FitResult wrappers
- metric.py: effect decomposition and evaluation utilities
- dataset/: NPZ datasets (features, adjacency, folds)
- experiment_*.ipynb: end-to-end experiments
- requirements.txt: Python dependencies

## Reproducibility notes

- Nuisance models (propensity, mean) always use full X
- Attention models use full X unless low_dimension=True
- Treatment simulation and baseline outcome always use full X

## License

Add a license of your choice (e.g., MIT) before open-sourcing.


