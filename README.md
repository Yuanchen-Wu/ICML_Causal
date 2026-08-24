# Smart Bootstrap Inference for Causal Effects on Graphs

This branch implements simulation, estimation, and bootstrap inference for
individual causal effects under network interference. It separates each
node's effect into:

- **Individual Main Effect (IME):** the direct/self-treatment effect.
- **Individual Spillover Effect (ISE):** the effect transmitted through neighbors.
- **Individual Total Effect (ITE):** IME + ISE.

Experiments can use an observed network from an NPZ file or a synthetic
stochastic block model (SBM).

## Features

- One-head (`with_self`) and two-head (`separate_self`) attention models.
- Full- and low-dimensional attention specifications.
- Oracle or fitted nuisance functions.
- Exact multiplier bootstrap and one-step infinitesimal jackknife (IJ)
  approximation.
- Paired exact-versus-IJ comparisons using shared bootstrap multipliers.
- Normal, Rademacher, and Poisson multipliers.
- Pointwise and uniform confidence intervals and coverage diagnostics.
- Separate train/test coverage evaluation and optional node-level CI snapshots.
- Analysis notebooks for coverage, interval length, damping sensitivity, and
  train-fraction comparisons.

## Installation

```bash
git clone https://github.com/Yuanchen-Wu/ICML_Causal.git
cd ICML_Causal
git checkout bootstrap
python -m pip install -r requirements.txt
```

The requirements install the CPU-compatible PyTorch package. Install the
appropriate PyTorch build separately if GPU acceleration is required.

## Quick start

Run the synthetic SBM experiment:

```bash
python run_experiment_sbm.py --config config_experiment_sbm.yaml
```

Run an experiment on an NPZ network in `dataset/`:

```bash
python run_experiment.py --config config_experiment.yaml
```

The scripts write uniquely numbered JSON metric files to the configured
`output.output_dir` (default: `results/`).

### Command-line overrides

Common SBM overrides include:

```bash
python run_experiment_sbm.py \
  --config config_experiment_sbm.yaml \
  --B 200 \
  --alpha 0.05 \
  --bootstrap_method ij \
  --multiplier_dist normal \
  --seed 41
```

Use `--bootstrap_method exact`, `ij`, or `both`. The `both` option runs a
paired comparison. Other overrides include `--fit_nuisance`,
`--train_fraction`, `--ij_damping`, and `--ij_diag`.

For observed-network experiments, `run_experiment.py` supports `--B`,
`--fit_nuisance`, `--multiplier_dist`, and `--path`.

## Configuration

The YAML files organize settings into five sections:

- `experiment`: graph/data source, outcome model, attention similarity, and
  data-generating parameters.
- `nuisance`: fitted (`true`) or oracle (`false`) nuisance functions.
- `bootstrap`: replicate count, confidence levels, multiplier distribution,
  exact/IJ method, solver settings, evaluation scope, and CI snapshots.
- `train_attn`: optimization, device, seed, and early-stopping settings.
- `output`: output directory and filename tag/prefix.

For SBM experiments, `experiment.sbm_splits` configures separate training and
evaluation graph sizes and edge probabilities. `bootstrap.alpha` accepts one
value or a list in the SBM runner.

## Data format

Observed-network datasets are loaded from `dataset/<experiment.path>`. Each
NPZ file is expected to provide:

- The configured feature array (for example, `lda_supervised`).
- `adj_matrix`, stored as a SciPy sparse matrix.
- `fold`, containing split labels used to construct train/validation/test sets.

## Project structure

- `run_experiment.py`: observed-network experiment runner.
- `run_experiment_sbm.py`: synthetic SBM runner with exact/IJ inference.
- `config_experiment.yaml`: observed-network configuration.
- `config_experiment_sbm.yaml`: SBM and IJ configuration.
- `addition.py`: SBM graph and covariate generation.
- `simulation.py`: treatment and outcome simulation.
- `model/interference.py`: one- and two-head graph attention models.
- `train.py`: model fitting and bootstrap implementations.
- `metric.py`: causal-effect evaluation metrics.
- `experiment_*.ipynb`: interactive end-to-end experiments.
- `aggregate_*.ipynb`, `plot_*.ipynb`: result aggregation and visualization.

## Reproducibility notes

- Set `train_attn.seed` or pass `--seed` to reproduce SBM runs.
- Nuisance models use the full covariate matrix.
- Attention models use the full covariate matrix unless
  `low_dimension: true`.
- `alpha_treat` controls treatment assignment; `bootstrap.alpha` controls the
  confidence level.

## License

No license has been added yet.
