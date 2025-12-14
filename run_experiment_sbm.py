import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from addition import generate_sbm_graph, generate_covariates
from simulation import simulate_treatment, compute_outcome
from train import (
    GraphData,
    TrainConfig,
    PropensityConfig,
    MeanConfig,
    AttentionConfig,
    fit_propensity,
    fit_mean,
    fit_attention,
    bootstrap_attention,
)
from metric import compute_individual_effects, evaluate


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(
    config_path: str,
    B_override: Optional[int] = None,
    fit_nuisance_override: Optional[bool] = None,
    multiplier_dist_override: Optional[str] = None,
) -> None:
    cfg = load_config(config_path)

    exp_cfg = cfg.get("experiment", {})
    nuis_cfg = cfg.get("nuisance", {})
    boot_cfg = cfg.get("bootstrap", {})
    train_attn_cfg = cfg.get("train_attn", {})
    attn_cfg_dict = cfg.get("attn_cfg", {})
    out_cfg = cfg.get("output", {})

    # ---- SBM experiment settings ----
    n = int(exp_cfg.get("n", 5000))
    num_communities = int(exp_cfg.get("num_communities", 10))
    p = float(exp_cfg.get("p", 0.005))
    q = float(exp_cfg.get("q", 0.0005))
    train_fraction = float(exp_cfg.get("train_fraction", 0.7))

    outcome_mode = exp_cfg.get("outcome_mode", "separate_self")
    low_dimension = bool(exp_cfg.get("low_dimension", False))
    include_self_loop = outcome_mode == "with_self"

    f_type = exp_cfg.get("f_type", "cosine")
    g_type = exp_cfg.get("g_type", "heter")
    attn_temperature = float(exp_cfg.get("attn_temperature", 0.0))

    beta = float(exp_cfg.get("beta", 1.0))
    alpha_treat = float(exp_cfg.get("alpha", 2.0))
    sigma = float(exp_cfg.get("sigma", 0.05))
    scale = float(exp_cfg.get("scale", 5.0))
    num_partition = int(exp_cfg.get("num_partition", 3))
    ratio_self = float(exp_cfg.get("ratio_self", 1.0))

    # If True: fit nuisances to get (mu_hat, e_hat).
    # If False: skip nuisance fitting and use oracle (m_star, e_star).
    fit_nuisance = bool(nuis_cfg.get("fit_nuisance", True))
    if fit_nuisance_override is not None:
        fit_nuisance = fit_nuisance_override

    B = int(B_override) if B_override is not None else int(boot_cfg.get("B", 200))
    alpha_ci = float(boot_cfg.get("alpha", 0.05))
    multiplier_dist = boot_cfg.get("multiplier_dist", "normal")
    if multiplier_dist_override is not None:
        multiplier_dist = multiplier_dist_override

    output_dir = out_cfg.get("output_dir", "results")
    tag = out_cfg.get("tag", "sbm_separate_self")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Set seeds ----
    seed = int(train_attn_cfg.get("seed", 41))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Generate separate SBM graphs and covariates for train and test ----
    # Optionally allow different sizes in the future; for now use n for both
    n_train = n
    n_test = n

    # Train graph
    A, community_labels_train = generate_sbm_graph(
        n=n_train,
        num_communities= num_communities,
        p=p,
        q=q,
        seed=seed,
    )
    X = generate_covariates(n=n_train, d=5, seed=seed)
    # Add self-loops before simulate_treatment (to mirror real-data pipeline)
    A = A + np.eye(n_train)

    # Test graph (same hyperparameters, different random seed)
    A_test, community_labels_test = generate_sbm_graph(
        n=n_test,
        num_communities=num_communities,
        p=p,
        q=q,
        seed=seed + 1,
    )
    X_test = generate_covariates(n=n_test, d=5, seed=seed + 1)
    A_test = A_test + np.eye(n_test)

    # ---- Simulate treatment and outcomes (train) ----
    nbrs_idx, n_train, d, partitions, t, e_star, treat_neighbor, treat_matrix = simulate_treatment(
        X,
        A,
        mode="train",
        beta=beta,
        alpha=alpha_treat,
        num_partitions=num_partition,
        include_self_loop=include_self_loop,
    )

    W_true, spillover_true, U_0, noise, y, m_star, self_effect_true, raw_scores = compute_outcome(
        X,
        A,
        treat_matrix,
        e_star,
        n_train,
        sigma,
        scale,
        attn_temperature,
        name=f_type,
        outcome_mode=outcome_mode,
        self_name=g_type,
        t=t,
        w_self=ratio_self,
        low_dimension=low_dimension,
    )

    IME, ISE, ITE = compute_individual_effects(W_true, A, outcome_mode=outcome_mode, g_self=self_effect_true)

    # ---- Simulate treatment and outcomes (test) ----
    nbrs_idx_test, n_test, d_test, partitions_test, t_test, e_star_test, treat_neighbor_test, treat_matrix_test = (
        simulate_treatment(
            X_test,
            A_test,
            mode="test",
            beta=beta,
            alpha=alpha_treat,
            include_self_loop=include_self_loop,
        )
    )

    (
        W_true_test,
        spillover_true_test,
        U_0_test,
        noise_test,
        y_test,
        m_star_test,
        self_effect_true_test,
        raw_scores_test,
    ) = compute_outcome(
        X_test,
        A_test,
        treat_matrix_test,
        e_star_test,
        n_test,
        sigma,
        scale,
        attn_temperature,
        name=f_type,
        outcome_mode=outcome_mode,
        self_name=g_type,
        t=t_test,
        w_self=ratio_self,
        low_dimension=low_dimension,
    )

    IME_test, ISE_test, ITE_test = compute_individual_effects(
        W_true_test, A_test, outcome_mode=outcome_mode, g_self=self_effect_true_test
    )

    # ---- Fit nuisance components (propensity and mean) ----
    data = GraphData(X=X, y=y, A=A, nbrs_idx=nbrs_idx, t=t, fold_assignments=partitions)

    if fit_nuisance:
        train_cfg_pm = TrainConfig(
            epochs=500,
            lr=5e-2,
            batch_size=n_train,
            patience=12,
            device="cpu",
            log_every=40,
        )

        prop_cfg = PropensityConfig(hidden_dim=128)
        mean_cfg = MeanConfig(hidden_dim=128)

        fr_p, data = fit_propensity(data, train_cfg_pm, prop_cfg)
        fr_m, data = fit_mean(data, train_cfg_pm, mean_cfg)
    else:
        # Directly use oracle nuisances from the data-generating process
        data.mu_hat = m_star
        data.e_hat = e_star

    # ---- Fit attention model ----
    train_cfg_attn = TrainConfig(
        epochs=int(train_attn_cfg.get("epochs", 400)),
        lr=float(train_attn_cfg.get("lr", 1.0e-4)),
        batch_size=int(train_attn_cfg.get("batch_size", 64)),
        patience=int(train_attn_cfg.get("patience", 12)),
        device=train_attn_cfg.get("device", "cpu"),
        log_every=int(train_attn_cfg.get("log_every", 1)),
        weight_decay=float(train_attn_cfg.get("weight_decay", 0.0)),
        seed=seed,
        verbose=bool(train_attn_cfg.get("verbose", True)),
    )

    attn_cfg = AttentionConfig(
        hidden_dim=int(attn_cfg_dict.get("hidden_dim", 64)),
        attn_temperature=float(attn_cfg_dict.get("attn_temperature", attn_temperature)),
        separate_self=bool(attn_cfg_dict.get("separate_self", outcome_mode == "separate_self")),
        low_dimension=bool(attn_cfg_dict.get("low_dimension", low_dimension)),
    )

    fr_a = fit_attention(
        data=data,
        train_cfg=train_cfg_attn,
        attn_cfg=attn_cfg,
        attn_true=W_true,
        attn_true_self=self_effect_true,
    )

    # ---- Evaluate causal effects from the fitted attention model ----
    with torch.no_grad():
        out = fr_a.model.cpu().predict(
            torch.tensor(X, dtype=torch.float32),
            nbrs_idx,
            torch.tensor(t, dtype=torch.float32),
        )
    W_pred, g_pred = (out[0], out[2]) if isinstance(out, tuple) and len(out) == 3 else (out[0], None)
    IME_est, ISE_est, ITE_est = compute_individual_effects(W_pred, A, outcome_mode, g_self=g_pred)
    result_train = evaluate(IME, ISE, ITE, IME_est, ISE_est, ITE_est, printing=True)

    with torch.no_grad():
        out_test = fr_a.model.predict(
            torch.tensor(X_test, dtype=torch.float32),
            nbrs_idx_test,
            torch.tensor(t_test, dtype=torch.float32),
        )
    W_pred_test, g_pred_test = (out_test[0], out_test[2]) if isinstance(out_test, tuple) and len(out_test) == 3 else (
        out_test[0],
        None,
    )
    IME_est_test, ISE_est_test, ITE_est_test = compute_individual_effects(
        W_pred_test, A_test, outcome_mode, g_self=g_pred_test
    )
    result_test = evaluate(IME_test, ISE_test, ITE_test, IME_est_test, ISE_est_test, ITE_est_test, printing=True)

    # ---- Bootstrap attention ----
    boot_res = bootstrap_attention(
        data=data,
        train_cfg=train_cfg_attn,
        attn_cfg=attn_cfg,
        attn_true=W_true,
        attn_true_self=self_effect_true,
        B=B,
        alpha=alpha_ci,
        multiplier_dist=multiplier_dist,
        X_test=X_test,
        nbrs_idx_test=nbrs_idx_test,
        t_test=t_test,
    )

    # ---- Coverage statistics for attention and raw scores over bootstrap prefixes ----
    def compute_coverage_over_prefixes(
        base: np.ndarray,
        boot: np.ndarray,
        truth: np.ndarray,
        alpha: float,
    ) -> Dict[str, Any]:
        """
        Compute coverage statistics for pointwise and uniform CIs
        using bootstrap prefixes of size k = step, 2*step, ..., B.
        """
        B_total = boot.shape[0]
        step = max(1, B_total // 10)
        prefixes = list(range(step, B_total + 1, step))
        if prefixes[-1] != B_total:
            prefixes.append(B_total)

        nonzero_mask = (truth != 0)
        n_nonzero = int(nonzero_mask.sum())

        results_by_prefix: Dict[str, Any] = {}

        for k in prefixes:
            diffs = boot[:k] - base[None, :, :]  # [k, n, n]

            # Pointwise CIs
            radius_pointwise = np.quantile(
                np.abs(diffs),
                1.0 - alpha / 2.0,
                axis=0,
            )  # [n, n]
            ci_lo_pw = base - radius_pointwise
            ci_hi_pw = base + radius_pointwise

            # Uniform band
            max_dev = np.max(np.abs(diffs).reshape(k, -1), axis=1)  # [k]
            c_uniform = np.quantile(max_dev, 1.0 - alpha)
            ci_lo_unif = base - c_uniform
            ci_hi_unif = base + c_uniform

            # Coverage (only over nonzero true entries)
            covered_pw = ((truth >= ci_lo_pw) & (truth <= ci_hi_pw)) & nonzero_mask
            covered_unif = ((truth >= ci_lo_unif) & (truth <= ci_hi_unif)) & nonzero_mask

            covered_pw_count = int(covered_pw.sum())
            covered_unif_count = int(covered_unif.sum())

            coverage_pw = covered_pw_count / n_nonzero if n_nonzero > 0 else float("nan")
            coverage_unif = covered_unif_count / n_nonzero if n_nonzero > 0 else float("nan")

            results_by_prefix[str(k)] = {
                "B_prefix": k,
                "n_nonzero": n_nonzero,
                "pointwise": {
                    "coverage": coverage_pw,
                    "covered": covered_pw_count,
                },
                "uniform": {
                    "coverage": coverage_unif,
                    "covered": covered_unif_count,
                },
            }

        return {
            "B_total": B_total,
            "alpha": alpha,
            "n_nonzero": n_nonzero,
            "by_prefix": results_by_prefix,
        }

    # Train coverage
    attn_coverage_train = compute_coverage_over_prefixes(
        base=boot_res.base_attn,
        boot=boot_res.boot_attn,
        truth=W_true,
        alpha=alpha_ci,
    )

    raw_coverage_train = compute_coverage_over_prefixes(
        base=boot_res.base_raw,
        boot=boot_res.boot_raw,
        truth=raw_scores,
        alpha=alpha_ci,
    )

    # Test coverage
    attn_coverage_test: Optional[Dict[str, Any]] = None
    raw_coverage_test: Optional[Dict[str, Any]] = None
    if (
        boot_res.base_attn_test is not None
        and boot_res.boot_attn_test is not None
        and boot_res.base_raw_test is not None
        and boot_res.boot_raw_test is not None
    ):
        attn_coverage_test = compute_coverage_over_prefixes(
            base=boot_res.base_attn_test,
            boot=boot_res.boot_attn_test,
            truth=W_true_test,
            alpha=alpha_ci,
        )
        raw_coverage_test = compute_coverage_over_prefixes(
            base=boot_res.base_raw_test,
            boot=boot_res.boot_raw_test,
            truth=raw_scores_test,
            alpha=alpha_ci,
        )

    # ---- Save scalar metrics (train/test causal effect evaluation + coverage statistics) ----
    nuis_suffix = "fitted" if fit_nuisance else "oracle"
    dataset_label = f"SBM_n{n}_K{num_communities}"
    tag_with_nuis_and_B = f"{dataset_label}_{tag}_nuis-{nuis_suffix}_B{B}"

    # Effective configs that reflect overrides
    effective_exp_cfg = dict(exp_cfg)
    effective_exp_cfg["n"] = n
    effective_exp_cfg["num_communities"] = num_communities
    effective_exp_cfg["p"] = p
    effective_exp_cfg["q"] = q
    effective_exp_cfg["train_fraction"] = train_fraction

    effective_boot_cfg = dict(boot_cfg)
    effective_boot_cfg["B"] = B
    effective_boot_cfg["alpha"] = alpha_ci
    effective_boot_cfg["multiplier_dist"] = multiplier_dist

    effective_nuis_cfg = dict(nuis_cfg)
    effective_nuis_cfg["fit_nuisance"] = fit_nuisance

    metrics = {
        "train": result_train,
        "test": result_test,
        "train_cfg_attn": train_attn_cfg,
        "attn_cfg": attn_cfg_dict,
        "bootstrap": effective_boot_cfg,
        "nuisance": effective_nuis_cfg,
        "experiment": effective_exp_cfg,
        "coverage": {
            "train": {
                "attention": attn_coverage_train,
                "raw_scores": raw_coverage_train,
            },
            "test": {
                "attention": attn_coverage_test,
                "raw_scores": raw_coverage_test,
            },
        },
    }
    metrics_path = os.path.join(output_dir, f"metrics_{tag_with_nuis_and_B}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic SBM spillover experiment with bootstrap.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_experiment_sbm.yaml",
        help="Path to SBM YAML config file.",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=None,
        help="Override bootstrap.B from config.",
    )
    parser.add_argument(
        "--fit_nuisance",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override nuisance.fit_nuisance in config (true/false).",
    )
    parser.add_argument(
        "--multiplier_dist",
        type=str,
        choices=["normal", "rademacher"],
        default=None,
        help="Override bootstrap.multiplier_dist in config (normal/rademacher).",
    )
    args = parser.parse_args()
    fit_nuisance_override = None
    if args.fit_nuisance is not None:
        fit_nuisance_override = args.fit_nuisance.lower() == "true"
    main(
        args.config,
        args.B,
        fit_nuisance_override=fit_nuisance_override,
        multiplier_dist_override=args.multiplier_dist,
    )


