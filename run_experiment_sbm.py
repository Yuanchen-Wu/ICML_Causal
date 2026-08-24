import argparse
import json
import os
import time
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
    ValidationData,
    fit_propensity,
    fit_mean,
    fit_attention,
    bootstrap_attention,
    lanczos_vs_cg_diagnostic,
)
from metric import compute_individual_effects, evaluate


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _subset_graph_for_diag(
    data: GraphData,
    attn_true: np.ndarray,
    attn_true_self: Optional[np.ndarray],
    max_n: int = 200,
) -> tuple[GraphData, np.ndarray, Optional[np.ndarray]]:
    n_sub = min(max_n, data.X.shape[0])
    idx = np.arange(n_sub)
    A_sub = data.A[np.ix_(idx, idx)].astype(np.float32)
    nbrs_sub = []
    for i in range(n_sub):
        neigh = np.where(A_sub[i] != 0)[0].tolist()
        nbrs_sub.append(np.array([i] + [j for j in neigh if j != i], dtype=np.int64))

    fold_sub = data.fold_assignments[idx] if data.fold_assignments is not None else np.zeros(n_sub, dtype=np.int64)
    data_sub = GraphData(
        X=data.X[idx].copy(),
        y=data.y[idx].copy(),
        A=A_sub,
        nbrs_idx=nbrs_sub,
        t=data.t[idx].copy(),
        fold_assignments=fold_sub.copy(),
        e_hat=data.e_hat[idx].copy() if data.e_hat is not None else None,
        mu_hat=data.mu_hat[idx].copy() if data.mu_hat is not None else None,
    )
    attn_true_sub = attn_true[np.ix_(idx, idx)].copy()
    attn_true_self_sub = attn_true_self[idx].copy() if attn_true_self is not None else None
    return data_sub, attn_true_sub, attn_true_self_sub


def _build_induced_graph_data(data: GraphData, node_idx: np.ndarray) -> GraphData:
    """Build an induced subgraph GraphData on selected node indices."""
    idx = np.asarray(node_idx, dtype=np.int64)
    A_sub = data.A[np.ix_(idx, idx)].astype(np.float32)
    nbrs_sub = []
    for i in range(A_sub.shape[0]):
        neigh = np.where(A_sub[i] != 0)[0].tolist()
        nbrs_sub.append(np.array([i] + [j for j in neigh if j != i], dtype=np.int64))
    fold_sub = data.fold_assignments[idx].copy()
    return GraphData(
        X=data.X[idx].copy(),
        y=data.y[idx].copy(),
        A=A_sub,
        nbrs_idx=nbrs_sub,
        t=data.t[idx].copy(),
        fold_assignments=fold_sub,
        e_hat=data.e_hat[idx].copy() if data.e_hat is not None else None,
        mu_hat=data.mu_hat[idx].copy() if data.mu_hat is not None else None,
    )


def _radius_summary(boot_res: Any, alpha: float) -> Dict[str, float]:
    diffs = boot_res.boot_attn - boot_res.base_attn[None, :, :]
    radius_pw = np.quantile(np.abs(diffs), 1.0 - alpha, axis=0)
    if boot_res.eval_mask is not None and boot_res.eval_mask.any():
        max_dev = np.max(np.abs(diffs)[:, boot_res.eval_mask], axis=1)
    else:
        max_dev = np.max(np.abs(diffs).reshape(diffs.shape[0], -1), axis=1)
    c_uniform = float(np.quantile(max_dev, 1.0 - alpha))
    return {
        "pointwise_mean": float(np.mean(radius_pw)),
        "pointwise_median": float(np.median(radius_pw)),
        "pointwise_max": float(np.max(radius_pw)),
        "c_uniform": c_uniform,
    }


def _draw_bootstrap_multipliers(B: int, n: int, multiplier_dist: str) -> np.ndarray:
    mats = []
    for _ in range(B):
        if multiplier_dist == "normal":
            xi = np.random.normal(loc=1.0, scale=1.0, size=n)
        elif multiplier_dist == "rademacher":
            eps = np.random.choice([-1.0, 1.0], size=n)
            xi = 1.0 + eps
        elif multiplier_dist == "poisson":
            xi = np.random.poisson(lam=1.0, size=n).astype(np.float32)
        else:
            raise ValueError(f"Unknown multiplier_dist: {multiplier_dist}")
        mats.append(xi.astype(np.float32))
    return np.stack(mats, axis=0)


def _bootstrap_method_summary(boot_res: Any, alpha: float) -> Dict[str, Any]:
    rad = _radius_summary(boot_res, alpha=alpha)
    losses = getattr(boot_res, "boot_weighted_loss", None)
    loss_summary: Dict[str, Any] = {"base_train_loss": getattr(boot_res, "base_train_loss", None)}
    if losses is not None and len(losses) > 0:
        loss_summary.update(
            {
                "boot_weighted_loss_mean": float(np.mean(losses)),
                "boot_weighted_loss_median": float(np.median(losses)),
                "boot_weighted_loss_min": float(np.min(losses)),
                "boot_weighted_loss_max": float(np.max(losses)),
            }
        )
    return {"radius": rad, "loss": loss_summary}


def main(
    config_path: str,
    B_override: Optional[int] = None,
    alpha_override: Optional[float] = None,
    fit_nuisance_override: Optional[bool] = None,
    multiplier_dist_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    train_fraction_override: Optional[float] = None,
    ij_damping_override: Optional[float] = None,
    bootstrap_method_override: Optional[str] = None,
    ij_diag_override: Optional[bool] = None,
) -> None:
    cfg = load_config(config_path)

    exp_cfg = cfg.get("experiment", {})
    nuis_cfg = cfg.get("nuisance", {})
    boot_cfg = cfg.get("bootstrap", {})
    eval_mask_cfg = boot_cfg.get("eval_mask")
    train_attn_cfg = cfg.get("train_attn", {})
    attn_cfg_dict = cfg.get("attn_cfg", {})
    out_cfg = cfg.get("output", {})

    # ---- SBM experiment settings ----
    n_default = int(exp_cfg.get("n", 5000))
    # Backward-compatible defaults for split-specific SBM config.
    num_communities_default = int(exp_cfg.get("num_communities", 10))
    p_default = float(exp_cfg.get("p", 0.005))
    q_default = float(exp_cfg.get("q", 0.0005))
    sbm_splits = exp_cfg.get("sbm_splits", {}) if isinstance(exp_cfg.get("sbm_splits"), dict) else {}
    sbm_train_cfg = sbm_splits.get("train", {}) if isinstance(sbm_splits.get("train"), dict) else {}
    sbm_eval_cfg = sbm_splits.get("eval", {}) if isinstance(sbm_splits.get("eval"), dict) else {}
    n_train_cfg = int(sbm_train_cfg.get("n", n_default))
    n_eval_cfg = int(sbm_eval_cfg.get("n", n_default))
    num_communities_train = int(sbm_train_cfg.get("num_communities", num_communities_default))
    p_train = float(sbm_train_cfg.get("p", p_default))
    q_train = float(sbm_train_cfg.get("q", q_default))
    num_communities_eval = int(sbm_eval_cfg.get("num_communities", num_communities_default))
    p_eval = float(sbm_eval_cfg.get("p", p_default))
    q_eval = float(sbm_eval_cfg.get("q", q_default))
    train_fraction = float(exp_cfg.get("train_fraction", 0.7))
    if train_fraction_override is not None:
        train_fraction = float(train_fraction_override)
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"experiment.train_fraction must satisfy 0 < train_fraction <= 1, got {train_fraction!r}")

    outcome_mode = exp_cfg.get("outcome_mode", "separate_self")
    low_dimension = bool(exp_cfg.get("low_dimension", False))
    include_self_loop = outcome_mode == "with_self"

    f_type = exp_cfg.get("f_type", "cosine")
    g_type = exp_cfg.get("g_type", "heter")
    attn_temperature = float(exp_cfg.get("attn_temperature", 0.0))

    beta = float(exp_cfg.get("beta", 1.0))
    # NOTE: renamed from experiment.alpha to experiment.alpha_treat to avoid
    # collision with bootstrap.alpha (CI level). Keep backwards compatibility.
    alpha_treat = float(exp_cfg.get("alpha_treat", exp_cfg.get("alpha", 2.0)))
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
    
    # Extract alpha values - support both single float and list
    alpha_raw = alpha_override if alpha_override is not None else boot_cfg.get("alpha", 0.05)
    if isinstance(alpha_raw, list):
        alpha_values = [float(a) for a in alpha_raw]
    elif isinstance(alpha_raw, (int, float)):
        alpha_values = [float(alpha_raw)]
    else:
        alpha_values = [float(alpha_raw)]
    
    # Use first alpha for bootstrap (bootstrap itself uses alpha for CI computation)
    alpha_ci = alpha_values[0]
    
    multiplier_dist = boot_cfg.get("multiplier_dist", "normal")
    if multiplier_dist_override is not None:
        multiplier_dist = multiplier_dist_override
    # Unified bootstrap method selector.
    # Preferred: bootstrap_method in {"exact", "ij", "both"}.
    # Backward-compat fallback: infer from legacy use_ij/run_both_use_ij flags.
    bootstrap_method = str(boot_cfg.get("bootstrap_method", "")).strip().lower()
    if bootstrap_method_override is not None:
        bootstrap_method = str(bootstrap_method_override).strip().lower()
    if bootstrap_method == "":
        legacy_use_ij = bool(boot_cfg.get("use_ij", False))
        legacy_run_both = bool(boot_cfg.get("run_both_use_ij", False))
        if legacy_run_both:
            bootstrap_method = "both"
        elif legacy_use_ij:
            bootstrap_method = "ij"
        else:
            bootstrap_method = "exact"
    if bootstrap_method not in {"exact", "ij", "both"}:
        raise ValueError(f"bootstrap_method must be one of exact/ij/both, got {bootstrap_method!r}")
    eval_scope = str(boot_cfg.get("eval_scope", "both")).strip().lower()
    if eval_scope not in {"train", "test", "both"}:
        raise ValueError(f"bootstrap.eval_scope must be one of train/test/both, got {eval_scope!r}")
    run_train_eval = eval_scope in {"train", "both"}
    run_test_eval = eval_scope in {"test", "both"}

    # In "both" mode we use IJ as the primary branch for downstream metrics and
    # run exact as the paired secondary branch on the same multipliers.
    use_ij = bootstrap_method in {"ij", "both"}
    ij_damping = float(boot_cfg.get("ij_damping", 1e-3))
    if ij_damping_override is not None:
        ij_damping = float(ij_damping_override)
    ij_cg_iters = int(boot_cfg.get("ij_cg_iters", 50))
    ij_cg_tol = float(boot_cfg.get("ij_cg_tol", 1e-6))
    ij_solver = str(boot_cfg.get("ij_solver", "cg")).strip().lower()
    ij_lanczos_m = int(boot_cfg.get("ij_lanczos_m", 100))
    ij_lanczos_k = int(boot_cfg.get("ij_lanczos_k", 50))
    ij_lanczos_reorth = bool(boot_cfg.get("ij_lanczos_reorth", True))
    ij_diag = bool(boot_cfg.get("ij_diag", False))
    if ij_diag_override is not None:
        ij_diag = ij_diag_override
    run_both_use_ij = bootstrap_method == "both"
    node_ci_cfg = boot_cfg.get("node_ci_snapshot", {})
    if isinstance(node_ci_cfg, dict):
        node_ci_snapshot_enabled = bool(node_ci_cfg.get("enabled", True))
        percentiles_raw = node_ci_cfg.get("percentiles", [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        node_ci_snapshot_enabled = bool(node_ci_cfg)
        percentiles_raw = [0.0, 0.25, 0.5, 0.75, 1.0]
    if isinstance(percentiles_raw, (int, float)):
        node_ci_snapshot_percentiles = [float(percentiles_raw)]
    else:
        node_ci_snapshot_percentiles = [float(p) for p in percentiles_raw]
    if node_ci_snapshot_enabled:
        if len(node_ci_snapshot_percentiles) == 0:
            raise ValueError("bootstrap.node_ci_snapshot.percentiles must be non-empty when enabled=true")
        if any((p < 0.0 or p > 1.0) for p in node_ci_snapshot_percentiles):
            raise ValueError(
                "bootstrap.node_ci_snapshot.percentiles must be in [0, 1], "
                f"got {node_ci_snapshot_percentiles!r}"
            )

    # ---- Eval-mask config (optional) ----
    # Simplified behavior: when eval_mask is present, always evaluate on edges and
    # always exclude the diagonal. Other legacy keys are ignored.
    eval_mask_cfg_effective: Optional[Dict[str, Any]] = None
    if eval_mask_cfg is not None:
        eval_mask_cfg_effective = {
            "mode": "edges",
            "exclude_diag": True,
        }

    output_dir = out_cfg.get("output_dir", "results")
    tag = out_cfg.get("tag", "sbm_separate_self")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Set seeds ----
    # `seed` controls randomness for treatment/outcomes/training.
    # Graph + covariates are fixed by a separate seed so they do not change when `seed` changes.
    seed = int(seed_override) if seed_override is not None else int(train_attn_cfg.get("seed", 41))
    graph_seed = int(exp_cfg.get("graph_seed", 41))

    # Fix graph/covariate generation
    np.random.seed(graph_seed)
    torch.manual_seed(graph_seed)

    # ---- Generate separate SBM graphs and covariates for train and test ----
    # Split-specific graph sizes (train vs shared eval config for val/test)
    n_train = n_train_cfg
    n_test = n_eval_cfg

    # Train graph
    A, community_labels_train = generate_sbm_graph(
        n=n_train,
        num_communities=num_communities_train,
        p=p_train,
        q=q_train,
        seed=graph_seed,
    )
    X = generate_covariates(n=n_train, d=5, seed=graph_seed)
    # Add self-loops before simulate_treatment (to mirror real-data pipeline)
    A = A + np.eye(n_train)

    # Test graph (eval hyperparameters, different random seed)
    A_test, community_labels_test = generate_sbm_graph(
        n=n_test,
        num_communities=num_communities_eval,
        p=p_eval,
        q=q_eval,
        seed=graph_seed + 1,
    )
    X_test = generate_covariates(n=n_test, d=5, seed=graph_seed + 1)
    A_test = A_test + np.eye(n_test)

    # Validation graph (used only for early stopping; seed = graph_seed + 2)
    n_val = n_eval_cfg
    A_val, community_labels_val = generate_sbm_graph(
        n=n_val,
        num_communities=num_communities_eval,
        p=p_eval,
        q=q_eval,
        seed=graph_seed + 2,
    )
    X_val = generate_covariates(n=n_val, d=5, seed=graph_seed + 2)
    A_val = A_val + np.eye(n_val)

    # Switch to experiment seed for treatment assignment, outcomes, and model training
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        verbose=bool(train_attn_cfg.get("verbose", True)),
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
        verbose=bool(train_attn_cfg.get("verbose", True)),
    )

    IME_test, ISE_test, ITE_test = compute_individual_effects(
        W_true_test, A_test, outcome_mode=outcome_mode, g_self=self_effect_true_test
    )

    # ---- Simulate treatment and outcomes (validation - oracle nuisances, early stopping only) ----
    # Save random state so inserting val simulation does not affect subsequent training reproducibility
    _rng_state_np = np.random.get_state()
    _rng_state_torch = torch.random.get_rng_state()

    nbrs_idx_val, n_val, d_val, partitions_val, t_val, e_star_val, treat_neighbor_val, treat_matrix_val = (
        simulate_treatment(
            X_val,
            A_val,
            mode="test",
            beta=beta,
            alpha=alpha_treat,
            include_self_loop=include_self_loop,
        )
    )

    (
        W_true_val,
        spillover_true_val,
        U_0_val,
        noise_val,
        y_val,
        m_star_val,
        self_effect_true_val,
        raw_scores_val,
    ) = compute_outcome(
        X_val,
        A_val,
        treat_matrix_val,
        e_star_val,
        n_val,
        sigma,
        scale,
        attn_temperature,
        name=f_type,
        outcome_mode=outcome_mode,
        self_name=g_type,
        t=t_val,
        w_self=ratio_self,
        low_dimension=low_dimension,
        verbose=bool(train_attn_cfg.get("verbose", True)),
    )

    # Restore random state
    np.random.set_state(_rng_state_np)
    torch.random.set_rng_state(_rng_state_torch)

    # Build ValidationData for early stopping in attention training (oracle nuisances)
    val_data = ValidationData(
        X=X_val,
        nbrs_idx=nbrs_idx_val,
        t=t_val,
        attn_true=W_true_val,
        attn_true_self=self_effect_true_val,
    )

    # ---- Build full training data and deterministic fitting subset ----
    data_full = GraphData(X=X, y=y, A=A, nbrs_idx=nbrs_idx, t=t, fold_assignments=partitions)
    train_nodes_total = int(n_train)
    train_nodes_used = int(np.clip(round(train_fraction * train_nodes_total), 1, train_nodes_total))
    if train_nodes_used == train_nodes_total:
        train_subset_idx = np.arange(train_nodes_total, dtype=np.int64)
    else:
        subset_rng = np.random.default_rng(graph_seed)
        train_subset_idx = np.sort(subset_rng.choice(train_nodes_total, size=train_nodes_used, replace=False).astype(np.int64))
    data = _build_induced_graph_data(data_full, train_subset_idx)
    W_true_fit = W_true[np.ix_(train_subset_idx, train_subset_idx)].copy()
    self_effect_true_fit = self_effect_true[train_subset_idx].copy() if self_effect_true is not None else None
    raw_scores_fit = raw_scores[np.ix_(train_subset_idx, train_subset_idx)].copy()
    ise_true_train = ISE[train_subset_idx].copy()

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
        data.mu_hat = m_star[train_subset_idx].copy()
        data.e_hat = e_star[train_subset_idx].copy()

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
        attn_true=W_true_fit,
        attn_true_self=self_effect_true_fit,
        val_data=val_data,
    )

    # ---- Evaluate causal effects from the fitted attention model ----
    result_train: Optional[Dict[str, Any]] = None
    result_test: Optional[Dict[str, Any]] = None
    if run_train_eval:
        with torch.no_grad():
            out = fr_a.model.cpu().predict(
                torch.tensor(X, dtype=torch.float32),
                nbrs_idx,
                torch.tensor(t, dtype=torch.float32),
            )
        W_pred, g_pred = (out[0], out[2]) if isinstance(out, tuple) and len(out) == 3 else (out[0], None)
        IME_est, ISE_est, ITE_est = compute_individual_effects(W_pred, A, outcome_mode, g_self=g_pred)
        result_train = evaluate(IME, ISE, ITE, IME_est, ISE_est, ITE_est, printing=True)

    if run_test_eval:
        with torch.no_grad():
            out_test = fr_a.model.predict(
                torch.tensor(X_test, dtype=torch.float32),
                nbrs_idx_test,
                torch.tensor(t_test, dtype=torch.float32),
            )
        W_pred_test, g_pred_test = (
            (out_test[0], out_test[2]) if isinstance(out_test, tuple) and len(out_test) == 3 else (out_test[0], None)
        )
        IME_est_test, ISE_est_test, ITE_est_test = compute_individual_effects(
            W_pred_test, A_test, outcome_mode, g_self=g_pred_test
        )
        result_test = evaluate(IME_test, ISE_test, ITE_test, IME_est_test, ISE_est_test, ITE_est_test, printing=True)

    # ---- Bootstrap attention ----
    eval_mask_cfg_for_bootstrap = eval_mask_cfg_effective
    
    primary_method_name = "ij" if use_ij else "exact"
    shared_bootstrap_multipliers = _draw_bootstrap_multipliers(B=B, n=data.X.shape[0], multiplier_dist=multiplier_dist)
    t_boot_primary = time.time()
    boot_res = bootstrap_attention(
        data=data,
        train_cfg=train_cfg_attn,
        attn_cfg=attn_cfg,
        attn_true=W_true_fit,
        attn_true_self=self_effect_true_fit,
        B=B,
        alpha=alpha_ci,
        multiplier_dist=multiplier_dist,
        eval_mask_cfg=eval_mask_cfg_for_bootstrap,
        X_test=X_test,
        nbrs_idx_test=nbrs_idx_test,
        t_test=t_test,
        A_test=A_test,
        attn_true_test=W_true_test,
        eval_mask_cfg_test=eval_mask_cfg_for_bootstrap,
        val_data=val_data,
        base_result=fr_a,
        bootstrap_multipliers=shared_bootstrap_multipliers,
        use_ij=use_ij,
        ij_damping=ij_damping,
        ij_cg_iters=ij_cg_iters,
        ij_cg_tol=ij_cg_tol,
        ij_solver=ij_solver,
        ij_lanczos_m=ij_lanczos_m,
        ij_lanczos_k=ij_lanczos_k,
        ij_lanczos_reorth=ij_lanczos_reorth,
    )
    boot_primary_time = time.time() - t_boot_primary

    bootstrap_compare: Dict[str, Any] = {
        "primary_method": primary_method_name,
        "methods": {
            primary_method_name: {
                "runtime_seconds": float(boot_primary_time),
                **_bootstrap_method_summary(boot_res, alpha=alpha_ci),
            }
        },
    }

    if run_both_use_ij:
        alt_use_ij = not use_ij
        alt_method_name = "ij" if alt_use_ij else "exact"
        if train_cfg_attn.verbose:
            print(f"[bootstrap_compare] Running secondary bootstrap method: {alt_method_name}")
        t_boot_alt = time.time()
        boot_res_alt = bootstrap_attention(
            data=data,
            train_cfg=train_cfg_attn,
            attn_cfg=attn_cfg,
            attn_true=W_true_fit,
            attn_true_self=self_effect_true_fit,
            B=B,
            alpha=alpha_ci,
            multiplier_dist=multiplier_dist,
            eval_mask_cfg=eval_mask_cfg_for_bootstrap,
            X_test=X_test,
            nbrs_idx_test=nbrs_idx_test,
            t_test=t_test,
            A_test=A_test,
            attn_true_test=W_true_test,
            eval_mask_cfg_test=eval_mask_cfg_for_bootstrap,
            val_data=val_data,
            base_result=fr_a,
            bootstrap_multipliers=shared_bootstrap_multipliers,
            use_ij=alt_use_ij,
            ij_damping=ij_damping,
            ij_cg_iters=ij_cg_iters,
            ij_cg_tol=ij_cg_tol,
            ij_solver=ij_solver,
            ij_lanczos_m=ij_lanczos_m,
            ij_lanczos_k=ij_lanczos_k,
            ij_lanczos_reorth=ij_lanczos_reorth,
        )
        boot_alt_time = time.time() - t_boot_alt
        bootstrap_compare["methods"][alt_method_name] = {
            "runtime_seconds": float(boot_alt_time),
            **_bootstrap_method_summary(boot_res_alt, alpha=alpha_ci),
        }
        if train_cfg_attn.verbose:
            p = bootstrap_compare["methods"][primary_method_name]
            q = bootstrap_compare["methods"][alt_method_name]
            print(
                f"[bootstrap_compare] {primary_method_name} vs {alt_method_name}: "
                f"runtime {p['runtime_seconds']:.3f}s vs {q['runtime_seconds']:.3f}s, "
                f"loss_mean {p['loss'].get('boot_weighted_loss_mean')} vs {q['loss'].get('boot_weighted_loss_mean')}"
            )

            # Compare per-replicate deviations ||boot - base|| (Frobenius) for attention.
            # Since we pass shared bootstrap multipliers, replicate b aligns across methods.
            def _fro_norms(boot: np.ndarray, base: np.ndarray, mask: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
                diffs = boot - base[None, :, :]  # [B, n, n]
                norms_all = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
                norms_mask = None
                if mask is not None and mask.any():
                    norms_mask = np.linalg.norm(diffs[:, mask], axis=1)
                return norms_all, norms_mask

            # Identify which object is ij vs exact for printing.
            if primary_method_name == "ij":
                boot_ij, boot_exact = boot_res, boot_res_alt
            else:
                boot_ij, boot_exact = boot_res_alt, boot_res

            n_ij_all, n_ij_mask = _fro_norms(boot_ij.boot_attn, boot_ij.base_attn, boot_ij.eval_mask)
            n_ex_all, n_ex_mask = _fro_norms(boot_exact.boot_attn, boot_exact.base_attn, boot_exact.eval_mask)

            def _summ(x: np.ndarray) -> str:
                return f"mean={float(np.mean(x)):.6g}, med={float(np.median(x)):.6g}, max={float(np.max(x)):.6g}"

            print(f"[bootstrap_compare] ||boot-base||_F (all entries): ij({_summ(n_ij_all)}), exact({_summ(n_ex_all)})")
            print(
                f"[bootstrap_compare] ratio ij/exact (all) mean={float(np.mean(n_ij_all / np.maximum(n_ex_all, 1e-12))):.6g}, "
                f"med={float(np.median(n_ij_all / np.maximum(n_ex_all, 1e-12))):.6g}"
            )

            if n_ij_mask is not None and n_ex_mask is not None:
                print(
                    f"[bootstrap_compare] ||boot-base||_2 (eval_mask entries): ij({_summ(n_ij_mask)}), exact({_summ(n_ex_mask)})"
                )
                print(
                    f"[bootstrap_compare] ratio ij/exact (mask) mean={float(np.mean(n_ij_mask / np.maximum(n_ex_mask, 1e-12))):.6g}, "
                    f"med={float(np.median(n_ij_mask / np.maximum(n_ex_mask, 1e-12))):.6g}"
                )

            # Print a few aligned replicates for sanity.
            show_k = min(5, int(n_ij_all.shape[0]))
            for i in range(show_k):
                print(
                    f"[bootstrap_compare] b={i+1}: ||ij-base||={float(n_ij_all[i]):.6g}, ||exact-base||={float(n_ex_all[i]):.6g}"
                )

    if ij_diag:
        print("[ij_diag] Running small exact-vs-IJ bootstrap diagnostic...")
        diag_data, diag_W_true, diag_self_true = _subset_graph_for_diag(
            data=data,
            attn_true=W_true_fit,
            attn_true_self=self_effect_true_fit,
            max_n=200,
        )
        diag_n = diag_data.X.shape[0]
        diag_train_cfg = TrainConfig(
            epochs=min(60, train_cfg_attn.epochs),
            lr=train_cfg_attn.lr,
            weight_decay=train_cfg_attn.weight_decay,
            batch_size=min(diag_n, train_cfg_attn.batch_size),
            patience=min(6, train_cfg_attn.patience),
            device=train_cfg_attn.device,
            seed=train_cfg_attn.seed,
            verbose=False,
            log_every=train_cfg_attn.log_every,
        )
        diag_B = 20

        t0 = time.time()
        diag_exact = bootstrap_attention(
            data=diag_data,
            train_cfg=diag_train_cfg,
            attn_cfg=attn_cfg,
            attn_true=diag_W_true,
            attn_true_self=diag_self_true,
            B=diag_B,
            alpha=alpha_ci,
            multiplier_dist=multiplier_dist,
            eval_mask_cfg=eval_mask_cfg_for_bootstrap,
            val_data=None,
            use_ij=False,
        )
        exact_time = time.time() - t0

        t1 = time.time()
        diag_ij = bootstrap_attention(
            data=diag_data,
            train_cfg=diag_train_cfg,
            attn_cfg=attn_cfg,
            attn_true=diag_W_true,
            attn_true_self=diag_self_true,
            B=diag_B,
            alpha=alpha_ci,
            multiplier_dist=multiplier_dist,
            eval_mask_cfg=eval_mask_cfg_for_bootstrap,
            val_data=None,
            use_ij=True,
            ij_damping=ij_damping,
            ij_cg_iters=ij_cg_iters,
            ij_cg_tol=ij_cg_tol,
            ij_solver=ij_solver,
            ij_lanczos_m=ij_lanczos_m,
            ij_lanczos_k=ij_lanczos_k,
            ij_lanczos_reorth=ij_lanczos_reorth,
        )
        ij_time = time.time() - t1

        exact_rad = _radius_summary(diag_exact, alpha=alpha_ci)
        ij_rad = _radius_summary(diag_ij, alpha=alpha_ci)

        print(
            f"[ij_diag] n={diag_n}, B={diag_B}, exact_time={exact_time:.3f}s, "
            f"ij_time={ij_time:.3f}s, speedup={exact_time / max(ij_time, 1e-12):.2f}x"
        )
        print(
            "[ij_diag] pointwise_mean "
            f"exact={exact_rad['pointwise_mean']:.6g}, ij={ij_rad['pointwise_mean']:.6g}, "
            f"ratio={ij_rad['pointwise_mean'] / max(exact_rad['pointwise_mean'], 1e-12):.4f}"
        )
        print(
            "[ij_diag] c_uniform "
            f"exact={exact_rad['c_uniform']:.6g}, ij={ij_rad['c_uniform']:.6g}, "
            f"ratio={ij_rad['c_uniform'] / max(exact_rad['c_uniform'], 1e-12):.4f}"
        )

        if ij_solver == "lanczos":
            diag_base_model = diag_ij.base_result.model
            diag_X_used = diag_data.X[:, :1] if attn_cfg.low_dimension else diag_data.X
            diag_x_t = torch.tensor(diag_X_used, dtype=torch.float32, device=train_cfg_attn.device)
            diag_t_t = torch.tensor(diag_data.t, dtype=torch.float32, device=train_cfg_attn.device)
            diag_e_t = torch.tensor(diag_data.e_hat, dtype=torch.float32, device=train_cfg_attn.device)
            diag_y_resid_t = torch.tensor(
                diag_data.y - diag_data.mu_hat, dtype=torch.float32, device=train_cfg_attn.device
            )
            lanczos_vs_cg_diagnostic(
                model_hat=diag_base_model,
                x=diag_x_t,
                nbrs_idx=diag_data.nbrs_idx,
                t_t=diag_t_t,
                e_t=diag_e_t,
                y_resid=diag_y_resid_t,
                separate_self=attn_cfg.separate_self,
                damping=ij_damping,
                cg_iters=ij_cg_iters,
                cg_tol=ij_cg_tol,
                lanczos_m=ij_lanczos_m,
                lanczos_k=ij_lanczos_k,
                lanczos_reorth=ij_lanczos_reorth,
            )

    def _log_eval_mask(name: str, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            print(f"[eval_mask] {name}: None (uniform band uses all entries; coverage defaults to truth!=0).")
            return
        total = int(mask.sum())
        per_row = mask.sum(axis=1)
        print(
            f"[eval_mask] {name}: mode=edges (exclude_diag=True), selected={total} entries, "
            f"per-row min/mean/max = {per_row.min()}/{per_row.mean():.2f}/{per_row.max()}"
        )

    # Log eval_mask from bootstrap
    if eval_mask_cfg_for_bootstrap is not None:
        _log_eval_mask("train", boot_res.eval_mask)
        _log_eval_mask("test", boot_res.eval_mask_test)

    # ---- Coverage statistics for attention and raw scores over bootstrap prefixes ----
    def compute_coverage_over_prefixes(
        base: np.ndarray,
        boot: np.ndarray,
        truth: np.ndarray,
        alpha: float,
        mask: Optional[np.ndarray] = None,
        store_uncovered: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute coverage statistics for pointwise and uniform CIs
        using bootstrap prefixes of size k = step, 2*step, ..., B.

        Notes
        -----
        - "Fraction covered" is the mean coverage over masked entries.
        - "All covered" is a 0/1 indicator of *simultaneous* coverage over the mask,
          matching the MultipleBootstrap-style "uniform coverage" event.
        - Uniform band is built using a studentized sup-statistic:
            M_b = sup_{(i,j) in mask} |boot_b[i,j] - base[i,j]| / se[i,j]
        """

        def _compute_studentized_bands(
            base_arr: np.ndarray,
            diffs_arr: np.ndarray,
            alpha_level: float,
            eval_mask: Optional[np.ndarray],
        ) -> Dict[str, Any]:
            """
            Build studentized pointwise and uniform (simultaneous) bands from diffs = boot - base.

            Inputs
            ------
            base_arr: [n, n]
            diffs_arr: [k, n, n]
            alpha_level: float
            eval_mask: optional boolean [n, n]; if provided, uniform statistic is taken over mask
            """
            k_local = diffs_arr.shape[0]
            ddof = 1 if k_local > 1 else 0

            # Per-entry bootstrap standard error (sigma-hat analogue)
            se = diffs_arr.std(axis=0, ddof=ddof) + 1e-12  # [n, n]

            # Studentized pointwise critical value per entry (two-sided via |T|)
            t_stat = np.abs(diffs_arr) / se[None, :, :]  # [k, n, n]
            z_pw = np.quantile(t_stat, 1.0 - alpha_level, axis=0)  # [n, n]
            radius_pw = z_pw * se
            ci_lo_pw = base_arr - radius_pw
            ci_hi_pw = base_arr + radius_pw

            # Studentized uniform critical value (single scalar)
            if eval_mask is None:
                Mb = np.max(t_stat.reshape(k_local, -1), axis=1)  # [k]
            else:
                mask_flat = eval_mask.reshape(-1)
                Mb = np.max(t_stat.reshape(k_local, -1)[:, mask_flat], axis=1)  # [k]
            c_uniform = float(np.quantile(Mb, 1.0 - alpha_level))
            radius_unif = c_uniform * se
            ci_lo_unif = base_arr - radius_unif
            ci_hi_unif = base_arr + radius_unif

            # Helpful diagnostics
            se_summary = {
                "min": float(np.min(se)),
                "median": float(np.median(se)),
                "max": float(np.max(se)),
            }
            z_pw_summary = {
                "min": float(np.min(z_pw)),
                "median": float(np.median(z_pw)),
                "max": float(np.max(z_pw)),
            }

            return {
                "se": se,
                "se_summary": se_summary,
                "z_pw_summary": z_pw_summary,
                "pointwise": {"lo": ci_lo_pw, "hi": ci_hi_pw},
                "uniform": {"lo": ci_lo_unif, "hi": ci_hi_unif, "c_uniform": c_uniform},
            }

        def _coverage_stats(
            lo: np.ndarray,
            hi: np.ndarray,
            target: np.ndarray,
            eval_mask: np.ndarray,
        ) -> Dict[str, Any]:
            within = (target >= lo) & (target <= hi)
            within_m = within[eval_mask]
            n_mask = int(within_m.size)
            covered_count = int(within_m.sum())
            fraction_covered = covered_count / n_mask if n_mask > 0 else float("nan")
            all_covered = int(bool(within_m.all())) if n_mask > 0 else 0
            return {
                "fraction_covered": fraction_covered,
                "all_covered": all_covered,
                "covered_count": covered_count,
                "n_mask": n_mask,
            }

        def _length_summary(
            lo: np.ndarray,
            hi: np.ndarray,
            eval_mask: np.ndarray,
        ) -> Dict[str, float]:
            length = hi - lo
            vals = length[eval_mask]
            if vals.size == 0:
                return {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "q25": float("nan"),
                    "q75": float("nan"),
                    "q90": float("nan"),
                    "max": float("nan"),
                }
            return {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
                "max": float(np.max(vals)),
            }

        B_total = boot.shape[0]
        step = max(1, B_total // 10)
        prefixes = list(range(step, B_total + 1, step))
        if prefixes[-1] != B_total:
            prefixes.append(B_total)

        # Default mask is the historical behavior: only evaluate on nonzero truth entries.
        eval_mask = (truth != 0) if mask is None else mask
        n_mask_default = int(eval_mask.sum())

        results_by_prefix: Dict[str, Any] = {}

        for k in prefixes:
            diffs = boot[:k] - base[None, :, :]  # [k, n, n]

            bands = _compute_studentized_bands(base, diffs, alpha, eval_mask)
            pw = _coverage_stats(
                lo=bands["pointwise"]["lo"],
                hi=bands["pointwise"]["hi"],
                target=truth,
                eval_mask=eval_mask,
            )
            unif = _coverage_stats(
                lo=bands["uniform"]["lo"],
                hi=bands["uniform"]["hi"],
                target=truth,
                eval_mask=eval_mask,
            )
            length_summary = {
                "pointwise": _length_summary(
                    bands["pointwise"]["lo"],
                    bands["pointwise"]["hi"],
                    eval_mask,
                ),
                "uniform": _length_summary(
                    bands["uniform"]["lo"],
                    bands["uniform"]["hi"],
                    eval_mask,
                ),
            }

            # NOTE: We intentionally do not store uncovered indices/details in JSON
            # to avoid large output files. Coverage aggregates are still preserved.

            results_by_prefix[str(k)] = {
                "B_prefix": k,
                # Backwards-compatible counters
                "n_nonzero": n_mask_default,
                "n_mask": n_mask_default,
                # New clear names (top-level convenience)
                "pointwise_fraction_covered": pw["fraction_covered"],
                "uniform_fraction_covered": unif["fraction_covered"],
                "pointwise_all_covered": pw["all_covered"],
                "uniform_all_covered": unif["all_covered"],
                # Structured schema
                "pointwise": {
                    "fraction_covered": pw["fraction_covered"],
                    "all_covered": pw["all_covered"],
                    "covered_count": pw["covered_count"],
                    "n_mask": pw["n_mask"],
                    # Backwards compatibility (old key names)
                    "coverage": pw["fraction_covered"],
                    "covered": pw["covered_count"],
                },
                "uniform": {
                    "fraction_covered": unif["fraction_covered"],
                    "all_covered": unif["all_covered"],
                    "covered_count": unif["covered_count"],
                    "n_mask": unif["n_mask"],
                    "c_uniform": bands["uniform"]["c_uniform"],
                    # Backwards compatibility (old key names)
                    "coverage": unif["fraction_covered"],
                    "covered": unif["covered_count"],
                },
                "se_summary": bands["se_summary"],
                "z_pw_summary": bands["z_pw_summary"],
                "length_summary": length_summary,
            }

        return {
            "B_total": B_total,
            "alpha": alpha,
            # Backwards compatibility
            "n_nonzero": n_mask_default,
            # Clear naming
            "n_mask": n_mask_default,
            "by_prefix": results_by_prefix,
        }

    def compute_vector_coverage_over_prefixes(
        base: np.ndarray,
        boot: np.ndarray,
        truth: np.ndarray,
        alpha: float,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Vector analogue of compute_coverage_over_prefixes for per-node targets,
        e.g., individual spillover effects (ISE).
        Shapes:
          base:  [n]
          boot:  [B, n]
          truth: [n]
        """

        def _compute_studentized_vector_bands(
            base_arr: np.ndarray,
            diffs_arr: np.ndarray,
            alpha_level: float,
            eval_mask: Optional[np.ndarray],
        ) -> Dict[str, Any]:
            k_local = diffs_arr.shape[0]
            ddof = 1 if k_local > 1 else 0
            se = diffs_arr.std(axis=0, ddof=ddof) + 1e-12  # [n]
            t_stat = np.abs(diffs_arr) / se[None, :]  # [k, n]
            z_pw = np.quantile(t_stat, 1.0 - alpha_level, axis=0)  # [n]
            radius_pw = z_pw * se
            ci_lo_pw = base_arr - radius_pw
            ci_hi_pw = base_arr + radius_pw

            if eval_mask is None:
                Mb = np.max(t_stat, axis=1)  # [k]
            else:
                Mb = np.max(t_stat[:, eval_mask], axis=1)  # [k]
            c_uniform = float(np.quantile(Mb, 1.0 - alpha_level))
            radius_unif = c_uniform * se
            ci_lo_unif = base_arr - radius_unif
            ci_hi_unif = base_arr + radius_unif
            return {
                "pointwise": {"lo": ci_lo_pw, "hi": ci_hi_pw},
                "uniform": {"lo": ci_lo_unif, "hi": ci_hi_unif, "c_uniform": c_uniform},
                "se_summary": {
                    "min": float(np.min(se)),
                    "median": float(np.median(se)),
                    "max": float(np.max(se)),
                },
                "z_pw_summary": {
                    "min": float(np.min(z_pw)),
                    "median": float(np.median(z_pw)),
                    "max": float(np.max(z_pw)),
                },
            }

        def _coverage_stats_vector(
            lo: np.ndarray,
            hi: np.ndarray,
            target: np.ndarray,
            eval_mask: np.ndarray,
        ) -> Dict[str, Any]:
            within = (target >= lo) & (target <= hi)
            within_m = within[eval_mask]
            n_mask = int(within_m.size)
            covered_count = int(within_m.sum())
            fraction_covered = covered_count / n_mask if n_mask > 0 else float("nan")
            all_covered = int(bool(within_m.all())) if n_mask > 0 else 0
            return {
                "fraction_covered": fraction_covered,
                "all_covered": all_covered,
                "covered_count": covered_count,
                "n_mask": n_mask,
            }

        def _length_summary_vector(
            lo: np.ndarray,
            hi: np.ndarray,
            eval_mask: np.ndarray,
        ) -> Dict[str, float]:
            length = hi - lo
            vals = length[eval_mask]
            if vals.size == 0:
                return {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "q25": float("nan"),
                    "q75": float("nan"),
                    "q90": float("nan"),
                    "max": float("nan"),
                }
            return {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
                "max": float(np.max(vals)),
            }

        def _build_node_ci_snapshot(
            truth_arr: np.ndarray,
            ci_lo_pw: np.ndarray,
            ci_hi_pw: np.ndarray,
            ci_lo_unif: np.ndarray,
            ci_hi_unif: np.ndarray,
        ) -> Dict[str, Any]:
            """Select configured percentile nodes by raw truth and extract both CI types."""
            n = truth_arr.shape[0]
            if n == 0:
                return {
                    "selection_rule": (
                        "Nodes selected by configured raw-truth percentile ranks; "
                        "no nodes available."
                    ),
                    "nodes": [],
                }

            sorted_idx = np.argsort(truth_arr, kind="stable")
            def _label_for_percentile(p: float) -> str:
                if np.isclose(p, 0.0):
                    return "smallest"
                if np.isclose(p, 1.0):
                    return "largest"
                pct = (f"{100.0 * p:.6f}").rstrip("0").rstrip(".")
                return f"p{pct}"

            target_ranks = [
                (_label_for_percentile(p), int(round(p * (n - 1))))
                for p in node_ci_snapshot_percentiles
            ]

            def _closest_unused_rank(target_rank: int, used_ranks: set[int]) -> Optional[int]:
                if target_rank not in used_ranks:
                    return target_rank
                for delta in range(1, n):
                    left = target_rank - delta
                    right = target_rank + delta
                    if left >= 0 and left not in used_ranks:
                        return left
                    if right < n and right not in used_ranks:
                        return right
                return None

            nodes = []
            used_ranks: set[int] = set()
            for label, rank in target_ranks:
                chosen_rank = _closest_unused_rank(rank, used_ranks)
                if chosen_rank is None:
                    continue
                used_ranks.add(chosen_rank)
                node_idx = int(sorted_idx[chosen_rank])
                nodes.append(
                    {
                        "label": label,
                        "node_index": node_idx,
                        "true_spillover": float(truth_arr[node_idx]),
                        "pointwise_ci_lower": float(ci_lo_pw[node_idx]),
                        "pointwise_ci_upper": float(ci_hi_pw[node_idx]),
                        "uniform_ci_lower": float(ci_lo_unif[node_idx]),
                        "uniform_ci_upper": float(ci_hi_unif[node_idx]),
                    }
                )

            selection_rule = (
                "Nodes selected by raw (signed) truth spillover at configured percentiles "
                f"{node_ci_snapshot_percentiles} with nearest-available-rank tie handling; "
                f"{len(nodes)} node(s) returned."
            )
            return {
                "selection_rule": selection_rule,
                "nodes": nodes,
            }

        B_total = boot.shape[0]
        step = max(1, B_total // 10)
        prefixes = list(range(step, B_total + 1, step))
        if prefixes[-1] != B_total:
            prefixes.append(B_total)

        eval_mask = (truth != 0) if mask is None else mask
        n_mask_default = int(eval_mask.sum())
        results_by_prefix: Dict[str, Any] = {}

        for k in prefixes:
            diffs = boot[:k] - base[None, :]  # [k, n]
            bands = _compute_studentized_vector_bands(base, diffs, alpha, eval_mask)
            pw = _coverage_stats_vector(
                lo=bands["pointwise"]["lo"],
                hi=bands["pointwise"]["hi"],
                target=truth,
                eval_mask=eval_mask,
            )
            unif = _coverage_stats_vector(
                lo=bands["uniform"]["lo"],
                hi=bands["uniform"]["hi"],
                target=truth,
                eval_mask=eval_mask,
            )
            length_summary = {
                "pointwise": _length_summary_vector(
                    bands["pointwise"]["lo"],
                    bands["pointwise"]["hi"],
                    eval_mask,
                ),
                "uniform": _length_summary_vector(
                    bands["uniform"]["lo"],
                    bands["uniform"]["hi"],
                    eval_mask,
                ),
            }
            entry = {
                "B_prefix": k,
                "n_nonzero": n_mask_default,
                "n_mask": n_mask_default,
                "pointwise_fraction_covered": pw["fraction_covered"],
                "uniform_fraction_covered": unif["fraction_covered"],
                "pointwise_all_covered": pw["all_covered"],
                "uniform_all_covered": unif["all_covered"],
                "pointwise": {
                    "fraction_covered": pw["fraction_covered"],
                    "all_covered": pw["all_covered"],
                    "covered_count": pw["covered_count"],
                    "n_mask": pw["n_mask"],
                    "coverage": pw["fraction_covered"],
                    "covered": pw["covered_count"],
                },
                "uniform": {
                    "fraction_covered": unif["fraction_covered"],
                    "all_covered": unif["all_covered"],
                    "covered_count": unif["covered_count"],
                    "n_mask": unif["n_mask"],
                    "c_uniform": bands["uniform"]["c_uniform"],
                    "coverage": unif["fraction_covered"],
                    "covered": unif["covered_count"],
                },
                "se_summary": bands["se_summary"],
                "z_pw_summary": bands["z_pw_summary"],
                "length_summary": length_summary,
            }
            if node_ci_snapshot_enabled and k == B_total:
                entry["node_ci_snapshot"] = _build_node_ci_snapshot(
                    truth_arr=truth,
                    ci_lo_pw=bands["pointwise"]["lo"],
                    ci_hi_pw=bands["pointwise"]["hi"],
                    ci_lo_unif=bands["uniform"]["lo"],
                    ci_hi_unif=bands["uniform"]["hi"],
                )
            results_by_prefix[str(k)] = entry

        return {
            "B_total": B_total,
            "alpha": alpha,
            "n_nonzero": n_mask_default,
            "n_mask": n_mask_default,
            "by_prefix": results_by_prefix,
        }

    # ---- Build per-node individual spillover effects from attention draws ----
    # ISE_i = sum_j W_ij * (A_ij - 1{i=j}) (matches metric.compute_individual_effects)
    A_offdiag_train = data.A - np.eye(data.A.shape[0], dtype=data.A.dtype)
    ise_base_train = np.sum(boot_res.base_attn * A_offdiag_train, axis=1)
    ise_boot_train = np.sum(boot_res.boot_attn * A_offdiag_train[None, :, :], axis=2)

    ise_true_test: Optional[np.ndarray] = None
    ise_base_test: Optional[np.ndarray] = None
    ise_boot_test: Optional[np.ndarray] = None
    if (
        A_test is not None
        and boot_res.base_attn_test is not None
        and boot_res.boot_attn_test is not None
    ):
        A_offdiag_test = A_test - np.eye(A_test.shape[0], dtype=A_test.dtype)
        ise_true_test = ISE_test
        ise_base_test = np.sum(boot_res.base_attn_test * A_offdiag_test, axis=1)
        ise_boot_test = np.sum(boot_res.boot_attn_test * A_offdiag_test[None, :, :], axis=2)

    # ---- Compute coverage for each alpha (single fixed eval mask mode) ----
    eval_mask_train = boot_res.eval_mask
    eval_mask_test = boot_res.eval_mask_test

    coverage_by_alpha_k: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    # Compute coverage for all alpha values with a single key "default"
    for alpha_val in alpha_values:
        alpha_str = (f"{alpha_val:.6f}".rstrip("0").rstrip(".")).replace(".", "p")
        alpha_key = f"alpha_{alpha_str}"
        coverage_by_alpha_k[alpha_key] = {}

        if alpha_val == alpha_values[0] and eval_mask_cfg_effective is not None:
            if run_train_eval:
                _log_eval_mask("train", eval_mask_train)
            if run_test_eval and eval_mask_test is not None:
                _log_eval_mask("test", eval_mask_test)

        raw_coverage_train: Optional[Dict[str, Any]] = None
        ise_coverage_train: Optional[Dict[str, Any]] = None
        if run_train_eval:
            raw_coverage_train = compute_coverage_over_prefixes(
                base=boot_res.base_raw,
                boot=boot_res.boot_raw,
                truth=raw_scores_fit,
                alpha=alpha_val,
                mask=eval_mask_train,
                store_uncovered=True,
            )
            ise_coverage_train = compute_vector_coverage_over_prefixes(
                base=ise_base_train,
                boot=ise_boot_train,
                truth=ise_true_train,
                alpha=alpha_val,
                mask=None,
            )

        # Test coverage
        raw_coverage_test: Optional[Dict[str, Any]] = None
        ise_coverage_test: Optional[Dict[str, Any]] = None
        if run_test_eval and (
            boot_res.base_attn_test is not None
            and boot_res.boot_attn_test is not None
            and boot_res.base_raw_test is not None
            and boot_res.boot_raw_test is not None
        ):
            raw_coverage_test = compute_coverage_over_prefixes(
                base=boot_res.base_raw_test,
                boot=boot_res.boot_raw_test,
                truth=raw_scores_test,
                alpha=alpha_val,
                mask=eval_mask_test,
                store_uncovered=True,
            )
            if (
                ise_base_test is not None
                and ise_boot_test is not None
                and ise_true_test is not None
            ):
                ise_coverage_test = compute_vector_coverage_over_prefixes(
                    base=ise_base_test,
                    boot=ise_boot_test,
                    truth=ise_true_test,
                    alpha=alpha_val,
                    mask=None,
                )

        split_block: Dict[str, Any] = {}
        if run_train_eval:
            split_block["train"] = {
                "raw_scores": raw_coverage_train,
                "individual_spillover": ise_coverage_train,
            }
        if run_test_eval:
            split_block["test"] = {
                "raw_scores": raw_coverage_test,
                "individual_spillover": ise_coverage_test,
            }
        coverage_by_alpha_k[alpha_key]["default"] = {
            "alpha": alpha_val,
            "k": None,
            **split_block,
        }
    
    # For backwards compatibility, store first alpha × first k coverage
    first_alpha_key = list(coverage_by_alpha_k.keys())[0]
    first_k_key_in_alpha = list(coverage_by_alpha_k[first_alpha_key].keys())[0]
    first_coverage = coverage_by_alpha_k[first_alpha_key][first_k_key_in_alpha]
    raw_coverage_train_first: Optional[Dict[str, Any]] = None
    raw_coverage_test_first: Optional[Dict[str, Any]] = None
    ise_coverage_train_first: Optional[Dict[str, Any]] = None
    ise_coverage_test_first: Optional[Dict[str, Any]] = None
    if "train" in first_coverage:
        raw_coverage_train_first = first_coverage["train"]["raw_scores"]
        ise_coverage_train_first = first_coverage["train"]["individual_spillover"]
    if "test" in first_coverage:
        raw_coverage_test_first = first_coverage["test"]["raw_scores"]
        ise_coverage_test_first = first_coverage["test"]["individual_spillover"]

    # ---- Save scalar metrics (train/test causal effect evaluation + coverage statistics) ----
    nuis_suffix = "fitted" if fit_nuisance else "oracle"
    # Optional override to control filename prefix; defaults to legacy dataset_label + tag
    filename_prefix = out_cfg.get("filename_prefix")
    if filename_prefix is None:
        dataset_label = f"SBM_n{n_train}_K{num_communities_train}"
        filename_prefix = f"{dataset_label}_{tag}"
    # Include alpha in filenames for clarity/reproducibility (e.g., a0p05 for alpha=0.05)
    alpha_str = (f"{alpha_ci:.6f}".rstrip("0").rstrip(".")).replace(".", "p")
    tag_with_nuis_and_B = f"{filename_prefix}_nuis-{nuis_suffix}_B{B}_a{alpha_str}_m{bootstrap_method}"

    # Effective configs that reflect overrides
    effective_exp_cfg = dict(exp_cfg)
    # Preserve legacy key for downstream notebooks; map to train split size.
    effective_exp_cfg["n"] = n_train
    # Preserve legacy keys for downstream notebooks; these map to train split.
    effective_exp_cfg["num_communities"] = num_communities_train
    effective_exp_cfg["p"] = p_train
    effective_exp_cfg["q"] = q_train
    effective_exp_cfg["sbm_splits"] = {
        "train": {
            "n": n_train,
            "num_communities": num_communities_train,
            "p": p_train,
            "q": q_train,
        },
        "eval": {
            "n": n_eval_cfg,
            "num_communities": num_communities_eval,
            "p": p_eval,
            "q": q_eval,
        },
    }
    effective_exp_cfg["train_fraction"] = train_fraction
    effective_exp_cfg["train_nodes_total"] = train_nodes_total
    effective_exp_cfg["train_nodes_used"] = train_nodes_used
    effective_exp_cfg["train_subset_seed"] = graph_seed
    effective_exp_cfg["train_subset_seed_source"] = "experiment.graph_seed"
    effective_exp_cfg["graph_seed"] = graph_seed

    effective_boot_cfg = dict(boot_cfg)
    effective_boot_cfg["B"] = B
    # Store alpha as list if multiple values, single value otherwise
    effective_boot_cfg["alpha"] = alpha_values if len(alpha_values) > 1 else alpha_values[0]
    effective_boot_cfg["multiplier_dist"] = multiplier_dist
    effective_boot_cfg["bootstrap_method"] = bootstrap_method
    effective_boot_cfg["use_ij"] = use_ij
    effective_boot_cfg["ij_damping"] = ij_damping
    effective_boot_cfg["ij_cg_iters"] = ij_cg_iters
    effective_boot_cfg["ij_cg_tol"] = ij_cg_tol
    effective_boot_cfg["ij_solver"] = ij_solver
    effective_boot_cfg["ij_lanczos_m"] = ij_lanczos_m
    effective_boot_cfg["ij_lanczos_k"] = ij_lanczos_k
    effective_boot_cfg["ij_lanczos_reorth"] = ij_lanczos_reorth
    effective_boot_cfg["ij_diag"] = ij_diag
    effective_boot_cfg["run_both_use_ij"] = run_both_use_ij
    effective_boot_cfg["eval_scope"] = eval_scope
    effective_boot_cfg["node_ci_snapshot"] = {
        "enabled": node_ci_snapshot_enabled,
        "percentiles": node_ci_snapshot_percentiles,
    }
    if eval_mask_cfg_effective is not None:
        effective_boot_cfg["eval_mask"] = eval_mask_cfg_effective

    effective_nuis_cfg = dict(nuis_cfg)
    effective_nuis_cfg["fit_nuisance"] = fit_nuisance

    # Record effective train_attn config (including seed override, if any)
    effective_train_attn_cfg = dict(train_attn_cfg)
    effective_train_attn_cfg["seed"] = seed

    # Build metrics dictionary
    # Structure: coverage_by_alpha_k contains all alpha × k combinations
    metrics = {
        "train_cfg_attn": effective_train_attn_cfg,
        "attn_cfg": attn_cfg_dict,
        "bootstrap": effective_boot_cfg,
        "nuisance": effective_nuis_cfg,
        "experiment": effective_exp_cfg,
        "coverage_by_alpha_k": coverage_by_alpha_k,  # Coverage statistics for each alpha × k combination
        "bootstrap_compare": bootstrap_compare,
    }
    if run_train_eval:
        metrics["train"] = result_train
    if run_test_eval:
        metrics["test"] = result_test
    
    # Only include redundant "coverage" field for backwards compatibility when single alpha.
    if len(alpha_values) == 1:
        coverage_block: Dict[str, Any] = {}
        if run_train_eval:
            coverage_block["train"] = {
                "raw_scores": raw_coverage_train_first,
                "individual_spillover": ise_coverage_train_first,
            }
        if run_test_eval:
            coverage_block["test"] = {
                "raw_scores": raw_coverage_test_first,
                "individual_spillover": ise_coverage_test_first,
            }
        metrics["coverage"] = coverage_block
    # Avoid overwriting existing outputs: append _{run_idx} where run_idx starts at 1.
    base_metrics_path = os.path.join(output_dir, f"metrics_{tag_with_nuis_and_B}.json")
    run_idx = 1
    metrics_path = base_metrics_path.replace(".json", f"_{run_idx}.json")
    while os.path.exists(metrics_path):
        run_idx += 1
        metrics_path = base_metrics_path.replace(".json", f"_{run_idx}.json")
    metrics["output_run_idx"] = run_idx
    metrics["output_path"] = metrics_path
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
        "--alpha",
        type=float,
        default=None,
        help="Override bootstrap.alpha from config (e.g., 0.05 for 95%% intervals).",
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
        choices=["normal", "rademacher", "poisson"],
        default=None,
        help="Override bootstrap.multiplier_dist in config (normal/rademacher/poisson).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override train_attn.seed from config (random seed for SBM generation + numpy/torch).",
    )
    parser.add_argument(
        "--ij_damping",
        type=float,
        default=None,
        help="Override bootstrap.ij_damping from config.",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=None,
        help="Override experiment.train_fraction from config (0 < value <= 1).",
    )
    parser.add_argument(
        "--bootstrap_method",
        type=str,
        choices=["exact", "ij", "both"],
        default=None,
        help="Bootstrap method: exact | ij | both (paired comparison).",
    )
    parser.add_argument(
        "--ij_diag",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Run print-only small exact-vs-IJ diagnostic (default false).",
    )
    args = parser.parse_args()
    fit_nuisance_override = None
    if args.fit_nuisance is not None:
        fit_nuisance_override = args.fit_nuisance.lower() == "true"
    ij_diag_override = None
    if args.ij_diag is not None:
        ij_diag_override = args.ij_diag.lower() == "true"
    main(
        args.config,
        args.B,
        alpha_override=args.alpha,
        fit_nuisance_override=fit_nuisance_override,
        multiplier_dist_override=args.multiplier_dist,
        seed_override=args.seed,
        train_fraction_override=args.train_fraction,
        ij_damping_override=args.ij_damping,
        bootstrap_method_override=args.bootstrap_method,
        ij_diag_override=ij_diag_override,
    )


