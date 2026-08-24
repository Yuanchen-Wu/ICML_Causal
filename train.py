from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time
from collections import OrderedDict

try:
    from torch.func import functional_call, jvp
except ImportError:  # pragma: no cover - fallback for older torch
    functional_call = None
    jvp = None

from utils import to_torch
from model.propensity import PropensityPredictor
from model.mean import MeanPredictor
from model.interference import (
    GCNWithAttentionOneHead,
    GCNWithAttentionTwoHead,
)


@dataclass
class GraphData:
    X: np.ndarray
    y: np.ndarray
    A: np.ndarray
    nbrs_idx: Sequence[np.ndarray]
    t: np.ndarray
    fold_assignments: np.ndarray
    e_hat: Optional[np.ndarray] = None
    mu_hat: Optional[np.ndarray] = None


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    patience: int = 12
    device: str = "cpu"
    seed: int = 41
    verbose: bool = True
    log_every: int = 1


@dataclass
class PropensityConfig:
    hidden_dim: int = 128
    dropout: float = 0.0


@dataclass
class MeanConfig:
    hidden_dim: int = 128
    dropout: float = 0.0
    agg_method: str = 'mean'  # 'mean' or 'gat'
    use_flexible_attention: bool = False
    low_dimension: bool = False


@dataclass
class AttentionConfig:
    hidden_dim: int = 128
    attn_temperature: float = 1.0
    separate_self: bool = False
    low_dimension: bool = False


@dataclass
class ValidationData:
    """Validation graph data used only for early stopping in fit_attention."""
    X: np.ndarray
    nbrs_idx: Sequence[np.ndarray]
    t: np.ndarray
    attn_true: np.ndarray
    attn_true_self: Optional[np.ndarray] = None


@dataclass
class FitResult:
    pred: np.ndarray
    model: nn.Module
    history: Dict[str, list]
    metrics: Dict[str, float]


@dataclass
class BootstrapResult:
    base_result: FitResult
    base_attn: np.ndarray
    boot_attn: np.ndarray  # shape [B, n, n]
    ci_pointwise_lower: np.ndarray
    ci_pointwise_upper: np.ndarray
    ci_uniform_lower: np.ndarray
    ci_uniform_upper: np.ndarray
    # Raw pre-softmax scores (if available from the model's predict)
    base_raw: Optional[np.ndarray] = None          # shape [n, n]
    boot_raw: Optional[np.ndarray] = None          # shape [B, n, n]
    ci_pointwise_lower_raw: Optional[np.ndarray] = None
    ci_pointwise_upper_raw: Optional[np.ndarray] = None
    ci_uniform_lower_raw: Optional[np.ndarray] = None
    ci_uniform_upper_raw: Optional[np.ndarray] = None
    # Optional test-side quantities (if test data is provided to bootstrap_attention)
    base_attn_test: Optional[np.ndarray] = None      # shape [n_test, n_test]
    boot_attn_test: Optional[np.ndarray] = None      # shape [B, n_test, n_test]
    base_raw_test: Optional[np.ndarray] = None       # shape [n_test, n_test]
    boot_raw_test: Optional[np.ndarray] = None       # shape [B, n_test, n_test]
    # Masks actually used for uniform bands / coverage (diagnostics)
    eval_mask: Optional[np.ndarray] = None
    eval_mask_test: Optional[np.ndarray] = None
    # Loss diagnostics on the training residual objective
    base_train_loss: Optional[float] = None
    boot_weighted_loss: Optional[np.ndarray] = None


# IJ / one-step bootstrap approximation:
# For each multiplier draw w, linearize the weighted optimum around the fitted base
# model and solve (H + damping * I) * delta = g_delta with HVP + CG, then update
# theta_new = theta_hat - delta. Enable via bootstrap_attention(..., use_ij=True).
def flatten_params(model: nn.Module) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in model.parameters():
        parts.append(p.detach().reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def set_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(flat[offset : offset + numel].view_as(p))
            offset += numel
    if offset != flat.numel():
        raise ValueError(f"Flat parameter size mismatch: consumed {offset}, provided {flat.numel()}")


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]], params: Sequence[torch.Tensor]) -> torch.Tensor:
    flat_parts: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            flat_parts.append(torch.zeros_like(p, device=p.device).reshape(-1))
        else:
            flat_parts.append(g.reshape(-1))
    if not flat_parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_parts, dim=0)


def hvp(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    grad_1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    grad_flat = _flatten_grads(grad_1, params)
    grad_v = torch.dot(grad_flat, v)
    grad_2 = torch.autograd.grad(grad_v, params, retain_graph=True, allow_unused=True)
    return _flatten_grads(grad_2, params)


def hvp_from_grad_flat(
    grad_flat: torch.Tensor,
    params: Sequence[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    grad_v = torch.dot(grad_flat, v)
    grad_2 = torch.autograd.grad(grad_v, params, retain_graph=True, allow_unused=True)
    return _flatten_grads(grad_2, params)


def cg_solve(
    A_mul: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b - A_mul(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    if torch.sqrt(rs_old).item() <= tol:
        return x
    for _ in range(max_iter):
        Ap = A_mul(p)
        denom = torch.dot(p, Ap)
        if torch.abs(denom).item() < 1e-12:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new).item() <= tol:
            break
        beta = rs_new / (rs_old + 1e-20)
        p = r + beta * p
        rs_old = rs_new
    return x


def lanczos_tridiag(
    H_mul: Callable[[torch.Tensor], torch.Tensor],
    dim_p: int,
    m: int,
    reorth: bool = True,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lanczos iteration to build a Krylov basis and tridiagonal projection of a
    symmetric operator ``H_mul``.

    Args:
        H_mul: matrix-free operator v -> Hv (must be symmetric; should NOT
               include damping).
        dim_p: dimensionality of the parameter space.
        m: number of Lanczos steps (Krylov subspace dimension).
        reorth: if True, perform full reorthogonalization at each step to
                maintain numerical orthogonality (recommended when m is small).
        device: torch device for allocated tensors.

    Returns:
        V: (dim_p, m) orthonormal Krylov basis.
        T: (m, m) symmetric tridiagonal matrix such that T = V^T H V.
    """
    m = min(m, dim_p)
    V = torch.zeros(dim_p, m, device=device, dtype=torch.float32)
    alphas = torch.zeros(m, device=device, dtype=torch.float32)
    betas = torch.zeros(m, device=device, dtype=torch.float32)

    v = torch.randn(dim_p, device=device, dtype=torch.float32)
    v = v / (torch.linalg.norm(v) + 1e-30)
    V[:, 0] = v

    w = H_mul(v)
    alpha_j = torch.dot(w, v)
    alphas[0] = alpha_j
    w = w - alpha_j * v

    for j in range(1, m):
        beta_j = torch.linalg.norm(w)
        betas[j] = beta_j
        if beta_j.item() < 1e-12:
            # Invariant subspace found; truncate.
            V = V[:, :j]
            alphas = alphas[:j]
            betas = betas[:j]
            break
        v_new = w / beta_j
        if reorth:
            # Full reorthogonalization against all previous basis vectors.
            coeffs = V[:, :j].T @ v_new  # (j,)
            v_new = v_new - V[:, :j] @ coeffs
            v_new = v_new / (torch.linalg.norm(v_new) + 1e-30)
        V[:, j] = v_new

        w = H_mul(v_new)
        alpha_j = torch.dot(w, v_new)
        alphas[j] = alpha_j
        w = w - alpha_j * v_new - beta_j * V[:, j - 1]

    actual_m = V.shape[1]
    T = torch.diag(alphas[:actual_m])
    if actual_m > 1:
        T += torch.diag(betas[1:actual_m], diagonal=1)
        T += torch.diag(betas[1:actual_m], diagonal=-1)
    return V, T


def build_lowrank_inverse_operator(
    model_hat: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    separate_self: bool,
    damping: float,
    m: int = 100,
    k: int = 50,
    reorth: bool = True,
    verbose: bool = False,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Dict[str, float]]:
    """Precompute a low-rank spectral approximation of ``(H + damping I)^{-1}``
    using Lanczos, then return a closure that applies the approximate inverse to
    an arbitrary right-hand-side vector.

    The Gauss-Newton Hessian H is approximated by its top-k Ritz pairs from the
    Lanczos decomposition.  For the orthogonal complement of the Ritz subspace
    the inverse simply equals ``1 / damping`` (Woodbury identity).

    Args:
        model_hat: base-fitted model (parameters are NOT mutated).
        x, nbrs_idx, t_t, e_t, y_resid, separate_self: data tensors needed by
            ``gauss_newton_hvp``.
        damping: Tikhonov regularisation added to H.
        m: number of Lanczos steps.
        k: number of top eigenpairs to retain (k <= m).
        reorth: whether to use full reorthogonalization in Lanczos.
        verbose: if True, print timing and spectral diagnostics.

    Returns:
        (solve, info) where ``solve(b)`` returns the approximate solution to
        ``(H + damping I) x = b``, and ``info`` is a dict of diagnostics.
    """
    device = next(model_hat.parameters()).device
    dim_p = sum(p.numel() for p in model_hat.parameters() if p.requires_grad)
    k = min(k, m, dim_p)

    model_hat.zero_grad(set_to_none=True)
    model_hat.train()

    def H_mul(v: torch.Tensor) -> torch.Tensor:
        return gauss_newton_hvp(
            model=model_hat,
            x=x,
            nbrs_idx=nbrs_idx,
            t_t=t_t,
            e_t=e_t,
            y_resid=y_resid,
            separate_self=separate_self,
            v=v,
        )

    t0 = time.time()
    V, T = lanczos_tridiag(H_mul, dim_p=dim_p, m=m, reorth=reorth, device=device)
    actual_m = V.shape[1]

    evals, evecs = torch.linalg.eigh(T)  # ascending order
    top_k = min(k, actual_m)
    lam = evals[-top_k:]       # (top_k,) largest eigenvalues
    S_k = evecs[:, -top_k:]    # (actual_m, top_k)
    Q = V @ S_k                # (dim_p, top_k) Ritz vectors
    build_time = time.time() - t0

    info: Dict[str, float] = {
        "lanczos_build_time": build_time,
        "dim_p": float(dim_p),
        "m_requested": float(m),
        "m_actual": float(actual_m),
        "k": float(top_k),
        "lam_max": float(lam[-1].item()),
        "lam_min": float(lam[0].item()),
        "cond_approx": float((lam[-1] / (lam[0] + 1e-30)).item()),
    }

    if verbose:
        print(
            f"[lanczos] build_time={build_time:.3f}s  dim_p={dim_p}  "
            f"m={actual_m}  k={top_k}  "
            f"lam_range=[{lam[0].item():.4e}, {lam[-1].item():.4e}]  "
            f"cond~={info['cond_approx']:.2f}"
        )

    inv_lam_damped = 1.0 / (lam + damping)  # (top_k,)

    def solve(b: torch.Tensor) -> torch.Tensor:
        proj = Q.T @ b                          # (top_k,)
        b_perp = b - Q @ proj                   # (dim_p,)
        return Q @ (proj * inv_lam_damped) + b_perp / damping

    return solve, info


def lanczos_vs_cg_diagnostic(
    model_hat: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    separate_self: bool,
    damping: float,
    cg_iters: int = 50,
    cg_tol: float = 1e-6,
    lanczos_m: int = 100,
    lanczos_k: int = 50,
    lanczos_reorth: bool = True,
    n_probes: int = 3,
) -> None:
    """Solve a few random RHS vectors with both CG and Lanczos and print
    relative errors.  Intended for use under ``ij_diag=True``."""
    device = next(model_hat.parameters()).device
    dim_p = sum(p.numel() for p in model_hat.parameters() if p.requires_grad)

    solve_lanczos, info = build_lowrank_inverse_operator(
        model_hat=model_hat, x=x, nbrs_idx=nbrs_idx, t_t=t_t, e_t=e_t,
        y_resid=y_resid, separate_self=separate_self, damping=damping,
        m=lanczos_m, k=lanczos_k, reorth=lanczos_reorth, verbose=False,
    )

    def A_mul(v: torch.Tensor) -> torch.Tensor:
        hv = gauss_newton_hvp(
            model=model_hat, x=x, nbrs_idx=nbrs_idx, t_t=t_t, e_t=e_t,
            y_resid=y_resid, separate_self=separate_self, v=v,
        )
        return hv + damping * v

    print(f"[lanczos_diag] Comparing CG (iters={cg_iters}) vs Lanczos (m={lanczos_m}, k={lanczos_k}) on {n_probes} random RHS:")
    for i in range(n_probes):
        b = torch.randn(dim_p, device=device, dtype=torch.float32)
        delta_cg = cg_solve(A_mul, b, max_iter=cg_iters, tol=cg_tol)
        delta_lz = solve_lanczos(b)
        cg_norm = torch.linalg.norm(delta_cg).item()
        rel_err = torch.linalg.norm(delta_cg - delta_lz).item() / max(cg_norm, 1e-30)
        print(f"  probe {i + 1}: ||delta_cg||={cg_norm:.4e}  rel_err=||cg-lanczos||/||cg||={rel_err:.4e}")


def dense_direct_vs_cg_diagnostic(
    model_hat: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    separate_self: bool,
    damping: float,
    cg_iters: int = 50,
    cg_tol: float = 1e-6,
    n_probes: int = 3,
) -> None:
    """Print-only diagnostic: compare dense_direct solves against CG."""
    device = next(model_hat.parameters()).device
    base = _ij_prepare_base_quantities(
        model_hat=model_hat,
        nbrs_idx=nbrs_idx,
        separate_self=separate_self,
        x=x,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
    )
    params = base["params"]
    L_base = base["L_base"]
    dim_p = sum(p.numel() for p in params)
    A_mul = _ij_build_A_mul(
        model_hat=model_hat,
        params=params,
        L_base=L_base,
        nbrs_idx=nbrs_idx,
        separate_self=separate_self,
        x=x,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
        damping=damping,
    )
    solve_dense, _ = build_dense_direct_inverse_operator(
        model_hat=model_hat,
        params=params,
        L_base=L_base,
        x=x,
        nbrs_idx=nbrs_idx,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
        separate_self=separate_self,
        damping=damping,
        verbose=False,
    )
    print(f"[dense_diag] Comparing CG (iters={cg_iters}) vs dense_direct on {n_probes} random RHS:")
    for i in range(n_probes):
        b = torch.randn(dim_p, device=device, dtype=torch.float32)
        delta_cg = cg_solve(A_mul, b, max_iter=cg_iters, tol=cg_tol)
        delta_dense = solve_dense(b)
        cg_norm = torch.linalg.norm(delta_cg).item()
        rel_err = torch.linalg.norm(delta_cg - delta_dense).item() / max(cg_norm, 1e-30)
        print(f"  probe {i + 1}: ||delta_cg||={cg_norm:.4e}  rel_err=||cg-dense||/||cg||={rel_err:.4e}")


def build_eval_mask(
    base_attn: np.ndarray,
    truth: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """
    Build a boolean [n, n] evaluation mask for uniform bands / coverage.
    The mask is always graph edges with diagonal excluded.
    """
    if base_attn.shape != truth.shape or base_attn.shape != A.shape:
        raise ValueError(f"Mask shapes mismatch: base_attn {base_attn.shape}, truth {truth.shape}, A {A.shape}")

    mask = (A != 0).copy()
    np.fill_diagonal(mask, False)
    return mask

def fit_propensity(data: GraphData, train_cfg: TrainConfig, prop_cfg: PropensityConfig) -> Tuple[FitResult, GraphData]:
    X_t, A_norm, t_t, y_t, d_var = to_torch(data.X, data.A, data.t, data.y, device=train_cfg.device)
    X_aug = torch.cat([X_t, d_var], dim=1)
    n = X_aug.shape[0]
    e_hat_np = np.zeros(n, dtype=np.float32)
    history = {"train_loss": [], "val_score": []}
    folds = np.unique(data.fold_assignments)
    bce = nn.BCEWithLogitsLoss()
    model: Optional[nn.Module] = None
    for fold in folds:
        train_idx = np.where(data.fold_assignments != fold)[0]
        test_idx  = np.where(data.fold_assignments == fold)[0]
        model = PropensityPredictor(in_features=X_aug.shape[1], hidden_dim=prop_cfg.hidden_dim, out_features=1).to(train_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        best_loss = float("inf")
        best_logits_test = None
        for epoch in range(train_cfg.epochs):
            model.train()
            # Mini-batch over train indices (compute full forward but accumulate batch losses)
            train_index_tensor = torch.tensor(train_idx, dtype=torch.long, device=train_cfg.device)
            loader = DataLoader(train_index_tensor, batch_size=train_cfg.batch_size, shuffle=True)
            running_loss = 0.0
            num_batches = 0
            for batch_indices in loader:
                optimizer.zero_grad()
                logits_full = model(X_aug, A_norm).squeeze(1)
                loss = bce(logits_full[batch_indices], t_t[batch_indices].float())
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                num_batches += 1
            loss = torch.tensor(running_loss / max(1, num_batches))
            model.eval()
            with torch.no_grad():
                logits = model(X_aug, A_norm).squeeze(1)
                test_loss = bce(logits[test_idx], t_t[test_idx].float())
            history["train_loss"].append(float(loss.item()))
            history["val_score"].append(float(test_loss.item()))
            if train_cfg.verbose and ((epoch + 1) % train_cfg.log_every == 0):
                print(f"[propensity] Fold {fold}, Epoch {epoch+1:03d} | train loss: {loss.item():.6f} | fold_loss: {test_loss.item():.6f}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                best_logits_test = logits[test_idx].detach().clone()
        with torch.no_grad():
            e_hat_np[test_idx] = torch.sigmoid(best_logits_test).cpu().numpy().astype(np.float32)
    data.e_hat = e_hat_np
    fr = FitResult(pred=e_hat_np, model=model, history=history, metrics={"best_fold_loss": float(np.min(history["val_score"])) if history["val_score"] else 0.0})
    return fr, data


def fit_mean(data: GraphData, train_cfg: TrainConfig, mean_cfg: MeanConfig, include_e_hat: bool = True) -> Tuple[FitResult, GraphData]:
    if include_e_hat and data.e_hat is None:
        raise ValueError("e_hat is required in GraphData when include_e_hat=True")

    X_t, A_norm, t_t, y_t, d_var = to_torch(data.X, data.A, data.t, data.y, device=train_cfg.device)
    feats_base = [X_t, d_var]
    if include_e_hat and data.e_hat is not None:
        e_t = torch.tensor(data.e_hat, dtype=torch.float32, device=train_cfg.device).view(-1, 1)
        feats_base.insert(1, e_t)
    X_aug = torch.cat(feats_base, dim=1)
    n = X_aug.shape[0]
    mu_hat_np = np.zeros(n, dtype=np.float32)
    history = {"train_loss": [], "val_score": []}
    mse = nn.MSELoss()
    folds = np.unique(data.fold_assignments)
    model: Optional[nn.Module] = None
    for fold in folds:
        train_idx = np.where(data.fold_assignments != fold)[0]
        test_idx  = np.where(data.fold_assignments == fold)[0]
        model = MeanPredictor(
            in_features=X_aug.shape[1],
            hidden_features=mean_cfg.hidden_dim,
            out_features=1,
            agg_method=mean_cfg.agg_method,
            use_flexible_attention=mean_cfg.use_flexible_attention,
            low_dimension=mean_cfg.low_dimension,
        ).to(train_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        best_loss = float("inf")
        best_pred_test = None
        for epoch in range(train_cfg.epochs):
            model.train()
            # Mini-batch over train indices (compute full forward but accumulate batch losses)
            train_index_tensor = torch.tensor(train_idx, dtype=torch.long, device=train_cfg.device)
            loader = DataLoader(train_index_tensor, batch_size=train_cfg.batch_size, shuffle=True)
            running_loss = 0.0
            num_batches = 0
            for batch_indices in loader:
                optimizer.zero_grad()
                preds_full = model(X_aug, A_norm).squeeze(1)
                loss = mse(preds_full[batch_indices], y_t[batch_indices].squeeze())
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                num_batches += 1
            loss = torch.tensor(running_loss / max(1, num_batches))
            model.eval()
            with torch.no_grad():
                preds = model(X_aug, A_norm).squeeze(1)
                test_loss = mse(preds[test_idx], y_t[test_idx].squeeze())
            history["train_loss"].append(float(loss.item()))
            history["val_score"].append(float(test_loss.item()))
            if train_cfg.verbose and ((epoch + 1) % train_cfg.log_every == 0):
                print(f"[mean] Fold {fold}, Epoch {epoch+1:03d} | train loss: {loss.item():.6f} | fold_loss: {test_loss.item():.6f}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                best_pred_test = preds[test_idx].detach().clone()
        mu_hat_np[test_idx] = best_pred_test.cpu().numpy().astype(np.float32)
    data.mu_hat = mu_hat_np
    fr = FitResult(pred=mu_hat_np, model=model, history=history, metrics={"best_fold_loss": float(np.min(history["val_score"])) if history["val_score"] else 0.0})
    return fr, data


def fit_attention(
    data: GraphData,
    train_cfg: TrainConfig,
    attn_cfg: AttentionConfig,
    attn_true: Optional[np.ndarray] = None,
    attn_true_self: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    val_data: Optional[ValidationData] = None,
) -> FitResult:
    # Requirements
    if data.e_hat is None or data.mu_hat is None:
        raise ValueError("GraphData must contain e_hat and mu_hat before fitting attention")
    if attn_true is None:
        raise ValueError("attn_true (ground-truth spillover weights) is required to compute and print spillover diff like run_gcn_experiment")

    device = train_cfg.device
    X_used = data.X[:, :1] if attn_cfg.low_dimension else data.X
    x = torch.tensor(X_used, dtype=torch.float32, device=device)
    y_resid = torch.tensor(data.y - data.mu_hat, dtype=torch.float32, device=device)
    t_t = torch.tensor(data.t, dtype=torch.float32, device=device)
    e_t = torch.tensor(data.e_hat, dtype=torch.float32, device=device)

    n = x.shape[0]
    weight_tensor: Optional[torch.Tensor] = None
    if sample_weights is not None:
        if len(sample_weights) != n:
            raise ValueError(f"sample_weights must have length {n}, got {len(sample_weights)}")
        weight_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device)
    if not attn_cfg.separate_self:
        model = GCNWithAttentionOneHead(
            input_dim=x.shape[1],
            hidden_dim=attn_cfg.hidden_dim,
            b=float(attn_cfg.attn_temperature),
        ).to(device)
    else:
        model = GCNWithAttentionTwoHead(
            input_dim=x.shape[1],
            hidden_dim=attn_cfg.hidden_dim,
            b=float(attn_cfg.attn_temperature),
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # Prepare validation tensors for early stopping (if provided)
    x_val: Optional[torch.Tensor] = None
    t_val: Optional[torch.Tensor] = None
    if val_data is not None:
        X_val_used = val_data.X[:, :1] if attn_cfg.low_dimension else val_data.X
        x_val = torch.tensor(X_val_used, dtype=torch.float32, device=device)
        t_val = torch.tensor(val_data.t, dtype=torch.float32, device=device)

    # DataLoader over (y_resid, idx) to mimic run_gcn_experiment batching
    indices = torch.arange(n, device=device)
    dl = DataLoader(TensorDataset(y_resid, indices), batch_size=train_cfg.batch_size, shuffle=True)

    start_time = time.time()
    best_spillover_loss = float("inf")
    best_predicted_attention: Optional[np.ndarray] = None
    best_model: Optional[nn.Module] = None

    no_improvement_count = 0
    patience = train_cfg.patience

    history: Dict[str, list] = {"train_loss": [], "spillover_diff": []}
    if val_data is not None:
        history["val_spillover_diff"] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        running_train_loss = 0.0

        # Epoch train
        for batch_Y, idx in dl:
            batch_Y = batch_Y.to(device)
            idx_cpu = idx.tolist()
            batch_nbrs = [data.nbrs_idx[i] for i in idx_cpu]

            optimizer.zero_grad()
            if not attn_cfg.separate_self:
                Y_pred, _ = model(x, batch_nbrs, t_t, e_t)
            else:
                Y_pred, _, _ = model(x, batch_nbrs, t_t, e_t)

            if weight_tensor is not None:
                w_batch = weight_tensor[idx].to(device)
                sq_err = (Y_pred - batch_Y) ** 2
                loss = (w_batch * sq_err).mean()
            else:
                loss = ((Y_pred - batch_Y) ** 2).mean()

            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())

        epoch_train_loss = running_train_loss / max(1, len(dl))
        history["train_loss"].append(epoch_train_loss)

        # Evaluate spillover loss on the full training dataset
        model.eval()
        with torch.no_grad():
            if not attn_cfg.separate_self:
                attn_pred, _ = model.predict(x, data.nbrs_idx, t_t)
                train_spillover_loss = float(np.sum((attn_pred - attn_true) ** 2))
            else:
                attn_pred, _, g_pred = model.predict(x, data.nbrs_idx, t_t)
                train_spillover_loss = float(np.sum((attn_pred - attn_true) ** 2))
                if attn_true_self is not None:
                    train_spillover_loss += float(np.sum((g_pred - attn_true_self) ** 2))
        history["spillover_diff"].append(train_spillover_loss)

        # Determine early-stopping metric: use validation if available, else train
        if val_data is not None:
            with torch.no_grad():
                if not attn_cfg.separate_self:
                    attn_pred_val, _ = model.predict(x_val, val_data.nbrs_idx, t_val)
                    val_spillover_loss = float(np.sum((attn_pred_val - val_data.attn_true) ** 2))
                else:
                    attn_pred_val, _, g_pred_val = model.predict(x_val, val_data.nbrs_idx, t_val)
                    val_spillover_loss = float(np.sum((attn_pred_val - val_data.attn_true) ** 2))
                    if val_data.attn_true_self is not None:
                        val_spillover_loss += float(np.sum((g_pred_val - val_data.attn_true_self) ** 2))
            history["val_spillover_diff"].append(val_spillover_loss)
            earlystop_metric = val_spillover_loss
        else:
            earlystop_metric = train_spillover_loss

        if earlystop_metric < best_spillover_loss:
            best_spillover_loss = earlystop_metric
            best_predicted_attention = attn_pred
            best_model = copy.deepcopy(model)
            no_improvement_count = 0
            if train_cfg.verbose:
                if val_data is not None:
                    print(f"Epoch {epoch+1:03d}, New best Val spillover diff: {val_spillover_loss:.4f} (Train: {train_spillover_loss:.4f})")
                else:
                    print(f"Epoch {epoch+1:03d}, New best Spillover diff: {train_spillover_loss:.4f}")
        else:
            no_improvement_count += 1
            if train_cfg.verbose:
                if val_data is not None:
                    print(f"No improve. Epoch {epoch+1:03d}, Val spillover diff: {val_spillover_loss:.4f} (Train: {train_spillover_loss:.4f})")
                else:
                    print(f"No improve. Epoch {epoch+1:03d}, Spillover diff: {train_spillover_loss:.4f}")
            if no_improvement_count >= patience:
                if train_cfg.verbose:
                    print(f"Stopping early at epoch {epoch+1} due to no improvement for {patience} consecutive epochs.")
                break

    if train_cfg.verbose:
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Use best model to compute spillover predictions (node-wise)
    assert best_model is not None, "Training did not produce a best model"
    best_model.eval()
    with torch.no_grad():
        if not attn_cfg.separate_self:
            preds, _ = best_model(x, data.nbrs_idx, t_t, e_t)
        else:
            preds, _, _ = best_model(x, data.nbrs_idx, t_t, e_t)
        spillover_pred = preds.detach().cpu().numpy().astype(np.float32)

    metrics: Dict[str, float] = {"best_spillover_diff": best_spillover_loss}
    fr = FitResult(pred=spillover_pred, model=best_model, history=history, metrics=metrics)
    return fr


def _draw_multiplier_weights(n: int, multiplier_dist: str) -> np.ndarray:
    if multiplier_dist == "normal":
        return np.random.normal(loc=1.0, scale=1.0, size=n)
    if multiplier_dist == "rademacher":
        eps = np.random.choice([-1.0, 1.0], size=n)
        return 1.0 + eps
    if multiplier_dist == "poisson":
        return np.random.poisson(lam=1.0, size=n).astype(np.float32)
    raise ValueError(f"Unknown multiplier_dist: {multiplier_dist}")


def _model_forward_preds(
    model: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    separate_self: bool,
) -> torch.Tensor:
    if not separate_self:
        y_pred, _ = model(x, nbrs_idx, t_t, e_t)
    else:
        y_pred, _, _ = model(x, nbrs_idx, t_t, e_t)
    return y_pred


def _flat_to_named_tangent(
    flat: torch.Tensor,
    named_params: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    tangent = OrderedDict()
    offset = 0
    for name, p in named_params.items():
        numel = p.numel()
        tangent[name] = flat[offset : offset + numel].view_as(p)
        offset += numel
    if offset != flat.numel():
        raise ValueError(f"Tangent size mismatch: consumed {offset}, provided {flat.numel()}")
    return tangent


def gauss_newton_hvp(
    model: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    separate_self: bool,
    v: torch.Tensor,
) -> torch.Tensor:
    if functional_call is None or jvp is None:
        raise RuntimeError("Gauss-Newton HVP requires torch.func.functional_call and torch.func.jvp")

    named_params = OrderedDict((name, p) for name, p in model.named_parameters() if p.requires_grad)
    named_buffers = OrderedDict(model.named_buffers())
    tangent = _flat_to_named_tangent(v, named_params)

    def resid_fn(param_dict: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
        out = functional_call(model, (param_dict, named_buffers), (x, nbrs_idx, t_t, e_t))
        y_pred = out[0] if not separate_self else out[0]
        return y_pred - y_resid

    resid, jv = jvp(resid_fn, (named_params,), (tangent,))
    gn_parts = torch.autograd.grad(resid, tuple(named_params.values()), grad_outputs=jv, retain_graph=False, allow_unused=True)
    # Match Hessian scaling for L_base = mean(r^2): H_GN = (2 / n) * J^T J.
    n_obs = max(1, int(resid.numel()))
    return (2.0 / float(n_obs)) * _flatten_grads(gn_parts, tuple(named_params.values()))


def _model_predict_attn_raw(
    model: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    separate_self: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not separate_self:
        attn, raw = model.predict(x, nbrs_idx, t_t)
    else:
        attn, raw, _ = model.predict(x, nbrs_idx, t_t)
    return attn.astype(np.float32), raw.astype(np.float32)


def _compute_spillover_diff(
    model: nn.Module,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    separate_self: bool,
    attn_true: np.ndarray,
    attn_true_self: Optional[np.ndarray],
) -> float:
    with torch.no_grad():
        if not separate_self:
            attn_pred, _ = model.predict(x, nbrs_idx, t_t)
            spillover_diff = float(np.sum((attn_pred - attn_true) ** 2))
        else:
            attn_pred, _, self_pred = model.predict(x, nbrs_idx, t_t)
            spillover_diff = float(np.sum((attn_pred - attn_true) ** 2))
            if attn_true_self is not None:
                spillover_diff += float(np.sum((self_pred - attn_true_self) ** 2))
    return spillover_diff


def _ij_prepare_base_quantities(
    model_hat: nn.Module,
    nbrs_idx: Sequence[np.ndarray],
    separate_self: bool,
    x: torch.Tensor,
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
    """Prepare base IJ quantities shared across solver backends."""
    model_hat.zero_grad(set_to_none=True)
    model_hat.train()
    params = [p for p in model_hat.parameters() if p.requires_grad]
    y_pred_hat = _model_forward_preds(
        model=model_hat,
        x=x,
        nbrs_idx=nbrs_idx,
        t_t=t_t,
        e_t=e_t,
        separate_self=separate_self,
    )
    resid = y_pred_hat - y_resid
    sq_resid = resid.square()
    L_base = sq_resid.mean()
    g_base = torch.autograd.grad(L_base, params, retain_graph=True, allow_unused=True)
    g_base_vec = _flatten_grads(g_base, params).detach()
    theta_hat = flatten_params(model_hat).detach()
    return {
        "params": params,
        "L_base": L_base,
        "g_base_vec": g_base_vec,
        "theta_hat": theta_hat,
    }


def _ij_rhs_g_total_vec(
    model_hat: nn.Module,
    params: Sequence[torch.Tensor],
    nbrs_idx: Sequence[np.ndarray],
    separate_self: bool,
    x: torch.Tensor,
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    weights: torch.Tensor,
    g_base_vec: torch.Tensor,
) -> torch.Tensor:
    """Compute IJ RHS g_total = g_base + g_delta for one replicate."""
    model_hat.zero_grad(set_to_none=True)
    model_hat.train()
    y_pred_hat = _model_forward_preds(
        model=model_hat,
        x=x,
        nbrs_idx=nbrs_idx,
        t_t=t_t,
        e_t=e_t,
        separate_self=separate_self,
    )
    resid = y_pred_hat - y_resid
    L_delta = ((weights - 1.0) * resid.square()).mean()
    g_delta = torch.autograd.grad(L_delta, params, retain_graph=False, allow_unused=True)
    g_delta_vec = _flatten_grads(g_delta, params).detach()
    return g_base_vec + g_delta_vec


def _ij_build_A_mul(
    model_hat: nn.Module,
    params: Sequence[torch.Tensor],
    L_base: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    separate_self: bool,
    x: torch.Tensor,
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    damping: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the IJ linear operator used by CG/Lanczos/dense_direct.

    This matches the current CG backend: GN-HVP when available, and fallback to
    full-Hessian HVP from grad(L_base) when GN path errors.
    """
    grad_base_flat: Optional[torch.Tensor] = None

    def A_mul(v: torch.Tensor) -> torch.Tensor:
        nonlocal grad_base_flat
        try:
            hv = gauss_newton_hvp(
                model=model_hat,
                x=x,
                nbrs_idx=nbrs_idx,
                t_t=t_t,
                e_t=e_t,
                y_resid=y_resid,
                separate_self=separate_self,
                v=v,
            )
        except Exception:
            # Fallback: cached full-Hessian HVP if torch.func is unavailable.
            if grad_base_flat is None:
                grad_base = torch.autograd.grad(L_base, params, create_graph=True, retain_graph=True, allow_unused=True)
                grad_base_flat = _flatten_grads(grad_base, params)
            hv = hvp_from_grad_flat(grad_base_flat, params, v)
        return hv + damping * v

    return A_mul


def build_dense_direct_inverse_operator(
    model_hat: nn.Module,
    params: Sequence[torch.Tensor],
    L_base: torch.Tensor,
    x: torch.Tensor,
    nbrs_idx: Sequence[np.ndarray],
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    separate_self: bool,
    damping: float,
    verbose: bool = False,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Dict[str, float | str]]:
    """Build dense direct solver for the SAME IJ linear system as CG.

    This is not an exact-Hessian inversion by itself; it directly solves the
    dense matrix representation of the current IJ operator.
    """
    device = next(model_hat.parameters()).device
    dim_p = int(sum(p.numel() for p in params))
    A_mul = _ij_build_A_mul(
        model_hat=model_hat,
        params=params,
        L_base=L_base,
        nbrs_idx=nbrs_idx,
        separate_self=separate_self,
        x=x,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
        damping=damping,
    )
    t0 = time.time()
    A = torch.zeros(dim_p, dim_p, device=device, dtype=torch.float64)
    for j in range(dim_p):
        e_j = torch.zeros(dim_p, device=device, dtype=torch.float32)
        e_j[j] = 1.0
        A[:, j] = A_mul(e_j).detach().to(dtype=torch.float64)
    dense_build_time = time.time() - t0
    sym_err = torch.linalg.norm(A - A.T) / (torch.linalg.norm(A) + 1e-30)
    A = 0.5 * (A + A.T)
    diag_A = torch.diag(A)
    chol_success = False
    factorization = "solve"
    L_chol: Optional[torch.Tensor] = None
    try:
        L_chol = torch.linalg.cholesky(A)
        chol_success = True
        factorization = "cholesky"
    except Exception:
        chol_success = False
        factorization = "solve"
    cond_est = float("inf")
    try:
        cond_est = float(torch.linalg.cond(A).item())
    except Exception:
        cond_est = float("inf")
    eig_min = float("nan")
    eig_max = float("nan")
    try:
        evals = torch.linalg.eigvalsh(A)
        if evals.numel() > 0:
            eig_min = float(evals[0].item())
            eig_max = float(evals[-1].item())
    except Exception:
        eig_min = float("nan")
        eig_max = float("nan")
    info: Dict[str, float | str] = {
        "dim_p": float(dim_p),
        "dense_build_time": float(dense_build_time),
        "dense_factorization": factorization,
        "diag_min": float(diag_A.min().item()) if dim_p > 0 else 0.0,
        "diag_max": float(diag_A.max().item()) if dim_p > 0 else 0.0,
        "symmetry_error": float(sym_err.item()),
        "cond_est": cond_est,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "chol_success": 1.0 if chol_success else 0.0,
    }
    if verbose:
        print(
            f"[dense_direct] build_time={dense_build_time:.3f}s dim_p={dim_p} "
            f"factorization={factorization} sym_err={sym_err.item():.3e} "
            f"diag_range=[{info['diag_min']:.3e}, {info['diag_max']:.3e}] "
            f"eig_range=[{eig_min:.3e}, {eig_max:.3e}] cond~={cond_est:.3e}"
        )
        if not chol_success and damping <= 0.0:
            print("[dense_direct] warning: Cholesky failed with non-positive damping; system may be ill-conditioned.")

    def solve_dense(b: torch.Tensor) -> torch.Tensor:
        b64 = b.to(device=device, dtype=torch.float64)
        if L_chol is not None:
            rhs = b64.unsqueeze(1)
            delta64 = torch.cholesky_solve(rhs, L_chol).squeeze(1)
        else:
            try:
                delta64 = torch.linalg.solve(A, b64)
            except Exception as exc:
                raise RuntimeError(
                    "dense_direct failed to solve IJ linear system; matrix may be singular/ill-conditioned. "
                    f"Try positive ij_damping. Original error: {exc}"
                ) from exc
        return delta64.to(dtype=b.dtype)

    return solve_dense, info


def _ij_one_step_theta(
    model_hat: nn.Module,
    nbrs_idx: Sequence[np.ndarray],
    separate_self: bool,
    x: torch.Tensor,
    t_t: torch.Tensor,
    e_t: torch.Tensor,
    y_resid: torch.Tensor,
    weights: torch.Tensor,
    damping: float,
    cg_iters: int,
    cg_tol: float,
) -> torch.Tensor:
    base = _ij_prepare_base_quantities(
        model_hat=model_hat,
        nbrs_idx=nbrs_idx,
        separate_self=separate_self,
        x=x,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
    )
    params = base["params"]
    L_base = base["L_base"]
    g_base_vec = base["g_base_vec"]
    theta_hat = base["theta_hat"]
    g_total_vec = _ij_rhs_g_total_vec(
        model_hat=model_hat,
        params=params,
        nbrs_idx=nbrs_idx,
        separate_self=separate_self,
        x=x,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
        weights=weights,
        g_base_vec=g_base_vec,
    )
    A_mul = _ij_build_A_mul(
        model_hat=model_hat,
        params=params,
        L_base=L_base,
        nbrs_idx=nbrs_idx,
        separate_self=separate_self,
        x=x,
        t_t=t_t,
        e_t=e_t,
        y_resid=y_resid,
        damping=damping,
    )
    delta = cg_solve(A_mul=A_mul, b=g_total_vec, max_iter=cg_iters, tol=cg_tol)
    theta_new = theta_hat.to(delta.device) - delta
    return theta_new


def bootstrap_attention(
    data: GraphData,
    train_cfg: TrainConfig,
    attn_cfg: AttentionConfig,
    attn_true: np.ndarray,
    attn_true_self: Optional[np.ndarray] = None,
    B: int = 200,
    alpha: float = 0.05,
    multiplier_dist: str = "normal",
    eval_mask_cfg: Optional[Dict] = None,
    # Optional test data for evaluating coverage on test graph
    X_test: Optional[np.ndarray] = None,
    nbrs_idx_test: Optional[Sequence[np.ndarray]] = None,
    t_test: Optional[np.ndarray] = None,
    A_test: Optional[np.ndarray] = None,
    attn_true_test: Optional[np.ndarray] = None,
    eval_mask_cfg_test: Optional[Dict] = None,
    # Optional validation data forwarded to fit_attention for early stopping
    val_data: Optional[ValidationData] = None,
    base_result: Optional[FitResult] = None,
    bootstrap_multipliers: Optional[np.ndarray] = None,
    use_ij: bool = False,
    ij_damping: float = 1e-3,
    ij_cg_iters: int = 50,
    ij_cg_tol: float = 1e-6,
    ij_solver: str = "cg",
    ij_lanczos_m: int = 100,
    ij_lanczos_k: int = 50,
    ij_lanczos_reorth: bool = True,
) -> BootstrapResult:
    """
    Multiplier bootstrap for the attention model.

    - Runs an unweighted baseline fit_attention.
    - Runs B weighted refits with random multipliers xi.
    - Returns pointwise and uniform confidence intervals for the attention matrix.
    """
    if data.e_hat is None or data.mu_hat is None:
        raise ValueError("GraphData must contain e_hat and mu_hat before bootstrap_attention")
    if use_ij and ij_solver not in {"cg", "lanczos", "dense_direct"}:
        raise ValueError(f"Unsupported ij_solver={ij_solver!r}. Expected one of: cg, lanczos, dense_direct.")
    device = train_cfg.device

    # Prepare tensors as in fit_attention
    X_used = data.X[:, :1] if attn_cfg.low_dimension else data.X
    x = torch.tensor(X_used, dtype=torch.float32, device=device)
    t_t = torch.tensor(data.t, dtype=torch.float32, device=device)
    e_t = torch.tensor(data.e_hat, dtype=torch.float32, device=device)
    y_resid_t = torch.tensor(data.y - data.mu_hat, dtype=torch.float32, device=device)
    n = x.shape[0]

    # Optional test tensors
    have_test = X_test is not None and nbrs_idx_test is not None and t_test is not None
    if have_test:
        X_test_used = X_test[:, :1] if attn_cfg.low_dimension else X_test
        x_test = torch.tensor(X_test_used, dtype=torch.float32, device=device)
        t_test_t = torch.tensor(t_test, dtype=torch.float32, device=device)

    if train_cfg.verbose:
        method_name = f"ij (solver={ij_solver})" if use_ij else "exact"
        print(f"[bootstrap] Running method={method_name}")

    # 1) Baseline unweighted fit (or reuse provided baseline)
    if base_result is None:
        if train_cfg.verbose:
            print("[bootstrap] Fitting baseline attention model (unweighted)...")
        base_res = fit_attention(
            data=data,
            train_cfg=train_cfg,
            attn_cfg=attn_cfg,
            attn_true=attn_true,
            attn_true_self=attn_true_self,
            sample_weights=None,
            val_data=val_data,
        )
    else:
        base_res = base_result
        if train_cfg.verbose:
            print("[bootstrap] Reusing provided baseline attention model.")
    base_model = base_res.model
    base_model.eval()
    with torch.no_grad():
        base_attn, base_raw = _model_predict_attn_raw(
            model=base_model,
            x=x,
            nbrs_idx=data.nbrs_idx,
            t_t=t_t,
            separate_self=attn_cfg.separate_self,
        )
        if have_test:
            base_attn_test, base_raw_test = _model_predict_attn_raw(
                model=base_model,
                x=x_test,
                nbrs_idx=nbrs_idx_test,
                t_t=t_test_t,
                separate_self=attn_cfg.separate_self,
            )
        base_pred = _model_forward_preds(
            model=base_model,
            x=x,
            nbrs_idx=data.nbrs_idx,
            t_t=t_t,
            e_t=e_t,
            separate_self=attn_cfg.separate_self,
        )
        base_train_loss = float(((base_pred - y_resid_t).square()).mean().item())
    base_attn_test_np: Optional[np.ndarray] = None
    base_raw_test_np: Optional[np.ndarray] = None
    if have_test:
        base_attn_test_np = base_attn_test
        base_raw_test_np = base_raw_test

    boot_attn_list = []
    boot_raw_list = []
    boot_attn_test_list: list[np.ndarray] = []
    boot_raw_test_list: list[np.ndarray] = []
    boot_weighted_loss_list: list[float] = []
    scratch_model: Optional[nn.Module] = copy.deepcopy(base_model) if use_ij else None

    # -- IJ precomputation (amortized across all B replicates) --
    lanczos_inv_op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    lanczos_g_base_vec: Optional[torch.Tensor] = None
    lanczos_theta_hat: Optional[torch.Tensor] = None
    dense_inv_op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    dense_g_base_vec: Optional[torch.Tensor] = None
    dense_theta_hat: Optional[torch.Tensor] = None
    dense_info: Optional[Dict[str, float | str]] = None
    ij_base_params: Optional[Sequence[torch.Tensor]] = None
    ij_base_L_base: Optional[torch.Tensor] = None

    base: Optional[Dict[str, torch.Tensor | list[torch.Tensor]]] = None
    if use_ij and ij_solver in {"lanczos", "dense_direct"}:
        base = _ij_prepare_base_quantities(
            model_hat=base_model,
            nbrs_idx=data.nbrs_idx,
            separate_self=attn_cfg.separate_self,
            x=x,
            t_t=t_t,
            e_t=e_t,
            y_resid=y_resid_t,
        )
        ij_base_params = base["params"]
        ij_base_L_base = base["L_base"]

    if use_ij and ij_solver == "lanczos":
        assert base is not None
        lanczos_g_base_vec = base["g_base_vec"]
        lanczos_theta_hat = base["theta_hat"]
        lanczos_inv_op, lanczos_info = build_lowrank_inverse_operator(
            model_hat=base_model,
            x=x,
            nbrs_idx=data.nbrs_idx,
            t_t=t_t,
            e_t=e_t,
            y_resid=y_resid_t,
            separate_self=attn_cfg.separate_self,
            damping=ij_damping,
            m=ij_lanczos_m,
            k=ij_lanczos_k,
            reorth=ij_lanczos_reorth,
            verbose=train_cfg.verbose,
        )
    elif use_ij and ij_solver == "dense_direct":
        assert base is not None
        assert ij_base_params is not None
        assert ij_base_L_base is not None
        dense_g_base_vec = base["g_base_vec"]
        dense_theta_hat = base["theta_hat"]
        dense_inv_op, dense_info = build_dense_direct_inverse_operator(
            model_hat=base_model,
            params=ij_base_params,
            L_base=ij_base_L_base,
            x=x,
            nbrs_idx=data.nbrs_idx,
            t_t=t_t,
            e_t=e_t,
            y_resid=y_resid_t,
            separate_self=attn_cfg.separate_self,
            damping=ij_damping,
            verbose=train_cfg.verbose,
        )

    # 2) Bootstrap refits
    for b in range(B):
        if train_cfg.verbose:
            print(f"[bootstrap] Fitting bootstrap replicate {b + 1}/{B}...")
        # draw multipliers xi of length n
        if bootstrap_multipliers is not None:
            if bootstrap_multipliers.shape != (B, n):
                raise ValueError(f"bootstrap_multipliers must have shape {(B, n)}, got {bootstrap_multipliers.shape}")
            xi = bootstrap_multipliers[b]
        else:
            xi = _draw_multiplier_weights(n=n, multiplier_dist=multiplier_dist)
        w_t = torch.tensor(xi, dtype=torch.float32, device=device)

        if use_ij:
            t_rep_start = time.time()

            if ij_solver == "lanczos":
                assert lanczos_inv_op is not None
                assert lanczos_g_base_vec is not None
                assert lanczos_theta_hat is not None
                assert ij_base_params is not None
                g_total_vec = _ij_rhs_g_total_vec(
                    model_hat=base_model,
                    params=ij_base_params,
                    nbrs_idx=data.nbrs_idx,
                    separate_self=attn_cfg.separate_self,
                    x=x,
                    t_t=t_t,
                    e_t=e_t,
                    y_resid=y_resid_t,
                    weights=w_t,
                    g_base_vec=lanczos_g_base_vec,
                )
                delta = lanczos_inv_op(g_total_vec)
                theta_new = lanczos_theta_hat - delta
            elif ij_solver == "dense_direct":
                assert dense_inv_op is not None
                assert dense_g_base_vec is not None
                assert dense_theta_hat is not None
                assert ij_base_params is not None
                g_total_vec = _ij_rhs_g_total_vec(
                    model_hat=base_model,
                    params=ij_base_params,
                    nbrs_idx=data.nbrs_idx,
                    separate_self=attn_cfg.separate_self,
                    x=x,
                    t_t=t_t,
                    e_t=e_t,
                    y_resid=y_resid_t,
                    weights=w_t,
                    g_base_vec=dense_g_base_vec,
                )
                delta = dense_inv_op(g_total_vec)
                theta_new = dense_theta_hat - delta
            else:
                theta_new = _ij_one_step_theta(
                    model_hat=base_model,
                    nbrs_idx=data.nbrs_idx,
                    separate_self=attn_cfg.separate_self,
                    x=x,
                    t_t=t_t,
                    e_t=e_t,
                    y_resid=y_resid_t,
                    weights=w_t,
                    damping=ij_damping,
                    cg_iters=ij_cg_iters,
                    cg_tol=ij_cg_tol,
                )

            assert scratch_model is not None
            set_params(scratch_model, theta_new)
            scratch_model.eval()
            with torch.no_grad():
                attn_b, raw_b = _model_predict_attn_raw(
                    model=scratch_model,
                    x=x,
                    nbrs_idx=data.nbrs_idx,
                    t_t=t_t,
                    separate_self=attn_cfg.separate_self,
                )
                y_pred_rep = _model_forward_preds(
                    model=scratch_model,
                    x=x,
                    nbrs_idx=data.nbrs_idx,
                    t_t=t_t,
                    e_t=e_t,
                    separate_self=attn_cfg.separate_self,
                )
                rep_loss = float((w_t * (y_pred_rep - y_resid_t).square()).mean().item())
                boot_weighted_loss_list.append(rep_loss)
            if have_test:
                with torch.no_grad():
                    attn_b_test, raw_b_test = _model_predict_attn_raw(
                        model=scratch_model,
                        x=x_test,
                        nbrs_idx=nbrs_idx_test,
                        t_t=t_test_t,
                        separate_self=attn_cfg.separate_self,
                    )
                boot_attn_test_list.append(attn_b_test)
                boot_raw_test_list.append(raw_b_test)
            if train_cfg.verbose:
                solver_tag = ij_solver
                if ij_solver == "dense_direct" and dense_info is not None:
                    solver_tag = f"dense_direct[{dense_info.get('dense_factorization', 'solve')}]"
                rep_spillover_diff = _compute_spillover_diff(
                    model=scratch_model,
                    x=x,
                    nbrs_idx=data.nbrs_idx,
                    t_t=t_t,
                    separate_self=attn_cfg.separate_self,
                    attn_true=attn_true,
                    attn_true_self=attn_true_self,
                )
                print(
                    f"[bootstrap][ij-{solver_tag}] replicate {b + 1}/{B} time: {time.time() - t_rep_start:.3f}s "
                    f"| spillover diff: {rep_spillover_diff:.6f} | weighted train loss: {rep_loss:.6f}"
                )
        else:
            boot_res = fit_attention(
                data=data,
                train_cfg=train_cfg,
                attn_cfg=attn_cfg,
                attn_true=attn_true,
                attn_true_self=attn_true_self,
                sample_weights=xi,
                val_data=val_data,
            )
            boot_model = boot_res.model
            boot_model.eval()
            with torch.no_grad():
                attn_b, raw_b = _model_predict_attn_raw(
                    model=boot_model,
                    x=x,
                    nbrs_idx=data.nbrs_idx,
                    t_t=t_t,
                    separate_self=attn_cfg.separate_self,
                )
                y_pred_rep = _model_forward_preds(
                    model=boot_model,
                    x=x,
                    nbrs_idx=data.nbrs_idx,
                    t_t=t_t,
                    e_t=e_t,
                    separate_self=attn_cfg.separate_self,
                )
                rep_loss = float((w_t * (y_pred_rep - y_resid_t).square()).mean().item())
                boot_weighted_loss_list.append(rep_loss)
                if have_test:
                    attn_b_test, raw_b_test = _model_predict_attn_raw(
                        model=boot_model,
                        x=x_test,
                        nbrs_idx=nbrs_idx_test,
                        t_t=t_test_t,
                        separate_self=attn_cfg.separate_self,
                    )
                    boot_attn_test_list.append(attn_b_test)
                    boot_raw_test_list.append(raw_b_test)
            if train_cfg.verbose:
                rep_spillover_diff = _compute_spillover_diff(
                    model=boot_model,
                    x=x,
                    nbrs_idx=data.nbrs_idx,
                    t_t=t_t,
                    separate_self=attn_cfg.separate_self,
                    attn_true=attn_true,
                    attn_true_self=attn_true_self,
                )
                print(
                    f"[bootstrap][exact] replicate {b + 1}/{B} "
                    f"spillover diff: {rep_spillover_diff:.6f} | weighted train loss: {rep_loss:.6f}"
                )

        boot_attn_list.append(attn_b)
        boot_raw_list.append(raw_b)

    boot_attn = np.stack(boot_attn_list, axis=0)  # [B, n, n]
    boot_raw = np.stack(boot_raw_list, axis=0)    # [B, n, n]
    boot_attn_test_np: Optional[np.ndarray] = None
    boot_raw_test_np: Optional[np.ndarray] = None
    if have_test and boot_attn_test_list:
        boot_attn_test_np = np.stack(boot_attn_test_list, axis=0)  # [B, n_test, n_test]
        boot_raw_test_np = np.stack(boot_raw_test_list, axis=0)    # [B, n_test, n_test]

    # Build evaluation masks (train/test) after baseline is available
    def _maybe_build_mask(cfg: Optional[Dict], base: np.ndarray, truth_mat: np.ndarray, A_mat: np.ndarray) -> Optional[np.ndarray]:
        if cfg is None:
            return None
        return build_eval_mask(
            base_attn=base,
            truth=truth_mat,
            A=A_mat,
        )

    eval_mask = _maybe_build_mask(eval_mask_cfg, base_attn, attn_true, data.A)
    eval_mask_test = None
    if have_test and attn_true_test is not None and A_test is not None:
        eval_mask_test = _maybe_build_mask(eval_mask_cfg_test or eval_mask_cfg, base_attn_test_np, attn_true_test, A_test)

    # 3) Pointwise CIs for attention
    diffs = boot_attn - base_attn[None, :, :]  # [B, n, n]
    radius_pointwise = np.quantile(
        np.abs(diffs),
        1.0 - alpha,
        axis=0,
    )  # [n, n]
    ci_pointwise_lower = base_attn - radius_pointwise
    ci_pointwise_upper = base_attn + radius_pointwise

    # 4) Uniform band for attention (optionally restricted to eval_mask)
    if eval_mask is not None and eval_mask.any():
        max_dev = np.max(np.abs(diffs)[:, eval_mask], axis=1)  # [B]
    else:
        max_dev = np.max(np.abs(diffs).reshape(B, -1), axis=1)  # [B]
    c_uniform = np.quantile(max_dev, 1.0 - alpha)
    ci_uniform_lower = base_attn - c_uniform
    ci_uniform_upper = base_attn + c_uniform

    # 5) Raw-score CIs (pre-softmax MLP outputs)
    diffs_raw = boot_raw - base_raw[None, :, :]  # [B, n, n]
    radius_pointwise_raw = np.quantile(
        np.abs(diffs_raw),
        1.0 - alpha,
        axis=0,
    )  # [n, n]
    ci_pointwise_lower_raw = base_raw - radius_pointwise_raw
    ci_pointwise_upper_raw = base_raw + radius_pointwise_raw

    if eval_mask is not None and eval_mask.any():
        max_dev_raw = np.max(np.abs(diffs_raw)[:, eval_mask], axis=1)  # [B]
    else:
        max_dev_raw = np.max(np.abs(diffs_raw).reshape(B, -1), axis=1)  # [B]
    c_uniform_raw = np.quantile(max_dev_raw, 1.0 - alpha)
    ci_uniform_lower_raw = base_raw - c_uniform_raw
    ci_uniform_upper_raw = base_raw + c_uniform_raw

    # 6) Optional test-side uniform bands are not returned; coverage will use boot_* with the same mask.

    return BootstrapResult(
        base_result=base_res,
        base_attn=base_attn,
        boot_attn=boot_attn,
        ci_pointwise_lower=ci_pointwise_lower,
        ci_pointwise_upper=ci_pointwise_upper,
        ci_uniform_lower=ci_uniform_lower,
        ci_uniform_upper=ci_uniform_upper,
        base_raw=base_raw,
        boot_raw=boot_raw,
        ci_pointwise_lower_raw=ci_pointwise_lower_raw,
        ci_pointwise_upper_raw=ci_pointwise_upper_raw,
        ci_uniform_lower_raw=ci_uniform_lower_raw,
        ci_uniform_upper_raw=ci_uniform_upper_raw,
        base_attn_test=base_attn_test_np,
        boot_attn_test=boot_attn_test_np,
        base_raw_test=base_raw_test_np,
        boot_raw_test=boot_raw_test_np,
        eval_mask=eval_mask,
        eval_mask_test=eval_mask_test,
        base_train_loss=base_train_loss,
        boot_weighted_loss=np.asarray(boot_weighted_loss_list, dtype=np.float32),
    )


