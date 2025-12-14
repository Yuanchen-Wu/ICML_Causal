from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time

from utils import to_torch
from model.propensity import PropensityPredictor
from model.mean import MeanPredictor
from model.interference import GCNWithAttentionOneHead, GCNWithAttentionTwoHead


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

        # Evaluate spillover loss on the full dataset (no separate val split)
        model.eval()
        with torch.no_grad():
            if not attn_cfg.separate_self:
                attn_pred, _ = model.predict(x, data.nbrs_idx, t_t)
                spillover_loss = float(np.sum((attn_pred - attn_true) ** 2))
            else:
                attn_pred, _, g_pred = model.predict(x, data.nbrs_idx, t_t)
                spillover_loss = float(np.sum((attn_pred - attn_true) ** 2))
                if attn_true_self is not None:
                    spillover_loss += float(np.sum((g_pred - attn_true_self) ** 2))
        history["spillover_diff"].append(spillover_loss)

        if spillover_loss < best_spillover_loss:
            best_spillover_loss = spillover_loss
            best_predicted_attention = attn_pred
            best_model = copy.deepcopy(model)
            no_improvement_count = 0
            if train_cfg.verbose:
                print(f"Epoch {epoch+1:03d}, New best Spillover diff: {spillover_loss:.4f}")
        else:
            no_improvement_count += 1
            if train_cfg.verbose:
                print(f"No improve. Epoch {epoch+1:03d}, Spillover diff: {spillover_loss:.4f}")
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


def bootstrap_attention(
    data: GraphData,
    train_cfg: TrainConfig,
    attn_cfg: AttentionConfig,
    attn_true: np.ndarray,
    attn_true_self: Optional[np.ndarray] = None,
    B: int = 200,
    alpha: float = 0.05,
    multiplier_dist: str = "normal",
    # Optional test data for evaluating coverage on test graph
    X_test: Optional[np.ndarray] = None,
    nbrs_idx_test: Optional[Sequence[np.ndarray]] = None,
    t_test: Optional[np.ndarray] = None,
) -> BootstrapResult:
    """
    Multiplier bootstrap for the attention model.

    - Runs an unweighted baseline fit_attention.
    - Runs B weighted refits with random multipliers xi.
    - Returns pointwise and uniform confidence intervals for the attention matrix.
    """
    device = train_cfg.device

    # Prepare tensors as in fit_attention
    X_used = data.X[:, :1] if attn_cfg.low_dimension else data.X
    x = torch.tensor(X_used, dtype=torch.float32, device=device)
    t_t = torch.tensor(data.t, dtype=torch.float32, device=device)
    n = x.shape[0]

    # Optional test tensors
    have_test = X_test is not None and nbrs_idx_test is not None and t_test is not None
    if have_test:
        X_test_used = X_test[:, :1] if attn_cfg.low_dimension else X_test
        x_test = torch.tensor(X_test_used, dtype=torch.float32, device=device)
        t_test_t = torch.tensor(t_test, dtype=torch.float32, device=device)

    # 1) Baseline unweighted fit
    if train_cfg.verbose:
        print("[bootstrap] Fitting baseline attention model (unweighted)...")
    base_res = fit_attention(
        data=data,
        train_cfg=train_cfg,
        attn_cfg=attn_cfg,
        attn_true=attn_true,
        attn_true_self=attn_true_self,
        sample_weights=None,
    )
    base_model = base_res.model
    base_model.eval()
    with torch.no_grad():
        if not attn_cfg.separate_self:
            base_attn, base_raw = base_model.predict(x, data.nbrs_idx, t_t)
        else:
            base_attn, base_raw, _ = base_model.predict(x, data.nbrs_idx, t_t)
        if have_test:
            if not attn_cfg.separate_self:
                base_attn_test, base_raw_test = base_model.predict(x_test, nbrs_idx_test, t_test_t)
            else:
                base_attn_test, base_raw_test, _ = base_model.predict(x_test, nbrs_idx_test, t_test_t)
    base_attn = base_attn.astype(np.float32)
    base_raw = base_raw.astype(np.float32)
    base_attn_test_np: Optional[np.ndarray] = None
    base_raw_test_np: Optional[np.ndarray] = None
    if have_test:
        base_attn_test_np = base_attn_test.astype(np.float32)
        base_raw_test_np = base_raw_test.astype(np.float32)

    boot_attn_list = []
    boot_raw_list = []
    boot_attn_test_list: list[np.ndarray] = []
    boot_raw_test_list: list[np.ndarray] = []

    # 2) Bootstrap refits
    for b in range(B):
        if train_cfg.verbose:
            print(f"[bootstrap] Fitting bootstrap replicate {b + 1}/{B}...")
        # draw multipliers xi of length n
        if multiplier_dist == "normal":
            xi = np.random.normal(loc=1.0, scale=1.0, size=n)
        elif multiplier_dist == "rademacher":
            eps = np.random.choice([-1.0, 1.0], size=n)
            xi = 1.0 + eps
        else:
            raise ValueError(f"Unknown multiplier_dist: {multiplier_dist}")

        boot_res = fit_attention(
            data=data,
            train_cfg=train_cfg,
            attn_cfg=attn_cfg,
            attn_true=attn_true,
            attn_true_self=attn_true_self,
            sample_weights=xi,
        )
        boot_model = boot_res.model
        boot_model.eval()
        with torch.no_grad():
            if not attn_cfg.separate_self:
                attn_b, raw_b = boot_model.predict(x, data.nbrs_idx, t_t)
            else:
                attn_b, raw_b, _ = boot_model.predict(x, data.nbrs_idx, t_t)
            if have_test:
                if not attn_cfg.separate_self:
                    attn_b_test, raw_b_test = boot_model.predict(x_test, nbrs_idx_test, t_test_t)
                else:
                    attn_b_test, raw_b_test, _ = boot_model.predict(x_test, nbrs_idx_test, t_test_t)
                boot_attn_test_list.append(attn_b_test.astype(np.float32))
                boot_raw_test_list.append(raw_b_test.astype(np.float32))
        boot_attn_list.append(attn_b.astype(np.float32))
        boot_raw_list.append(raw_b.astype(np.float32))

    boot_attn = np.stack(boot_attn_list, axis=0)  # [B, n, n]
    boot_raw = np.stack(boot_raw_list, axis=0)    # [B, n, n]
    boot_attn_test_np: Optional[np.ndarray] = None
    boot_raw_test_np: Optional[np.ndarray] = None
    if have_test and boot_attn_test_list:
        boot_attn_test_np = np.stack(boot_attn_test_list, axis=0)  # [B, n_test, n_test]
        boot_raw_test_np = np.stack(boot_raw_test_list, axis=0)    # [B, n_test, n_test]

    # 3) Pointwise CIs for attention
    diffs = boot_attn - base_attn[None, :, :]  # [B, n, n]
    radius_pointwise = np.quantile(
        np.abs(diffs),
        1.0 - alpha / 2.0,
        axis=0,
    )  # [n, n]
    ci_pointwise_lower = base_attn - radius_pointwise
    ci_pointwise_upper = base_attn + radius_pointwise

    # 4) Uniform band for attention
    max_dev = np.max(np.abs(diffs).reshape(B, -1), axis=1)  # [B]
    c_uniform = np.quantile(max_dev, 1.0 - alpha)
    ci_uniform_lower = base_attn - c_uniform
    ci_uniform_upper = base_attn + c_uniform

    # 5) Raw-score CIs (pre-softmax MLP outputs)
    diffs_raw = boot_raw - base_raw[None, :, :]  # [B, n, n]
    radius_pointwise_raw = np.quantile(
        np.abs(diffs_raw),
        1.0 - alpha / 2.0,
        axis=0,
    )  # [n, n]
    ci_pointwise_lower_raw = base_raw - radius_pointwise_raw
    ci_pointwise_upper_raw = base_raw + radius_pointwise_raw

    max_dev_raw = np.max(np.abs(diffs_raw).reshape(B, -1), axis=1)  # [B]
    c_uniform_raw = np.quantile(max_dev_raw, 1.0 - alpha)
    ci_uniform_lower_raw = base_raw - c_uniform_raw
    ci_uniform_upper_raw = base_raw + c_uniform_raw

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
    )


