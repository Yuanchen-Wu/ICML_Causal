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

    loss_fn = nn.MSELoss()
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
            loss = loss_fn(Y_pred, batch_Y)
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


