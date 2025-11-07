import numpy as np
import torch


def compute_individual_effects(
    source,
    A,
    outcome_mode: str = "with_self",
    g_self=None,
    *,
    X=None,
    nbrs_idx=None,
    t=None,
):
    """Compute IME/ISE/ITE in a unified way.

    Usage:
      - From attention matrix: compute_individual_effects(W, A, outcome_mode, g_self=...)
      - From model: compute_individual_effects(model, A, outcome_mode, X=X, nbrs_idx=nbrs_idx, t=t)

    with_self:
      IME = diag(W); ISE = sum(W * (A - I)); ITE = IME + ISE
    separate_self:
      IME = g (self-treatment); ISE = sum(W * (A - I)); ITE = IME + ISE
    """
    # Resolve attention matrix and optional self weights
    if hasattr(source, "predict") and X is not None and nbrs_idx is not None and t is not None:
        # Model path
        with torch.no_grad():
            # Align X width with model's expected input dimension
            X_arr = X
            try:
                if hasattr(source, "self_mlp") and isinstance(source.self_mlp[0], torch.nn.Linear):
                    in_feats = int(source.self_mlp[0].in_features)
                    if X_arr.shape[1] > in_feats:
                        X_arr = X_arr[:, :in_feats]
                elif hasattr(source, "attention_mlp") and isinstance(source.attention_mlp[0], torch.nn.Linear):
                    pair_in = int(source.attention_mlp[0].in_features)
                    in_feats = max(1, pair_in // 2)
                    if X_arr.shape[1] > in_feats:
                        X_arr = X_arr[:, :in_feats]
            except Exception:
                # Fallback: use X as-is if inspection fails
                X_arr = X

            X_t = torch.tensor(X_arr, dtype=torch.float32)
            t_t = torch.tensor(t, dtype=torch.float32)
            out = source.cpu().predict(X_t, nbrs_idx, t_t)
        if isinstance(out, tuple) and len(out) == 3:
            W, _, g = out
        else:
            W, _ = out
            g = None
    else:
        # Matrix path
        W = source
        g = g_self

    # Ensure numpy arrays
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if g is not None and isinstance(g, torch.Tensor):
        g = g.detach().cpu().numpy()

    n = A.shape[0]
    if outcome_mode == "with_self":
        IME = np.diag(W)
        ISE = np.sum(W * (A - np.eye(n)), axis=1)
        ITE = IME + ISE
        return IME, ISE, ITE

    # separate_self
    if g is None:
        raise ValueError("Self-treatment weights g are required for outcome_mode='separate_self'")
    IME = np.asarray(g)
    ISE = np.sum(W * (A - np.eye(n)), axis=1)
    ITE = IME + ISE
    return IME, ISE, ITE


def evaluate(IME, ISE, ITE, IME_est, ISE_est, ITE_est, printing=True):
    # Compute AME (Absolute Mean Error)
    IME_AME = np.abs(np.mean(IME) - np.mean(IME_est))
    ISE_AME = np.abs(np.mean(ISE) - np.mean(ISE_est))
    ITE_AME = np.abs(np.mean(ITE) - np.mean(ITE_est))

    # Also report mean differences (signed)
    IME_AME = np.mean(IME) - np.mean(IME_est)
    ISE_AME = np.mean(ISE) - np.mean(ISE_est)
    ITE_AME = np.mean(ITE) - np.mean(ITE_est)

    # Compute PEHE (sqrt of mean squared error)
    IME_PEHE = np.sqrt(np.mean((IME - IME_est) ** 2))
    ISE_PEHE = np.sqrt(np.mean((ISE - ISE_est) ** 2))
    ITE_PEHE = np.sqrt(np.mean((ITE - ITE_est) ** 2))

    if printing:
        print("IME: AME = {:.5f}, PEHE = {:.5f}".format(IME_AME, IME_PEHE))
        print("ISE: AME = {:.5f}, PEHE = {:.5f}".format(ISE_AME, ISE_PEHE))
        print("ITE: AME = {:.5f}, PEHE = {:.5f}".format(ITE_AME, ITE_PEHE))

    results = {
        "IME_AME": round(float(IME_AME), 5),
        "ISE_AME": round(float(ISE_AME), 5),
        "ITE_AME": round(float(ITE_AME), 5),
        "IME_PEHE": round(float(IME_PEHE), 5),
        "ISE_PEHE": round(float(ISE_PEHE), 5),
        "ITE_PEHE": round(float(ITE_PEHE), 5),
    }
    return results


def compute_dbml_baseline(y, mu_hat, e_hat, A, n, t):
    first_column = (t - e_hat)
    # first_column=(treat_binary-e_star)
    second_column = (A - np.eye(n)).dot(first_column)
    predictor = np.column_stack((first_column, second_column)) / np.sum(A, 1).reshape(-1, 1)
    response = y - mu_hat
    # predictor=np.column_stack((first_column,second_column))
    beta_hat = np.linalg.inv(predictor.T @ predictor) @ (predictor.T @ response)
    # DBML method (two parameter estimate)
    IME_est = beta_hat[0] / np.sum(A, 1)
    ISE_est = np.sum(beta_hat[1] * (A - np.eye(n)), 1) / np.sum(A, 1)
    ITE_est = IME_est + ISE_est
    return IME_est, ISE_est, ITE_est, beta_hat


