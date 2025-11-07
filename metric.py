import numpy as np
import torch


def compute_individual_effects(
    attention_matrix,
    A,
    outcome_mode: str = "with_self",
    g_self: np.ndarray | torch.Tensor | None = None,
):
    """Compute IME/ISE/ITE from an attention matrix and adjacency.

    Args:
      attention_matrix: np.ndarray or torch.Tensor of shape (n, n)
      A: adjacency matrix (np.ndarray) of shape (n, n)
      outcome_mode: "with_self" or "separate_self"
      g_self: per-node self-treatment weights (required if outcome_mode == "separate_self")
    """
    W = attention_matrix
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if g_self is not None and isinstance(g_self, torch.Tensor):
        g_self = g_self.detach().cpu().numpy()

    n = A.shape[0]
    if outcome_mode == "with_self":
        IME = np.diag(W)
        ISE = np.sum(W * (A - np.eye(n)), axis=1)
        ITE = IME + ISE
        return IME, ISE, ITE

    if g_self is None:
        raise ValueError("g_self must be provided when outcome_mode='separate_self'")
    IME = np.asarray(g_self)
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


