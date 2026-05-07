import numpy as np
import torch

# Fit dynamics in LM core
# Used for Markov Chain Transformer
def fit_core_dynamics(H_seq, Q, center=True):
    Z = H_seq.detach().cpu().float().numpy() @ Q.detach().cpu().float().numpy()
    Zt = Z[:, :-1].reshape(-1, Z.shape[-1])
    Zn = Z[:, 1:].reshape(-1, Z.shape[-1])
    if center:
        Zt -= Zt.mean(axis=0)
        Zn -= Zn.mean(axis=0)
    A_T, *_ = np.linalg.lstsq(Zt, Zn, rcond=None)
    resid = Zn - Zt @ A_T
    r2 = 1 - (resid ** 2).sum(0) / ((Zn ** 2).sum(0) + 1e-12)
    evals = np.linalg.eigvals(A_T.T)
    return A_T.T, evals, r2.mean()



@torch.no_grad()
def token_core_reps(model, U_core, loader, device, P):
    """
    Mean core-projected activation per target token.
    Returns (P, r) array.
    (was: get_token_core_reps)
    """
    model.eval()
    Uc = U_core.to(device)
    Z_all, C_all = [], []
    for x, y in loader:
        _, _, h2 = model(x.to(device), return_states=True)
        Z_all.append((h2[:, -1, :] @ Uc).detach().cpu().numpy())
        C_all.append(y[:, -1].numpy())
    Z = np.concatenate(Z_all)
    C = np.concatenate(C_all)
    reps = np.zeros((P, Z.shape[1]), dtype=np.float64)
    for t in range(P):
        mask = C == t
        if mask.any():
            reps[t] = Z[mask].mean(axis=0)
    return reps

# Fit operator, first using SVD
# Used for Mod Add analysis
def fit_operator_svd(reps, *, k_max=None, svd_tol=1e-12,
                            ridge=0.0, center=True,
                            holdout_style="none", holdout_frac=0.075,
                            holdout_seed=0):
    """
    Fit shift operator reps[t] @ A ≈ reps[(t+1) % P] in SVD-truncated subspace.
    Returns (A, r2_train, r2_test, k, evals).
    r2_test is nan when holdout_style="none".
    (was: fit_shift_operator_svd, cleaned up)
    """
    X0 = np.asarray(reps, dtype=np.float64)
    P, r = X0.shape
    if P < 3:
        return np.zeros((1, 1)), 0.0, np.nan, 1, np.array([0.0])

    t = np.arange(P)
    tp1 = (t + 1) % P

    # Train/test edge split
    if holdout_style == "block_edges":
        rng = np.random.RandomState(int(holdout_seed))
        m = int(np.clip(np.round(holdout_frac * P), 1, P - 2))
        start = int(rng.randint(P))
        test_mask = np.zeros(P, dtype=bool)
        test_mask[(start + np.arange(m)) % P] = True
        train_mask = ~test_mask
        tr_idx = train_mask[t] & train_mask[tp1]
        te_idx = test_mask[t] & test_mask[tp1]
    else:
        tr_idx = np.ones(P, dtype=bool)
        te_idx = np.zeros(P, dtype=bool)

    Xtr, Ytr = X0[t[tr_idx]], X0[tp1[tr_idx]]
    Xte, Yte = X0[t[te_idx]], X0[tp1[te_idx]]

    if Xtr.shape[0] < 3:
        return np.zeros((1, 1)), 0.0, np.nan, 1, np.array([0.0])

    # Center using train mean only
    if center:
        mu = Xtr.mean(axis=0, keepdims=True)
        Xtr, Ytr = Xtr - mu, Ytr - mu
        if Xte.shape[0] > 0:
            Xte, Yte = Xte - mu, Yte - mu

    if k_max is None:
        k_max = min(P - 1, r)
    k_max = max(1, min(k_max, r))

    _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((1, 1)), 0.0, np.nan, 1, np.array([0.0])

    k = max(1, min(int(np.sum(s >= svd_tol * s[0])), k_max))
    Vk = Vt[:k].T
    Xk_tr, Yk_tr = Xtr @ Vk, Ytr @ Vk

    if ridge > 0:
        A = np.linalg.solve(Xk_tr.T @ Xk_tr + ridge * np.eye(k), Xk_tr.T @ Yk_tr)
    else:
        A, *_ = np.linalg.lstsq(Xk_tr, Yk_tr, rcond=None)

    def _r2(Yt, Yh):
        if Yt.size == 0:
            return np.nan
        return 1 - np.sum((Yt - Yh) ** 2) / (np.sum((Yt - Yt.mean(0)) ** 2) + 1e-12)

    r2_tr = _r2(Yk_tr, Xk_tr @ A)
    r2_te = _r2(Yte @ Vk, Xte @ Vk @ A) if Xte.shape[0] >= 3 else np.nan

    evals = np.linalg.eigvals(A)
    return A, float(r2_tr), float(r2_te), k, evals