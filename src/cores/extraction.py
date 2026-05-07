from __future__ import annotations
import numpy as np
import torch
from cores.models import eval_accuracy
from cores.ablations import ablate

"""
Algorithmic Core Extraction (ACE).

"""

def _balanced_svd(Cov, Obs, *, max_rank=None, eps=1e-8):
    """
    Core math: Lc Lc^T = Cov, Lo Lo^T = Obs, SVD(Lc^T Lo), basis = orth(Lc @ U).
    Returns (Q_full, singular_values_squared).
    """
    D = Cov.shape[0]

    def _factor(A):
        tr = float(torch.trace(A).item())
        jitter = eps * (tr / max(D, 1) if tr > 0 else 1.0)
        w, U = torch.linalg.eigh(A)
        return U @ torch.diag(torch.sqrt(torch.clamp(w, min=0.0) + jitter))

    Lc, Lo = _factor(Cov), _factor(Obs)
    U, s, _ = torch.linalg.svd(Lc.T @ Lo, full_matrices=False)

    r = D if max_rank is None else min(D, int(max_rank))
    Q, _ = torch.linalg.qr(Lc @ U[:, :r], mode="reduced")
    return Q, s[:r].square()


def _select_rank(w, energy_threshold, min_rank=1):
    """Pick rank from cumulative energy of weights w."""
    w = torch.clamp(w, min=0.0)
    total = w.sum()
    if total <= 0:
        return max(min_rank, 1)
    cum = torch.cumsum(w, dim=0) / (total + 1e-12)
    hit = (cum >= energy_threshold).nonzero(as_tuple=True)[0]
    r = (int(hit[0].item()) + 1) if hit.numel() else w.numel()
    return max(r, min_rank)


def ace_from_matrices(Cov, Obs, *, r=None, energy_threshold=0.99,
                      min_rank=1, max_rank=None, eps=1e-8):
    """ACE from precomputed Cov and Obs (e.g. SVA where Obs is built via autograd)."""
    Cov = Cov.to(torch.float64)
    Obs = Obs.to(torch.float64)
    Q, w = _balanced_svd(Cov, Obs, max_rank=max_rank, eps=eps)
    if r is None:
        r = _select_rank(w, energy_threshold, min_rank)
    return Q[:, :r].float(), w.cpu()


def ace(activations, tail_factory, *, sample_size=128, energy_threshold=0.99,
        min_rank=1, max_rank=None, return_full=False, eps=1e-8):
    """
    ACE from activations + tail function (e.g. lm_head).
    Builds Cov from activations, Obs via vmap(jacrev(tail)) on a subsample.

    tail_factory: callable returning f: R^D -> R^K
        e.g. lambda: (lambda h: model.lm_head(h))
    """
    N, D = activations.shape
    dev = activations.device
    la = torch.device("cpu") if dev.type == "mps" else dev

    # Cov
    #H = activations.detach().to(la, torch.float32)
    H = activations.detach().to(la).to(torch.float64)
    Hc = H - H.mean(0, keepdim=True)
    Cov = (Hc.T @ Hc) / max(1, N - 1)

    # Obs via jacrev
    acts = activations.detach().float()
    S = min(sample_size, N)
    h_sub = acts[torch.randperm(N, device=dev)[:S]].clone().requires_grad_(True)
    f = tail_factory()

    def gram(h):
        J = torch.func.jacrev(f)(h).reshape(-1, D)
        return J.T @ J

    #Obs = torch.vmap(gram)(h_sub).mean(0).to(la, torch.float32)
    Obs = torch.vmap(gram)(h_sub).mean(0).to(la).to(torch.float64)

    # Extract
    Q, w = _balanced_svd(Cov, Obs, max_rank=max_rank, eps=eps)
    Q = Q.to(dev, torch.float32)
    w = w.cpu()

    if return_full:
        return Q, w
    r = _select_rank(w, energy_threshold, min_rank)
    return Q[:, :r], r


def _bisect_nec(model, Q_full, target_floor, loader, device):
    """
    Binary search for smallest r where remove-accuracy <= target_floor.
    Assumes remove-accuracy is monotonically non-increasing in r.
    (was: linear scan in compute_nec_suf_refine_from_r0)
    """
    r_max = Q_full.shape[1]
    lo, hi = 1, r_max
    best_r = r_max
    while lo <= hi:
        mid = (lo + hi) // 2
        U = Q_full[:, :mid].contiguous()
        acc = ablate(model, model.layer2, U, "remove", loader, device,  last_token=True)
        if acc <= target_floor:
            best_r = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best_r


def _bisect_suf(model, Q_full, base_acc, loader, device, tol=1e-3):
    """
    Binary search for smallest r where keep-accuracy >= base_acc - tol.
    Assumes keep-accuracy is monotonically non-decreasing in r.
    """
    r_max = Q_full.shape[1]
    lo, hi = 1, r_max
    best_r = r_max
    while lo <= hi:
        mid = (lo + hi) // 2
        U = Q_full[:, :mid].contiguous()
        acc = ablate(model, model.layer2, U, "keep", loader, device, last_token=True)
        if acc >= base_acc - tol:
            best_r = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best_r


def _scan_nec(model, Q_full, target_floor, loader, device, start=1):
    """Linear scan"""
    r_max = Q_full.shape[1]
    for r in range(start, r_max + 1):
        U = Q_full[:, :r].contiguous()
        acc = ablate(model, model.layer2, U, "remove", loader, device, last_token=True)
        if acc <= target_floor:
            return r
    return r_max

def _scan_suf(model, Q_full, base_acc, loader, device, start=1, tol=1e-3):
    """Linear scan"""
    r_max = Q_full.shape[1]
    for r in range(start, r_max + 1):
        U = Q_full[:, :r].contiguous()
        acc = ablate(model, model.layer2, U, "keep", loader, device, last_token=True)
        if acc >= base_acc - tol:
            return r
    return r_max

def ace_ablate(model, H_last, device, test_loader, *, vocab_size=53, d_model=128,
               sample_size=1000, energy_threshold=0.99):
    H = H_last.to(device=device, dtype=torch.float32)
    target_func_factory = lambda: (lambda h: model.lm_head(h))
    Q_full, vals = ace(H, target_func_factory, sample_size=sample_size,
                       min_rank=1, max_rank=d_model, return_full=True)

    v = vals.numpy() if hasattr(vals, 'numpy') else np.asarray(vals, dtype=np.float64)
    v = np.clip(v, 0, None)
    total = v.sum()
    if total > 0:
        cum = np.cumsum(v) / (total + 1e-12)
        hit = np.where(cum >= energy_threshold)[0]
        r_energy = int(hit[0] + 1) if hit.size else len(v)
    else:
        r_energy = 1

    base_acc = eval_accuracy(model, test_loader, device, last_token=True)
    target_floor = 2.0 / vocab_size

    r_nec_strict = _scan_nec(model, Q_full, target_floor, test_loader, device, start=r_energy)
    r_suf_strict = _scan_suf(model, Q_full, base_acc,    test_loader, device, start=1)

    return Q_full, r_energy, r_nec_strict, r_suf_strict, base_acc
