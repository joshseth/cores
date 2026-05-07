import numpy as np
import torch

def principal_angles(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    A = A.detach().cpu()
    B = B.detach().cpu()
    s = torch.linalg.svdvals(A.T @ B)
    s = torch.clamp(s, 0.0, 1.0)
    return torch.rad2deg(torch.acos(s)).numpy()


def projector_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    A = A.detach().cpu()
    B = B.detach().cpu()
    # PA = A @ A.T
    # PB = B @ B.T
    # r = A.shape[1]
    # return float(torch.trace(PA @ PB) / max(1, r))
    return (torch.linalg.norm(A.T @ B, ord="fro") ** 2) / np.sqrt(A.shape[1]*B.shape[1])
        
def cca_corrs(Hi, Hj, Ui, Uj, k=3):
    Ui = Ui.detach().cpu()
    Uj = Uj.detach().cpu()
    Zi = (Hi @ Ui).numpy()
    Zj = (Hj @ Uj).numpy()
    Zi -= Zi.mean(axis=0)
    Zj -= Zj.mean(axis=0)
    k = min(k, Zi.shape[1], Zj.shape[1])
    
    Cii = Zi.T @ Zi / len(Zi)
    Cjj = Zj.T @ Zj / len(Zj)
    Cij = Zi.T @ Zj / len(Zi)
    
    Wi = np.linalg.inv(np.linalg.cholesky(Cii + 1e-8 * np.eye(Zi.shape[1])))
    Wj = np.linalg.inv(np.linalg.cholesky(Cjj + 1e-8 * np.eye(Zj.shape[1])))
    
    _, s, _ = np.linalg.svd(Wi @ Cij @ Wj.T)
    return np.clip(s[:k], 0, 1)