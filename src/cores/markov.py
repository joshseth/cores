import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from itertools import combinations
from cores.utils import get_device, seed_all, fmt_eig
from cores.models import (
    OneLayerTransformer, train_markov_lm, eval_accuracy,
    MarkovDataset, collect_markov_activations
    )
from cores.extraction import ace
from cores.ablations import ablate
from cores.compare_cores import principal_angles, projector_overlap, cca_corrs
from cores.infer_mechanism import fit_core_dynamics

device = get_device()
print("device:", device)

@dataclass(frozen=True)
class E1Args:
    d_model: int = 64
    epochs: int = 40
    lr: float = 1e-4
    seq_len: int = 32
    num_samples: int = 3000
    batch_size: int = 64
    k_models: int = 3
    seed: int = 0
    reuse_ckpt: bool = False
    force_retrain: bool = False
    energy_cut: float = 0.999
    output_dir: str = "experiments/markov_chain/results"  

# Markov chain stationary distribution, pi
def stat_dist(matrix):
    w, v = np.linalg.eig(matrix.T)
    pi = np.real(v[:, np.argmax(w)])
    return pi/pi.sum(), w

# Dynamic fit baseline
def oracle_r2(T):
    pi, _ = stat_dist(T)
    ones = np.ones_like(pi)
    m = (np.diag(pi) @ (T * (1 - T))).T @ ones
    v = pi * (1-pi)
    return 1 - 1/len(pi)* (ones.T @ (m/v))

# Helpers for serialization
def _evals_to_pairs(evals):
    """Convert a 1D array of complex eigenvalues to a list of [re, im] pairs."""
    arr = np.asarray(evals)
    return [[float(z.real), float(z.imag)] for z in arr]

@torch.no_grad()
def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(torch.float32).cpu()


args = E1Args()

# Markov transition probability matrix, T
alpha = 0.75
beta = 0.25
T = np.array([
    [alpha, beta, 0, 0],
    [0, alpha, beta, 0],
    [0, 0, alpha, beta],
    [beta, 0, 0, alpha],
], dtype=float)

r2o = oracle_r2(T)

print(f"Markov Transition Matrix:\n{T}")
piT, gtvals = stat_dist(T)
pi_str = ", ".join(fmt_eig(z) for z in piT)
gtvals_str = ", ".join(fmt_eig(z) for z in sorted(gtvals, reverse=True))
print(f"\nπ: [{pi_str}]")
print(f"λ: [{gtvals_str}]\n")

# Data
train_data = MarkovDataset(T=T, seed=0, num_samples=args.num_samples, seq_len=args.seq_len)
test_data = MarkovDataset(T=T, seed=999, num_samples=args.num_samples, seq_len=args.seq_len)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Train
models: list[torch.nn.Module] = []
Xseqs: list[torch.Tensor] = []
Hseqs: list[torch.Tensor] = []
HBseqs: list[torch.Tensor] = []
Yseqs: list[torch.Tensor] = []
print(f"\nTraining {args.k_models} Models ...")
for m_idx in range(args.k_models):
    seed_all(args.seed + 100 * m_idx)
    print(f"\n  Model {m_idx}")
    model = OneLayerTransformer(T.shape[0], args.d_model).to(device)

    for ep in train_markov_lm(model, train_loader, device, epochs=args.epochs, lr=args.lr):
        if ep % 10 == 0 or ep == args.epochs:
            acc = eval_accuracy(model, test_loader, device)
            print(f"  Epoch {ep:03d}, Acc={acc:.4f}")
    models.append(model)
    X, H, Y = collect_markov_activations(model, test_loader, device)
    Xseqs.append(X)
    HBseqs.append(H)
    Hseqs.append(H.reshape(-1, args.d_model))
    Yseqs.append(Y)

# ACE
print("\nAlgorithmic Core Extraction (ACE)")
bases, ranks = [], []
for i, m in enumerate(models):
    acts = Hseqs[i]

    U, r = ace(acts.to(next(m.parameters()).device, dtype=torch.float32),
               lambda: (lambda h: m.lm_head(h)),
    sample_size = 128,
    energy_threshold = args.energy_cut)
    bases.append(U)
    ranks.append(r)

for i, (U, r) in enumerate(zip(bases, ranks)):
    print(f"  Model {i}: Core rank={r}  d_model={U.shape[0]}")

# Ablate
print("\nAblations: Core Neccessity & Sufficiency")
ablations = []
for i, m in enumerate(models):
    acc_base = eval_accuracy(m, test_loader, device)
    U = bases[i]
    acc_keep = ablate(m, m.layer, U, "keep", test_loader, device, last_token=False)
    acc_rem  = ablate(m, m.layer, U, "remove", test_loader, device, last_token=False)
    ablations.append({"model": i,
                      "base": float(acc_base),
                      "keep": float(acc_keep),
                      "remove": float(acc_rem)})
    print(f"  Model {i}: Base={acc_base:.4f} Keep={acc_keep:.4f} Remove={acc_rem:.4f}")

# Compare Cores
metrics = {"Principal Angles": [], "Projector Overlap": [], "CCA Correlations": []}
pairwise_records = []  

for (i, (Ui, Hi)), (j, (Uj, Hj)) in combinations(enumerate(zip(bases, Hseqs)), 2):
    label = f"M{i}--M{j}"
    pa = principal_angles(Ui, Uj)
    metrics["Principal Angles"].append(f"  {label}: [{', '.join(f'{a:.1f}' for a in pa[:3])}]")
    ov = projector_overlap(Ui, Uj)
    metrics["Projector Overlap"].append(f"  {label}: {ov:.3f}")
    cca_ij = cca_corrs(Hi, Hj, Ui, Uj)
    metrics["CCA Correlations"].append(f"  {label}: [{', '.join(f'{c:.3f}' for c in cca_ij)}]")

    pairwise_records.append({
        "pair": f"M{i}-M{j}",
        "overlap": float(ov),
        "principal_angles": [float(a) for a in np.asarray(pa).ravel().tolist()],
        "cca_corrs": [float(c) for c in np.asarray(cca_ij).ravel().tolist()],
    })

for name, lines in metrics.items():
    print(f"\n{name}")
    print("\n".join(lines))


# Infer Mechanisms
print("Mechanism Discovery in Cores\nby Fitting Dynamics")
dynamics_records = []  # per-model ACE dynamics
for i, (H, U) in enumerate(zip(HBseqs, bases)):
    _, eigs, r2 = fit_core_dynamics(H, U)
    eigs_sorted = eigs[np.argsort(-np.abs(eigs))]
    eig_str = ", ".join(fmt_eig(z) for z in eigs_sorted)
    print(f"\nModel {i}")
    print(f"  λ: [{eig_str}]")
    print(f"  R²: {r2:.3f}   R²/R_oracle²: {r2/r2o:.3f}")
    dynamics_records.append({
        "model": i,
        "evals": _evals_to_pairs(eigs_sorted),
        "r2_mean": float(r2),
        "r2_over_oracle": float(r2 / r2o),
    })

# Compare Full Models / Consensus
def gcca(H_list, r):
    """Consensus coords from centered activation matrices."""
    M = np.concatenate(H_list, axis=1)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :r] * s[:r]  # (n, r)
print("\nEnsemble GCCA")
n_use = min(20000, Hseqs[0].shape[0])
rng = np.random.RandomState(args.seed)
idx = rng.permutation(Hseqs[0].shape[0])[:n_use]

H_list, mu_list = [], []
for H in Hseqs:
    H_np = H[idx].numpy().astype(np.float64)
    mu = H_np.mean(axis=0)
    H_list.append(H_np - mu)
    mu_list.append(mu)

M = np.concatenate(H_list, axis=1)
_, S, _ = np.linalg.svd(M, full_matrices=False)
cum = np.cumsum(S**2) / (np.sum(S**2) + 1e-12)
r_ens = min(int(args.d_model * 0.75), int(np.searchsorted(cum, args.energy_cut)) + 1)
print(f"  rank r={r_ens}")

G = gcca(H_list, r_ens)
U_ens_list = []
for Hc in H_list:
    W = np.linalg.pinv(Hc) @ G
    Q, _ = np.linalg.qr(W)
    U_ens_list.append(torch.tensor(Q, dtype=torch.float32))

ensemble_ablations = []  # used as "Consensus" 
for i, m in enumerate(models):
    U_ens = U_ens_list[i].to(device)
    acc_base = eval_accuracy(m, test_loader, device)
    acc_keep = ablate(m, m.layer, U_ens, "keep", test_loader, device, last_token=False)
    acc_rem = ablate(m, m.layer, U_ens, "remove", test_loader, device, last_token=False)
    print(f"  M{i}: Base={acc_base:.4f} Keep={acc_keep:.4f} Remove={acc_rem:.4f}")
    ensemble_ablations.append({
        "model": i,
        "base": float(acc_base),
        "keep": float(acc_keep),
        "remove": float(acc_rem),
    })


# =====
# Save 
# =====
print("\nSaving artifacts ...")

# 1) Cosine similarity between flattened model parameters
flats = [_flat_params(m) for m in models]
n_models = len(flats)
cos_mat = np.zeros((n_models, n_models), dtype=np.float32)
for i in range(n_models):
    for j in range(n_models):
        cos_mat[i, j] = float(F.cosine_similarity(flats[i], flats[j], dim=0).item())

# 2) Dataset summary (token counts) -> used to compute oracle / chance baselines
X0 = Xseqs[0].cpu().numpy().reshape(-1).astype(np.int64)
Y0 = Yseqs[0].cpu().numpy().reshape(-1).astype(np.int64)
V = T.shape[0]
counts_x = np.bincount(X0, minlength=V).astype(np.int64).tolist()
counts_y = np.bincount(Y0, minlength=V).astype(np.int64).tolist()
N = int(X0.size)

# 3) Bayes-optimal ("oracle") and chance accuracy baselines
T_max_row = T.max(axis=1)                       # (V,)
w_x = np.asarray(counts_x, dtype=float)
w_x = w_x / w_x.sum()
oracle_acc = float((w_x * T_max_row).sum())
y_star = int(np.argmax(piT))
chance_acc = float(counts_y[y_star] / sum(counts_y))

# 4) gcca
gcca_dynamics_canonical = {
    "evals": dynamics_records[0]["evals"],
    "r2_mean": dynamics_records[0]["r2_mean"],
    "source": "model_0_ace_dynamics",
}

# 5) final results dict
results = {
    "config": {
        "vocab_size": int(V),
        "d_model": args.d_model,
        "k_models": args.k_models,
        "seq_len": args.seq_len,
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "energy_cut": args.energy_cut,
    },
    "ground_truth": {
        "transition_matrix": T.tolist(),
        "stationary": piT.tolist(),
        "transition_evals": {
            "values": _evals_to_pairs(np.asarray(sorted(gtvals, key=lambda z: -np.abs(z)))),
        },
    },
    "dataset_summary": {
        "N": N,
        "counts_x": counts_x,
        "counts_y": counts_y,
    },
    "oracle": oracle_acc,
    "chance": chance_acc,
    "oracle_r2": float(r2o),
    "weight_cosine_similarity": cos_mat.tolist(),
    # per-model ACE ablations 
    "ablations": ablations,
    # geometric / statistical comparisons
    "pairwise": pairwise_records,
    # per-model ACE dynamics 
    "dynamics": dynamics_records,
    "gcca": {
        "ablations": ablations,
        "dynamics": gcca_dynamics_canonical,
    },
    # `ensemble_gcca` slot: Panel E "Consensus+/Consensus-"
    "ensemble_gcca": {
        "ablations": ensemble_ablations,
        "rank": int(r_ens),
    },
}

# 6) Write to disk
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
results_path = out_dir / "results.json"
with results_path.open("w") as f:
    json.dump(results, f, indent=2)
print(f"  wrote {results_path.resolve()}")

config_path = out_dir / "config.json"
with config_path.open("w") as f:
    json.dump(asdict(args), f, indent=2)
print(f"  wrote {config_path.resolve()}")
