"""Full-state Markov control for the Markov null-control figure.

The existing ladder compares rank-three subspaces. This control instead uses
the complete 64-dimensional trained hidden state. It asks what CCA, causal
ablation, operator recovery, and ACE coverage look like before any compact
subspace is extracted.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from cores.ablations import ablate  # noqa: E402
from cores.infer_mechanism import fit_core_dynamics  # noqa: E402
from cores.models import MarkovDataset, collect_markov_activations, eval_accuracy  # noqa: E402
from cores.utils import get_device  # noqa: E402
from markov_model_cache import load_or_train  # noqa: E402


OUT = HERE / "markov_full_model_control" / "results.json"
SEEDS = [100 * index for index in range(12)]
D_MODEL = 64
DISPLAY_CCS = 3
T = np.array(
    [
        [0.75, 0.25, 0.0, 0.0],
        [0.0, 0.75, 0.25, 0.0],
        [0.0, 0.0, 0.75, 0.25],
        [0.25, 0.0, 0.0, 0.75],
    ],
    dtype=float,
)


def to_pairs(values):
    return [[float(value.real), float(value.imag)] for value in np.asarray(values)]


def cca_coordinates(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Regularized CCA matching the existing ladder implementation."""
    n_examples = len(first)
    c_first = first.T @ first / n_examples
    c_second = second.T @ second / n_examples
    c_cross = first.T @ second / n_examples
    eye_first = np.eye(c_first.shape[0])
    eye_second = np.eye(c_second.shape[0])
    whiten_first = np.linalg.inv(np.linalg.cholesky(c_first + 1e-8 * eye_first))
    whiten_second = np.linalg.inv(
        np.linalg.cholesky(c_second + 1e-8 * eye_second)
    )
    singular = np.linalg.svd(
        whiten_first @ c_cross @ whiten_second.T,
        compute_uv=False,
    )
    return np.clip(singular[:DISPLAY_CCS], 0.0, 1.0)


def matched_truth_error(fitted, truth):
    """Best injective match of the three true modes into a larger spectrum."""
    fitted = np.asarray(fitted, dtype=complex)
    truth = np.asarray(truth, dtype=complex)
    costs = np.abs(truth[:, None] - fitted[None, :])
    truth_rows, fitted_cols = linear_sum_assignment(costs)
    scale = np.max(np.abs(truth))
    return float(costs[truth_rows, fitted_cols].mean() / scale)


def main():
    device = get_device()
    print(f"device={device}", flush=True)

    train = MarkovDataset(T=T, seed=0, num_samples=3000, seq_len=32)
    test = MarkovDataset(T=T, seed=999, num_samples=3000, seq_len=32)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    truth = np.linalg.eigvals(T)
    truth = truth[np.argsort(-np.abs(truth))][1:]
    identity = torch.eye(D_MODEL, dtype=torch.float32)
    hidden_states: list[np.ndarray] = []
    evaluations = []

    for model_index, seed in enumerate(SEEDS):
        model, from_cache = load_or_train(
            seed,
            train_loader,
            device,
            d_model=D_MODEL,
            epochs=40,
            lr=1e-4,
        )
        base = eval_accuracy(model, test_loader, device)
        _, hidden_sequence, _ = collect_markov_activations(
            model, test_loader, device
        )
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach().cpu().numpy()
        hidden = hidden.astype(np.float64)
        hidden -= hidden.mean(axis=0, keepdims=True)
        hidden_states.append(hidden)

        full_basis = identity.to(device)
        keep = ablate(
            model,
            model.layer,
            full_basis,
            "keep",
            test_loader,
            device,
            last_token=False,
        )
        remove = ablate(
            model,
            model.layer,
            full_basis,
            "remove",
            test_loader,
            device,
            last_token=False,
        )
        _, eigenvalues, r2 = fit_core_dynamics(hidden_sequence, identity)
        evaluations.append(
            {
                "model": model_index,
                "seed": seed,
                "subspace": "Full state (64D)",
                "rank": D_MODEL,
                "base": float(base),
                "keep": float(keep),
                "remove": float(remove),
                "r2": float(r2),
                "spectral_distance": matched_truth_error(eigenvalues, truth),
                "evals": to_pairs(eigenvalues),
                "ace_coverage": 1.0,
            }
        )
        print(
            f"{'loaded' if from_cache else 'trained'} "
            f"{model_index + 1:02d}/{len(SEEDS)} seed={seed:4d} "
            f"keep={keep:.4f} remove={remove:.4f} "
            f"spectrum={evaluations[-1]['spectral_distance']:.5f}",
            flush=True,
        )

    cca_profiles = []
    for first, second in combinations(range(len(hidden_states)), 2):
        correlations = cca_coordinates(hidden_states[first], hidden_states[second])
        cca_profiles.append(
            {
                "pair": f"M{first}-M{second}",
                "cca_corrs": [float(value) for value in correlations],
            }
        )
        print(
            f"CCA M{first}-M{second}: "
            + ", ".join(f"{value:.4f}" for value in correlations),
            flush=True,
        )

    payload = {
        "protocol": {
            "seeds": SEEDS,
            "n_models": len(SEEDS),
            "n_model_pairs": len(cca_profiles),
            "state_dimension": D_MODEL,
            "displayed_canonical_components": DISPLAY_CCS,
            "num_samples": 3000,
            "seq_len": 32,
            "all_models_use_identical_held_out_examples": True,
            "spectrum_note": (
                "Minimum injective match of the three true nonstationary "
                "eigenvalues into the full 64-dimensional fitted spectrum; "
                "extra fitted eigenvalues are not penalized."
            ),
            "overlap_note": (
                "The full hidden space contains ACE-3 by definition, so "
                "normalized ACE projector coverage equals one."
            ),
        },
        "truth_evals": to_pairs(truth),
        "cca_profiles": cca_profiles,
        "evaluations": evaluations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
