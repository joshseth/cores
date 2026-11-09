"""Pairwise full-state CCA control for the Markov null-control figure.

For every pair of independently trained seeds, CCA is fit between the two
complete 64-dimensional hidden states on identical held-out sequences.  The
top three canonical directions on each side are then treated as rank-three
subspaces and evaluated for causal concentration, operator-spectrum recovery,
and within-model overlap with ACE-3.

Panel-A CCA values use all 66 different-seed model pairs.  For the intervention
panels, each seed's reported point is its mean over the 11 partner-derived
CCA-3 subspaces, so the displayed sampling unit remains the trained model.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from cores.extraction import ace  # noqa: E402
from cores.infer_mechanism import fit_core_dynamics  # noqa: E402
from cores.models import MarkovDataset, collect_markov_activations, eval_accuracy  # noqa: E402
from cores.utils import get_device  # noqa: E402
from markov_model_cache import load_or_train  # noqa: E402


OUT = HERE / "markov_pairwise_cca3_control" / "results.json"
SEEDS = [100 * index for index in range(12)]
D_MODEL = 64
RANK = 3
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


def matched_spectrum_error(fitted, truth):
    fitted = np.asarray(fitted, dtype=complex)
    truth = np.asarray(truth, dtype=complex)
    scale = np.max(np.abs(truth))
    return min(
        sum(
            abs(fitted[index] - truth[order[index]])
            for index in range(len(fitted))
        )
        / len(fitted)
        / scale
        for order in permutations(range(len(truth)))
    )


def pairwise_cca_bases(first: np.ndarray, second: np.ndarray):
    """Return top correlations and Euclidean bases for both canonical spans."""
    n_examples = len(first)
    c_first = first.T @ first / n_examples
    c_second = second.T @ second / n_examples
    c_cross = first.T @ second / n_examples
    eye = np.eye(D_MODEL)
    whiten_first = np.linalg.inv(np.linalg.cholesky(c_first + 1e-8 * eye))
    whiten_second = np.linalg.inv(np.linalg.cholesky(c_second + 1e-8 * eye))
    left, singular, right_t = np.linalg.svd(
        whiten_first @ c_cross @ whiten_second.T,
        full_matrices=False,
    )

    # CCA coefficients in the original feature coordinates.  QR preserves each
    # top-three span while producing an orthonormal basis for intervention.
    coefficients_first = whiten_first.T @ left[:, :RANK]
    coefficients_second = whiten_second.T @ right_t.T[:, :RANK]
    basis_first, _ = np.linalg.qr(coefficients_first)
    basis_second, _ = np.linalg.qr(coefficients_second)
    return (
        np.clip(singular[:RANK], 0.0, 1.0),
        torch.tensor(basis_first[:, :RANK], dtype=torch.float32),
        torch.tensor(basis_second[:, :RANK], dtype=torch.float32),
    )


def direct_intervention_accuracy(hidden, labels, weight, bias, basis):
    """Exact post-layer keep/remove intervention for the one-layer model."""
    hidden = hidden.to(torch.float32)
    labels = labels.reshape(-1)
    basis = basis.to(torch.float32)
    projected = (hidden @ basis) @ basis.T
    keep_logits = projected @ weight.T + bias
    remove_logits = (hidden - projected) @ weight.T + bias
    keep = (keep_logits.argmax(-1) == labels).to(torch.float32).mean().item()
    remove = (remove_logits.argmax(-1) == labels).to(torch.float32).mean().item()
    return keep, remove


def operator_metrics(hidden_sequence, basis, truth):
    _, eigenvalues, r2 = fit_core_dynamics(hidden_sequence, basis)
    eigenvalues = np.asarray(eigenvalues)
    eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]
    return {
        "r2": float(r2),
        "spectral_distance": matched_spectrum_error(eigenvalues, truth),
        "evals": to_pairs(eigenvalues),
    }


def projector_overlap(first, second):
    first = first.to(torch.float64)
    second = second.to(torch.float64)
    return float(torch.linalg.norm(first.T @ second, ord="fro") ** 2 / RANK)


def main():
    device = get_device()
    print(f"device={device}", flush=True)
    train = MarkovDataset(T=T, seed=0, num_samples=3000, seq_len=32)
    test = MarkovDataset(T=T, seed=999, num_samples=3000, seq_len=32)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    truth = np.linalg.eigvals(T)
    truth = truth[np.argsort(-np.abs(truth))][1:]
    models = []

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
        _, hidden_sequence, labels = collect_markov_activations(
            model, test_loader, device
        )
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach().cpu()
        centered = hidden.to(torch.float64)
        centered = centered - centered.mean(0, keepdim=True)
        q_full, _ = ace(
            hidden.to(device, torch.float32),
            lambda current=model: (lambda h: current.lm_head(h)),
            sample_size=128,
            energy_threshold=0.999,
            max_rank=D_MODEL,
            return_full=True,
        )
        models.append(
            {
                "model": model_index,
                "seed": seed,
                "base": float(base),
                "hidden": hidden,
                "centered": centered.numpy(),
                "sequence": hidden_sequence,
                "labels": labels,
                "weight": model.lm_head.weight.detach().cpu(),
                "bias": model.lm_head.bias.detach().cpu(),
                "ace": q_full[:, :RANK].detach().cpu(),
            }
        )
        print(
            f"{'loaded' if from_cache else 'trained'} "
            f"{model_index + 1:02d}/{len(SEEDS)} seed={seed:4d} base={base:.4f}",
            flush=True,
        )

    pair_profiles = []
    pair_sides = [[] for _ in models]
    for first, second in combinations(range(len(models)), 2):
        correlations, first_basis, second_basis = pairwise_cca_bases(
            models[first]["centered"],
            models[second]["centered"],
        )
        pair_name = f"M{first}-M{second}"
        pair_profiles.append(
            {
                "pair": pair_name,
                "first_seed": models[first]["seed"],
                "second_seed": models[second]["seed"],
                "cca_corrs": [float(value) for value in correlations],
            }
        )

        for model_index, partner_index, basis in (
            (first, second, first_basis),
            (second, first, second_basis),
        ):
            current = models[model_index]
            keep, remove = direct_intervention_accuracy(
                current["hidden"],
                current["labels"],
                current["weight"],
                current["bias"],
                basis,
            )
            operator = operator_metrics(current["sequence"], basis, truth)
            pair_sides[model_index].append(
                {
                    "pair": pair_name,
                    "partner_model": partner_index,
                    "partner_seed": models[partner_index]["seed"],
                    "keep": keep,
                    "remove": remove,
                    "overlap_with_ace": projector_overlap(current["ace"], basis),
                    **operator,
                }
            )
        print(
            f"{pair_name}: "
            + ", ".join(f"{value:.4f}" for value in correlations),
            flush=True,
        )

    per_model = []
    for model_index, records in enumerate(pair_sides):
        current = models[model_index]
        per_model.append(
            {
                "model": model_index,
                "seed": current["seed"],
                "base": current["base"],
                "n_partner_subspaces": len(records),
                "keep": float(np.mean([row["keep"] for row in records])),
                "remove": float(np.mean([row["remove"] for row in records])),
                "spectral_distance": float(
                    np.mean([row["spectral_distance"] for row in records])
                ),
                "overlap_with_ace": float(
                    np.mean([row["overlap_with_ace"] for row in records])
                ),
                "partner_sd": {
                    "keep": float(np.std([row["keep"] for row in records], ddof=1)),
                    "remove": float(
                        np.std([row["remove"] for row in records], ddof=1)
                    ),
                    "spectral_distance": float(
                        np.std(
                            [row["spectral_distance"] for row in records], ddof=1
                        )
                    ),
                    "overlap_with_ace": float(
                        np.std([row["overlap_with_ace"] for row in records], ddof=1)
                    ),
                },
                "partner_records": records,
            }
        )

    payload = {
        "protocol": {
            "seeds": SEEDS,
            "n_models": len(SEEDS),
            "n_model_pairs": len(pair_profiles),
            "all_pairs_are_different_training_seeds": True,
            "state_dimension": D_MODEL,
            "canonical_subspace_rank": RANK,
            "num_samples": 3000,
            "seq_len": 32,
            "all_models_use_identical_held_out_examples": True,
            "panel_a_note": (
                "Top three canonical correlations from CCA fit between the "
                "complete 64D hidden states of each different-seed model pair."
            ),
            "intervention_note": (
                "For each pair, the top three canonical directions on each "
                "model side define a rank-3 Euclidean projection subspace. "
                "Each model-level point averages its 11 partner-derived spaces."
            ),
        },
        "truth_evals": to_pairs(truth),
        "cca_profiles": pair_profiles,
        "per_model": per_model,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
