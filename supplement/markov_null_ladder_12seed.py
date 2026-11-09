"""Twelve-seed Markov null ladder.

Every CCA arm uses the same rank (3), architecture, held-out sequences, and
number of examples.  The ladder separates four distinct controls:

* ambient random rank-3 subspaces in trained activation spaces;
* ACE rank-3 subspaces in untrained networks;
* the top three activation PCs in trained networks;
* the centered rank-3 output-readout contrast span.

The trained causal/operator battery is also evaluated at rank three, and the
ordered ACE prefix curve is extended from three to twelve training seeds.
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

from cores.ablations import ablate  # noqa: E402
from cores.extraction import ace  # noqa: E402
from cores.infer_mechanism import fit_core_dynamics  # noqa: E402
from cores.models import (  # noqa: E402
    MarkovDataset,
    collect_markov_activations,
    eval_accuracy,
)
from cores.utils import get_device, seed_all  # noqa: E402
from markov_model_cache import load_or_train  # noqa: E402


OUT = HERE / "markov_null_ladder_12seed" / "results.json"
SEEDS = [100 * index for index in range(12)]
D_MODEL = 64
RANK = 3
PREFIX_RANKS = list(range(1, 7))
N_RANDOM_DRAWS = 100
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


def random_basis(rng: np.random.Generator):
    q, _ = np.linalg.qr(rng.standard_normal((D_MODEL, RANK)))
    return torch.tensor(q[:, :RANK], dtype=torch.float32)


def pca_basis(hidden):
    centered = hidden.to(torch.float64)
    centered = centered - centered.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:RANK].T.to(torch.float32)


def readout_basis(model):
    weights = model.lm_head.weight.detach().cpu().to(torch.float64)
    weights = weights - weights.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(weights, full_matrices=False)
    return vh[:RANK].T.to(torch.float32)


def project(hidden, basis):
    coordinates = (hidden @ basis).numpy().astype(np.float64)
    return coordinates - coordinates.mean(0, keepdims=True)


def cca_coordinates(first, second):
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
    return np.clip(singular[:RANK], 0.0, 1.0)


def pairwise_cca(coordinates, prefix):
    rows = []
    for first, second in combinations(range(len(coordinates)), 2):
        rows.append(
            {
                "pair": f"{prefix}{first}-{prefix}{second}",
                "cca_corrs": [
                    float(value)
                    for value in cca_coordinates(
                        coordinates[first], coordinates[second]
                    )
                ],
            }
        )
    return rows


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


def evaluate_operator(hidden_sequence, basis, truth):
    _, eigenvalues, r2 = fit_core_dynamics(hidden_sequence, basis.cpu())
    eigenvalues = np.asarray(eigenvalues)
    eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]
    return {
        "r2": float(r2),
        "spectral_distance": matched_spectrum_error(eigenvalues, truth),
        "evals": to_pairs(eigenvalues),
    }


def main():
    device = get_device()
    print(f"device={device}", flush=True)
    train = MarkovDataset(T=T, seed=0, num_samples=3000, seq_len=32)
    test = MarkovDataset(T=T, seed=999, num_samples=3000, seq_len=32)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    truth = np.linalg.eigvals(T)
    truth = truth[np.argsort(-np.abs(truth))][1:]
    rng = np.random.default_rng(404)

    trained_hidden = []
    trained_sequences = []
    trained_models = []
    bases = {"ACE-3": [], "PCA-3": [], "Readout-3": [], "Random-3": []}
    prefix_rows = []
    evaluations = []
    native_ranks = []

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
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach().cpu()
        q_full, weights = ace(
            hidden.to(device, torch.float32),
            lambda current=model: (lambda h: current.lm_head(h)),
            sample_size=128,
            energy_threshold=0.999,
            max_rank=D_MODEL,
            return_full=True,
        )
        q_full = q_full.detach()
        nonnegative = np.clip(weights.detach().cpu().numpy(), 0.0, None)
        cumulative = np.cumsum(nonnegative) / nonnegative.sum()
        native_rank = int(np.searchsorted(cumulative, 0.999) + 1)
        native_ranks.append(native_rank)

        ace_three = q_full[:, :RANK].cpu()
        model_bases = {
            "ACE-3": ace_three,
            "PCA-3": pca_basis(hidden),
            "Readout-3": readout_basis(model),
            "Random-3": random_basis(rng),
        }
        for name, basis in model_bases.items():
            bases[name].append(basis)

        for rank in PREFIX_RANKS:
            basis = q_full[:, :rank]
            keep = ablate(
                model,
                model.layer,
                basis,
                "keep",
                test_loader,
                device,
                last_token=False,
            )
            remove = ablate(
                model,
                model.layer,
                basis,
                "remove",
                test_loader,
                device,
                last_token=False,
            )
            prefix_rows.append(
                {
                    "model": model_index,
                    "seed": seed,
                    "rank": rank,
                    "base": float(base),
                    "keep": float(keep),
                    "remove": float(remove),
                    "cumulative_balanced_energy": float(cumulative[rank - 1]),
                }
            )

        for name, basis in model_bases.items():
            if name == "ACE-3":
                prefix = next(
                    row
                    for row in prefix_rows
                    if row["model"] == model_index and row["rank"] == RANK
                )
                keep, remove = prefix["keep"], prefix["remove"]
            else:
                basis_device = basis.to(device)
                keep = ablate(
                    model,
                    model.layer,
                    basis_device,
                    "keep",
                    test_loader,
                    device,
                    last_token=False,
                )
                remove = ablate(
                    model,
                    model.layer,
                    basis_device,
                    "remove",
                    test_loader,
                    device,
                    last_token=False,
                )
            operator = evaluate_operator(hidden_sequence, basis, truth)
            evaluations.append(
                {
                    "model": model_index,
                    "seed": seed,
                    "subspace": name,
                    "rank": RANK,
                    "base": float(base),
                    "keep": float(keep),
                    "remove": float(remove),
                    **operator,
                }
            )

        trained_models.append(model)
        trained_sequences.append(hidden_sequence)
        trained_hidden.append(hidden)
        print(
            f"{'loaded' if from_cache else 'trained'} "
            f"{model_index + 1:02d}/{len(SEEDS)} seed={seed:4d} "
            f"native_rank={native_rank} base={base:.4f}",
            flush=True,
        )

    untrained_hidden = []
    untrained_bases = []
    for model_index, seed in enumerate(SEEDS):
        seed_all(seed + 9000)
        model = OneLayerTransformer(4, D_MODEL).to(device)
        model.eval()
        _, hidden_sequence, _ = collect_markov_activations(
            model, test_loader, device
        )
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach().cpu()
        q_full, _ = ace(
            hidden.to(device, torch.float32),
            lambda current=model: (lambda h: current.lm_head(h)),
            sample_size=128,
            energy_threshold=0.999,
            max_rank=D_MODEL,
            return_full=True,
        )
        untrained_hidden.append(hidden)
        untrained_bases.append(q_full[:, :RANK].detach().cpu())
        print(f"untrained {model_index + 1:02d}/{len(SEEDS)}", flush=True)

    trained_coordinates = {
        name: [
            project(hidden, basis)
            for hidden, basis in zip(trained_hidden, method_bases)
        ]
        for name, method_bases in bases.items()
    }
    untrained_coordinates = [
        project(hidden, basis)
        for hidden, basis in zip(untrained_hidden, untrained_bases)
    ]
    cca_profiles = {
        name: pairwise_cca(coordinates, "M")
        for name, coordinates in trained_coordinates.items()
        if name != "Random-3"
    }
    cca_profiles["Untrained ACE-3"] = pairwise_cca(untrained_coordinates, "U")
    trained_to_untrained = []
    for index in range(len(SEEDS)):
        trained_to_untrained.append(
            {
                "pair": f"M{index}-U{index}",
                "cca_corrs": [
                    float(value)
                    for value in cca_coordinates(
                        trained_coordinates["ACE-3"][index],
                        untrained_coordinates[index],
                    )
                ],
            }
        )
    cca_profiles["Trained→untrained"] = trained_to_untrained

    random_cca_draws = []
    for draw in range(N_RANDOM_DRAWS):
        draw_coordinates = [
            project(hidden, random_basis(rng)) for hidden in trained_hidden
        ]
        for record in pairwise_cca(draw_coordinates, "M"):
            random_cca_draws.append({"draw": draw, **record})
        if (draw + 1) % 10 == 0:
            print(f"random CCA draws {draw + 1}/{N_RANDOM_DRAWS}", flush=True)

    payload = {
        "protocol": {
            "seeds": SEEDS,
            "n_models": len(SEEDS),
            "n_model_pairs": len(SEEDS) * (len(SEEDS) - 1) // 2,
            "random_draws": N_RANDOM_DRAWS,
            "rank": RANK,
            "prefix_ranks": PREFIX_RANKS,
            "d_model": D_MODEL,
            "epochs": 40,
            "num_samples": 3000,
            "seq_len": 32,
            "jacobian_samples": 128,
            "all_cca_arms_use_identical_held_out_examples": True,
        },
        "truth_evals": to_pairs(truth),
        "native_ranks": native_ranks,
        "prefix_rows": prefix_rows,
        "evaluations": evaluations,
        "cca_profiles": cca_profiles,
        "ambient_random_cca_draws": random_cca_draws,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
