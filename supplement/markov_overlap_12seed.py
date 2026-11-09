"""Within-coordinate projector overlaps with trained Markov ACE-3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from cores.extraction import ace  # noqa: E402
from cores.models import (  # noqa: E402
    MarkovDataset,
    OneLayerTransformer,
    collect_markov_activations,
)
from cores.utils import get_device, seed_all  # noqa: E402
from markov_model_cache import load_or_train  # noqa: E402


OUT = HERE / "markov_overlap_12seed" / "results.json"
SEEDS = [100 * index for index in range(12)]
D_MODEL = 64
RANK = 3
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


def projector_overlap(first, second):
    first = first.detach().cpu().to(torch.float64)
    second = second.detach().cpu().to(torch.float64)
    return float(torch.linalg.norm(first.T @ second, ord="fro") ** 2 / RANK)


def pca_basis(hidden):
    centered = hidden.detach().cpu().to(torch.float64)
    centered -= centered.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:RANK].T.to(torch.float32)


def readout_basis(model):
    weights = model.lm_head.weight.detach().cpu().to(torch.float64)
    weights -= weights.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(weights, full_matrices=False)
    return vh[:RANK].T.to(torch.float32)


def random_basis(rng):
    q, _ = np.linalg.qr(rng.standard_normal((D_MODEL, RANK)))
    return torch.tensor(q[:, :RANK], dtype=torch.float32)


def ace_basis(model, hidden, device):
    full, _ = ace(
        hidden.to(device, torch.float32),
        lambda current=model: (lambda h: current.lm_head(h)),
        sample_size=128,
        energy_threshold=0.999,
        max_rank=D_MODEL,
        return_full=True,
    )
    return full[:, :RANK].detach().cpu()


def main():
    device = get_device()
    print(f"device={device}", flush=True)
    train = MarkovDataset(T=T, seed=0, num_samples=3000, seq_len=32)
    test = MarkovDataset(T=T, seed=999, num_samples=3000, seq_len=32)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)
    rng = np.random.default_rng(1701)
    records = []
    trained_ace = []

    for model_index, seed in enumerate(SEEDS):
        model, from_cache = load_or_train(
            seed,
            train_loader,
            device,
            d_model=D_MODEL,
            epochs=40,
            lr=1e-4,
        )
        _, hidden_sequence, _ = collect_markov_activations(
            model, test_loader, device
        )
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach().cpu()
        ace_three = ace_basis(model, hidden, device)
        trained_ace.append(ace_three)
        comparators = {
            "ACE-3": ace_three,
            "PCA-3": pca_basis(hidden),
            "Centered readout-3": readout_basis(model),
        }
        for name, basis in comparators.items():
            records.append(
                {
                    "model": model_index,
                    "seed": seed,
                    "condition": name,
                    "draw": None,
                    "projector_overlap": projector_overlap(ace_three, basis),
                }
            )
        for draw in range(N_RANDOM_DRAWS):
            records.append(
                {
                    "model": model_index,
                    "seed": seed,
                    "condition": "Ambient random-3",
                    "draw": draw,
                    "projector_overlap": projector_overlap(
                        ace_three, random_basis(rng)
                    ),
                }
            )
        print(
            f"{'loaded' if from_cache else 'trained'} "
            f"{model_index + 1:02d}/{len(SEEDS)}",
            flush=True,
        )

    for model_index, seed in enumerate(SEEDS):
        seed_all(seed + 9000)
        model = OneLayerTransformer(4, D_MODEL).to(device)
        model.eval()
        _, hidden_sequence, _ = collect_markov_activations(
            model, test_loader, device
        )
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach().cpu()
        untrained = ace_basis(model, hidden, device)
        records.append(
            {
                "model": model_index,
                "seed": seed,
                "condition": "Untrained ACE-3",
                "draw": None,
                "projector_overlap": projector_overlap(
                    trained_ace[model_index], untrained
                ),
            }
        )
        print(f"untrained {model_index + 1:02d}/{len(SEEDS)}", flush=True)

    payload = {
        "protocol": {
            "seeds": SEEDS,
            "n_models": len(SEEDS),
            "rank": RANK,
            "d_model": D_MODEL,
            "random_draws_per_model": N_RANDOM_DRAWS,
            "random_overlap_expectation": RANK / D_MODEL,
            "untrained_note": (
                "Raw ambient-coordinate overlap with the paired trained seed; "
                "cross-model hidden coordinates are not identifiable."
            ),
        },
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
