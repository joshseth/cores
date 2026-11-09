"""Compare cumulative ACE balanced energy with PCA variance on Markov models."""

from __future__ import annotations

import csv
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
from cores.models import MarkovDataset, collect_markov_activations  # noqa: E402
from cores.utils import get_device, seed_all  # noqa: E402
from markov_model_cache import load_or_train  # noqa: E402


OUT_DIR = HERE / "markov_energy_spectra_12seed"
RESULTS_JSON = OUT_DIR / "results.json"
EVIDENCE_CSV = (
    HERE / "outputs"
    / "24_markov_energy_evidence.csv"
)

SEEDS = [100 * index for index in range(12)]
D_MODEL = 64
THRESHOLDS = (0.99, 0.999)
T = np.array(
    [
        [0.75, 0.25, 0.0, 0.0],
        [0.0, 0.75, 0.25, 0.0],
        [0.0, 0.0, 0.75, 0.25],
        [0.25, 0.0, 0.0, 0.75],
    ],
    dtype=float,
)


def normalized_cumulative(weights: torch.Tensor) -> torch.Tensor:
    weights = torch.clamp(weights.detach().cpu().to(torch.float64), min=0)
    total = weights.sum()
    if total <= 0:
        raise RuntimeError("energy spectrum has zero total mass")
    return torch.cumsum(weights, dim=0) / total


def cutoff_rank(cumulative: torch.Tensor, threshold: float) -> int:
    target = torch.tensor(threshold, dtype=cumulative.dtype)
    return int(torch.searchsorted(cumulative, target).item()) + 1


def main() -> None:
    device = get_device()
    print(f"device={device}", flush=True)

    train = MarkovDataset(T=T, seed=0, num_samples=3000, seq_len=32)
    test = MarkovDataset(T=T, seed=999, num_samples=3000, seq_len=32)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    curve_rows: list[dict[str, float | int | str]] = []
    cutoff_rows: list[dict[str, float | int | str]] = []

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
        hidden = hidden_sequence.reshape(-1, D_MODEL).detach()

        centered = hidden.cpu().to(torch.float64)
        centered = centered - centered.mean(0, keepdim=True)
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        pca_values = torch.linalg.eigvalsh(covariance).flip(0)
        pca_cumulative = normalized_cumulative(pca_values)

        seed_all(seed)
        _, ace_weights = ace(
            hidden.to(device, torch.float32),
            lambda current=model: (lambda h: current.lm_head(h)),
            sample_size=128,
            max_rank=D_MODEL,
            return_full=True,
        )
        ace_cumulative = normalized_cumulative(ace_weights)

        for method, cumulative, energy_name in (
            ("ACE", ace_cumulative, "balanced energy"),
            ("PCA", pca_cumulative, "activation variance"),
        ):
            for rank, value in enumerate(cumulative.tolist(), start=1):
                curve_rows.append(
                    {
                        "record_type": "curve",
                        "model": model_index,
                        "seed": seed,
                        "method": method,
                        "energy_definition": energy_name,
                        "rank": rank,
                        "cumulative_energy": float(value),
                        "threshold": "",
                        "cutoff_rank": "",
                    }
                )
            for threshold in THRESHOLDS:
                cutoff_rows.append(
                    {
                        "record_type": "cutoff",
                        "model": model_index,
                        "seed": seed,
                        "method": method,
                        "energy_definition": energy_name,
                        "rank": "",
                        "cumulative_energy": "",
                        "threshold": threshold,
                        "cutoff_rank": cutoff_rank(cumulative, threshold),
                    }
                )

        print(
            f"{'loaded' if from_cache else 'trained'} "
            f"{model_index + 1:02d}/{len(SEEDS)} seed={seed:4d} | "
            f"99.9% ranks ACE={cutoff_rank(ace_cumulative, 0.999):2d}, "
            f"PCA={cutoff_rank(pca_cumulative, 0.999):2d}",
            flush=True,
        )

    all_rows = curve_rows + cutoff_rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_type",
        "model",
        "seed",
        "method",
        "energy_definition",
        "rank",
        "cumulative_energy",
        "threshold",
        "cutoff_rank",
    ]
    with EVIDENCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    RESULTS_JSON.write_text(
        json.dumps(
            {
                "protocol": {
                    "seeds": SEEDS,
                    "n_models": len(SEEDS),
                    "d_model": D_MODEL,
                    "thresholds": THRESHOLDS,
                    "ace_energy": (
                        "squared balanced singular values of covariance and "
                        "output-sensitivity Gramians"
                    ),
                    "pca_energy": "eigenvalues of activation covariance",
                    "ace_jacobian_prompts": 128,
                    "note": (
                        "The Markov tail is linear, so its Jacobian is constant "
                        "and the ACE spectrum is insensitive to sampled prompts."
                    ),
                },
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {RESULTS_JSON}", flush=True)
    print(f"saved {EVIDENCE_CSV}", flush=True)


if __name__ == "__main__":
    main()
