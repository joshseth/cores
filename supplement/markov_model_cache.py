"""Shared deterministic checkpoint cache for the 12-seed Markov controls."""

from __future__ import annotations

from pathlib import Path

import torch

from cores.models import OneLayerTransformer, train_markov_lm
from cores.utils import seed_all


HERE = Path(__file__).resolve().parent
CACHE_DIR = (
    HERE
    / "markov_model_cache"
    / "t075_n3000_seq32_d64_e40_lr1e-4"
)


def checkpoint_path(seed: int) -> Path:
    return CACHE_DIR / f"seed_{seed:04d}.pt"


def load_or_train(
    seed: int,
    train_loader,
    device: torch.device,
    *,
    d_model: int = 64,
    epochs: int = 40,
    lr: float = 1e-4,
) -> tuple[OneLayerTransformer, bool]:
    """Return a trained model and whether it was loaded from cache."""
    seed_all(seed)
    model = OneLayerTransformer(4, d_model).to(device)
    path = checkpoint_path(seed)
    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state"])
        return model, True

    for _ in train_markov_lm(
        model,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
    ):
        pass
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "seed": seed,
            "d_model": d_model,
            "epochs": epochs,
            "lr": lr,
        },
        path,
    )
    return model, False
