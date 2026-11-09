#!/usr/bin/env python3
"""Modular-addition ACE prefix intervention curves and ordering controls.

It loads the three WD=1 checkpoints, reproduces the ACE basis from the
full held-out activation covariance and the exact linear-head Jacobian Gramian,
and saves cached arrays and the figure in the working directory.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
CKPTS = REPO / "experiments/modular_addition/ckpts/wd1"
OUT = REPO / "supplement/outputs"
CACHE = OUT / "26_modadd_prefix_curves.npz"
CSV_OUT = OUT / "26_modadd_prefix_curves.csv"
NOTES_OUT = OUT / "26_modadd_prefix_curves_notes.md"
FIG_STEM = OUT / "26_modadd_prefix_curves"

sys.path.insert(0, str(REPO / "src"))
from cores.extraction import _balanced_svd  # noqa: E402
from cores.models import ModAddDataset, TwoLayerTransformer  # noqa: E402


VOCAB = 53
D_MODEL = 128
R_EVAL = 53
EPOCHS = np.arange(0, 2001, 100, dtype=int)
MODEL_IXS = np.arange(3, dtype=int)
ENERGY_THRESHOLD = 0.99
SUF_TOL = 1e-3
NEAR_CHANCE = 2.0 / VOCAB
N_PERM = 50
ORDER_EPOCH = 2000
RNG_SEED = 260727


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_test_tensors() -> tuple[torch.Tensor, torch.Tensor, float]:
    ds = ModAddDataset(p=VOCAB, train=False)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([int(ds[i][1][-1]) for i in range(len(ds))])
    counts = torch.bincount(ys, minlength=VOCAB)
    empirical_chance = float(counts.max().item() / len(ys))
    return xs, ys, empirical_chance


def load_model(epoch: int, model_ix: int, device: torch.device) -> TwoLayerTransformer:
    path = CKPTS / f"epoch_{epoch:06d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TwoLayerTransformer(vocab_size=VOCAB, d_model=D_MODEL).to(device)
    model.load_state_dict(payload["model_state_dicts"][model_ix])
    model.eval()
    return model


@torch.no_grad()
def collect_last_hidden(
    model: TwoLayerTransformer, xs: torch.Tensor, device: torch.device
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(xs), 512):
        _, _, h2 = model(xs[start : start + 512].to(device), return_states=True)
        chunks.append(h2[:, -1, :].detach().cpu())
    return torch.cat(chunks, dim=0).float()


def ace_basis(
    hidden: torch.Tensor, model: TwoLayerTransformer
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Match ACE for a linear tail, using its exact (constant) Jacobian."""
    h = hidden.double()
    hc = h - h.mean(dim=0, keepdim=True)
    cov = (hc.T @ hc) / max(1, len(h) - 1)
    weight = model.lm_head.weight.detach().cpu().double()
    obs = weight.T @ weight
    q, spectrum = _balanced_svd(cov, obs, max_rank=D_MODEL, eps=1e-8)

    w = np.clip(spectrum.detach().cpu().numpy(), 0.0, None)
    cumulative = np.cumsum(w) / max(float(w.sum()), 1e-30)
    hits = np.flatnonzero(cumulative >= ENERGY_THRESHOLD)
    r_energy = int(hits[0] + 1) if len(hits) else len(w)
    return q.float(), w, r_energy


def prefix_accuracies(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    model: TwoLayerTransformer,
    q_ordered: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact last-layer prefix interventions, including r=0."""
    q = q_ordered.float()
    weight = model.lm_head.weight.detach().cpu().float()
    bias = model.lm_head.bias.detach().cpu().float()

    base_logits = hidden @ weight.T + bias
    base_acc = float((base_logits.argmax(dim=-1) == labels).float().mean().item())

    # Contribution of each orthonormal direction to every output logit.
    coordinates = hidden @ q
    direction_readout = q.T @ weight.T
    per_direction = coordinates[:, :, None] * direction_readout[None, :, :]
    cumulative = torch.cumsum(per_direction, dim=1)

    keep = np.empty(q.shape[1] + 1, dtype=np.float64)
    remove = np.empty(q.shape[1] + 1, dtype=np.float64)
    keep[0] = float((bias.argmax().expand_as(labels) == labels).float().mean().item())
    remove[0] = base_acc
    for r in range(1, q.shape[1] + 1):
        contribution = cumulative[:, r - 1, :]
        keep_logits = contribution + bias
        remove_logits = base_logits - contribution
        keep[r] = float((keep_logits.argmax(dim=-1) == labels).float().mean().item())
        remove[r] = float((remove_logits.argmax(dim=-1) == labels).float().mean().item())
    return keep, remove, base_acc


def first_crossing(values: np.ndarray, predicate, *, start: int = 1) -> int:
    for rank in range(start, len(values)):
        if predicate(float(values[rank])):
            return rank
    return len(values) - 1


def compute() -> dict[str, np.ndarray | float | int]:
    device = choose_device()
    xs, labels, empirical_chance = load_test_tensors()
    shape = (len(MODEL_IXS), len(EPOCHS), R_EVAL + 1)
    keep = np.full(shape, np.nan)
    remove = np.full(shape, np.nan)
    base = np.full(shape[:2], np.nan)
    r_energy = np.full(shape[:2], np.nan)
    r_suf = np.full(shape[:2], np.nan)
    r_nec_near = np.full(shape[:2], np.nan)
    r_nec_emp = np.full(shape[:2], np.nan)
    spectra = np.full((len(MODEL_IXS), len(EPOCHS), D_MODEL), np.nan)

    order_keep = np.full((len(MODEL_IXS), 2 + N_PERM, R_EVAL + 1), np.nan)
    order_remove = np.full_like(order_keep, np.nan)
    rng = np.random.default_rng(RNG_SEED)

    for si, model_ix in enumerate(MODEL_IXS):
        print(f"model {model_ix}", flush=True)
        for ti, epoch in enumerate(EPOCHS):
            model = load_model(int(epoch), int(model_ix), device)
            hidden = collect_last_hidden(model, xs, device)
            q, spectrum, energy_rank = ace_basis(hidden, model)
            k, rem, acc = prefix_accuracies(
                hidden, labels, model, q[:, :R_EVAL]
            )
            keep[si, ti], remove[si, ti], base[si, ti] = k, rem, acc
            spectra[si, ti] = spectrum
            r_energy[si, ti] = energy_rank
            r_suf[si, ti] = first_crossing(
                k, lambda value, a=acc: value >= a - SUF_TOL
            )
            if acc <= NEAR_CHANCE:
                r_nec_near[si, ti] = 0
            else:
                r_nec_near[si, ti] = first_crossing(
                    rem, lambda value: value <= NEAR_CHANCE
                )
            if acc <= empirical_chance:
                r_nec_emp[si, ti] = 0
            else:
                r_nec_emp[si, ti] = first_crossing(
                    rem, lambda value, c=empirical_chance: value <= c
                )

            if epoch == ORDER_EPOCH:
                orders = [
                    np.arange(R_EVAL),
                    np.arange(R_EVAL)[::-1].copy(),
                    *[rng.permutation(R_EVAL) for _ in range(N_PERM)],
                ]
                for oi, order in enumerate(orders):
                    ok, orem, _ = prefix_accuracies(
                        hidden, labels, model, q[:, np.asarray(order).copy()]
                    )
                    order_keep[si, oi] = ok
                    order_remove[si, oi] = orem

            print(
                f"  ep {epoch:4d} acc={acc:.3f} energy={energy_rank:2d} "
                f"suf={int(r_suf[si, ti]):2d} "
                f"nec(2/V)={int(r_nec_near[si, ti]):2d} "
                f"nec(emp)={int(r_nec_emp[si, ti]):2d}",
                flush=True,
            )
            del model

    result: dict[str, np.ndarray | float | int] = {
        "epochs": EPOCHS,
        "model_ixs": MODEL_IXS,
        "ranks": np.arange(R_EVAL + 1),
        "keep": keep,
        "remove": remove,
        "base": base,
        "r_energy": r_energy,
        "r_suf": r_suf,
        "r_nec_near": r_nec_near,
        "r_nec_emp": r_nec_emp,
        "spectra": spectra,
        "order_keep": order_keep,
        "order_remove": order_remove,
        "near_chance": NEAR_CHANCE,
        "empirical_chance": empirical_chance,
        "suf_tol": SUF_TOL,
        "n_perm": N_PERM,
        "order_epoch": ORDER_EPOCH,
    }
    np.savez_compressed(CACHE, **result)
    return result


def load_or_compute() -> dict[str, np.ndarray | float | int]:
    if CACHE.exists():
        print(f"loading cache {CACHE}", flush=True)
        data = np.load(CACHE, allow_pickle=False)
        return {key: data[key] for key in data.files}
    return compute()


def sem(values: np.ndarray, axis=0) -> np.ndarray:
    return np.nanstd(values, axis=axis, ddof=1) / math.sqrt(values.shape[axis])


def plot(result: dict[str, np.ndarray | float | int]) -> None:
    epochs = np.asarray(result["epochs"])
    ranks = np.asarray(result["ranks"])
    keep = np.asarray(result["keep"])
    remove = np.asarray(result["remove"])
    base = np.asarray(result["base"])
    r_energy = np.asarray(result["r_energy"])
    r_suf = np.asarray(result["r_suf"])
    r_nec_near = np.asarray(result["r_nec_near"])
    r_nec_emp = np.asarray(result["r_nec_emp"])
    order_keep = np.asarray(result["order_keep"])
    order_remove = np.asarray(result["order_remove"])
    empirical_chance = float(result["empirical_chance"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(11.0, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.08, 1.0])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    mean_keep = keep.mean(axis=0)
    mean_remove = remove.mean(axis=0)
    extent = [ranks[0] - 0.5, ranks[-1] + 0.5, epochs[0] - 50, epochs[-1] + 50]
    im_a = ax_a.imshow(
        mean_keep,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="viridis",
        interpolation="nearest",
    )
    im_b = ax_b.imshow(
        mean_remove,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="viridis",
        interpolation="nearest",
    )
    for ax in (ax_a, ax_b):
        ax.axhline(800, color="white", ls="--", lw=1.0, alpha=0.9)
        ax.text(52, 830, "grok", color="white", ha="right", va="bottom", fontsize=7)
        ax.set_xlabel("ACE prefix rank $r$")
        ax.set_ylabel("Epoch")
        ax.set_xticks([0, 10, 20, 30, 40, 53])
    ax_a.set_title("Core-only accuracy (sufficiency)")
    ax_b.set_title("Core-removed accuracy (necessity)")
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], location="bottom", shrink=0.55, pad=0.02)
    cbar.set_label("Held-out accuracy")

    colors = {"energy": "#555555", "suf": "#2378b5", "nec": "#d84b40", "strict": "#984ea3"}
    for values, key, label, marker in [
        (r_energy, "energy", "99% ACE energy", "o"),
        (r_suf, "suf", "Sufficient: keep $\\geq$ baseline - .001", "s"),
        (r_nec_near, "nec", r"Necessary: remove $\leq 2/|V|$", "^"),
    ]:
        mu = values.mean(axis=0)
        se = sem(values, axis=0)
        ax_c.plot(epochs, mu, color=colors[key], lw=1.8, marker=marker, ms=3, label=label)
        ax_c.fill_between(epochs, mu - se, mu + se, color=colors[key], alpha=0.13)
    ax_c.axvline(800, color="black", ls="--", lw=0.8, alpha=0.55)
    ax_c.set_title("Threshold-crossing dimensions")
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Dimension")
    ax_c.set_xlim(0, 2000)
    ax_c.set_ylim(-1, 55)
    ax_c.legend(loc="upper right", frameon=False)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(epochs, base.mean(axis=0), color="#aaaaaa", ls=":", lw=1.2)
    ax_c2.set_ylabel("Baseline accuracy", color="#888888")
    ax_c2.set_ylim(-0.02, 1.02)
    ax_c2.tick_params(axis="y", colors="#888888")
    ax_c2.spines["top"].set_visible(False)

    def ordering_panel(ax, values: np.ndarray, title: str, mode: str) -> None:
        # Indices: 0 ordered, 1 reverse, 2: random permutations.
        ordered_mu = values[:, 0, :].mean(axis=0)
        ordered_se = sem(values[:, 0, :], axis=0)
        reverse_mu = values[:, 1, :].mean(axis=0)
        reverse_se = sem(values[:, 1, :], axis=0)
        random_flat = values[:, 2:, :].reshape(-1, values.shape[-1])
        random_mu = random_flat.mean(axis=0)
        random_sd = random_flat.std(axis=0, ddof=1)

        ax.plot(ranks, ordered_mu, color="#2378b5", lw=2.0, label="ACE order")
        ax.fill_between(ranks, ordered_mu - ordered_se, ordered_mu + ordered_se,
                        color="#2378b5", alpha=0.16)
        ax.plot(ranks, random_mu, color="#777777", lw=1.5, label="Random order")
        ax.fill_between(ranks, random_mu - random_sd, random_mu + random_sd,
                        color="#777777", alpha=0.13)
        ax.plot(ranks, reverse_mu, color="#d84b40", lw=1.5, ls="--",
                label="Reverse ACE order")
        ax.fill_between(ranks, reverse_mu - reverse_se, reverse_mu + reverse_se,
                        color="#d84b40", alpha=0.10)
        if mode == "keep":
            ax.axhline(base[:, -1].mean() - SUF_TOL, color="black", ls=":", lw=1.0,
                       label="Baseline - .001")
        else:
            ax.axhline(NEAR_CHANCE, color="black", ls=":", lw=1.0,
                       label=r"Near chance $2/|V|$")
            ax.axhline(empirical_chance, color="#984ea3", ls=":", lw=0.9,
                       label="Empirical chance")
        ax.set_title(title)
        ax.set_xlabel("Prefix rank $r$ within the same ACE-53 span")
        ax.set_ylabel("Held-out accuracy")
        ax.set_xlim(0, R_EVAL)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 10, 20, 30, 40, 53])
        ax.legend(loc="best", frameon=False, ncol=2)

    ordering_panel(
        ax_d,
        order_keep,
        f"Ordering control at epoch {ORDER_EPOCH:,}: core only",
        "keep",
    )
    ordering_panel(
        ax_e,
        order_remove,
        f"Ordering control at epoch {ORDER_EPOCH:,}: core removed",
        "remove",
    )

    for label, ax in zip("ABCDE", [ax_a, ax_b, ax_c, ax_d, ax_e]):
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top")

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG_STEM.with_suffix(f".{suffix}"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_evidence(result: dict[str, np.ndarray | float | int]) -> None:
    epochs = np.asarray(result["epochs"])
    ranks = np.asarray(result["ranks"])
    keep = np.asarray(result["keep"])
    remove = np.asarray(result["remove"])
    base = np.asarray(result["base"])
    r_energy = np.asarray(result["r_energy"])
    r_suf = np.asarray(result["r_suf"])
    r_nec_near = np.asarray(result["r_nec_near"])
    r_nec_emp = np.asarray(result["r_nec_emp"])
    order_keep = np.asarray(result["order_keep"])
    order_remove = np.asarray(result["order_remove"])
    empirical_chance = float(result["empirical_chance"])

    def crossing(values: np.ndarray, threshold: float, relation: str) -> float:
        candidates = np.flatnonzero(
            values[1:] >= threshold if relation == "ge" else values[1:] <= threshold
        )
        return float(candidates[0] + 1) if len(candidates) else float("nan")

    ordering_summary: dict[str, dict[str, float]] = {}
    for label, order_indices in {
        "ACE": [0],
        "reverse": [1],
        "random": list(range(2, 2 + N_PERM)),
    }.items():
        suf_crossings: list[float] = []
        nec_crossings: list[float] = []
        for si in range(len(MODEL_IXS)):
            for oi in order_indices:
                suf_crossings.append(
                    crossing(order_keep[si, oi], base[si, -1] - SUF_TOL, "ge")
                )
                nec_crossings.append(
                    crossing(order_remove[si, oi], NEAR_CHANCE, "le")
                )
        ordering_summary[label] = {
            "sufficient_mean": float(np.nanmean(suf_crossings)),
            "sufficient_sd": float(np.nanstd(suf_crossings, ddof=1))
            if len(suf_crossings) > 1
            else 0.0,
            "necessary_mean": float(np.nanmean(nec_crossings)),
            "necessary_sd": float(np.nanstd(nec_crossings, ddof=1))
            if len(nec_crossings) > 1
            else 0.0,
        }

    with CSV_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_ix",
                "epoch",
                "rank",
                "baseline_accuracy",
                "keep_accuracy",
                "remove_accuracy",
                "r_energy_99",
                "r_sufficient",
                "r_necessary_2_over_V",
                "r_necessary_empirical_chance",
            ]
        )
        for si, model_ix in enumerate(MODEL_IXS):
            for ti, epoch in enumerate(epochs):
                for ri, rank in enumerate(ranks):
                    writer.writerow(
                        [
                            int(model_ix),
                            int(epoch),
                            int(rank),
                            base[si, ti],
                            keep[si, ti, ri],
                            remove[si, ti, ri],
                            int(r_energy[si, ti]),
                            int(r_suf[si, ti]),
                            int(r_nec_near[si, ti]),
                            int(r_nec_emp[si, ti]),
                        ]
                    )

    last = -1
    summary = {
        "epoch": int(epochs[last]),
        "baseline_mean": float(base[:, last].mean()),
        "energy_rank_mean": float(r_energy[:, last].mean()),
        "sufficient_rank_mean": float(r_suf[:, last].mean()),
        "necessary_rank_2_over_V_mean": float(r_nec_near[:, last].mean()),
        "necessary_rank_empirical_chance_mean": float(r_nec_emp[:, last].mean()),
        "operational_rank_mean": float(
            np.maximum.reduce([r_energy[:, last], r_suf[:, last], r_nec_near[:, last]]).mean()
        ),
        "per_model": [
            {
                "model_ix": int(model_ix),
                "energy": int(r_energy[si, last]),
                "sufficient": int(r_suf[si, last]),
                "necessary_2_over_V": int(r_nec_near[si, last]),
                "necessary_empirical_chance": int(r_nec_emp[si, last]),
            }
            for si, model_ix in enumerate(MODEL_IXS)
        ],
        "ordering_control": ordering_summary,
    }
    NOTES_OUT.write_text(
        "# Modular-addition necessity/sufficiency prefix analysis\n\n"
        f"- Standard WD=1 branch; epochs 0-2,000 every 100; three checkpointed models.\n"
        f"- Full held-out activation covariance; exact linear-head Jacobian Gramian.\n"
        f"- Sufficiency dimension: smallest ordered ACE prefix with core-only accuracy "
        f"`>= baseline - {SUF_TOL}`.\n"
        f"- Necessity dimension: smallest ordered ACE prefix with core-removed accuracy "
        f"`<= 2/|V| = {NEAR_CHANCE:.6f}`. A stricter empirical-majority chance "
        f"threshold `{empirical_chance:.6f}` is shown as a sensitivity analysis.\n"
        f"- The operational core rank is the maximum of the 99%-energy rank and "
        f"the two ablation threshold crossings, ensuring the selected prefix passes "
        f"both causal tests. Pre-grokking necessity is vacuous when baseline accuracy "
        f"is already below the near-chance threshold and is recorded as zero.\n"
        f"- Ordering controls at epoch {ORDER_EPOCH}: ACE order, reverse order, and "
        f"{N_PERM} random permutations, all within the identical top-{R_EVAL} ACE span.\n\n"
        "## Epoch-2,000 summary\n\n"
        "```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = load_or_compute()
    save_evidence(result)
    plot(result)
    print(f"saved {FIG_STEM}.png/.pdf/.svg")
    print(f"saved {CSV_OUT}")
    print(f"saved {NOTES_OUT}")


if __name__ == "__main__":
    main()
