#!/usr/bin/env python3
"""Supplement figure: main-text Fig. 2C (eigenvalues of the core's cyclic operator snapping to the unit circle at
grokking) for the two other seeds of the same run, drawn on the same circles (seed 2 blue, seed 3 orange).

Same checkpoints, epochs, analysis (analyze_epoch with block-edge holdout, ridge 1.0, ring eps = sin(pi/53)) and style
(plot_mod_add.apply_style) as the main panel, which shows seed 1 (model index 0) of this run; its printed values
reproduce from model 0 here at all five epochs. (The arXiv v1 version of that panel was drawn from a different run.)
Eigenvalues within eps of the unit circle are drawn in the seed's color, the rest in a pale tint of it.

  python supplement/plot_mod_add_clock_seeds.py \
      --run-dir experiments/modular_addition/ckpts/wd1 --out supplement/outputs/32_mod_add_clock_seeds.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec

from cores.models import build_modadd_dataloader
from cores.modular_addition_analysis import analyze_epoch
from cores.plot_mod_add import apply_style
from cores.utils import get_device

SEEDS = {1: ("Seed 2", "#1f77b4"), 2: ("Seed 3", "#ff7f0e")}     # model index -> label, color


def pale(c, w=0.35):
    return tuple(w * np.array(to_rgb(c)) + (1 - w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("experiments/modular_addition/ckpts/wd1"))
    ap.add_argument("--epochs", default="0,300,800,900,2000")
    ap.add_argument("--vocab-size", type=int, default=53)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs" / "32_mod_add_clock_seeds.pdf")
    a = ap.parse_args()
    device = get_device()
    test_loader = build_modadd_dataloader(train=False, batch_size=512)
    ring_eps = np.sin(np.pi / a.vocab_size)
    epochs = [int(e) for e in a.epochs.split(",")]

    res = {}
    for m_ix in SEEDS:
        for ep in epochs:
            out = analyze_epoch(a.run_dir, ep=ep, m_ix=m_ix, test_loader=test_loader, device=device,
                                vocab_size=a.vocab_size, d_model=128, ring_eps=ring_eps,
                                holdout_style="block_edges", ridge=1.0)
            res[m_ix, ep] = out
            print(f"model {m_ix} epoch {ep}: acc {out['acc']:.2f} R2_h {out['r2_test']:.2f} core {out['r_nec']}", flush=True)

    apply_style()
    fig = plt.figure(figsize=(5.5, 1.75))
    gs = GridSpec(2, len(epochs), figure=fig, height_ratios=[0.6, 1.0], wspace=0.05, hspace=0.15)
    info = fig.add_subplot(gs[0, :]); info.axis("off")
    info.text(0.0, 0.95, "Cyclic Mechanism Emerges in Core at Grokking (Seeds 2 and 3)", ha="left", va="top",
              fontsize=6, fontweight="bold", transform=info.transAxes)
    rows = [(0.62, [c for _, c in SEEDS.values()], r"Eigenvalue near unit circle $||\lambda|-1| < \epsilon$"),
            (0.42, [pale(c) for _, c in SEEDS.values()], "Off-circle")]
    for y, cols, txt in rows:
        for i, c in enumerate(cols):
            info.scatter([0.005 + 0.012 * i], [y], s=5, color=c, transform=info.transAxes, clip_on=False)
        info.text(0.03, y, txt, ha="left", va="center", fontsize=5.5, transform=info.transAxes)
    x = 0.005
    for lab, c in SEEDS.values():
        info.scatter([x], [0.22], s=5, color=c, transform=info.transAxes, clip_on=False)
        info.text(x + 0.01, 0.22, lab, ha="left", va="center", fontsize=5.5, transform=info.transAxes)
        x += 0.085

    for j, ep in enumerate(epochs):
        ax = fig.add_subplot(gs[1, j])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, color="gray", alpha=0.5, lw=0.5))
        for m_ix, (lab, c) in SEEDS.items():
            ev = res[m_ix, ep]["evals"]
            near = np.abs(np.abs(ev) - 1.0) <= ring_eps
            ax.scatter(ev.real[~near], ev.imag[~near], s=7, color=pale(c), lw=0.0)
        for m_ix, (lab, c) in SEEDS.items():
            ev = res[m_ix, ep]["evals"]
            near = np.abs(np.abs(ev) - 1.0) <= ring_eps
            ax.scatter(ev.real[near], ev.imag[near], s=7, color=c, lw=0.0, alpha=0.85)
        if j == len(epochs) // 2:
            L = 0.35
            akw = dict(arrowstyle="-|>", lw=0.35, color="0.25", mutation_scale=6, shrinkA=0, shrinkB=0)
            ax.annotate("", xy=(L, 0), xytext=(0, 0), arrowprops=akw, zorder=3)
            ax.annotate("", xy=(0, L), xytext=(0, 0), arrowprops=akw, zorder=3)
            ax.text(L + 0.02, 0, "Re", ha="left", va="center", fontsize=6, color="0.25")
            ax.text(0, L + 0.02, "Im", ha="center", va="bottom", fontsize=6, color="0.25")
        ax.set_title(f"Epoch {ep:,}", fontsize=6, pad=2)
        rows = [("Acc:", lambda o: f"{o['acc']:.2f}"),
                (r"$R_h^2$:", lambda o: f"{(o['r2_test'] if not np.isnan(o['r2_test']) else o['r2']):.2f}"),
                (r"Core$_{\mathrm{dim}}$:", lambda o: f"{o['r_nec']}")]
        for i, (name, val) in enumerate(rows):
            y = -0.2 - 0.13 * i
            ax.text(0.47, y, name, transform=ax.transAxes, ha="right", va="center", fontsize=5, clip_on=False)
            for k, (m_ix, (lab, c)) in enumerate(SEEDS.items()):
                ax.text(0.51 + 0.27 * k, y, val(res[m_ix, ep]), transform=ax.transAxes, ha="left", va="center",
                        fontsize=5, color=c, clip_on=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, format="pdf", bbox_inches="tight", dpi=500)
    fig.savefig(a.out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
