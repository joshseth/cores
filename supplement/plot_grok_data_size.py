#!/usr/bin/env python3
"""Appendix figure: grok time vs. training examples at training fraction 0.5: the modulus sweep, copies of Z_31, and
aliased Z_31 (Fig. A11 of the paper), in the style of the paper's grokking figure.
Input: grok_mechanism_control/results.json (grok_mechanism_control.py on MPS; two-layer model, full batch, AdamW,
weight decay 1.0, 3 seeds; delay tau_grok = steps from memorization (train acc >= 0.99) to grokking (held-out
acc >= 0.99), at most 30,000 steps; stored as "drift").
  31_grok_data_size.{pdf,png}  tau_grok vs distinct training examples n. The modulus sweep with the theory curve at
                               p = sqrt(2 n) (the sweep's p for n examples); k copies of Z_31 and a names per number in
                               Z_31, with the theory value at p = 31 (dashed)
Usage
  python supplement/plot_grok_data_size.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from grok_sweep import _set_plot_style  # noqa: E402  (the paper figure's style)

D = HERE / "grok_mechanism_control"
OMEGA0, PCRIT0 = 3803.40, 21.34      # main-text two-layer fit (grok_sweep_two_layer.py, 12 seeds), fraction 0.5


def load():
    return json.loads((D / "results.json").read_text())["runs"]


def law(p, O, pc):
    p = np.asarray(p, float)
    return np.where(p > pc, -O * np.log(np.clip(1 - pc / p, 1e-12, None)), np.nan)


def ebar(ax, x, ys, color, marker, label=None, open_=False):
    ys = np.asarray(ys, float)
    ax.errorbar([x], [ys.mean()], yerr=[ys.std()] if len(ys) > 1 else None, fmt=marker, color=color, ms=2.2,
                capsize=2, elinewidth=0.5, capthick=0.5, zorder=5, label=label,
                mfc="white" if open_ else color, mew=0.6)


def panel_data_size(ax, runs):
    ns = np.geomspace(440, 5400, 300)
    ax.plot(ns, law(np.sqrt(2 * ns), OMEGA0, PCRIT0), "-", color="green", lw=1.0, label=r"theory, $p=\sqrt{2n}$")
    line, = ax.plot(ns, np.full_like(ns, float(law(31, OMEGA0, PCRIT0))), ":", color="grey", lw=1.0,
                    label=r"theory, $p=31$")
    line.set_dashes([1.0, 0.5])
    groups = {}
    for r in runs:
        if r["arm"] == "base" and r["p"] >= 31:
            groups.setdefault(("base", r["n_train"]), []).append(r["drift"])
        elif r["arm"] in ("C_copies", "D_alias") and r["drift"]:
            groups.setdefault((r["arm"], r["n_train"]), []).append(r["drift"])
    spec = {"base": ("k", "o", "modulus sweep"), "C_copies": ("tab:red", "s", r"copies of $\mathbb{Z}_{31}$"),
            "D_alias": ("tab:blue", "D", r"aliased $\mathbb{Z}_{31}$")}
    seen = set()
    for (arm, n), ys in sorted(groups.items()):
        color, marker, lab = spec[arm]
        ebar(ax, n, ys, color, marker, None if arm in seen else lab)
        seen.add(arm)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([500, 1000, 2000, 4000])
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([300, 1000, 3000]); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(260, 9000)
    ax.legend(loc="lower left", handlelength=0.6, labelspacing=0.0, handletextpad=0.25, fontsize=5)
    ax.set_xlabel(r"Training examples $n$", size=8)
    ax.set_ylabel(r"Delay $\tau_{\mathrm{grok}}$ (steps)", size=8)
    ax.set_title("Grok Time vs. Data Size", size=8, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(1.9, 1.5), constrained_layout=True)
    panel_data_size(ax, load())
    ax.tick_params(axis="both", labelsize=7)
    out = args.outdir / "31_grok_data_size"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=500, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
