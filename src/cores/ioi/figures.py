#!/usr/bin/env python3
"""
figures.py: the IOI figures, from the saved results of golden.py and wang.py.

Fig. 7        (A) Flip AUC of each site's core against relative depth, markers at the chosen layers. (B) The chosen core on the test prompts, as normalized AUC.
Appendix      Each site's core flipped alone against its rank, stars at the sufficient rank; removal at both sites against core size, on train and on the prompts of Wang et al.,
              markers at the chosen core. Keep against remove for the core and other directions on the prompts of Wang et al. (wang_controls.py).

  PYTHONPATH=src python -m cores.ioi.figures --dir results/ioi/golden --out figures
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODELS = [("gpt2", "GPT-2 Small"), ("gpt2-medium", "GPT-2 Medium"), ("gpt2-large", "GPT-2 Large"),
          ("llama-3.1-8b", "LLaMA-3.1-8B"), ("gemma-2-9b", "Gemma-2-9B"), ("qwen2.5-32b", "Qwen2.5-32B")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
OLD_STYLE = {"gpt2": (":", "^"), "gpt2-medium": ("-", "o"), "gpt2-large": ("--", "s"),
             "llama-3.1-8b": ("-.", "D"), "gemma-2-9b": ((0, (5, 1, 1, 1)), "v"), "qwen2.5-32b": ((0, (1, 1)), "P")}
SITE_COL = {"S2": "#009E73", "END": "#0072B2"}
FS_T, FS_L, FS_K = 7.0, 6.8, 6.0
RULE = "flip at both sites"
GROUPS = (("keep", "Keep\n(ideal 1)", 1), ("remove", "Remove\n(ideal 0)", 0), ("flip_S2", "Flip S2\n(ideal −1)", -1), ("flip_END", "Flip END\n(ideal −1)", -1))


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.2, pad=1.5, length=2, width=0.5)


def schematic(ax):
    """One of the prompts, with the two sites marked."""
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    toks = [("Mary and John were playing in the park.", None), ("John", "S2"), ("kicked the ball", None), ("to", "END"), ("→ Mary", "ans")]
    r, x = ax.figure.canvas.get_renderer(), 0.0
    for t, role in toks:
        x += 0.008 if role in SITE_COL else 0.0
        if role in SITE_COL:
            h = ax.text(x, 0.3, t, ha="left", va="center", fontsize=5.0, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.2", fc=SITE_COL[role], ec="none"))
        else:
            h = ax.text(x, 0.3, t, ha="left", va="center", fontsize=5.0, color="#555555" if role is None else "#222222", style="italic" if role == "ans" else "normal")
        x = h.get_window_extent(r).transformed(ax.transAxes.inverted()).x1 + (0.02 if role in SITE_COL else 0.012)


def chosen(r):
    """The layer pair with the smallest core; a tie goes to the lower removal on train."""
    total = {q: sum(v["core"][RULE]["ranks"]) for q, v in r["pairs"].items() if v["core"][RULE]}
    return min(total, key=lambda k: (total[k], r["pairs"][k]["core"][RULE]["train"]["remove"]))


def load(root, name):
    r = json.loads((root / name / f"{name}.json").read_text()); w = json.loads((root / name / f"{name}_wang.json").read_text())
    q = chosen(r); P = r["pairs"][q]; c = P["core"][RULE]; a, b = c["ranks"]
    sweep = r["sweep"]
    fill = root / name / f"{name}_fill.json"
    if fill.exists():
        extra = json.loads(fill.read_text())
        sweep = {t: {**sweep[t], **extra[t]} for t in ("S2", "END")}
    return {"sweep": sweep, "n_layers": r["n_layers"], "layers": P["layers"], "ranks": [a, b], "train": c["train"], "test": c["held-out"], "test_clean": r["clean"]["held-out"], "pair": P,
            "wang": w["pairs"][q]["cores"][f"{a}+{b}"], "wang_curve": w["pairs"][q]["curve"]}


def fig7(R, out):
    fig = plt.figure(figsize=(5.5, 2.55))
    top = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.3, left=0.08, right=0.99, top=0.86, bottom=0.2)
    sub = top[0].subgridspec(3, 1, height_ratios=[0.26, 1, 1], hspace=0.12)
    sch, axS, axE = fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), fig.add_subplot(sub[2])
    schematic(sch)
    for i, (name, label, v, c) in enumerate(R):
        ls, mk = OLD_STYLE[name]
        for si, (site, ax) in enumerate((("S2", axS), ("END", axE))):
            pts = sorted((int(l), x["flip"]) for l, x in v["sweep"][site].items())
            ax.plot([l / v["n_layers"] for l, _ in pts], [a for _, a in pts], linestyle=ls, color=SITE_COL[site], lw=0.9, alpha=0.9)
            sel = v["layers"][si]
            ax.plot(sel / v["n_layers"], dict(pts)[sel], mk, ms=4.2, color=c, mec="black", mew=0.5, zorder=10 - i)
    for site, ax in (("S2", axS), ("END", axE)):
        ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.08); ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"{site} core\nflip AUC", fontsize=FS_L, color=SITE_COL[site], fontweight="bold", labelpad=2)
        tidy(ax)
    plt.setp(axS.get_xticklabels(), visible=False)
    axE.set_xlabel("Relative layer depth", fontsize=FS_L)
    sch.set_title("Two Sites, Two Depths", loc="left", fontsize=FS_T, fontweight="bold", pad=3)

    ax = fig.add_subplot(top[1]); w = 0.8 / len(R)
    nz = lambda auc, base: (auc - 0.5) / (base - 0.5)
    for gi, (m, _, ideal) in enumerate(GROUPS):
        ax.plot([gi - 0.45, gi + 0.45], [ideal, ideal], color="0.2", lw=0.7, ls=(0, (3, 2)), zorder=1)
        for ri, (n, l, v, c) in enumerate(R):
            ax.bar(gi - 0.4 + w / 2 + ri * w, nz(v["test"][m], v["test_clean"]), width=w * 0.9, color=c, edgecolor="none", zorder=2)
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g[1] for g in GROUPS])
    ax.axhline(0, color="0.4", lw=0.5); ax.set_ylim(-1.15, 1.15); ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("Normalized AUC", fontsize=FS_L, labelpad=1)
    ax.set_title("Two-Site Core Controls IOI", loc="left", fontsize=FS_T, fontweight="bold", pad=3)
    tidy(ax)
    ax.tick_params(axis="x", labelsize=5.3)

    for a_, L in ((sch, "A"), (ax, "B")):
        pos = a_.get_position()
        fig.text(pos.x0 - 0.065, pos.y1 + 0.035, L, fontsize=9, fontweight="bold", va="bottom")
    hs = [Line2D([], [], color="0.35", ls=OLD_STYLE[n][0], marker=OLD_STYLE[n][1], ms=4, mfc=c, mec="black", mew=0.5, lw=0.9, label=l) for n, l, v, c in R]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=FS_K, handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.53, -0.01))
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ioi_fig7.{ext}", dpi=400)
    plt.close(fig)


def app_curves(R, out):
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.95), gridspec_kw={"wspace": 0.36, "left": 0.08, "right": 0.99, "top": 0.87, "bottom": 0.3})
    for n, l, v, c in R:
        P = v["pair"]; done = {tuple(e[:3]): e[3] for e in P["evaluated"]}
        for si, site in enumerate(("S2", "END")):
            pts = sorted((int(r), a) for r, a in P["control"][site]["flip"].items())
            ax[si].plot([r for r, _ in pts], [a for _, a in pts], "-", color=c, lw=0.9, marker=OLD_STYLE[n][1], ms=2.2)
            r0 = P["sufficient"]["flip"][site]
            ax[si].plot([r0], [done[(site, r0, "flip")]], "*", color=c, ms=6, mec="black", mew=0.4, zorder=5)
        for (curve, ls) in (({int(r): x["remove"] for r, x in P["curve"].items()}, "-"), ({int(r): a for r, a in v["wang_curve"].items()}, ":")):
            rs = sorted(curve)
            ax[2].plot([2 * r for r in rs], [curve[r] for r in rs], ls, color=c, lw=0.9)
        ax[2].plot([sum(v["ranks"])], [v["train"]["remove"]], OLD_STYLE[n][1], color=c, ms=3.5, mec="black", mew=0.4, zorder=5)
    titles = ("S2 Core Flipped Alone", "END Core Flipped Alone", "Removal at Both Sites")
    for x, L, t in zip(ax, "ABC", titles):
        x.set_xscale("log", base=2); tidy(x); x.set_title(t, loc="left", fontsize=FS_T, fontweight="bold", pad=3)
        x.text(-0.2, 1.08, L, transform=x.transAxes, fontsize=9, fontweight="bold", va="bottom")
        x.axhline(0.5, color="0.6", lw=0.5, ls=":"); x.set_ylim(-0.03 if L != "C" else 0.3, 1.05)
    for x, t in zip(ax[:2], ("S2 core rank", "END core rank")):
        x.axhline(0.1, color="0.2", lw=0.5, ls=(0, (3, 2))); x.set_xticks([2, 8, 32, 128]); x.set_xticklabels(["2", "8", "32", "128"]); x.set_xlabel(t, fontsize=FS_L)
    ax[0].set_ylabel("Flip AUC", fontsize=FS_L, labelpad=1); ax[2].set_ylabel("Removal AUC", fontsize=FS_L, labelpad=1)
    ax[2].axhline(0.55, color="0.2", lw=0.5, ls=(0, (3, 2))); ax[2].set_xticks([2, 8, 32, 128, 512]); ax[2].set_xticklabels(["2", "8", "32", "128", "512"]); ax[2].set_xlabel("Core dimensions (S2 + END)", fontsize=FS_L)
    hs = [Line2D([], [], color=c, lw=0.9, marker=OLD_STYLE[n][1], ms=3, label=l) for n, l, v, c in R] + [Line2D([], [], color="0.3", lw=0.9, ls="-", label="train"), Line2D([], [], color="0.3", lw=0.9, ls=":", label="Wang et al.")]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=FS_K - 0.5, handlelength=1.6, columnspacing=0.9, handletextpad=0.3, bbox_to_anchor=(0.5, -0.02))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"app_ioi_curves.{ext}", dpi=400)
    plt.close(fig)


def app_wang_controls(root, out):
    """Sufficiency against necessity, as in the SVA figure, with IOI's cutoffs: mean and s.d. across the GPT-2 models."""
    rows = [json.loads((root / f"wang_controls_{n}.json").read_text())["rows"] for n, _ in MODELS if n.startswith("gpt2")]
    style = {"ACE": ("#1f77b4", "ACE", (-5, 5), "right"), "PCA": ("#A9A9A9", "PCA", (-5, 5), "right"), "mean diff": ("#ff7f0e", "Mean difference", (-5, -9), "right"),
             "random": ("#8c564b", "Random", (5, -9), "left"), "unembedding": ("#9467bd", "Unembedding", (5, 4), "left")}
    fig, ax = plt.subplots(figsize=(2.3, 2.1), gridspec_kw={"left": 0.2, "right": 0.97, "top": 0.88, "bottom": 0.2})
    ax.add_patch(plt.Rectangle((0.95, 0.44), 0.09, 0.11, color="#1f77b4", alpha=0.08, lw=0))
    ax.text(0.935, 0.455, "sufficient +\nnecessary", fontsize=5.5, color="#1f77b4", va="bottom", ha="right")
    for b, (c, label, off, ha) in style.items():
        k, r = ([v[b][q] for v in rows] for q in ("keep", "remove"))
        ax.errorbar(sum(k) / len(k), sum(r) / len(r), xerr=statistics.pstdev(k), yerr=statistics.pstdev(r), fmt="o", color=c, ms=4, elinewidth=0.6, capsize=1.5)
        ax.annotate(label, (sum(k) / len(k), sum(r) / len(r)), xytext=off, textcoords="offset points", fontsize=5.8, color="0.2", ha=ha)
    ax.axvline(0.95, color="0.5", lw=0.5, ls="--"); ax.axhline(0.55, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(0.45, 1.04); ax.set_ylim(0.44, 1.06); ax.set_xlabel("Keep AUC (higher = sufficient)", fontsize=FS_L)
    ax.set_ylabel("Remove AUC (lower = necessary)", fontsize=FS_L); tidy(ax); ax.set_title("Sufficiency vs. Necessity", loc="left", fontsize=FS_T, fontweight="bold", pad=3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"app_ioi_wang_controls.{ext}", dpi=400)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("results/ioi/golden"))
    p.add_argument("--wang-controls", type=Path, default=Path("results/ioi"), help="folder with wang_controls_<model>.json")
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5, "mathtext.fontset": "dejavusans"})
    R = [(n, l.replace("-8B", "").replace("-9B", "").replace("-32B", ""), load(args.dir, n), COLORS[i]) for i, (n, l) in enumerate(MODELS)]
    fig7(R, args.out)
    app_curves(R, args.out)
    app_wang_controls(args.wang_controls, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
