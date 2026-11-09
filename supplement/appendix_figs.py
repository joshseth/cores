#!/usr/bin/env python3
"""Appendix figures redrawn in the main-text style (sans-serif, bold Title Case panel titles, bold uppercase panel letters,
no grid, main-figure colors, sized for the 5.5 in NeurIPS text width). Data sources:

  app_markov_ladder     <- supplement/outputs/28_markov_pairwise_cca3_evidence.csv (includes full-state CCA-3)
  app_markov_energy     <- supplement/outputs/24_markov_energy_evidence.csv
  app_sva_controls      <- supplement/outputs/21_sva_controls_evidence.csv
  app_sva_keep_remove   <- supplement/outputs/21_sva_controls_evidence.csv
  app_llm_nulls         <- supplement/outputs/14_llm_null_ladders_evidence.csv
  app_ioi_single        <- supplement/outputs/22_ioi_controls_evidence.csv
  app_ioi_single_core   <- src/cores/experiments/ioi_all/<model>/results.json

  python supplement/appendix_figs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "supplement/outputs"
OUT = EV
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5, "mathtext.fontset": "dejavusans",
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.frameon": False, "legend.fontsize": 6})
BLUE, GRAY, ORANGE, PURPLE, BROWN, RED = "#1f77b4", "#A9A9A9", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"
FS_T, FS_L = 7.0, 6.8
METHOD_COL = {"ACE": BLUE, "ACE selected": BLUE, "ACE rank 1": "#9ecae1", "Difference of means": ORANGE, "PCA-1": GRAY,
              "Unembedding": PURPLE}
LLM_MODELS = ["GPT-2 Small", "GPT-2 Medium", "GPT-2 Large", "LLaMA-3.1-8B", "Gemma-2-9B", "Qwen-2.5-32B"]
LLM_COL = {"GPT-2 Small": "#1f77b4", "GPT-2 Medium": "#2ca02c", "GPT-2 Large": "#d62728", "LLaMA-3.1-8B": "#e377c2",
           "Gemma-2-9B": "#9467bd", "Qwen-2.5-32B": "#8c564b"}
rng = np.random.default_rng(0)


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", pad=1.5, length=2, width=0.5)


def panel(ax, letter, title, dx=-0.22, dy=1.1):
    ax.set_title(title, loc="left", fontsize=FS_T, fontweight="bold", pad=4)
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def bars(ax, groups, colors, values, ideal=None, ylim=(0, 1.05), chance=None):
    """One bar per group (mean) with the per-model points."""
    for i, (g, c) in enumerate(zip(groups, colors)):
        v = np.asarray(values[g], float)
        ax.bar(i, v.mean(), width=0.68, color=c, alpha=0.9, lw=0)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color="k", alpha=0.6, lw=0, zorder=3)
    if chance is not None:
        ax.axhline(chance, color="0.4", lw=0.6, ls=":")
    ax.set_xticks(range(len(groups))); ax.set_ylim(*ylim); tidy(ax)


# --------------------------------------------------------------------------- Markov
def markov_ladder():
    d = pd.read_csv(EV / "28_markov_pairwise_cca3_evidence.csv")
    fig = plt.figure(figsize=(5.5, 2.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 0.95, 1.0, 0.85, 0.9], wspace=0.8, left=0.12, right=0.99, top=0.8, bottom=0.3)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    # A: CCA ladder heatmap
    conds = [("ACE-3 (trained)", "ACE"), ("Full-state CCA-3", "Full-state CCA"), ("PCA-3 (trained)", "PCA"),
             ("Untrained ACE-3", "Untrained ACE"), ("Random-3", "Random")]
    cca = d[d.section == "cca"]
    M = np.array([[cca[(cca.condition == c) & (cca.metric == m)].value.mean() for m in ("CC1", "CC2", "CC3")] for c, _ in conds])
    ax[0].imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(3):
            ax[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.3, color="white" if M[i, j] > 0.8 or M[i, j] < 0.2 else "k")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["CC1", "CC2", "CC3"]); ax[0].set_yticks(range(len(conds)))
    ax[0].set_yticklabels([n for _, n in conds], fontsize=5.8); ax[0].tick_params(length=0, pad=1.5)
    panel(ax[0], "A", "CCA Across Seeds", dx=-0.85, dy=1.16)
    # B: prefix rank
    pre = d[d.section == "prefix"]; ranks = np.arange(1, 7)
    for met, col, lab in (("keep", BLUE, "keep"), ("remove", GRAY, "remove")):
        v = np.array([pre[(pre.condition == f"rank-{r}") & (pre.metric == met)].value.to_numpy() for r in ranks])
        ax[1].plot(ranks, v.mean(1), "-o", color=col, ms=2.5, lw=1, label=lab)
        ax[1].fill_between(ranks, v.min(1), v.max(1), color=col, alpha=0.15, lw=0)
    ax[1].axhline(0.748, color="0.3", lw=0.5, ls="--"); ax[1].axhline(0.25, color="0.4", lw=0.6, ls=":"); ax[1].axvline(3, color=RED, lw=0.6, ls="--")
    ax[1].set_xticks(ranks); ax[1].set_ylim(0.15, 0.82); ax[1].set_xlabel("ACE prefix rank", fontsize=FS_L)
    ax[1].set_ylabel("Accuracy", fontsize=FS_L, labelpad=1); ax[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=5.3, handlelength=1.0)
    tidy(ax[1]); panel(ax[1], "B", "Rank Cutoff")
    # C: causal, rank-matched
    cau = d[d.section == "causal"]; cs = [("ACE-3", "ACE"), ("PCA-3", "PCA"), ("Full-state CCA-3", "Full-state CCA"), ("Random-3", "Random")]
    for i, (c, _) in enumerate(cs):
        for k, (met, col) in enumerate((("keep", BLUE), ("remove", GRAY))):
            v = cau[(cau.condition == c) & (cau.metric == met)].value.to_numpy()
            if len(v):
                ax[2].bar(i + (k - 0.5) * 0.36, v.mean(), width=0.34, color=col, lw=0)
                ax[2].scatter(i + (k - 0.5) * 0.36 + rng.uniform(-0.06, 0.06, len(v)), v, s=2, color="k", alpha=0.5, lw=0, zorder=3)
    ax[2].axhline(0.748, color="0.3", lw=0.5, ls="--"); ax[2].axhline(0.25, color="0.4", lw=0.6, ls=":")
    ax[2].set_xticks(range(len(cs))); ax[2].set_xticklabels([n for _, n in cs], fontsize=5.6, rotation=50, ha="right", rotation_mode="anchor")
    ax[2].set_ylim(0.15, 0.82); ax[2].set_ylabel("Accuracy", fontsize=FS_L, labelpad=1); tidy(ax[2]); panel(ax[2], "C", "Ablations")
    # D: spectrum error
    sp = d[d.section == "spectrum"]; ss = [c for c, _ in cs if len(sp[sp.condition == c])]
    for i, c in enumerate(ss):
        v = sp[sp.condition == c].value.to_numpy()
        ax[3].scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=4, color=GRAY, lw=0); ax[3].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=BLUE, lw=1.2)
    ax[3].set_yscale("log"); ax[3].set_xticks(range(len(ss))); ax[3].set_xticklabels([dict(cs)[c] for c in ss], fontsize=5.6, rotation=50, ha="right", rotation_mode="anchor")
    ax[3].set_ylabel("Spectral error", fontsize=FS_L, labelpad=1); tidy(ax[3]); panel(ax[3], "D", "Spectrum")
    # E: overlap with ACE
    ov = d[d.section == "overlap"]; oc = [("ACE-3", "ACE"), ("PCA-3", "PCA"), ("Full-state CCA-3", "Full-state CCA"), ("Random-3", "Random"), ("Untrained ACE-3", "Untrained")]
    oc = [x for x in oc if len(ov[ov.condition == x[0]])]
    for i, (c, _) in enumerate(oc):
        v = ov[ov.condition == c].value.to_numpy()
        ax[4].scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=3, color=GRAY, lw=0, alpha=0.6); ax[4].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=BLUE, lw=1.2)
    ax[4].set_xticks(range(len(oc))); ax[4].set_xticklabels([n for _, n in oc], fontsize=5.6, rotation=50, ha="right", rotation_mode="anchor"); ax[4].set_ylim(-0.03, 1.05)
    ax[4].set_ylabel("Overlap with ACE", fontsize=FS_L, labelpad=1); tidy(ax[4]); panel(ax[4], "E", "Subspace Overlap")
    save(fig, "app_markov_ladder")


def markov_energy():
    d = pd.read_csv(EV / "24_markov_energy_evidence.csv"); cur = d[d.record_type == "curve"]; cut = d[d.record_type == "cutoff"]
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.75), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.45})
    for meth, col in (("ACE", BLUE), ("PCA", GRAY)):
        c = cur[cur.method == meth]
        for _, g in c.groupby(["model", "seed"]):
            ax[0].plot(g["rank"], g.cumulative_energy, color=col, lw=0.3, alpha=0.4)
        m = c.groupby("rank").cumulative_energy.mean()
        ax[0].plot(m.index, m.values, color=col, lw=1.2, label=meth)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks([1, 2, 4, 8, 16, 32, 64]); ax[0].set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax[0].axvline(3, color=RED, lw=0.6, ls="--"); ax[0].set_xlabel("Prefix rank", fontsize=FS_L); ax[0].set_ylabel("Cumulative energy", fontsize=FS_L)
    ax[0].legend(loc="lower right"); tidy(ax[0]); panel(ax[0], "A", "Cumulative Spectrum")
    for k, (thr, mk) in enumerate(((0.99, "o"), (0.999, "s"))):
        for meth, col in (("ACE", BLUE), ("PCA", GRAY)):
            v = cut[(cut.method == meth) & (np.isclose(cut.threshold, thr))].cutoff_rank.to_numpy()
            ax[1].scatter(k + (0.15 if meth == "PCA" else -0.15) + rng.uniform(-0.05, 0.05, len(v)), v, s=6, marker=mk, color=col, lw=0)
    ax[1].set_yscale("log", base=2); ax[1].set_yticks([2, 4, 8, 16, 32, 64]); ax[1].set_yticklabels(["2", "4", "8", "16", "32", "64"])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["99%", "99.9%"]); ax[1].set_xlim(-0.6, 1.6)
    ax[1].set_xlabel("Energy threshold", fontsize=FS_L); ax[1].set_ylabel("Rank needed", fontsize=FS_L); tidy(ax[1]); panel(ax[1], "B", "Cutoff Rank")
    save(fig, "app_markov_energy")


# --------------------------------------------------------------------------- LLM controls
def llm_controls(csv, methods, labels, name, title_prefix):
    d = pd.read_csv(EV / csv)
    fig, ax = plt.subplots(1, 4, figsize=(5.5, 1.7), gridspec_kw={"wspace": 0.55})
    for k, (met, ttl, chance, ylab) in enumerate((("Keep", "Keep (Sufficiency)", 0.5, "AUC"), ("Remove", "Remove (Necessity)", 0.5, "AUC"),
                                                 ("Flip", "Flip (Inversion)", 0.5, "AUC"), ("Overlap", "Overlap with ACE", None, "|cos| with ACE axis"))):
        ms = [m for m in methods if len(d[(d.method == m) & (d.metric == met)])]
        vals = {m: d[(d.method == m) & (d.metric == met)].value.to_numpy() for m in ms}
        bars(ax[k], ms, [METHOD_COL[m] for m in ms], vals, chance=chance)
        ax[k].set_xticklabels([labels[m] for m in ms], rotation=35, ha="right", fontsize=5.8)
        ax[k].set_ylabel(ylab, fontsize=FS_L, labelpad=1); panel(ax[k], "ABCD"[k], ttl, dx=-0.38)
    save(fig, name)


def sva_keep_remove():
    d = pd.read_csv(EV / "21_sva_controls_evidence.csv")
    fig, ax = plt.subplots(figsize=(2.3, 2.1))
    ax.add_patch(plt.Rectangle((0.75, 0.4), 0.33, 0.35, color=BLUE, alpha=0.08, lw=0))
    ax.text(0.765, 0.415, "sufficient +\nnecessary", fontsize=5.5, color=BLUE, va="bottom")
    for m, lab in (("ACE", "ACE"), ("Difference of means", "Mean difference"), ("PCA-1", "PCA-1"), ("Unembedding", "Unembedding")):
        k = d[(d.method == m) & (d.metric == "Keep")].value.to_numpy(); r = d[(d.method == m) & (d.metric == "Remove")].value.to_numpy()
        ax.errorbar(k.mean(), r.mean(), xerr=k.std(), yerr=r.std(), fmt="o", color=METHOD_COL[m], ms=4, elinewidth=0.6, capsize=1.5)
        ax.annotate(lab, (k.mean(), r.mean()), xytext=(4, 4), textcoords="offset points", fontsize=5.8, color="0.2")
    ax.axvline(0.75, color="0.5", lw=0.5, ls="--"); ax.axhline(0.75, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(0.45, 1.05); ax.set_ylim(0.4, 1.05); ax.set_xlabel("Keep AUC (higher = sufficient)", fontsize=FS_L)
    ax.set_ylabel("Remove AUC (lower = necessary)", fontsize=FS_L); tidy(ax); panel(ax, "", "Sufficiency vs. Necessity", dx=0)
    save(fig, "app_sva_keep_remove")


def llm_nulls():
    d = pd.read_csv(EV / "14_llm_null_ladders_evidence.csv")
    fig, ax = plt.subplots(1, 4, figsize=(5.5, 1.85), gridspec_kw={"wspace": 0.8, "width_ratios": [1.1, 1, 1.1, 1]})
    s = d[(d.task == "SVA") & (d.evidence_type == "CCA")]
    conds = ["Raw ACE", "Label-only expectation", "Labels removed", "Prompt-shuffled"]
    for i, c in enumerate(conds):
        v = s[s.method == c].value.abs().to_numpy()
        ax[0].scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=4, color=GRAY, lw=0)
        ax[0].errorbar(i, v.mean(), yerr=v.std(), fmt="o", color=BLUE, ms=3.5, elinewidth=0.7, capsize=1.5)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Raw", "Label only", "Labels removed", "Shuffled"], fontsize=5.8, rotation=35, ha="right", rotation_mode="anchor")
    ax[0].set_ylim(0, 1.02); ax[0].set_ylabel("|Correlation|", fontsize=FS_L, labelpad=1); tidy(ax[0]); panel(ax[0], "A", "SVA: Label Removal", dx=-0.35)
    def flip(axx, task, ms, labels, letter, title):
        c = d[(d.task == task) & (d.evidence_type == "Causal") & (d.metric == "flip_auc")]
        for i, m in enumerate(ms):
            v = c[c.method == m].value.to_numpy()
            axx.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=4, color=GRAY, lw=0)
            axx.plot([i - 0.25, i + 0.25], [np.exp(np.log(np.clip(v, 1e-3, None)).mean())] * 2, color=METHOD_COL[m], lw=1.5)
        axx.set_yscale("log"); axx.set_ylim(1e-3, 1.5); axx.axhline(0.5, color="0.4", lw=0.6, ls=":")
        axx.set_xticks(range(len(ms))); axx.set_xticklabels([labels[m] for m in ms], rotation=35, ha="right", fontsize=5.8)
        axx.set_ylabel("Flip AUC (lower = stronger)", fontsize=FS_L, labelpad=1); tidy(axx); panel(axx, letter, title, dx=-0.35)
    flip(ax[1], "SVA", ["ACE", "PCA-1", "Difference of means", "Unembedding"],
         {"ACE": "ACE", "PCA-1": "PCA-1", "Difference of means": "Mean diff.", "Unembedding": "Unembed."}, "B", "SVA: Flip")
    i_ = d[(d.task == "IOI") & (d.evidence_type == "CCA")]
    for c, col in (("Raw ACE", BLUE), ("Answer label removed", "#9ecae1"), ("Task covariates removed", "#2ca02c"), ("Prompt-shuffled", GRAY)):
        m = [i_[(i_.method == c) & (i_.metric == k)].value for k in ("CC1", "CC2", "CC3")]
        ax[2].errorbar([1, 2, 3], [x.mean() for x in m], yerr=[x.std() for x in m], fmt="-o", color=col, ms=2.5, lw=0.9, elinewidth=0.6,
                       capsize=1.5, label={"Raw ACE": "raw", "Answer label removed": "answer removed", "Task covariates removed": "covariates removed",
                                           "Prompt-shuffled": "shuffled"}[c])
    ax[2].set_xticks([1, 2, 3]); ax[2].set_xticklabels(["CC1", "CC2", "CC3"]); ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("Canonical correlation", fontsize=FS_L, labelpad=1)
    for y, t, col in ((0.93, "raw, answer label removed", BLUE), (0.345, "task covariates removed", "#2ca02c"), (0.11, "prompts shuffled", GRAY)):
        ax[2].text(2.0, y, t, ha="center", fontsize=5.4, color=col)
    tidy(ax[2]); panel(ax[2], "C", "IOI: Residual CCA", dx=-0.35)
    flip(ax[3], "IOI", ["ACE selected", "ACE rank 1", "PCA-1", "Difference of means"],
         {"ACE selected": "ACE sel.", "ACE rank 1": "ACE-1", "PCA-1": "PCA-1", "Difference of means": "Mean diff."}, "D", "IOI: Flip")
    save(fig, "app_llm_nulls")


# --------------------------------------------------------------------------- IOI companions
def modadd_prefix():
    d = np.load(EV / "26_modadd_prefix_curves.npz")
    ep, rk = d["epochs"], d["ranks"]
    fig = plt.figure(figsize=(5.5, 3.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.75, wspace=0.45, left=0.09, right=0.98, top=0.92, bottom=0.11)
    for j, (key, ttl) in enumerate((("keep", "Core Only (Sufficiency)"), ("remove", "Core Removed (Necessity)"))):
        ax = fig.add_subplot(gs[0, j]); M = d[key].mean(0)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=(rk[0] - 0.5, rk[-1] + 0.5, ep[0] - 50, ep[-1] + 50))
        ax.axhline(800, color="white", lw=0.6, ls="--"); ax.text(rk[-1] - 1, 830, "grok", color="white", fontsize=5.3, ha="right")
        ax.set_xlabel("ACE prefix rank $r$", fontsize=FS_L); ax.set_ylabel("Epoch", fontsize=FS_L) if j == 0 else None
        ax.tick_params(length=2, width=0.5); panel(ax, "AB"[j], ttl, dx=-0.3)
    cb = fig.colorbar(im, ax=[fig.axes[0], fig.axes[1]], location="right", fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.set_label("Held-out accuracy", fontsize=6)
    ax = fig.add_subplot(gs[0, 2])
    for key, col, lab in (("r_energy", "0.3", "99% energy"), ("r_suf", BLUE, "sufficient"), ("r_nec_near", RED, "necessary")):
        m = d[key].mean(0); ax.plot(ep, m, "-o", ms=1.8, lw=0.9, color=col, label=lab)
        ax.fill_between(ep, d[key].min(0), d[key].max(0), color=col, alpha=0.12, lw=0)
    ax.axvline(800, color="0.4", lw=0.6, ls="--"); ax.set_xlabel("Epoch", fontsize=FS_L); ax.set_ylabel("Dimension", fontsize=FS_L, labelpad=1)
    ax.legend(loc="upper right", fontsize=5.3, handlelength=1); tidy(ax); panel(ax, "C", "Threshold Crossings", dx=-0.3)
    for j, (key, ttl, chance) in enumerate((("order_keep", "Ordering: Core Only", None), ("order_remove", "Ordering: Core Removed", float(d["near_chance"])))):
        ax = fig.add_subplot(gs[1, 0] if j == 0 else gs[1, 1:])
        O = d[key]; r = np.arange(O.shape[-1])
        rnd = O[:, 2:, :].reshape(-1, O.shape[-1])
        ax.fill_between(r, np.percentile(rnd, 5, 0), np.percentile(rnd, 95, 0), color=GRAY, alpha=0.3, lw=0)
        ax.plot(r, rnd.mean(0), color="0.4", lw=0.9, label="random order")
        ax.plot(r, O[:, 1, :].mean(0), color=RED, lw=0.9, ls="--", label="reversed ACE order")
        ax.plot(r, O[:, 0, :].mean(0), color=BLUE, lw=1.2, label="ACE order")
        if chance:
            ax.axhline(chance, color="0.3", lw=0.5, ls=":")
        ax.set_xlim(0, r[-1]); ax.set_ylim(-0.02, 1.03); ax.set_xlabel("Prefix rank within the same ACE span", fontsize=FS_L)
        ax.set_ylabel("Held-out accuracy", fontsize=FS_L, labelpad=1); tidy(ax); panel(ax, "DE"[j], ttl, dx=-0.3 if j == 0 else -0.14)
        ax.legend(loc="center right" if j == 0 else "center left", fontsize=5.3, handlelength=1.4)
    save(fig, "app_modadd_prefix")


def sva_jacobian():
    p = json.loads((ROOT / "supplement/sva_jacobian_robustness/results.json").read_text()); rows = p["rows"]
    ns = sorted({int(r["jacobian_samples"]) for r in rows})
    agg = lambda k: [np.array([float(r[k]) for r in rows if int(r["jacobian_samples"]) == n]) for n in ns]
    fig, ax = plt.subplots(1, 2, figsize=(4.4, 1.75), gridspec_kw={"wspace": 0.45})
    for k, col, lab in (("keep_auc", BLUE, "core only"), ("remove_auc", RED, "core removed")):
        v = agg(k); ax[0].errorbar(ns, [x.mean() for x in v], yerr=[x.std() for x in v], fmt="-o", ms=2.5, lw=0.9, color=col, capsize=1.5,
                                   elinewidth=0.6, label=lab)
    ax[0].axhline(p["baseline_auc"], color="0.3", lw=0.5, ls="--", label="clean baseline"); ax[0].axhline(0.5, color="0.5", lw=0.5, ls=":")
    ax[0].set_xscale("log"); ax[0].set_xticks(ns); ax[0].set_xticklabels([str(n) for n in ns]); ax[0].minorticks_off(); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("Jacobian samples", fontsize=FS_L); ax[0].set_ylabel("AUC", fontsize=FS_L)
    ax[0].legend(loc="center right", fontsize=5.3, handlelength=1.2); tidy(ax[0]); panel(ax[0], "A", "Causal Scores Converge", dx=-0.22)
    v = agg("alignment_to_full"); ax[1].errorbar(ns, [x.mean() for x in v], yerr=[x.std() for x in v], fmt="-o", ms=2.5, lw=0.9, color=BLUE,
                                                  capsize=1.5, elinewidth=0.6)
    ax[1].set_xscale("log"); ax[1].set_xticks(ns); ax[1].set_xticklabels([str(n) for n in ns]); ax[1].minorticks_off(); ax[1].set_xlabel("Jacobian samples", fontsize=FS_L); ax[1].set_ylabel("|cos| with 600-sample axis", fontsize=FS_L)
    tidy(ax[1]); panel(ax[1], "B", "Axis Is Stable", dx=-0.3)
    save(fig, "app_sva_jacobian")


def sva_alignment_heatmap():
    """Cross-model core alignment (15 pairs) as heatmaps; values as in the appendix table of the original submission."""
    names = ["GPT-2 Small", "GPT-2 Medium", "GPT-2 Large", "LLaMA-3.1", "Gemma-2", "Qwen2.5"]
    pairs = {("Qwen2.5", "LLaMA-3.1"): (0.921, 0.924), ("Qwen2.5", "Gemma-2"): (0.934, 0.943), ("Qwen2.5", "GPT-2 Large"): (0.844, 0.857),
             ("Qwen2.5", "GPT-2 Medium"): (0.829, 0.842), ("Qwen2.5", "GPT-2 Small"): (0.701, 0.757), ("LLaMA-3.1", "Gemma-2"): (0.888, 0.880),
             ("LLaMA-3.1", "GPT-2 Large"): (0.760, 0.752), ("LLaMA-3.1", "GPT-2 Medium"): (0.727, 0.718), ("LLaMA-3.1", "GPT-2 Small"): (0.585, 0.616),
             ("Gemma-2", "GPT-2 Large"): (0.893, 0.909), ("Gemma-2", "GPT-2 Medium"): (0.894, 0.911), ("Gemma-2", "GPT-2 Small"): (0.790, 0.846),
             ("GPT-2 Large", "GPT-2 Medium"): (0.923, 0.951), ("GPT-2 Large", "GPT-2 Small"): (0.878, 0.924), ("GPT-2 Medium", "GPT-2 Small"): (0.919, 0.968)}
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.35), gridspec_kw={"wspace": 0.08})
    for k, (ttl, letter) in enumerate((("Spearman's $\\rho$", "A"), ("Pearson's $r$ (CCA$_{1D}$)", "B"))):
        M = np.eye(6)
        for (a, b), v in pairs.items():
            i, j = names.index(a), names.index(b); M[i, j] = M[j, i] = v[k]
        im = ax[k].imshow(M, cmap="RdBu_r", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax[k].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if M[i, j] > 0.8 else "k")
        ax[k].set_xticks(range(6)); ax[k].set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.8)
        ax[k].set_yticks(range(6)); ax[k].set_yticklabels(names if k == 0 else [], fontsize=5.8); ax[k].tick_params(length=0, pad=1.5)
        panel(ax[k], letter, ttl, dx=-0.62 if k == 0 else -0.1)
    cb = fig.colorbar(im, ax=list(ax), fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=5.5, length=1.5)
    save(fig, "app_sva_alignment")


def ioi_single_core():
    """Single-site IOI core: localization, causal rank, controls, state transplant."""
    from cores.ioi.figures import COLORS, MODELS, OLD_STYLE
    dirs = {"gpt2": "gpt2", "gpt2-medium": "gpt2medium", "gpt2-large": "gpt2large", "llama-3.1-8b": "llama318b", "gemma-2-9b": "gemma29b", "qwen2.5-32b": "qwen2532b"}
    R = []
    for i, (n, l) in enumerate(MODELS):
        if n in dirs:
            r = json.loads((ROOT / "src/cores/experiments/ioi_all" / dirs[n] / "results.json").read_text())
            L, rs = r["final_layer_idx"], r["selected_rank"]
            c = r["rank_curves"].get(str(L)) or r["rank_curve"]
            R.append((n, l.replace("-8B", "").replace("-9B", "").replace("-32B", ""), r, L, rs, c, next(x for x in c if x["rank"] == rs), COLORS[i]))
    fig, ax = plt.subplots(1, 4, figsize=(5.5, 1.5), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.05, 1.05]})
    for n, l, r, L, rs, c, row, col in R:
        m, nl = OLD_STYLE[n][1], r["n_layers"]
        pts = sorted((x["layer_idx"], x["auc_flip"]) for x in r["sweep"])
        ax[0].plot([p[0] / nl for p in pts], [p[1] for p in pts], color=col, lw=0.8)
        ax[0].scatter(L / nl, dict(pts)[L], s=14, marker=m, color=col, edgecolor="k", lw=0.3, zorder=3)
        ax[1].plot([x["rank"] for x in c], [x["auc_flip"] for x in c], color=col, lw=0.8)
        ax[1].scatter(rs, row["auc_flip"], s=14, marker=m, color=col, edgecolor="k", lw=0.3, zorder=3)
    ax[0].set_xlabel("Relative layer depth", fontsize=FS_L); ax[0].set_ylabel("Flip AUC", fontsize=FS_L)
    ax[1].set_xlabel("Core rank", fontsize=FS_L); ax[1].set_ylabel("Flip AUC", fontsize=FS_L)
    w = 0.8 / len(R)
    for i, (n, l, r, L, rs, c, row, col) in enumerate(R):
        ax[2].bar(np.arange(4) + (i - (len(R) - 1) / 2) * w, [r["base_auc"], row["auc_keep"], row["auc_remove"], row["auc_flip"]], width=w * 0.9, color=col, lw=0)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["Base", "Keep", "Remove", "Flip"], rotation=35, ha="right", fontsize=5.8); ax[2].set_ylabel("AUC", fontsize=FS_L)
    so = [x[2]["state_operator"] for x in R]
    for k, (vals, col, lab) in enumerate((([s["auc_factual_transplant"] for s in so], GRAY, "Factual state"), ([x[6]["auc_flip"] for x in R], BLUE, "Flip"),
                                          ([s["auc_counterfactual_transplant"] for s in so], RED, "Counterfactual state"))):
        ax[3].bar(np.arange(len(R)) + (k - 1) * 0.27, vals, width=0.25, color=col, lw=0, label=lab)
    ax[3].set_xticks(range(len(R))); ax[3].set_xticklabels([x[1] for x in R], rotation=35, ha="right", fontsize=5.4); ax[3].set_ylabel("AUC", fontsize=FS_L)
    for a, s, t in zip(ax, "ABCD", ("Localization", "Causal Rank", "Core Controls", "State Transplant")):
        a.axhline(0.5, color="0.4", lw=0.6, ls=":"); a.set_ylim(-0.03, 1.05); tidy(a); panel(a, s, t, dx=-0.3)
    fig.legend(handles=[Line2D([], [], color=x[7], lw=1, marker=OLD_STYLE[x[0]][1], ms=3, label=x[1]) for x in R], loc="lower center", ncol=len(R),
               fontsize=5.6, bbox_to_anchor=(0.36, -0.28), columnspacing=0.9, handlelength=1.4)
    ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=5.4, ncol=1, handlelength=0.8, labelspacing=0.15)
    save(fig, "app_ioi_single_core")


def main():
    markov_ladder(); markov_energy()
    llm_controls("21_sva_controls_evidence.csv", ["ACE", "Difference of means", "PCA-1", "Unembedding"],
                 {"ACE": "ACE", "Difference of means": "Mean diff.", "PCA-1": "PCA-1", "Unembedding": "Unembed."}, "app_sva_controls", "SVA")
    sva_keep_remove(); llm_nulls()
    llm_controls("22_ioi_controls_evidence.csv", ["ACE selected", "ACE rank 1", "Difference of means", "PCA-1"],
                 {"ACE selected": "ACE full", "ACE rank 1": "ACE-1", "Difference of means": "Mean diff.", "PCA-1": "PCA-1"}, "app_ioi_single", "IOI")
    ioi_single_core(); modadd_prefix(); sva_jacobian(); sva_alignment_heatmap()


if __name__ == "__main__":
    main()
