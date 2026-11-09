#!/usr/bin/env python3
"""Fig. 5 (subject-verb agreement) redrawn for the NeurIPS layout: one row at the 5.5 in text width, one shared model key
(marker + color) instead of three per-panel legends, direct labels on the condition curves (A) and the prompt classes (B),
and the R^2 range in place of a six-row legend. Data, analysis and statistics are those of
src/cores/plot_sva0.py (loaders and helpers imported from its copy, sva_plot_legible.py).

  python paper_figures/sva_fig5.py experiments/sva/gpt2small experiments/sva/gpt2med experiments/sva/gpt2large \
      experiments/sva/llama31 experiments/sva/gemma2 experiments/sva/qwen25_32b
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sva_plot_legible as S  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
COND = ["base", "keep", "remove", "flip"]
COND_NAME = {"base": "Base", "keep": "Core only", "remove": "Core removed", "flip": "Core flipped"}
FS_T, FS_L, FS_K = 7.5, 7.0, 6.3


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.3, pad=1.5, length=2, width=0.5)


def title(ax, letter, text):
    ax.set_title(text, loc="left", fontsize=FS_T, fontweight="bold", pad=4)


def schematic(ax):
    """The choice the model makes: a singular subject with a plural attractor, and the two candidate verbs. Colors are those
    of the singular and plural prompts in panel B."""
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(-0.14, 0.5, "The", ha="left", va="center", fontsize=5.6, color="#444444", clip_on=False)
    ax.text(0.0, 0.5, "dog", ha="left", va="center", fontsize=5.6, color="#222222", fontweight="bold", clip_on=False)
    ax.text(0.15, 0.5, "near the cats", ha="left", va="center", fontsize=5.6, color="#444444", clip_on=False)
    ax.annotate("", xy=(0.7, 0.5), xytext=(0.6, 0.5), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.6))
    ax.text(0.74, 0.78, "is", ha="left", va="center", fontsize=5.8, fontweight="bold", color="white", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.2", fc=S.BLUE, ec="none"))
    ax.text(0.74, 0.18, "are", ha="left", va="center", fontsize=5.8, fontweight="bold", color=S.ORANGE, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=S.ORANGE, lw=0.6))
    ax.text(0.92, 0.78, "singular", ha="left", va="center", fontsize=5, color=S.BLUE, clip_on=False)
    ax.text(0.92, 0.18, "plural", ha="left", va="center", fontsize=5, color=S.ORANGE, clip_on=False)


def stroke(mk):
    return mk in ["+", "x", "1", "2", "3", "4", "|", "_"]


def main():
    runs = [S.load_run(Path(d)) for d in sys.argv[1:] if Path(d).exists()]
    models = [R["model_code"] for R in runs]
    fig = plt.figure(figsize=(5.5, 2.45))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.95], wspace=0.5, left=0.08, right=0.99, top=0.84, bottom=0.25)
    subA = gs[0, 0].subgridspec(2, 1, height_ratios=[0.2, 1.0], hspace=0.08)
    sch, axA = fig.add_subplot(subA[0]), fig.add_subplot(subA[1])
    axB, axC = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    schematic(sch)

    # ---------------- A: layer sweep
    grid = np.linspace(0, 1, 101); curves = {k: [] for k in COND}
    for R in runs:
        sw = R["res"].get("layer_sweep", [])
        if not sw:
            continue
        L = np.array([r["layer_idx"] for r in sw], float); d = L / max(L.max(), 1)
        mk = S.MODEL_MARKER.get(R["model_code"], "o")
        for k, fld in zip(COND, ["behavior_auc_base", "behavior_auc_keep", "behavior_auc_remove", "behavior_auc_flip"]):
            y = np.array([r.get(fld, np.nan) for r in sw], float); m = np.isfinite(y)
            o = np.argsort(d[m]); curves[k].append(np.interp(grid, d[m][o], y[m][o], left=np.nan, right=np.nan))
            c = S.COND_COLOR[k]
            axA.scatter(d, y, s=6, marker=mk, alpha=0.28, linewidths=0.5, zorder=3,
                        **({"c": c} if stroke(mk) else {"facecolors": "none", "edgecolors": c}))
    ends = {}
    for k in COND:
        Y = np.vstack(curves[k]); mu = np.nanmean(Y, 0)
        axA.plot(grid, mu, lw=1.0, color=S.COND_COLOR[k], zorder=4)
        axA.fill_between(grid, np.nanmin(Y, 0), np.nanmax(Y, 0), alpha=0.08, lw=0, color=S.COND_COLOR[k])
        ends[k] = mu[np.isfinite(mu)][-1]
    axA.legend(handles=[Line2D([], [], color=S.COND_COLOR[k], lw=1.2, label=COND_NAME[k]) for k in COND], loc="lower left",
               frameon=False, fontsize=FS_K, handlelength=1.2, labelspacing=0.25, borderaxespad=0.1)
    axA.axhline(0.5, color="0.6", lw=0.5, ls=":")
    axA.set(xlim=(0, 1.0), ylim=(0, 1.08), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    axA.set_xlabel("Relative layer depth", fontsize=FS_L); axA.set_ylabel("Agreement AUC", fontsize=FS_L, labelpad=1)
    tidy(axA); title(sch, "A", "Layer Sweep for\nNumber-Agreement Core")

    # ---------------- B: core coordinate vs margin
    zs = [S.zscore(R["z"]) for R in runs]; xmin, xmax = min(z.min() for z in zs), max(z.max() for z in zs)
    r2 = []
    for R in runs:
        zZ, mZ, y, mc = S.zscore(R["z"]), S.zscore(R["m"]), R["y"], R["model_code"]
        sing, plur = y == 0, y == 1
        if mZ[plur].mean() - mZ[sing].mean() < 0:
            mZ = -mZ
        if np.mean(zZ * mZ) < 0:
            zZ = -zZ
        a, b, rr, _ = S.fit_affine(zZ, mZ); r2.append(rr)
        mk = S.MODEL_MARKER.get(mc, "o")
        for msk, col in ((sing, S.BLUE), (plur, S.ORANGE)):
            axB.scatter(zZ[msk], mZ[msk], s=2.5, alpha=0.18, color=col, marker=mk, linewidths=0.4 if stroke(mk) else 0, rasterized=True)
        xs = np.linspace(xmin, xmax, 50)
        axB.plot(xs, a * xs + b, color=S.MODEL_COLOR.get(mc, "0.4"), lw=0.9, ls=S.MODEL_LS.get(mc, "-"))
    axB.axhline(0, color="0.75", lw=0.4, ls="--"); axB.axvline(0, color="0.75", lw=0.4, ls="--")
    axB.text(0.03, 0.97, "plural", color=S.ORANGE, fontsize=FS_K, fontweight="bold", transform=axB.transAxes, va="top")
    axB.text(0.97, 0.03, "singular", color=S.BLUE, fontsize=FS_K, fontweight="bold", transform=axB.transAxes, ha="right")
    axB.text(0.97, 0.97, f"$R^2$ {min(r2):.2f}–{max(r2):.2f}", fontsize=FS_K, transform=axB.transAxes, ha="right", va="top", color="0.25")
    axB.set_xlabel("Core coordinate ($z$)", fontsize=FS_L)
    axB.set_ylabel("Plural $-$ singular logit ($z$)", fontsize=FS_L, labelpad=1)
    axB.xaxis.set_major_locator(MaxNLocator(3)); axB.yaxis.set_major_locator(MaxNLocator(3))
    tidy(axB); title(axB, "B", "Single Core Axis Controls\nAgreement in LLMs")
    # plural points sit upper right; move the plural label there if the corner is free
    axB.texts[0].set_position((0.03, 0.97))

    # ---------------- C: perturbations
    modes, labels = ["clean", "remove", "flip"], ["Base", "Remove", "Flip"]
    sc = {}
    for R in runs:
        rows = R["res"].get("flip_demo", [])
        bp = {}
        for r in rows:
            bp.setdefault(r["prompt"], {})[r["mode"]] = {k: float(r[k]) for k in ["is", "are", "was", "were"]}
        out = {m: [] for m in modes}
        for pr in sorted(bp):
            cs = S.infer_correct_set_from_prompt(pr)
            for m in modes:
                out[m].append(S.log_ratio_correct_over_max(bp[pr][m], cs) if m in bp[pr] else np.nan)
        sc[R["model_code"]] = {m: np.array(v, float) for m, v in out.items()}
    pos0 = np.array([1.0, 2.0, 3.0]); off = np.linspace(-0.3, 0.3, len(runs)); w = 0.6 / len(runs)
    rng = np.random.default_rng(0); allv = []
    for j, m in enumerate(modes):
        for i, mc in enumerate(models):
            v = sc[mc][m]; v = v[np.isfinite(v)]; allv.append(v)
            b = axC.boxplot([v], positions=[pos0[j] + off[i]], widths=w * 0.9, showfliers=False, patch_artist=True,
                            medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                            boxprops=dict(lw=0))
            b["boxes"][0].set(facecolor=S.MODEL_COLOR.get(mc, "0.5"), alpha=0.75)
            axC.scatter(rng.normal(pos0[j] + off[i], 0.012, len(v)), v, s=2, color="k", alpha=0.35, lw=0, zorder=5)
    allv = np.concatenate(allv); top, bot = np.nanmax(allv), np.nanmin(allv); yr = top - bot
    for k, (ma, mb, ia, ib) in enumerate([("clean", "remove", 0, 1), ("clean", "flip", 0, 2), ("remove", "flip", 1, 2)]):
        p = S.fisher_combine_p([S.paired_wilcoxon(sc[mc][ma], sc[mc][mb])[1] for mc in models])
        yy = top + 0.05 * yr + k * 0.09 * yr
        axC.plot([pos0[ia], pos0[ia], pos0[ib], pos0[ib]], [yy - 0.015 * yr, yy, yy, yy - 0.015 * yr], lw=0.4, c="k")
        axC.text((pos0[ia] + pos0[ib]) / 2, yy, (f"$p$={p:.0e}" if p < 1e-3 else f"$p$={p:.3f}").replace("e-0", "e-"),
                 ha="center", va="bottom", fontsize=5.6)
    axC.set_xticks(pos0); axC.set_xticklabels(labels, fontsize=FS_L)
    axC.set_ylabel(r"$\log\,\mathbb{P}(\mathrm{correct})/\mathbb{P}(\mathrm{max})$", fontsize=FS_L, labelpad=1)
    axC.set_ylim(bot - 0.05 * yr, top + 0.36 * yr); axC.yaxis.set_major_locator(MaxNLocator(4))
    tidy(axC); title(axC, "C", "Perturbing Core\nDisrupts Agreement")

    # ---------------- shared model key
    hs = [Line2D([], [], color=S.MODEL_COLOR.get(mc, "0.4"), ls=S.MODEL_LS.get(mc, "-"), lw=1.0,
                 marker=S.MODEL_MARKER.get(mc, "o"), ms=3.2, mfc="none" if not stroke(S.MODEL_MARKER.get(mc, "o")) else None,
                 mew=0.7, label=R["pretty"]) for mc, R in zip(models, runs)]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=FS_K, handlelength=2.0,
               columnspacing=1.3, bbox_to_anchor=(0.52, -0.01))
    for ax, L in ((sch, "A"), (axB, "B"), (axC, "C")):
        p = ax.get_position()
        fig.text(p.x0 - 0.06, p.y1 + 0.075, L, fontsize=9, fontweight="bold", va="bottom", ha="left")
    out = Path(__file__).resolve().parent / "figures" / "sva_fig5"
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=400)
    print("wrote", out, "R2", [round(x, 2) for x in r2])


if __name__ == "__main__":
    main()
