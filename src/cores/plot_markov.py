"""
Plot Markov Fig
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# -------------------------------------------------------------------------
# Style 
# -------------------------------------------------------------------------
BLUE = "#1f77b4"
GRAY = "#A9A9A9"
DOT_C = "k"

RNG_SEED = 0
BAR_W = 0.75
JIT = 0.50          
DOT_S = 0.25       

LEN = 6.5
HI = 2.7
HMAP_COL = "RdBu_r"

USE_TEX = False


# -------------------------------------------------------------------------
# Helpers 
# -------------------------------------------------------------------------
def mean_std(x):
    x = np.asarray(x, float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1)) if np.sum(~np.isnan(x)) > 1 else 0.0
    return mu, sd


def get_base_keep_remove(rows, k):
    base = np.full(k, np.nan)
    keep = np.full(k, np.nan)
    rem = np.full(k, np.nan)
    for r in rows:
        i = int(r["model"])
        base[i] = r["base"]
        keep[i] = r["keep"]
        rem[i] = r["remove"]
    return base, keep, rem


def to_complex(vals):
    """Convert a list of [re, im] pairs to a 1D complex array."""
    v = np.asarray(vals, float)
    return v[:, 0] + 1j * v[:, 1]


# -------------------------------------------------------------------------
# Main plotting
# -------------------------------------------------------------------------
def make_figure(results, outpath: Path):
    # ---- Pull everything we need out of `results` -----------------------
    cos = np.asarray(results["weight_cosine_similarity"], dtype=float)
    k = len(results["ablations"])

    base, keep, rem = get_base_keep_remove(results["ablations"], k)
    oracle = float(results["oracle"])
    chance = float(results["chance"])

    # ---- Global mpl style ------------------------------------------------
    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.2})
    if USE_TEX:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage{cmbright}",
        })
    else:
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Latin Modern Sans",
                "Computer Modern Sans Serif",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        })

    # ---- Figure layout: cos | cbar | abl | ov | cbar | cca | cbar | E | F
    fig = plt.figure(figsize=(LEN, HI), dpi=500, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 8,
        height_ratios=[1.0, 1.1],
        width_ratios=[0.75, 0.04, 0.6, 0.75, 0.04, 0.75, 0.04, 0.85],
    )
    ax_cos = fig.add_subplot(gs[0, 0])
    cax_cos = fig.add_subplot(gs[0, 1])
    ax_abl = fig.add_subplot(gs[0, 2])
    ax_ov = fig.add_subplot(gs[0, 3])
    cax_ov = fig.add_subplot(gs[0, 4])
    ax_cca = fig.add_subplot(gs[0, 5])
    cax_cca = fig.add_subplot(gs[0, 6])
    #axP = fig.add_subplot(gs[0, 7])
    ax_eig = fig.add_subplot(gs[0, 7])

    # =====================================================================
    # A) Cosine sim heatmap
    # =====================================================================
    n = cos.shape[0]
    labels_cos = (
        [r"M$_1$", r"M$_2$", r"M$_3$"] if n == 3 else [f"M{i}" for i in range(n)]
    )

    hm_cos = sns.heatmap(
        cos,
        annot=True, fmt=".2f",
        vmin=0.0, vmax=1.0,
        cmap=HMAP_COL, square=True,
        annot_kws={"size": 5},
        ax=ax_cos, cbar=True, cbar_ax=cax_cos,
    )
    ax_cos.set_xlabel("Full Model", fontsize=6)
    ax_cos.text(
        0.5, -0.4, r"Seeds $\left\{1, 2, 3\right\}$",
        transform=ax_cos.transAxes, ha="center", va="top",
        fontsize=5, fontstyle="italic",
    )
    ax_cos.set_ylabel("Full Model", fontsize=6)
    ax_cos.set_xticklabels(labels_cos, fontsize=6)
    ax_cos.set_yticklabels(labels_cos, fontsize=6, rotation=0)
    for which in ("x", "y"):
        ax_cos.tick_params(axis=which, which="both",
                           labelsize=5, pad=1, length=1, width=0.5)
    cbar = hm_cos.collections[0].colorbar
    cbar.set_ticks([0.0, 0.0, 1.0])
    cbar.set_ticklabels(["0", "0", "1"])
    cbar.ax.tick_params(labelsize=5, length=0)
    cbar.ax.set_ylabel("Cosine similarity", fontsize=6, labelpad=0, rotation=270)

    # =====================================================================
    # B) Ablations overlay (per-model ACE)
    # =====================================================================
    rng = np.random.RandomState(RNG_SEED)
    labelsA = ["Optimal", "Full Model", "Core Only", "Core Removed", "Chance"]
    xs = np.arange(len(labelsA), dtype=float)

    def bar_with_dots(ax, x, arr, cap=2, alph=1.0, col=GRAY):
        mu, sd = mean_std(arr)
        ax.bar([x], [mu], yerr=[sd], capsize=cap, width=0.9, color=col, alpha=alph)
        vals = np.asarray(arr, float)
        vals = vals[~np.isnan(vals)]
        jit = (rng.rand(len(vals)) - 0.5) * (BAR_W * JIT)
        ax.scatter(np.full(len(vals), x) + jit, vals,
                   s=DOT_S, color=DOT_C, zorder=3)

    ax_abl.bar([0], [oracle], capsize=3, width=BAR_W, color=GRAY, alpha=0.7)
    bar_with_dots(ax_abl, 1, base, alph=0.7)
    bar_with_dots(ax_abl, 2, keep, alph=1.0, col=BLUE)
    bar_with_dots(ax_abl, 3, rem,  alph=1.0, col=BLUE)
    ax_abl.bar([4], [chance], capsize=3, width=BAR_W, color=GRAY, alpha=0.7)

    ax_abl.set_yticks(np.linspace(0, 1, 2), [0, 1], fontsize=5)
    ax_abl.set_xticks(xs, labelsA, fontsize=6)
    plt.setp(ax_abl.get_xticklabels(),
             rotation=45, fontsize=5, ha="right", rotation_mode="anchor")
    ax_abl.set_ylim(0, 1)
    ax_abl.set_ylabel("Test accuracy", labelpad=-3, fontsize=6)
    for which in ("x", "y"):
        ax_abl.tick_params(axis=which, which="both",
                           labelsize=5, pad=1, length=1, width=0.5)
    ax_abl.legend(
        handles=[
            Patch(facecolor=GRAY, edgecolor="none", label="Controls", alpha=0.7),
            Patch(facecolor=BLUE, edgecolor="none", label="Ablations", alpha=1.0),
        ],
        loc="upper left", framealpha=0.0, edgecolor="none",
        fontsize=4, labelspacing=0.0,
    )

    # =====================================================================
    # C) Projector overlap heatmap   
    # =====================================================================
    rows = results["pairwise"]
    overlap_mat = np.eye(k, dtype=float)
    cca_mat = np.eye(k, dtype=float)
    for row in rows:
        i, j = map(int, row["pair"].replace("M", "").split("-"))
        overlap_mat[i, j] = overlap_mat[j, i] = float(row["overlap"])
        cca = np.asarray(row.get("cca_corrs", []), dtype=float)
        m_ = min(3, cca.size)
        cca_val = float(np.nanmean(cca[:m_])) if m_ > 0 else 0.0
        cca_mat[i, j] = cca_mat[j, i] = cca_val

    hm_ov = sns.heatmap(
        overlap_mat, annot=True, fmt=".2f", vmin=0, vmax=1,
        cmap=HMAP_COL, square=True, annot_kws={"size": 5},
        ax=ax_ov, cbar=True, cbar_ax=cax_ov,
    )
    ax_ov.set_xlabel("Core", fontsize=6)
    ax_ov.set_xticklabels([r"C$_1$", r"C$_2$", r"C$_3$"])
    ax_ov.set_yticklabels([r"C$_1$", r"C$_2$", r"C$_3$"], rotation=0)
    for which in ("x", "y"):
        ax_ov.tick_params(axis=which, which="both",
                          labelsize=5, pad=1, length=1, width=0.5)
    cbar1 = hm_ov.collections[0].colorbar
    cbar1.set_ticks([0, 1.0])
    cbar1.set_ticklabels(["0", "1"])
    cbar1.ax.tick_params(labelsize=5, length=0)
    cbar1.ax.set_ylabel("Projector Overlap", labelpad=0, rotation=270, fontsize=6)

    hm_c = sns.heatmap(
        cca_mat, annot=True, fmt=".2f", vmin=0.0, vmax=1.0,
        cmap=HMAP_COL, square=True, annot_kws={"size": 5},
        ax=ax_cca, cbar=True, cbar_ax=cax_cca,
    )
    ax_cca.set_xlabel("Core", fontsize=6)
    ax_cca.set_xticklabels([r"C$_1$", r"C$_2$", r"C$_3$"])
    ax_cca.set_yticklabels([r"C$_1$", r"C$_2$", r"C$_3$"], rotation=0)
    for which in ("x", "y"):
        ax_cca.tick_params(axis=which, which="both",
                           labelsize=5, pad=1, length=1, width=0.5)
    cbar2 = hm_c.collections[0].colorbar
    cbar2.set_ticks([0, 1.0])
    cbar2.set_ticklabels(["0", "1"])
    cbar2.ax.tick_params(labelsize=5, length=0)
    cbar2.ax.set_ylabel("Mean CCA", labelpad=0, rotation=270, fontsize=6)

    # =====================================================================
    # E) Core vs. Consensus alignment
    # =====================================================================
#    p_base, p_keep, p_rem = get_base_keep_remove(
#        results["ensemble_gcca"]["ablations"], k)
#    c_base, c_keep, c_rem = get_base_keep_remove(
#        results["gcca"]["ablations"], k)
#
#    labelsP = ["Optimal", r"Consensus$^{+}$", r"Consensus$^{-}$",
#               r"Core$^{+}$", r"Core$^{-}$", "Chance"]
#    xsP = np.arange(len(labelsP), dtype=float)
#
#    def bar_with_dots2(x, arr, cap=2, alph=1.0, col=GRAY, axP=axP):
#        mu, sd = mean_std(arr)
#        axP.bar([x], [mu], yerr=[sd], capsize=cap, width=BAR_W,
#                color=col, alpha=alph)
#        vals = np.asarray(arr, float)
#        vals = vals[~np.isnan(vals)]
#        jit = (rng.rand(len(vals)) - 0.5) * (BAR_W * JIT)
#        axP.scatter(np.full(len(vals), x) + jit, vals,
#                    s=DOT_S, color=DOT_C, zorder=3)
#
#    axP.bar([0], [oracle], capsize=3, width=BAR_W, color=GRAY, alpha=0.7)
#    bar_with_dots2(1, p_keep, alph=1.0, col="#4A5568", axP=axP)
#    bar_with_dots2(2, p_rem,  alph=1.0, col="#4A5568", axP=axP)
#    bar_with_dots2(3, c_keep, alph=1.0, col=BLUE,     axP=axP)
#    bar_with_dots2(4, c_rem,  alph=1.0, col=BLUE,     axP=axP)
#    axP.bar([5], [chance], capsize=3, width=BAR_W, color=GRAY, alpha=0.7)
#
#    axP.set_yticks(np.linspace(0, 1, 2), [0, 1], fontsize=5)
#    axP.set_xticks(xsP, labelsP, fontsize=6)
#    plt.setp(axP.get_xticklabels(),
#             rotation=60, fontsize=5, ha="right", rotation_mode="anchor")
#    axP.set_ylim(0, 1)
#    axP.set_ylabel("Test accuracy", labelpad=-3, fontsize=6)
#    for which in ("x", "y"):
#        axP.tick_params(axis=which, which="both",
#                        labelsize=5, pad=1, length=1, width=0.5)
#    axP.legend(
#        handles=[
#            Patch(facecolor=GRAY, edgecolor="none", label="Controls", alpha=0.7),
#            Patch(facecolor="#4A5568", edgecolor="none",
#                  label="Consensus ablations", alpha=1.0),
#            Patch(facecolor=BLUE, edgecolor="none",
#                  label="Core ablations", alpha=1.0),
#        ],
#        loc="upper left", framealpha=0.0, edgecolor="none",
#        fontsize=4, labelspacing=0.0,
#    )
#
    # =====================================================================
    # F) Eigenvalues — ground truth vs. inferred
    # =====================================================================
    TOPK = 4
    z_true = to_complex(results["ground_truth"]["transition_evals"]["values"])
    z_true = z_true[np.argsort(-np.abs(z_true))][:TOPK]

    z_inf = to_complex(results["gcca"]["dynamics"]["evals"])
    z_inf = z_inf[np.argsort(-np.abs(z_inf))][:TOPK]
    _ = float(results["gcca"]["dynamics"].get("r2_mean", np.nan))

    th = np.linspace(0, 2 * np.pi, 400)
    ax_eig.plot(np.cos(th), np.sin(th),
                color="0.7", lw=0.6, alpha=0.9, zorder=1)
    ax_eig.set_anchor("C")
    ax_eig.margins(0)
    ax_eig.text(0.0, -1.03, "Unit circle",
                color="0.7", fontsize=6, ha="center", va="top", alpha=1.0)

    ax_eig.scatter(z_true.real, z_true.imag, s=20, color=BLUE,
                   marker="o", edgecolors="white", linewidths=0.5, zorder=3)
    ax_eig.scatter(z_inf.real, z_inf.imag, s=25, color="red",
                   marker="x", linewidths=0.5, zorder=4)

    ax_eig.set_aspect("equal", "box")
    ax_eig.set_xlim(-1.1, 1.1)
    ax_eig.set_ylim(-1.1, 1.1)
    ax_eig.set_xticks([])
    ax_eig.set_yticks([])
    ax_eig.tick_params(length=0)
    for sp in ax_eig.spines.values():
        sp.set_visible(False)

    L = 0.15
    x0, y0 = 0.5, 0.5
    arrow_kw = dict(arrowstyle='-|>', lw=0.5, color="0.25",
                    mutation_scale=7, shrinkA=0, shrinkB=0)
    ax_eig.annotate("", xy=(x0 + L, y0), xytext=(x0, y0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=arrow_kw, zorder=5)
    ax_eig.annotate("", xy=(x0, y0 + L), xytext=(x0, y0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=arrow_kw, zorder=5)
    ax_eig.text(x0 + 0.4 * L, y0 - 0.02, "Re",
                transform=ax_eig.transAxes,
                ha="center", va="top", fontsize=6, color="0.25", zorder=6)
    ax_eig.text(x0 - 0.02, y0 + 0.4 * L, "Im",
                transform=ax_eig.transAxes,
                ha="right", va="center", fontsize=6, color="0.25", zorder=6)

    truth_proxy = Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor=BLUE, markeredgecolor="white",
                         markeredgewidth=0.5, markersize=4)
    core_proxy = Line2D([0], [0], marker="x", linestyle="None",
                        color="red", markersize=3, linewidth=1.0)
    ax_eig.legend(
        handles=[truth_proxy, core_proxy],
        labels=["Ground truth", "Inferred from core"],
        frameon=False, fontsize=6,
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=1, handletextpad=0.0, columnspacing=0.0, borderaxespad=0.0,
    )

    # =====================================================================
    # Titles + panel letters
    # =====================================================================
    fig.canvas.draw()
    axes = [ax_cos, ax_abl, ax_ov, ax_cca, ax_eig]
    titles = [
        "Model Weight\nSimilarity",
        "Core Necessity\n& Sufficiency",
        "Core Geometric\nSimilarity",
        "Core Statistical\nSimilarity",
        #"Core vs. Consensus\n Alignment",
        "Core Recovers\nMarkov Spectrum",
    ]
    y_title = max(ax.get_position().y1 for ax in axes) + 0.025
    for ax, t in zip(axes, titles):
        pos = ax.get_position()
        x = 0.5 * (pos.x0 + pos.x1)
        fig.text(x, y_title, t, ha="center", va="bottom",
                 fontweight="bold", fontsize=6)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letters = ["A", "B", "C", "D", "E"]
    bboxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in axes
    ]
    y0 = max(bb.y1 for bb in bboxes) + 0.15
    dx0 = 0.013
    tweak = {
        "A": (0.040, 0.0),
        "C": (0.020, 0.0),
        "D": (0.020, 0.0),
       # "E": (-0.010, 0.0),
        "E": (0.018, 0.0),
    }
    for bb, L_ in zip(bboxes, letters):
        dx, dy = tweak.get(L_, (0.0, 0.0))
        x = bb.x0 - dx0 + dx
        y = y0 + dy
        fig.text(x, y, L_, ha="left", va="top",
                 fontweight="bold", fontsize=8)

    # =====================================================================
    # Save
    # =====================================================================
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=500)
    print(f"wrote {outpath.resolve()}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/markov_chain/results",
                   help="Directory containing results.json (default: results)")
    p.add_argument("--out",
                   default=None,
                   help="Output PDF path "
                        "(default: <results-dir>/plots/Markov_fig.pdf)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find {results_path}. "
            "Run Markov_new.py first to generate it."
        )

    with results_path.open() as f:
        results = json.load(f)

    out = Path(args.out) if args.out else results_dir / "plots" / "Markov_fig.pdf"
    make_figure(results, out)


if __name__ == "__main__":
    main()
