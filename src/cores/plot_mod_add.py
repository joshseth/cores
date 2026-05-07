"""
Mod-add grokking: plotting script.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from cores.modular_addition_analysis import analyze_epoch, load_core_result, CoreFormationResult
from cores.models import build_modadd_dataloader
from cores.utils import get_device


# ═══════════════════════════════════════════════════════════════════════
# STYLE + COLORS
# ═══════════════════════════════════════════════════════════════════════

C_ACC  = "#D62728"; C_CORE = "#4D4D4D"; C_KEEP = "#1F77B4"
C_REM  = "#FF7F0E"; C_WD0  = "#9467bd"; C_NEC  = "#ff7f0e"
C_SUF  = "#1f77b4"; C_NEAR = "#1f77b4"; C_FAR  = "#A9A9A9"
LEG_KW = dict(handlelength=0.8, borderpad=0.2, labelspacing=0.25)

def apply_style():
    mpl.rcParams.update({
        "font.size": 6, "axes.titlesize": 6, "axes.titlepad": 2.0,
        "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 5, "axes.linewidth": 0.5, "lines.linewidth": 1.0,
        "legend.frameon": False, "font.family": "sans-serif",
        "font.sans-serif": ["Latin Modern Sans", "Computer Modern Sans Serif", "DejaVu Sans"],
        "mathtext.fontset": "cm", "axes.unicode_minus": False,
    })


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def mean_se(x, axis=0):
    x = np.asarray(x, float)
    mu = np.nanmean(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1) if np.any(n > 1) else np.zeros_like(mu)
    return mu, sd / np.maximum(1, np.sqrt(n))

def plot_with_traces(ax, x, data, color, label=None, ls="-",
                     band_alpha=0.12, max_epoch=None):
    x = np.asarray(x); arr = np.asarray(data)
    if max_epoch is not None:
        mask = x <= max_epoch; x = x[mask]
        arr = arr[:, mask] if arr.ndim == 2 else arr[mask]
    mu, se = mean_se(arr, axis=0)
    line, = ax.plot(x, mu, color=color, ls=ls, lw=1.0, label=label)
    ax.fill_between(x, mu - se, mu + se, color=color, alpha=band_alpha, edgecolor="none")
    return line

def draw_markers(ax, grok=None, split=None, y_text=0.45, ym = 1.0):
    eff = [pe.withStroke(linewidth=2, foreground="white")]
    if grok is not None:
        ax.axvline(int(grok), ls=":", color="black", alpha=0.4, lw=0.7, zorder=1, ymax=ym)
        ax.text(int(grok), y_text, "Grokked", transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="right", fontsize=5, path_effects=eff)
    if split is not None:
        ax.axvline(int(split), ls="--", color="0.4", alpha=0.5, lw=0.7, zorder=1)
        ax.text(int(split), 0.02, "Split", transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="right", fontsize=5, color="0.4", path_effects=eff)

def retitle(ax, t, pad=4.0):
    ax.set_title(t, loc="left", fontsize=6, pad=pad, fontweight="bold")

def panel_label(fig, ax, s, *, dx=0.018, dy=0.006):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, s, ha="right", va="bottom", fontsize=7, fontweight="bold")

def tighten(ax):
    ax.tick_params(axis="both", which="both", pad=1.0, length=2.0, width=0.5)

def sci_x(ax, nloc=5):
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nloc))

def align_epochs(a, b):
    ea, eb = np.asarray(a, int), np.asarray(b, int)
    c = np.intersect1d(ea, eb)
    return c, np.searchsorted(ea, c), np.searchsorted(eb, c)

def _fmt_epoch_sci(ep):
    ep = int(ep)
    if ep == 0: return r"$0$"
    k = int(np.floor(np.log10(ep)))
    m = ep / 10**k
    if np.isclose(m, round(m)) and 1 <= m <= 9:
        m = int(round(m))
        return rf"$10^{{{k}}}$" if m == 1 else rf"${m}\times 10^{{{k}}}$"
    return f"{ep:,}"


# ═══════════════════════════════════════════════════════════════════════
# CLOCK SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def draw_clock(ax, *, run_dir, ep, test_loader, ring_eps, device,
               prefix_dir=None, m_ix=0, vocab_size=53, d_model=128,
               show_axes=False, show_r2_holdout=False,
               holdout_style="none", ridge=0.0, yd=-0.33, **kw):
    dirs = (Path(run_dir), prefix_dir) if prefix_dir else (Path(run_dir),)
    out = analyze_epoch(
        *dirs, ep=ep, m_ix=m_ix, test_loader=test_loader, device=device,
        vocab_size=vocab_size, d_model=d_model,
        ring_eps=ring_eps, holdout_style=holdout_style, ridge=ridge, **kw)

    ev = out["evals"]
    near = np.abs(np.abs(ev) - 1.0) <= ring_eps

    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)

    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, color="gray", alpha=0.5, lw=0.5))
    ax.scatter(ev.real[~near], ev.imag[~near], s=7, color=C_FAR, lw=0.0)
    ax.scatter(ev.real[near],  ev.imag[near],  s=7, color=C_NEAR, lw=0.0)

    if show_axes:
        L = 0.35
        akw = dict(arrowstyle="-|>", lw=0.35, color="0.25", mutation_scale=6, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(L, 0), xytext=(0, 0), arrowprops=akw, zorder=3)
        ax.annotate("", xy=(0, L), xytext=(0, 0), arrowprops=akw, zorder=3)
        ax.text(L + 0.02, 0, "Re", ha="left", va="center", fontsize=6, color="0.25")
        ax.text(0, L + 0.02, "Im", ha="center", va="bottom", fontsize=6, color="0.25")

    r2_show = out["r2_test"] if show_r2_holdout and not np.isnan(out["r2_test"]) else out["r2"]
    if show_r2_holdout:
        ax.text(0.5, yd,
                f"Acc: {out['acc']:.2f}\n" rf"$R_h^2$: {r2_show:.2f}" "\n"
                rf"Core$_{{\mathrm{{dim}}}}$: {out['r_nec']}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5,
                bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=1), clip_on=False)
    else:
        ax.text(0.5, yd,
                f"Acc: {out['acc']:.2f}\n" rf"$R^2$: {r2_show:.2f}" "\n"
                rf"Core$_{{\mathrm{{dim}}}}$: {out['r_nec']}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5,
                bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=1), clip_on=False)
    return out


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1
# ═══════════════════════════════════════════════════════════════════════

def figure1(result: CoreFormationResult, *, run_dir, test_loader, device, ring_eps,
            prefix_dir=None, m_ix=0, vocab_size=53, d_model=128,
            clock_epochs=(0, 300, 800, 900, 2000),
            max_epoch=2000, save_path="mod_add_fig1.pdf"):
    apply_style()
    x, grok = result.epochs, result.grok_ep

    fig = plt.figure(figsize=(5.5, 3.1))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.3], hspace=0.15, wspace=0.35)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    # A: accuracy + core dim
    l_acc = plot_with_traces(axA, x, result.test_acc, C_ACC, label="Model acc", ls="--", max_epoch=max_epoch)
    axA_r = axA.twinx()
    l_r = plot_with_traces(axA_r, x, result.r_nec, C_CORE, label="Core dim", max_epoch=max_epoch)
    retitle(axA, "Mod Add Core Forms at Grokking")
    axA.set_ylabel("Test accuracy"); axA.set_xlabel("Epoch"); axA.set_ylim(-0.02, 1.02)
    draw_markers(axA, grok=grok)
    axA_r.set_ylim(-0.5, 53); axA_r.set_ylabel("Core dimension", rotation=270, labelpad=0, fontsize=6)
    axA_r.yaxis.label.set_verticalalignment("bottom")
    axA.legend(handles=[l_acc, l_r], loc="lower right", **LEG_KW)

    # B: necessity & sufficiency
    plot_with_traces(axB, x, result.keep_acc, C_KEEP, label="Core only", max_epoch=max_epoch)
    plot_with_traces(axB, x, result.remove_acc, C_REM, label="Core removed", max_epoch=max_epoch)
    axB.axhline(1/vocab_size, ls=":", lw=0.7, color="black", alpha=0.35, label=r"Chance $|V|^{-1}$")
    retitle(axB, "Core is Necessary and Sufficient")
    axB.set_ylabel("Test accuracy"); axB.set_xlabel("Epoch"); axB.set_ylim(-0.02, 1.02)
    draw_markers(axB, grok=grok); axB.legend(loc="lower right", **LEG_KW)

    # C: clock strip
    clock_spec = gs[1, :]
    gs_clk = GridSpecFromSubplotSpec(2, len(clock_epochs), subplot_spec=clock_spec, wspace=0.05, hspace=0.15)
    ax_info = fig.add_subplot(gs_clk[0, :]); ax_info.axis("off")
    ax_info.text(0.0, 0.75, "Cyclic Mechanism Emerges in Core at Grokking",
                 ha="left", va="top", fontsize=6, fontweight="bold", transform=ax_info.transAxes)
    ax_info.scatter([0.005], [0.32], s=5, color=C_NEAR, transform=ax_info.transAxes, clip_on=False)
    ax_info.text(0.015, 0.32, r"Eigenvalue near unit circle $||\lambda|-1| < \epsilon$",
                 ha="left", va="center", fontsize=5.5, transform=ax_info.transAxes)
    ax_info.scatter([0.005], [0.17], s=5, color=C_FAR, transform=ax_info.transAxes, clip_on=False)
    ax_info.text(0.015, 0.17, "Off-circle", ha="left", va="center", fontsize=5.5, transform=ax_info.transAxes)

    clock_axes = []
    for j, ep in enumerate(clock_epochs):
        ax = fig.add_subplot(gs_clk[1, j])
        draw_clock(ax, run_dir=run_dir, ep=int(ep), test_loader=test_loader,
                   ring_eps=ring_eps, device=device, prefix_dir=prefix_dir,
                   m_ix=m_ix, vocab_size=vocab_size, d_model=d_model, yd=-0.5,
                   show_axes=(j == len(clock_epochs) // 2),
                   show_r2_holdout=True, holdout_style="block_edges", ridge=1.0)
        ax.set_title(f"Epoch {int(ep):,}", fontsize=6, pad=2)
        clock_axes.append(ax)

    panel_label(fig, axA, "A", dy=0.035); panel_label(fig, axB, "B", dy=0.035)
    panel_label(fig, clock_axes[0], "C", dy=0.15, dx=0.04)
    for a in [axA, axB, axA_r]: tighten(a)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=500)
    print(f"Saved: {save_path}"); plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2
# ═══════════════════════════════════════════════════════════════════════

def figure2(wd1: CoreFormationResult, branch: CoreFormationResult,
            ms_wd1: dict, ms_branch: dict,
            *, run_dir_wd1, run_dir_wd0, prefix_dir=None,
            test_loader, device, ring_eps,
            m_ix=0, vocab_size=53, d_model=128,
            clock_epoch=20000, split_epoch=None,
            save_path="mod_add_fig2.pdf"):
    apply_style()
    grok = wd1.grok_ep

    fig = plt.figure(figsize=(5.5, 3.3))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.60, wspace=0.25)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    clock_spec = gs[1, 0]; axD = fig.add_subplot(gs[1, 1])

    # A: core dim comparison
    common, ia, ib = align_epochs(wd1.epochs, branch.epochs)
    m1, s1 = mean_se(wd1.r_nec); mb, sb = mean_se(branch.r_nec)
    axA.plot(common, m1[ia], color="black", lw=1.0, label=r"WD =1 (weight decay on)")
    axA.fill_between(common, m1[ia]-s1[ia], m1[ia]+s1[ia], color="black", alpha=0.10, edgecolor="none")
    axA.plot(common, mb[ib], color=C_WD0, lw=1.0, label="WD 1→0 (weight decay off, post-grokking)", zorder=0)
    axA.fill_between(common, mb[ib]-sb[ib], mb[ib]+sb[ib], color=C_WD0, alpha=0.10, edgecolor="none")
    retitle(axA, "Mod Add Core Inflates After Grokking")
    axA.set_ylabel("Core size (dimension)"); axA.set_xlabel("Epoch"); axA.set_ylim(-0.5, 100)
    draw_markers(axA, grok=grok, split=split_epoch, y_text = 0.3, ym = 0.65); axA.legend(loc="upper left", **LEG_KW); sci_x(axA)

    # B: redundancy gap (WD=1)
    xx = wd1.epochs
    mask = xx >= grok
    suf_mu, _ = mean_se(wd1.r_suf_strict)   
    nec_mu, _ = mean_se(wd1.r_nec_strict)   
    axB.plot(xx, suf_mu, lw=1.0, color=C_SUF, label="Sufficient dim (keep → baseline accuracy)")
    axB.plot(xx, nec_mu, lw=1.0, color=C_NEC, label="Necessary dim (remove → chance accuracy)")
    axB.fill_between(xx[mask], suf_mu[mask], nec_mu[mask], color=C_SUF, alpha=0.08, label="Redundancy")
    retitle(axB, "Core Inflation is Driven by Redundancy")
    axB.set_ylabel("Core size (dimension)"); axB.set_xlabel("Epoch"); axB.set_ylim(-0.5, 100)
    draw_markers(axB, grok=grok, split=split_epoch, y_text = 0.3, ym = 0.65); axB.legend(loc="upper left", **LEG_KW); sci_x(axB)

    # C: two clock snapshots
    gs_clk = GridSpecFromSubplotSpec(1, 2, subplot_spec=clock_spec, wspace=0.3)
    axC1 = fig.add_subplot(gs_clk[0, 0]); axC2 = fig.add_subplot(gs_clk[0, 1])
    draw_clock(axC1, run_dir=run_dir_wd1, ep=clock_epoch, test_loader=test_loader,
               ring_eps=ring_eps, device=device, prefix_dir=prefix_dir,
               m_ix=m_ix, show_axes=True, holdout_style="none", ridge=0.0)
    draw_clock(axC2, run_dir=run_dir_wd0, ep=clock_epoch, test_loader=test_loader,
               ring_eps=ring_eps, device=device, prefix_dir=prefix_dir,
               m_ix=m_ix, show_axes=False, holdout_style="none", ridge=0.0)
    axC1.set_title(r"WD =1", fontsize=6, pad=1, fontweight="bold")
    axC2.set_title(r"WD 1→0", fontsize=6, pad=1, color=C_WD0, fontweight="bold")

    # Row title
    boxes = [a.get_position() for a in [axC1, axC2, axD]]
    fig.text(min(b.x0 for b in boxes), max(b.y1 for b in boxes) + 0.05,
             "Cyclic Operator Saturates with Core Inflation",
             ha="left", va="bottom", fontsize=6, fontweight="bold")
    # Epoch label between clocks
    bL, bR = axC1.get_position(), axC2.get_position()
    fig.text(0.5*(bL.x1+bR.x0), bL.y0 + 0.05*(bL.y1-bL.y0),
             f"Epoch:\n {_fmt_epoch_sci(clock_epoch)}", ha="center", va="center", fontsize=5)

    # D: mode count overlay
    eA, eB = ms_wd1["epochs"], ms_branch["epochs"]
    common_m, ia_m, ib_m = align_epochs(eA, eB)
    countsA = np.stack([p.sum(axis=0) for p in ms_wd1["per_model"]["present"]]).astype(float)
    countsB = np.stack([p.sum(axis=0) for p in ms_branch["per_model"]["present"]]).astype(float)
    muA, seA = mean_se(countsA); muB, seB = mean_se(countsB)
    axD.plot(common_m, muA[ia_m], color="black", lw=1.0, label="WD =1")
    axD.fill_between(common_m, muA[ia_m]-seA[ia_m], muA[ia_m]+seA[ia_m], color="black", alpha=0.12, edgecolor="none")
    axD.plot(common_m, muB[ib_m], color=C_WD0, lw=1.0, label="WD 1→0", zorder=0)
    axD.fill_between(common_m, muB[ib_m]-seB[ib_m], muB[ib_m]+seB[ib_m], color=C_WD0, alpha=0.12, edgecolor="none")
    K = int(ms_wd1["K_norm"])
    axD.set_ylabel("Rotational modes"); axD.set_xlabel("Epoch")
    axD.set_ylim(-0.5, K+0.5); axD.set_yticks(np.arange(0, K+1, 9))
    draw_markers(axD, grok=grok, split=split_epoch, y_text=0.3, ym = 0.65)
    axD.legend(loc="best", **LEG_KW); sci_x(axD)

    # Legend + labels
    x0, y0 = 0.12, 0.0
    fig.text(x0, y0, "●", color=C_NEAR, fontsize=5, va="center", ha="left")
    fig.text(x0+0.018, y0, r"Near unit circle $||\lambda|-1|<\epsilon$", fontsize=5, va="center", ha="left")
    fig.text(x0, y0-0.022, "●", color=C_FAR, fontsize=5, va="center", ha="left")
    fig.text(x0+0.018, y0-0.022, "Off-circle", fontsize=5, va="center", ha="left")

    panel_label(fig, axA, "A", dy=0.035); panel_label(fig, axB, "B", dy=0.035)
    panel_label(fig, axC1, "C", dy=0.06)
    for a in [axA, axB, axD]: tighten(a)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=500)
    print(f"Saved: {save_path}"); plt.show()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--figure", choices=["1", "2", "both"], default="both")
    p.add_argument("--run-dir-wd1", type=Path, required=True)
    p.add_argument("--run-dir-wd0", type=Path, default=None)
    p.add_argument("--prefix-dir", type=Path, default=None)
    p.add_argument("--core-npz-wd1", type=Path, default=None)
    p.add_argument("--core-npz-wd0", type=Path, default=None)
    p.add_argument("--mode-npz-wd1", type=Path, default=None)
    p.add_argument("--mode-npz-wd0", type=Path, default=None)
    p.add_argument("--vocab-size", type=int, default=53)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--clock-epoch-fig2", type=int, default=20000)
    p.add_argument("--split-epoch", type=int, default=None)
    cli = p.parse_args()

    device = get_device()
    test_loader = build_modadd_dataloader(train=False, batch_size=512)
    ring_eps = np.sin(np.pi / cli.vocab_size)

    if cli.figure in ("1", "both"):
        wd1 = load_core_result(cli.core_npz_wd1)
        figure1(wd1, run_dir=cli.run_dir_wd1, test_loader=test_loader,
                device=device, ring_eps=ring_eps, prefix_dir=cli.prefix_dir,
                vocab_size=cli.vocab_size, d_model=cli.d_model)

    if cli.figure in ("2", "both"):
        wd1 = load_core_result(cli.core_npz_wd1)
        br = load_core_result(cli.core_npz_wd0)
        ms_wd1 = dict(np.load(cli.mode_npz_wd1, allow_pickle=False))
        ms_br = dict(np.load(cli.mode_npz_wd0, allow_pickle=False))
        for ms in [ms_wd1, ms_br]:
            pr = ms.pop("present")
            ms["per_model"] = {"present": [pr[s] for s in range(pr.shape[0])],
                               "frac_near": ms.pop("frac_near"), "r2": ms.pop("r2")}
            ms["K_norm"] = int(ms["K_norm"])
            ms["grok_ep_plot"] = int(ms.get("grok_ep", wd1.grok_ep))
        figure2(wd1, br, ms_wd1, ms_br,
                run_dir_wd1=cli.run_dir_wd1, run_dir_wd0=cli.run_dir_wd0,
                prefix_dir=cli.prefix_dir,
                test_loader=test_loader, device=device, ring_eps=ring_eps,
                vocab_size=cli.vocab_size, d_model=cli.d_model,
                clock_epoch=cli.clock_epoch_fig2, split_epoch=cli.split_epoch)
