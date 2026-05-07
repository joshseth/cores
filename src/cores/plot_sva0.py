# ============================================================
# Combined A/B/C figure for SVA 
# - Panel A: normalized layer depth, one line per condition
# - Panel B: Single Core Axis Controls Agreement (Scatter)
# - Panel C: grouped boxplots with Fisher-combined p-values
# ============================================================
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import wilcoxon, chi2, pearsonr, spearmanr
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerTuple
import matplotlib as mpl

USE_TEX = False  # set True for real cmbright via LaTeX

if USE_TEX:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble": r"\usepackage{cmbright}",
    })
else:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Latin Modern Sans", "Computer Modern Sans Serif", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

mpl.rcParams["axes.linewidth"] = 0.4  

# -------------------------
# Dynamic Styles
# -------------------------
MODEL_MARKER = {
    "gpt2small": "o", "gpt2med": "s", "gpt2large": "x",
    "gemma": "D", "gemma2": "D", "llama": "^", "qwen": "v"
}
MODEL_COLOR = {
    "gpt2small": "#1f77b4", "gpt2med": "#2ca02c", "gpt2large": "#d62728",
    "gemma": "#9467bd", "gemma2": "#9467bd", "llama": "#e377c2", "qwen": "#8c564b"
}
MODEL_LS = {
    "gpt2small": ":", "gpt2med": ":", "gpt2large": ":",
    "gemma": "--", "gemma2": "--", "llama": "-.", "qwen": "-"
}

COND_COLOR = {
    "base": "black",
    "keep": "#7393B3",
    "remove": "#A47DAB",
    "flip": "#d62728",
}
COND_LABEL = {
    "base": "Base LLMs",
    "keep": r"Core Only ($+$)",
    "remove": r"Core Removed ($-$)",
    "flip": r"Core Flipped ($\leftrightarrow$)",
}

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
TITLE_PAD = 16

# -------------------------
# Helpers
# -------------------------
def load_json(p: Path):
    return json.loads(p.read_text())

def pretty_model_name(code: str) -> str:
    code = code.lower()
    mapping = {"gpt2med": "GPT-2 Medium", "gpt2medium": "GPT-2 Medium"}
    if code in mapping: return mapping[code]
    if code.startswith("gpt2"):
        size = code[4:]
        return "GPT-2" + (f" {size.capitalize()}" if size else "")
    if "gemma" in code: return "Gemma-2"
    if "llama" in code: return "LLaMA-3.1"
    if "qwen" in code: return "Qwen2.5"
    return code.replace("-", " ").title()

def base_model_code(run_code: str) -> str:
    return run_code.split("_")[0].lower()

def zscore(x: np.ndarray, eps: float = 1e-12):
    x = np.asarray(x, dtype=float).reshape(-1)
    return (x - np.nanmean(x)) / (np.nanstd(x) + eps)

def fit_affine(x: np.ndarray, y: np.ndarray):
    x, y = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    x_rank, y_rank = x.argsort().argsort(), y.argsort().argsort()
    xr_c, yr_c = x_rank - x_rank.mean(), y_rank - y_rank.mean()
    spearman = float((xr_c @ yr_c) / (np.sqrt((xr_c @ xr_c) * (yr_c @ yr_c)) + 1e-12))
    return float(a), float(b), float(r2), float(spearman)

def log_ratio_correct_over_max(p: dict, correct_set: str, eps: float = 1e-12) -> float:
    p_is, p_are, p_was, p_were = float(p["is"]), float(p["are"]), float(p["was"]), float(p["were"])
    p_max = max(p_is, p_are, p_was, p_were)
    p_corr = (p_is + p_was) if correct_set == "sing" else (p_are + p_were)
    return float(np.log((p_corr + eps) / (p_max + eps)))

def infer_correct_set_from_prompt(prompt: str) -> str:
    plural_markers = [" keys", " children", " men", " women", " mice", " authors", " pilots", " teachers", " labels"]
    return "plur" if any(m in " " + prompt.lower().strip() for m in plural_markers) else "sing"

def panel_label(ax, letter: str):
    ax.text(-0.15, 1.3, letter, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top", ha="left")

def paired_wilcoxon(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 1: return np.nan, np.nan
    return wilcoxon(x[mask], y[mask])

def fisher_combine_p(pvals, eps=1e-300):
    pvals = np.clip(np.asarray(pvals, float), eps, 1.0)
    return float(chi2.sf(-2.0 * np.sum(np.log(pvals)), df=2*pvals.size))

def add_sig_bar(ax, x1, x2, y, text, lw=0.5):
    ax.plot([x1, x1, x2, x2], [y, y+0.0, y+0.0, y], lw=lw, c="k", clip_on=False)
    ax.text((x1+x2)/2, y, text, ha="center", va="bottom", fontsize=4)

# -------------------------
# Load Data
# -------------------------
def load_run(run_dir: Path):
    dir0 = run_dir.name
    cfg = load_json(run_dir / "config.json")
    res = load_json(run_dir / "results.json")
    
    sweep_file = run_dir / "layer_sweep.json"
    sweep = load_json(sweep_file).get("results", []) if sweep_file.exists() else res.get("layer_sweep", [])
    res["layer_sweep"] = sweep

    args = cfg.get("args", {})
    npz = run_dir / "plot_payload" / "knob_fit_test.npz"
    dat = np.load(npz, allow_pickle=True)
    
    return {
        "dir0": dir0,
        "model_code": base_model_code(dir0),
        "pretty": pretty_model_name(base_model_code(dir0)),
        "run_dir": run_dir,
        "res": res,
        "z": np.asarray(dat["z"], dtype=float).reshape(-1),
        "m": np.asarray(dat["m"], dtype=float).reshape(-1),
        "y": np.asarray(dat.get("y", np.zeros_like(dat["z"])), dtype=int),
    }

def print_correlations(runs):
    """Computes cross-model core alignment (Table 6)"""
    print("\n" + "="*60)
    print("Table 6: Subject-Verb Agreement Core Alignment")
    print("="*60)
    print(f"{'Pair':<35} | {'Spearman':<10} | {'Pearson':<10}")
    print("-" * 60)
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            z1, z2 = runs[i]["z"], runs[j]["z"]
            p_val, _ = pearsonr(z1, z2)
            if p_val < 0: z2, p_val = -z2, -p_val
            s_val, _ = spearmanr(z1, z2)
            print(f"{runs[i]['pretty']} x {runs[j]['pretty']:<18} | {s_val:<10.3f} | {p_val:<10.3f}")
    print("="*60 + "\n")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=str, help="Paths to experiment directories")
    args = parser.parse_args()

    runs = [load_run(Path(d)) for d in args.dirs if Path(d).exists()]
    if not runs:
        print("No valid directories found."); exit()
        
    model_list = [R["model_code"] for R in runs]
    
    print_correlations(runs)

    fig = plt.figure(figsize=(7.0, 3.0), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.95, 0.95, 0.45], wspace=0.0)
    axA, axC, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])

    # ============================================================
    # PANEL A: Layer Sweep
    # ============================================================
    depth_grid = np.linspace(0.0, 1.0, 101)
    cond_curves = {k: [] for k in ["base", "keep", "remove", "flip"]}
    
    for R in runs:
        sweep = R["res"].get("layer_sweep", [])
        if not sweep: continue
        
        layers = np.array([r["layer_idx"] for r in sweep], dtype=float)
        depth = layers / (np.nanmax(layers) if np.nanmax(layers) > 0 else 1.0)

        for key, field in [("base", "behavior_auc_base"), ("keep", "behavior_auc_keep"), 
                           ("remove", "behavior_auc_remove"), ("flip", "behavior_auc_flip")]:
            yvals = np.array([r.get(field, np.nan) for r in sweep], dtype=float)
            
            # Safe interpolation
            m = np.isfinite(depth) & np.isfinite(yvals)
            if m.sum() > 1:
                xs, ys = depth[m], yvals[m]
                order = np.argsort(xs)
                cond_curves[key].append(np.interp(depth_grid, xs[order], ys[order], left=np.nan, right=np.nan))

            mk, col = MODEL_MARKER.get(R["model_code"], "o"), COND_COLOR[key]
            is_stroke = mk in ["+", "x", "1", "2", "3", "4", "|", "_"]
            
            if is_stroke:
                axA.scatter(depth, yvals, s=5, marker=mk, alpha=0.3, c=col, linewidths=0.5, zorder=3)
            else:
                axA.scatter(depth, yvals, s=5, marker=mk, alpha=0.3, facecolors="none", edgecolors=col, linewidths=0.5, zorder=3)

    for cond_key in ["base", "keep", "remove", "flip"]:
        Y = np.vstack(cond_curves[cond_key]) if cond_curves[cond_key] else np.empty((0, len(depth_grid)))
        if len(Y):
            ymean = np.nanmean(Y, axis=0)
            axA.plot(depth_grid, ymean, lw=0.75, color=COND_COLOR[cond_key])
            axA.fill_between(depth_grid, np.nanmin(Y, axis=0), np.nanmax(Y, axis=0), alpha=0.075, linewidth=0, color=COND_COLOR[cond_key])

    axA.set(ylim=(0.0, 1.1), xlim=(-0.025, 1.1), xlabel="Normalized layer depth", ylabel="Agreement performance (AUC)", yticks=[0.0, 0.5, 1.0], xticks=[0.0, 0.5, 1.0])
    axA.set_title("Layer Sweep for Number-Agreement Core\nin Transformer Models", fontweight="bold", fontsize=6, pad=TITLE_PAD)
    axA.text(0.5, 1.02, r"Number agreement $\it{e.g.}\!$, $is$ vs. $are$", transform=axA.transAxes, ha="center", va="bottom", fontsize=6, color="0.25")
    for spine in ["top", "right"]: axA.spines[spine].set_visible(False)
    axA.tick_params(axis="both", labelsize=5, pad=1, length=1, width=0.5)

    handles = [Line2D([], [], color=COND_COLOR[k], lw=1.0, label=COND_LABEL[k]) for k in ["base", "keep", "remove", "flip"]]
    handles += [Line2D([], [], linestyle="None", marker=MODEL_MARKER.get(mc, "o"), markersize=2.5, markerfacecolor="none", color="0.2", label=R["pretty"], markeredgewidth=0.5, alpha=0.75) for mc, R in zip(model_list, runs)]
    axA.legend(handles=handles, frameon=False, fontsize=5, loc="best", handlelength=0.5)
    panel_label(axA, "A")

    # ============================================================
    # PANEL B (axC in grid): Single Core Axis Controls Agreement
    # ============================================================
    zZ_all = [zscore(R["z"]) for R in runs]
    xmin, xmax = min(np.min(z) for z in zZ_all), max(np.max(z) for z in zZ_all)
    r2_by_model = {}
    #rho_by_model = {}

    for R in runs:
        zZ, mZ, y, mc = zscore(R["z"]), zscore(R["m"]), R["y"], R["model_code"]
        mk, line_c, ls = MODEL_MARKER.get(mc, "o"), MODEL_COLOR.get(mc, "0.5"), MODEL_LS.get(mc, "-")
        is_stroke = mk in ["+", "x", "1", "2", "3", "4", "|", "_"]
        
        mask_sing, mask_plur = (y == 0), (y == 1)

        # Align logic: plural above singular, positive slope
        if (np.nanmean(mZ[mask_plur]) - np.nanmean(mZ[mask_sing])) < 0: mZ = -mZ
        if np.nanmean(zZ * mZ) < 0: zZ = -zZ

        a, b, r2, _ = fit_affine(zZ, mZ)
        r2_by_model[mc] = r2
        #a, b, _, rho = fit_affine(zZ, mZ)
        #rho_by_model[mc] = rho

        axC.scatter(zZ[mask_sing], mZ[mask_sing], s=3, alpha=0.25, color=BLUE, marker=mk, linewidths=0.5 if is_stroke else 0)
        axC.scatter(zZ[mask_plur], mZ[mask_plur], s=3, alpha=0.25, color=ORANGE, marker=mk, linewidths=0.5 if is_stroke else 0)
        axC.plot(np.linspace(xmin, xmax, 200), a * np.linspace(xmin, xmax, 200) + b, color=line_c, lw=1.0, alpha=0.9, ls=ls)

    axC.axhline(0, ls="--", lw=0.2, color="0.7", alpha=0.25)
    axC.axvline(0, ls="--", lw=0.2, color="0.7", alpha=0.25)
    axC.set(xlabel=r"Core coordinate ($z$ score)", ylabel=r"Plural $-$ singular logits ($z$ score)")
    axC.set_title("Single Core Axis Controls\nAgreement in LLMs", fontweight="bold", pad=TITLE_PAD, fontsize=6)
    for spine in ["top", "right"]: axC.spines[spine].set_visible(False)
    axC.tick_params(axis="both", labelsize=5, pad=1, length=1, width=0.5)
    axC.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axC.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axC.set_box_aspect(1)

    # Legends for Panel B
    axC.add_artist(axC.legend(handles=[
        Line2D([], [], linestyle="None", marker="o", markersize=3, markerfacecolor=BLUE, markeredgecolor="none", label="Singular prompt"),
        Line2D([], [], linestyle="None", marker="o", markersize=3, markerfacecolor=ORANGE, markeredgecolor="none", label="Plural prompt")
    ], loc="lower right", frameon=False, fontsize=5, handletextpad=0.6, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.0)))

    combo_handles, combo_labels = [], []
    for mc, R in zip(model_list, runs):
        combo_handles.append((
            Line2D([], [], linestyle="None", marker=MODEL_MARKER.get(mc, "o"), markersize=2.5, markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.5, alpha=0.75),
            Line2D([], [], linestyle=MODEL_LS.get(mc, "-"), lw=1.0, color=MODEL_COLOR.get(mc, "0.5"))
        ))
        combo_labels.append(f"{R['pretty']} ($R^2$={r2_by_model[mc]:.2f})")
        #combo_labels.append(f"{R['pretty']} ($\\rho$={rho_by_model[mc]:.2f})")
        
    axC.legend(combo_handles, combo_labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper left", frameon=False, fontsize=5, handletextpad=0.6, borderaxespad=0.2, bbox_to_anchor=(0.0, 1.0))
    panel_label(axC, "B")

    # ============================================================
    # PANEL C (axB in grid): Perturbation Boxplots
    # ============================================================
    modes, mode_labels = ["clean", "remove", "flip"], ["Base\nLLM", "Core\nRemoved", "Core\nFlipped"]
    scores_by_model = {}

    for R in runs:
        flip_rows = R["res"].get("flip_demo", [])
        if not flip_rows: continue
        by_prompt = {}
        for r in flip_rows: by_prompt.setdefault(r["prompt"], {})[r["mode"]] = {k: float(r[k]) for k in ["is", "are", "was", "were"]}
        
        scores = {m: [] for m in modes}
        for pr in sorted(by_prompt.keys()):
            corr = infer_correct_set_from_prompt(pr)
            for m in modes: scores[m].append(log_ratio_correct_over_max(by_prompt[pr][m], corr) if m in by_prompt[pr] else np.nan)
        scores_by_model[R["model_code"]] = {m: np.array(scores[m], float) for m in modes}

    if scores_by_model:
        base_pos, offsets = np.array([1, 2, 3], dtype=float), np.linspace(-0.25, 0.25, len(runs))
        data_list, pos_list, color_list = [], [], []

        for j, mode in enumerate(modes):
            for i, mc in enumerate(model_list):
                if mc not in scores_by_model: continue
                ys = scores_by_model[mc][mode]
                data_list.append(ys[np.isfinite(ys)])
                pos_list.append(base_pos[j] + offsets[i])
                color_list.append(MODEL_COLOR.get(mc, "0.5"))

        bp = axB.boxplot(data_list, positions=pos_list, widths=0.4/len(runs), showfliers=False, patch_artist=True, medianprops=dict(color="black", linewidth=1.0), boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5))
        for patch, c in zip(bp["boxes"], color_list): patch.set(facecolor=c, edgecolor="black", linewidth=0.0, alpha=0.7)

        rng = np.random.default_rng(0)
        for pos, ys in zip(pos_list, data_list):
            axB.scatter(rng.normal(loc=pos, scale=0.03, size=len(ys)), ys, s=3, alpha=0.35, color="k", linewidths=0, zorder=5)

        axB.set(xticks=base_pos, xticklabels=mode_labels, ylabel=r"Agreement score  $\log \frac{\mathbb{P}(\text{correct})}{\mathbb{P}(\text{max})}$")
        axB.set_title("Perturbing Core Disrupts\nAgreement", fontsize=6, fontweight="bold", pad=TITLE_PAD)
        for spine in ["top", "right"]: axB.spines[spine].set_visible(False)
        axB.tick_params(axis="both", labelsize=5, pad=1, length=1, width=0.5)
        axB.yaxis.set_major_locator(MaxNLocator(nbins=4))

        # Significance brackets (Fisher-combined p-values)
        comps = [("clean", "remove", 0, 1, "Base vs Removed"), ("clean", "flip", 0, 2, "Base vs Flipped"), ("remove", "flip", 1, 2, "Removed vs Flipped")]
        all_y = np.concatenate([ys for ys in data_list if len(ys)])
        y_max, y_min = np.nanmax(all_y), np.nanmin(all_y)
        yr = (y_max - y_min) if (y_max - y_min) > 0 else 1.0

        for k, (ma, mb, ia, ib, _) in enumerate(comps):
            pvals = [paired_wilcoxon(scores_by_model[mc][ma], scores_by_model[mc][mb])[1] for mc in model_list if mc in scores_by_model]
            if pvals:
                p_f = fisher_combine_p(pvals)
                txt = f"p={p_f:.1e}" if p_f < 0.001 else f"p={p_f:.3f}"
                add_sig_bar(axB, base_pos[ia] + offsets.mean(), base_pos[ib] + offsets.mean(), y_max + 0.02*yr + k*(0.06*yr), txt, lw=0.33)

        axB.legend(handles=[Line2D([], [], linestyle="None", marker="s", markersize=3, markerfacecolor=MODEL_COLOR.get(mc, "0.5"), markeredgecolor="none", label=runs[i]["pretty"], alpha=0.7) for i, mc in enumerate(model_list)], frameon=False, fontsize=5, loc="lower left", handletextpad=-0.3, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.05))
        panel_label(axB, "C")


    for ax in [axA, axB, axC]:
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(6)
    # -------------------------
    # Save & Show
    # -------------------------
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "experiments" / "sva" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "SVA_figure.pdf"

    fig.savefig(out_file, dpi=500, bbox_inches="tight")
    print(f"Wrote publication figure to: {out_file}")
    plt.close()