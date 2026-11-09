#!/usr/bin/env python3
"""
tables.py: the IOI appendix tables from the saved results: cores, held-out prompts, controls, reduced model.

  PYTHONPATH=src python -m cores.ioi.tables --results results/ioi --out tables
"""
import argparse
import json
from pathlib import Path

RULE = "flip at both sites"
MODELS = [("gpt2", "GPT-2 Small"), ("gpt2-medium", "GPT-2 Medium"), ("gpt2-large", "GPT-2 Large"), ("llama-3.1-8b", "LLaMA-3.1-8B"), ("gemma-2-9b", "Gemma-2-9B"), ("qwen2.5-32b", "Qwen2.5-32B")]

REDUCED_HEAD = r"""\begin{table}[h]
\centering
\footnotesize
\caption{Reduced model on held-out frames and 40 new names (test; validation in parentheses; AUC and shuffled pairs on test, with AUC 0.998--1.000 and shuffled pairs within 0.02 of zero on validation). No fit uses an intervention. Same answer: the sign of the predicted margin matches the network's after the same change to the S2 core. Shuffled pairs and shuffled alignment match differences from different pairs, in random order; on true pairs the fit does not depend on the order within a pair. Beyond one fixed direction: the share of the error left by the best single fixed direction that the map removes. One direction at S2: the mean within-pair difference. Size between sentences: correlation of the sentence-mean size of the swap with its prediction.}
\label{tab:ioi_reduced}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
 & GPT-2 S & GPT-2 M & GPT-2 L & LLaMA-3.1 & Gemma-2 & Qwen2.5 \\
\midrule
"""


def chosen(r):
    """The layer pair with the smallest core; a tie goes to the lower removal on train."""
    total = {q: sum(v["core"][RULE]["ranks"]) for q, v in r["pairs"].items() if v["core"][RULE]}
    return min(total, key=lambda k: (total[k], r["pairs"][k]["core"][RULE]["train"]["remove"]))


def table(label, caption, spec, head, rows):
    return ("\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{" + spec + "}\n\\toprule\n"
            + head + " \\\\\n\\midrule\n" + "\n".join(r + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/ioi"))
    p.add_argument("--out", type=Path, default=Path("tables"))
    args = p.parse_args()
    GOLD, MODEL, OUT = args.results / "golden", args.results / "model", args.out
    OUT.mkdir(parents=True, exist_ok=True)
    cores, held, ctrl = [], [], []
    for m, label in MODELS:
        r = json.loads((GOLD / m / f"{m}.json").read_text()); w = json.loads((GOLD / m / f"{m}_wang.json").read_text())
        b = json.loads((MODEL / m / f"{m}_controls.json").read_text())
        q = chosen(r); P = r["pairs"][q]; c = P["core"][RULE]; a, e = c["ranks"]; lS, lE = P["layers"]; nl, D = r["n_layers"], r["D"]
        assert b["layers"] == [lS, lE] and b["ranks"] == [a, e], m
        t, h, g = c["train"], c["held-out"], w["pairs"][q]["cores"][f"{a}+{e}"]
        s = P["sufficient"]["flip"]
        cores.append(f"{label} & {D} & {lS}/{lE} ({lS / nl:.2f}/{lE / nl:.2f}) & {s['S2']} + {s['END']} & {a} + {e} ({100 * (a + e) / D:.1f}\\%) & {t['keep']:.3f} & {t['remove']:.3f} & {t['flip_S2']:.3f} & {t['flip_END']:.3f}")
        held.append(f"{label} & " + " & ".join(f"{v[k]:.3f}" for v in (h, g) for k in ("keep", "remove", "flip_S2", "flip_END")))
        C = b["controls"]
        ctrl.append(f"{label} & {a} + {e} & " + " & ".join(f"{C[x][k]:.2f}" for k in ("keep", "remove", "flip_S2", "flip_END") for x in ("ACE", "PCA", "random")))
    (OUT / "t_golden_cores.tex").write_text(table(
        "tab:ioi_cores",
        "IOI cores in six models, on the extraction set, where their ranks were set. Layers: S2 and END sites (relative depth). Sufficient: at each site, the smallest rank that alone preserves and flips the behavior. "
        "Smallest core: the smallest $r_{S2} + r_{END}$ that also passes removal. AUC at the smallest core: core-only and core-removed at both sites, and flip at each site alone.",
        "lrccccccc", "Model & $D$ & Layers S2/END & Sufficient & Smallest core (\\% of $D$) & Core-only & Core-removed & Flip S2 & Flip END", cores))
    (OUT / "t_golden_heldout.tex").write_text(table(
        "tab:ioi_heldout",
        "The cores of Table~\\ref{tab:ioi_cores} on prompts not used to extract them or to set their ranks (AUC). Test: 27 held-out frames and 40 new names (1,026 prompts; clean 0.999--1.000). "
        "Wang et al.: their 15 templates, 99 names, places, and objects (1,020 prompts; clean 1.000).",
        "l|cccc|cccc", " & \\multicolumn{4}{c|}{Test} & \\multicolumn{4}{c}{Wang et al.} \\\\ Model & Core-only & Core-removed & Flip S2 & Flip END & Core-only & Core-removed & Flip S2 & Flip END", held))
    (OUT / "t_golden_controls.tex").write_text(table(
        "tab:ioi_controls",
        "Controls at the ranks of the smallest core, on the extraction set (AUC). PCA: principal components of the activations at each site, unpaired, over the same prompts as the core, mean subtracted. Random: a random subspace of the same size. "
        "Core-only and core-removed at both sites, and flip at each site alone.",
        "lc|ccc|ccc|ccc|ccc", " & & \\multicolumn{3}{c|}{Core-only} & \\multicolumn{3}{c|}{Core-removed} & \\multicolumn{3}{c|}{Flip S2} & \\multicolumn{3}{c}{Flip END} \\\\ Model & Ranks & ACE & PCA & Rand. & ACE & PCA & Rand. & ACE & PCA & Rand. & ACE & PCA & Rand.", ctrl))

    J = json.loads((args.results / "reduced_model.json").read_text()); ms = [m for m, _ in MODELS]; X = J["across"]
    C = lambda x: f"${x:.2f}$" if round(x, 2) < 0 else f"{abs(x):.2f}"
    pair = lambda k: " & ".join(f"{C(J['models'][m]['test'][k])} ({C(J['models'][m]['val'][k])})" for m in ms)
    three = lambda k: " & ".join(f"{J['models'][m]['test'][k]:.3f}" for m in ms)
    test = lambda k: " & ".join(C(J["models"][m]["test"][k]) for m in ms)
    into = lambda d: " & ".join(C(sum(v for k, v in d.items() if k.endswith(f">{m}")) / sum(k.endswith(f">{m}") for k in d)) for m in ms)
    lines = [
        r"\multicolumn{7}{l}{\emph{$\mathbf{A}$: END core from the S2 core (share explained)}} \\",
        rf"Role difference within a pair & {pair('role')} \\",
        rf"Change after swapping the S2 core & {pair('change')} \\",
        rf"\quad control: $\mathbf{{A}}$ from shuffled pairs & {test('change_shuffled')} \\",
        rf"\quad beyond one fixed direction & {pair('beyond')} \\",
        rf"\quad an operator fit on these swaps & {pair('change_swapfit')} \\", r"\midrule",
        r"\multicolumn{7}{l}{\emph{$\mathbf{c}$: margin from the END core}} \\",
        rf"AUC & {three('g_auc')} \\",
        rf"Size between sentences & {pair('g_between')} \\", r"\midrule",
        r"\multicolumn{7}{l}{\emph{Both: answer from the S2 core alone}} \\",
        rf"AUC & {three('auc')} \\",
        rf"\quad control: one direction at S2 & {three('line_auc')} \\",
        rf"Same answer as the network & {pair('name')} \\",
        rf"Same, S2 core swapped & {pair('same_swap')} \\",
        rf"Same, S2 core flipped & {pair('same_flip')} \\",
        rf"\quad control: ignoring the swap & {pair('ignore_swap')} \\",
        rf"Size between sentences & {pair('between')} \\", r"\midrule",
        rf"\multicolumn{{7}}{{l}}{{\emph{{Across models: change after swapping the S2 core, {X['n_pairs']} shared pairs}}}} \\",
        rf"$\mathbf{{A}}$ from the other five (mean) & {into(X['carried'])} \\",
        r"\quad the model's own $\mathbf{A}$ & " + " & ".join(C(X["own"][m]) for m in ms) + r" \\",
        rf"\quad control: shuffled alignment & {into(X['shuffled'])} \\",
        rf"\quad beyond one fixed direction & {into(X['carried_beyond'])} \\",
    ]
    (OUT / "t_golden_reduced.tex").write_text(REDUCED_HEAD + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("wrote", OUT, "t_golden_cores, t_golden_heldout, t_golden_controls, t_golden_reduced")


if __name__ == "__main__":
    main()
