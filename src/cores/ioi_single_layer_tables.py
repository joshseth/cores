#!/usr/bin/env python3
"""
ioi_single_layer_tables.py: rebuild the single-site IOI results of the paper from the saved runs
(ioi_core.py results and ioi_cross_model.py alignment, six models) and check them against the
values printed in the paper.

Tables
  Table A7   single-site core: layer (depth), causal rank, core-only, core-removed and flipped
             AUC, counterfactual transplant
  Fig. A8D  flip AUC by subspace: selected-rank ACE, ACE rank 1, PCA-1, difference of means
  Fig. A8C  cross-model canonical correlations (mean over the 15 pairs of CC1-3): raw, answer
             label removed, all task covariates removed, prompts shuffled
  plus the numbers quoted in the text (depth and rank ranges, 99%-energy ranks, CC ranges)

Reads only saved results; loads no model.

Usage
  python src/cores/ioi_single_layer_tables.py --root src/cores/experiments/ioi_all
  python src/cores/ioi_single_layer_tables.py --root <rerun root> --out tables.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODELS = [("gpt2", "GPT-2 Small"), ("gpt2medium", "GPT-2 Medium"), ("gpt2large", "GPT-2 Large"),
          ("llama318b", "LLaMA-3.1-8B"), ("gemma29b", "Gemma-2-9B"), ("qwen2532b", "Qwen2.5-32B")]

# The paper's values, as printed (strings keep the printed precision).
PAPER_TABLE_A7 = {  # layer, n_layers, depth, rank, core-only, core-removed, flipped, transplant
    "gpt2":       (9, 12, "0.75", 7, "1.000", "0.951", "0.012", "0.003"),
    "gpt2medium": (20, 24, "0.83", 21, "1.000", "0.797", "0.015", "0.001"),
    "gpt2large":  (22, 36, "0.61", 17, "1.000", "0.985", "0.015", "0.006"),
    "llama318b":  (23, 32, "0.72", 15, "1.000", "0.998", "0.002", "0.002"),
    "gemma29b":   (31, 42, "0.74", 15, "1.000", "1.000", "0.015", "0.019"),
    "qwen2532b":  (56, 64, "0.88", 17, "1.000", "0.998", "0.009", "0.002"),
}
PAPER_TEXT = {
    "energy_ranks": (8, 24, 17, 17, 16, 25),            # 99%-energy ranks, in MODELS order
    "depth_range": ("0.61", "0.88"),
    "rank_range": (7, 21),
    "core_removed_range": ("0.80", "1"),
    "pca1_flip": "1",                                    # PCA-1 never moves the answer
    "gpt2_ace1_dm_flip": ("0.36", "0.37"),               # ACE rank 1 and difference of means, GPT-2 Small
    "others_ace1_dm_flip_above": 0.94,                   # both, in the other five models
    "cc_covariates_removed": ("0.65", "0.58", "0.52"),
    "cc1_covariates_removed_range": ("0.47", "0.82"),
    "cc_shuffled": ("0.29", "0.25", "0.23"),
}


def matches(ours, printed):
    """ours rounds to the printed value (a string, or an int for counts)."""
    if isinstance(printed, int):
        return int(round(ours)) == printed
    d = len(printed.split(".")[1]) if "." in printed else 0
    return abs(ours - float(printed)) <= 0.5 * 10 ** -d + 1e-9


def curve_at_selected(d):
    curves = d.get("rank_curves") or {}
    return curves.get(str(d["final_layer_idx"])) or curves.get(d["final_layer_idx"]) or d["rank_curve"]


def load(root):
    R = {}
    for key, _ in MODELS:
        f = root / key / "results.json"
        if f.exists():
            R[key] = json.loads(f.read_text())
    cm = root / "cross_model_alignment.json"
    C = json.loads(cm.read_text()) if cm.exists() else []
    return R, [c for c in C if "cca_raw" in c]


def table_a7(R):
    rows = {}
    for key, _ in MODELS:
        if key not in R:
            continue
        d = R[key]
        curve = {x["rank"]: x for x in curve_at_selected(d)}
        sel = curve[d["selected_rank"]]
        so = d.get("state_operator", {})
        rows[key] = dict(layer=d["final_layer_idx"], n_layers=d["n_layers"],
                         depth=d["final_layer_idx"] / d["n_layers"], rank=d["selected_rank"],
                         energy_rank=d.get("r_energy", np.nan), base=d["base_auc"],
                         keep=sel["auc_keep"], remove=sel["auc_remove"], flip=sel["auc_flip"],
                         factual=so.get("auc_factual_transplant", np.nan),
                         cf=so.get("auc_counterfactual_transplant", np.nan),
                         ace1=curve[1]["auc_flip"],
                         pca1=d.get("baselines", {}).get("pca1", {}).get("flip", np.nan),
                         dm=d.get("baselines", {}).get("diff_means", {}).get("flip", np.nan))
    return rows


def alignment(C):
    if not C:
        return None
    get = lambda k: np.array([c[k][:3] for c in C])
    return dict(n_pairs=len(C), raw=get("cca_raw"), label_removed=get("cca_label_removed"),
                covariates_removed=get("cca_all_task_vars_removed"),
                shuffled=get("cca_all_task_vars_null_mean"))


def render(T, A):
    name = dict(MODELS)
    L = ["### Table A7. Single-site IOI core", "",
         "| Model | Layer (depth) | Rank | Core-only | Core-removed | Flipped | Transplant |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for k, r in T.items():
        L.append(f"| {name[k]} | {r['layer']}/{r['n_layers']} ({r['depth']:.2f}) | {r['rank']} | {r['keep']:.3f} | "
                 f"{r['remove']:.3f} | {r['flip']:.3f} | {r['cf']:.3f} |")
    L += ["", "### Fig. A8D. Flip AUC by subspace", "",
          "| Model | Selected-rank ACE | ACE rank 1 | PCA-1 | Difference of means |", "|---|---:|---:|---:|---:|"]
    for k, r in T.items():
        L.append(f"| {name[k]} | {r['flip']:.3f} | {r['ace1']:.3f} | {r['pca1']:.3f} | {r['dm']:.3f} |")
    if A:
        L += ["", f"### Fig. A8C. Cross-model canonical correlations ({A['n_pairs']} pairs, mean)", "",
              "| | CC1 | CC2 | CC3 |", "|---|---:|---:|---:|"]
        for k, lab in [("raw", "Raw ACE"), ("label_removed", "Answer label removed"),
                       ("covariates_removed", "All task covariates removed"), ("shuffled", "Prompts shuffled")]:
            L.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in A[k].mean(0)) + " |")
    return "\n".join(L)


def check(T, A):
    """Every value printed in the paper's single-site results, against the saved runs."""
    bad, n = [], 0

    def test(label, ours, printed):
        nonlocal n
        n += 1
        if not matches(ours, printed):
            bad.append(f"{label}: ours {ours:.4f}, paper {printed}")

    for k, (layer, n_layers, depth, rank, keep, remove, flip, cf) in PAPER_TABLE_A7.items():
        if k not in T:
            bad.append(f"Table A7 {k}: missing"); n += 1; continue
        r = T[k]
        for lab, ours, p in [("layer", r["layer"], layer), ("n_layers", r["n_layers"], n_layers),
                             ("depth", r["depth"], depth), ("rank", r["rank"], rank), ("core-only", r["keep"], keep),
                             ("core-removed", r["remove"], remove), ("flipped", r["flip"], flip), ("transplant", r["cf"], cf)]:
            test(f"Table A7 {k} {lab}", ours, p)
    t = PAPER_TEXT
    keys = [k for k, _ in MODELS]
    for k, e in zip(keys, t["energy_ranks"]):
        if k in T:
            test(f"99%-energy rank {k}", T[k]["energy_rank"], e)
    vals = lambda f: np.array([T[k][f] for k in keys if k in T])
    for lab, arr, (lo, hi) in [("depth range", vals("depth"), t["depth_range"]),
                               ("rank range", vals("rank"), t["rank_range"]),
                               ("core-removed range", vals("remove"), t["core_removed_range"])]:
        test(f"{lab} min", arr.min(), lo)
        test(f"{lab} max", arr.max(), hi)
    for k in keys:
        if k in T:
            test(f"PCA-1 flip {k}", T[k]["pca1"], t["pca1_flip"])
    if "gpt2" in T:
        test("ACE rank-1 flip, GPT-2 Small", T["gpt2"]["ace1"], t["gpt2_ace1_dm_flip"][0])
        test("difference-of-means flip, GPT-2 Small", T["gpt2"]["dm"], t["gpt2_ace1_dm_flip"][1])
    for k in keys[1:]:
        if k in T:
            for f, lab in [("ace1", "ACE rank-1"), ("dm", "difference-of-means")]:
                n += 1
                if not T[k][f] > t["others_ace1_dm_flip_above"]:
                    bad.append(f"{lab} flip {k}: ours {T[k][f]:.4f}, paper > {t['others_ace1_dm_flip_above']}")
    if A:
        for i, p in enumerate(t["cc_covariates_removed"]):
            test(f"CC{i + 1}, task covariates removed", A["covariates_removed"].mean(0)[i], p)
        for i, p in enumerate(t["cc_shuffled"]):
            test(f"CC{i + 1}, prompts shuffled", A["shuffled"].mean(0)[i], p)
        lo, hi = t["cc1_covariates_removed_range"]
        test("CC1 per pair min, task covariates removed", A["covariates_removed"][:, 0].min(), lo)
        test("CC1 per pair max, task covariates removed", A["covariates_removed"][:, 0].max(), hi)
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("src/cores/experiments/ioi_all"))
    ap.add_argument("--out", type=Path, default=None, help="write the markdown tables here")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    R, C = load(args.root)
    if not R:
        sys.exit(f"no results.json under {args.root}")
    T, A = table_a7(R), alignment(C)
    md = render(T, A)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n")
        print(f"\nwrote {args.out}")
    if not args.no_check:
        n, bad = check(T, A)
        print(f"\nCHECK against the paper (Table A7, Fig. A8C-D and the numbers in the text): {n - len(bad)}/{n} match")
        for b in bad:
            print("  MISMATCH", b)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
