#!/usr/bin/env python3
"""
ioi_cross_model.py: run the IOI core analysis across architectures on
identical prompts, then report label-controlled cross-model alignment.

Single-token name filtering depends on the tokenizer, so without a shared name
list each model would get a different prompt set and the coordinate
correlations would compare different data. This script (1) asks every model which names are single-token, (2) takes
the intersection, (3) runs each model with that shared list, (4) verifies
the prompt sets are byte-identical via saved hashes, (5) reports alignment
with the IO-position label removed.

Alignment statistics (same logic as the SVA label control):
  observed     -- Pearson/Spearman correlation of core coordinates
  label-null   -- what two unrelated models with these label sensitivities
                  would show: corr(z_i, s) * corr(z_j, s)
  partial      -- correlation of label-residualized coordinates
  within-class -- correlation restricted to each label class (label constant)

Usage:
  python src/cores/ioi_cross_model.py \
      --models gpt2 gpt2-medium meta-llama/Llama-3.1-8B google/gemma-2-9b \
      --out-root experiments/ioi --dtype float32
  # add --skip-runs to only redo the alignment from existing payloads
"""
from __future__ import annotations
import argparse, itertools, json, re, subprocess, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
IOI = HERE / "ioi_core.py"


def rebuild_design(out_root, seed, n_samples):
    """Reconstruct the per-prompt task variables (IO name, S name, position)
    for the test split, from the deterministic dataset builder. Name token ids
    are replaced by indices, which does not change any rng draw, so the
    rebuilt prompts and split match the payloads exactly (verified against the
    saved label vector by the caller)."""
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    from ioi_core import build_ioi_dataset, TEMPLATES
    shared = json.loads((Path(out_root) / "shared_names.json").read_text())
    names = [(nm, i) for i, nm in enumerate(shared)]
    rng = np.random.default_rng(seed)
    texts, y, first_ix, second_ix = build_ioi_dataset(n_samples, rng, names)
    N = len(texts)
    te = rng.permutation(N)[N // 2:]
    yt = y[te]
    fi, si = np.asarray(first_ix)[te], np.asarray(second_ix)[te]
    # IO is the first name when y==0, the second when y==1
    io_ix = np.where(yt == 0, fi, si)
    s_ix = np.where(yt == 0, si, fi)
    # Template identity must be in the design: it is identical across models
    # and encoded almost deterministically (it fixes the sentence frame and the
    # prompt length), so leaving it out lets input-determined structure appear
    # as shared computation; without it, canonical correlations are near 1
    # even between models that share no computation.
    keys = []
    for t in TEMPLATES:
        seg = t.split("{Y}", 1)[1].split("{S}", 1)[0]
        keys.append(seg)
    tix = np.full(len(te), -1, dtype=int)
    texts_te = [texts[int(i)] for i in te]
    for j, txt in enumerate(texts_te):
        hits = [k for k, seg in enumerate(keys) if seg in txt]
        if len(hits) == 1:
            tix[j] = hits[0]
    if (tix < 0).any():
        raise ValueError(f"could not identify the template for "
                         f"{(tix < 0).sum()} prompts")
    K, T = len(shared), len(TEMPLATES)
    D = np.zeros((len(te), 2 * K + T + 2))
    D[np.arange(len(te)), io_ix] = 1.0
    D[np.arange(len(te)), K + s_ix] = 1.0
    D[np.arange(len(te)), 2 * K + tix] = 1.0
    D[:, 2 * K + T] = 2.0 * yt - 1.0
    D[:, 2 * K + T + 1] = 1.0
    return D, yt


def short(name):
    return re.sub(r"[^A-Za-z0-9]+", "", name.split("/")[-1]).lower()


def dump_names(model, dtype):
    out = subprocess.run([sys.executable, str(IOI), "--model-name", model,
                          "--dump-names", "--dtype", dtype],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout, out.stderr)
        raise RuntimeError(f"--dump-names failed for {model}")
    line = [l for l in out.stdout.splitlines() if l.startswith("SINGLE_TOKEN_NAMES:")]
    assert line, f"no name line from {model}: {out.stdout[-400:]}"
    return [n.strip() for n in line[0].split(":", 1)[1].split(",") if n.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("experiments/ioi"))
    ap.add_argument("--n-samples", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-runs", action="store_true")
    ap.add_argument("--sweep-ranks", type=str, default="1,2,4,8")
    ap.add_argument("--fast-sweep", action="store_true",
                    help="two-stage: cheap flip-only localization, then full "
                         "rank curves at the top-k layers")
    ap.add_argument("--probe-rank", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--coarse-frac", type=float, default=0.5)
    ap.add_argument("--keep-tol", type=float, default=0.02)
    ap.add_argument("--causal-tol", type=float, default=0.02)
    ap.add_argument("--cascade-removal", action="store_true")
    ap.add_argument("--rank-rule", type=str, default="causal",
                    choices=["causal", "cliff", "cliff2", "energy"])
    ap.add_argument("--rank-fine", type=int, default=12)
    ap.add_argument("--rank-stride", type=int, default=2)
    ap.add_argument("--rank-max", type=int, default=10)
    ap.add_argument("--n-jac", type=int, default=128)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-state-operator", action="store_true")
    args = ap.parse_args()

    # ---- 1. shared single-token name list ---------------------------------
    if not args.skip_runs:
        sets = {}
        for m in args.models:
            sets[m] = set(dump_names(m, args.dtype))
            print(f"{m}: {len(sets[m])} single-token names")
        shared = sorted(set.intersection(*sets.values()))
        print(f"\nshared across all {len(args.models)} models: {len(shared)} names -> {shared}")
        if len(shared) < 8:
            sys.exit(f"only {len(shared)} shared names; extend CANDIDATE_NAMES in ioi_core.py")
        (args.out_root).mkdir(parents=True, exist_ok=True)
        (args.out_root / "shared_names.json").write_text(json.dumps(shared, indent=2))

        # ---- 2. per-model runs on identical prompts -----------------------
        for m in args.models:
            d = args.out_root / short(m)
            cmd = [sys.executable, str(IOI), "--model-name", m,
                   "--n-samples", str(args.n_samples),
                   "--batch-size", str(args.batch_size),
                   "--dtype", args.dtype, "--seed", str(args.seed),
                   "--sweep-ranks", args.sweep_ranks,
                   "--rank-max", str(args.rank_max), "--n-jac", str(args.n_jac),
                   "--names", ",".join(shared), "--out-dir", str(d)]
            if args.fast_sweep:
                cmd += ["--fast-sweep", "--probe-rank", str(args.probe_rank),
                        "--top-k", str(args.top_k),
                        "--coarse-frac", str(args.coarse_frac)]
            cmd += ["--keep-tol", str(args.keep_tol),
                    "--rank-rule", args.rank_rule,
                    "--causal-tol", str(args.causal_tol),
                    "--rank-fine", str(args.rank_fine),
                    "--rank-stride", str(args.rank_stride)]
            if not args.no_baselines:
                cmd.append("--baselines")
            if args.cascade_removal:
                cmd.append("--cascade-removal")
            if not args.no_state_operator:
                cmd.append("--state-operator")
            print("\n" + "=" * 70 + f"\nRUN: {m}\n" + "=" * 70)
            if subprocess.run(cmd).returncode != 0:
                sys.exit(f"run failed for {m}")

    # ---- 3. alignment on identical prompts --------------------------------
    data = {}
    for m in args.models:
        f = args.out_root / short(m) / "knob_fit_test.npz"
        if not f.exists():
            print(f"missing payload: {f}; skipping {m}")
            continue
        d = np.load(f)
        z, y = d["z"].astype(float), d["y"].astype(int)
        s = 2.0 * y - 1.0
        if np.corrcoef(z, s)[0, 1] < 0:       # sign-align: +z = IO in second slot
            z = -z
        Zm = d["Z_multi"].astype(float) if "Z_multi" in d.files else None
        data[short(m)] = {"z": (z - z.mean()) / z.std(), "y": y, "Z": Zm,
                          "hash": d["prompt_hash"] if "prompt_hash" in d.files else None,
                          "r_energy": None}
        rj = args.out_root / short(m) / "results.json"
        if rj.exists():
            r = json.loads(rj.read_text())
            data[short(m)]["r_energy"] = r.get("r_energy")
            data[short(m)]["layer"] = r.get("final_layer_idx")
            data[short(m)]["rank_curve"] = r.get("rank_curve")
            data[short(m)]["state_op"] = r.get("state_operator")
    if len(data) < 2:
        sys.exit("need at least two completed runs for alignment")

    keys = list(data)
    h0 = data[keys[0]]["hash"]
    for k in keys[1:]:
        hk = data[k]["hash"]
        if h0 is not None and hk is not None and not np.array_equal(h0, hk):
            print(f"  NOTE: prompt hashes differ between {keys[0]} and {k}. Runs "
                  "made before the stable-hash fix used a process-salted hash, so "
                  "this is expected for legacy payloads; label vectors are checked "
                  "below and must match.")
        assert np.array_equal(data[keys[0]]["y"], data[k]["y"]), "label vectors differ"
    print("\nprompt sets verified identical across models"
          if h0 is not None else "\n(prompt hashes unavailable; labels match)")

    print("\n=== PER-MODEL SUMMARY ===")
    print(f"{'model':14s} {'core layer':>10s} {'energy rank':>12s} "
          f"{'flip r=1':>9s} {'flip r=max':>11s} {'cf transplant':>14s}")
    for k in keys:
        d = data[k]
        rc = d.get("rank_curve") or []
        f1 = rc[0]["auc_flip"] if rc else float("nan")
        fm = rc[-1]["auc_flip"] if rc else float("nan")
        so = d.get("state_op") or {}
        cf = so.get("auc_counterfactual_transplant", float("nan"))
        print(f"{k:14s} {str(d.get('layer')):>10s} {str(d.get('r_energy')):>12s} "
              f"{f1:9.3f} {fm:11.3f} {cf:14.3f}")

    s_lab = 2.0 * data[keys[0]]["y"] - 1.0

    def resid(z):
        X = np.stack([np.ones_like(z), s_lab], 1)
        b, *_ = np.linalg.lstsq(X, z, rcond=None)
        return z - X @ b

    inert = [k for k in keys
             if ((data[k].get("rank_curve") or [{}])[0].get("auc_flip", 0) or 0) > 0.9]
    print("\n=== LEADING-DIRECTION ALIGNMENT (IO-position label removed) ===")
    if inert:
        print(f"  CAUTION: the leading direction is causally inert for {inert} "
              "(rank-1 flip > 0.9), so single-coordinate alignment does not "
              "compare decision axes for those models. Use the subspace CCA below.")
    print(f"{'pair':26s} {'observed':>9s} {'(rank)':>7s} {'label-null':>11s} "
          f"{'partial':>8s} {'within-0':>9s} {'within-1':>9s}")
    rows = []
    for a, b in itertools.combinations(keys, 2):
        zi, zj = data[a]["z"], data[b]["z"]
        r = float(np.corrcoef(zi, zj)[0, 1])
        rs = float(spearmanr(zi, zj).statistic)
        riy = float(np.corrcoef(zi, s_lab)[0, 1]); rjy = float(np.corrcoef(zj, s_lab)[0, 1])
        null = riy * rjy
        ei, ej = resid(zi), resid(zj)
        part = float(np.corrcoef(ei, ej)[0, 1])
        y0 = data[keys[0]]["y"]
        w0 = float(np.corrcoef(zi[y0 == 0], zj[y0 == 0])[0, 1])
        w1 = float(np.corrcoef(zi[y0 == 1], zj[y0 == 1])[0, 1])
        rows.append({"pair": f"{a}x{b}", "observed": r, "spearman": rs,
                     "label_null": null, "partial": part,
                     "within_0": w0, "within_1": w1})
        print(f"{a+'x'+b:26s} {r:9.3f} {rs:7.3f} {null:11.3f} {part:8.3f} "
              f"{w0:9.3f} {w1:9.3f}")
    # ---- subspace-level: label-controlled CCA between coordinate sets -------
    # Cores may differ in dimension and depth across models, so rank-1
    # correlation is not the right statistic. Canonical correlations between
    # the models' coordinate matrices (on identical prompts) generalize it;
    # residualizing both on the label first removes the shared-answer channel.
    D_full = None
    try:
        D_full, y_rebuilt = rebuild_design(args.out_root, args.seed, args.n_samples)
        if not np.array_equal(y_rebuilt, data[keys[0]]["y"]):
            print("  NOTE: rebuilt task design does not match payload labels; "
                  "skipping the all-task-variables control")
            D_full = None
    except Exception as e:
        print(f"  NOTE: could not rebuild task design ({e}); skipping the "
              "all-task-variables control")

    def resid_full(Z):
        B, *_ = np.linalg.lstsq(D_full, Z, rcond=None)
        return Z - D_full @ B

    def resid_mat(Z):
        X = np.stack([np.ones(len(Z)), s_lab], 1)
        B, *_ = np.linalg.lstsq(X, Z, rcond=None)
        return Z - X @ B

    def cca_corrs(A, B):
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        Qa, _ = np.linalg.qr(Ac); Qb, _ = np.linalg.qr(Bc)
        return np.linalg.svd(Qa.T @ Qb, compute_uv=False)

    def cca_null(A, B, n_perm=50, seed=0):
        """Chance level for these ranks and this sample size: destroy the
        prompt-to-prompt correspondence by shuffling rows of B. Without this,
        canonical correlations between two rank-16 sets on 600 prompts look
        high whether or not the models share structure."""
        prng = np.random.default_rng(seed)
        tops = []
        for _ in range(n_perm):
            tops.append(cca_corrs(A, B[prng.permutation(len(B))])[:3])
        T = np.array(tops)
        return T.mean(0), T.std(0)

    have_Z = [k for k in keys if data[k]["Z"] is not None]
    if len(have_Z) >= 2:
        print("\n=== SUBSPACE ALIGNMENT: label-controlled CCA between coordinate "
              "sets ===")
        print(f"{'pair':22s} {'ranks':>7s} {'CCA raw':>21s} "
              f"{'CCA label-removed':>21s} {'shuffled null':>21s}")
        fmt = lambda v: ", ".join(f"{x:.3f}" for x in v[:3])
        for a, b in itertools.combinations(have_Z, 2):
            A, B = data[a]["Z"], data[b]["Z"]
            raw = cca_corrs(A, B)
            ctl = cca_corrs(resid_mat(A), resid_mat(B))
            nm, ns = cca_null(resid_mat(A), resid_mat(B))
            fullA = resid_full(A) if D_full is not None else None
            fullB = resid_full(B) if D_full is not None else None
            ctl2 = cca_corrs(fullA, fullB) if D_full is not None else None
            nm2, ns2 = (cca_null(fullA, fullB) if D_full is not None
                        else (None, None))
            print(f"{a+'x'+b:22s} {f'{A.shape[1]}x{B.shape[1]}':>7s} "
                  f"{fmt(raw):>21s} {fmt(ctl):>21s} "
                  f"{', '.join(f'{m:.2f}±{sd:.2f}' for m, sd in zip(nm, ns)):>21s}")
            if ctl2 is not None:
                print(f"{'':22s} {'':>7s} {'task vars + template:':>21s} "
                      f"{fmt(ctl2):>21s} "
                      f"{', '.join(f'{m:.2f}±{sd:.2f}' for m, sd in zip(nm2, ns2)):>21s}")
            rows.append({"pair": f"{a}x{b}", "cca_raw": [float(x) for x in raw],
                         "cca_label_removed": [float(x) for x in ctl],
                         "cca_shuffled_null_mean": [float(x) for x in nm],
                         "cca_shuffled_null_sd": [float(x) for x in ns],
                         "cca_all_task_vars_removed": ([float(x) for x in ctl2]
                                                       if ctl2 is not None else None),
                         "cca_all_task_vars_null_mean": ([float(x) for x in nm2]
                                                         if nm2 is not None else None),
                         "ranks": [int(A.shape[1]), int(B.shape[1])]})
        print("Reading: canonical correlations near 1 mean the two coordinate sets "
              "span a shared subspace of task-relevant variation; the label-removed "
              "column excludes agreement that follows from both models simply "
              "answering correctly.")

    (args.out_root / "cross_model_alignment.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {args.out_root}/cross_model_alignment.json")
    print("Reading: 'partial' and the within-class columns are alignment with the "
          "IO-position label's contribution removed. High values mean models agree "
          "on the graded fine structure, not merely on the binary answer.")


if __name__ == "__main__":
    main()
