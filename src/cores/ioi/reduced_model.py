#!/usr/bin/env python3
"""
reduced_model.py: the reduced model of the IOI core and every number reported for it, from the per-prompt data (model_dump.py output). No GPU and no model weights.

    S2 to END      z_E_hat = mean_E + A (z_S - mean_S)     A: least squares on within-pair differences, r_E = A r_S
    END to answer  m_hat   = c' z_E + d                    c, d: least squares on single prompts

Both are fitted on natural activations from the extraction frames and names, with no interventions. Scored on test and validation (held-out frames, 40 new names),
and on saved interventions that the fit never sees. Across models, A is carried from one model into another through linear maps between their cores, fitted on natural
within-pair differences on half of the prompts shared by all six models and scored on the other half.

Pairs: members adjacent, partner = i ^ 1; s = -1 on the first member (B repeated, answer A) and +1 on the second. Margin m = logit(A) - logit(B), with A mentioned first.
The tokenizers are needed to match prompts across models; they download once, without the weights.

  PYTHONPATH=src python -m cores.ioi.reduced_model --dumps results/ioi/model --out results/ioi/reduced_model.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from transformers import AutoTokenizer

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "llama-3.1-8b", "gemma-2-9b", "qwen2.5-32b"]
HF = {"gpt2": "gpt2", "gpt2-medium": "gpt2-medium", "gpt2-large": "gpt2-large", "llama-3.1-8b": "meta-llama/Llama-3.1-8B", "gemma-2-9b": "google/gemma-2-9b", "qwen2.5-32b": "Qwen/Qwen2.5-32B"}
SETS = ("fit", "test", "val")
HELD = ("test", "val")

share = lambda y, p: float(1 - ((y - p) ** 2).sum() / (y ** 2).sum())
r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
beyond = lambda y, p, base: float(1 - ((y - p) ** 2).sum() / ((y - base) ** 2).sum())   # the share of the error left by one fixed direction that p removes
agree = lambda p, m: float(np.mean(np.sign(p) == np.sign(m)))
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def auc(m, y):
    r = rankdata(m); pos = y == 0; n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ridge(X, Y, lam=1e-2):
    G = X.T @ X
    return np.linalg.solve(G + lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), X.T @ Y)


class Dump:
    """One model's per-prompt data, read lazily."""

    def __init__(self, root, model):
        self.model = model
        self.data = np.load(f"{root}/{model}/{model}_dump.npz", allow_pickle=False)
        self.row = {int(i): k for k, i in enumerate(self.data["cS_ids"])}
        self.cache = {}

    def get(self, s, key):
        k = f"{s}_{key}"
        if k not in self.cache:
            self.cache[k] = self.data[k].astype(np.float64) if key not in ("y", "frame", "AB", "pos") else self.data[k]
        return self.cache[k]

    def n(self, s):
        return len(self.get(s, "m"))

    def partner(self, s):
        return np.arange(self.n(s)) ^ 1

    def first(self, s):
        return np.arange(self.n(s)) % 2 == 0

    def sign(self, s):
        return np.where(self.first(s), -1.0, 1.0)

    def name_difference(self, s):
        """Per prompt, c(B) - c(A), with c(n) each name alone after the start token, at the S2 layer, in S2 core coordinates."""
        c = self.data["cS"].astype(np.float64); AB = self.get(s, "AB")
        a, b = (np.repeat([self.row[int(x)] for x in AB[:, j]], 2) for j in (0, 1))
        return c[b] - c[a]

    def swap(self, s):
        m = self.get(s, "m")
        return m[self.partner(s)] - m


def size_scores(D, s, pred, scale):
    """Between-sentence correlation of the swap's size with its prediction, and magnitude R^2 with the scale from the fit set."""
    e = D.first(s); y, p, f = np.abs(D.swap(s)[e]), np.abs(pred[e]), D.get(s, "frame")[e]; fs = [k for k in np.unique(f) if (f == k).sum() >= 5]
    return corr([y[f == k].mean() for k in fs], [p[f == k].mean() for k in fs]), float(1 - ((y - scale * p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def size_scale(D, pred):
    e = D.first("fit"); y, p = np.abs(D.swap("fit")[e]), np.abs(pred[e])
    return float((y * p).sum() / (p ** 2).sum())


def fit(D):
    """A, c, d from natural activations. Control: A on shuffled pairs in random order (the fit does not depend on the order within a pair, so the control must not keep it)."""
    zS, zE, m = D.get("fit", "zS"), D.get("fit", "zE"), D.get("fit", "m")
    pt, e = D.partner("fit"), D.first("fit")
    rS, rE = 0.5 * (zS - zS[pt])[e], 0.5 * (zE - zE[pt])[e]
    A = np.linalg.lstsq(rS, rE, rcond=None)[0].T
    g = np.random.default_rng(0); A_shuffled = np.linalg.lstsq(rS, rE[g.permutation(len(rS))] * g.choice([-1.0, 1.0], len(rS))[:, None], rcond=None)[0].T
    dE = D.get("fit", "zE_sw") - zE; A_swaps = np.linalg.lstsq(zS[pt] - zS, dE, rcond=None)[0].T   # for comparison: a map fitted on the swaps themselves
    direction = (D.sign("fit")[:, None] * dE).mean(0)                                                # the best single fixed direction for the END change after an S2 swap
    G = np.linalg.lstsq(np.column_stack([zE, np.ones(len(zE))]), m, rcond=None)[0]
    v = rS.mean(0); v = v / np.linalg.norm(v); mS, mE = zS.mean(0), zE.mean(0)
    line = np.polyfit((zS - mS) @ v, m, 1)                                                           # one direction at S2: the mean within-pair difference
    return dict(A=A, A_shuffled=A_shuffled, A_swaps=A_swaps, direction=direction, f=lambda x: mE + (x - mS) @ A.T, g=lambda x: x @ G[:-1] + G[-1], line=lambda x: np.polyval(line, (x - mS) @ v))


def score_model(D, F):
    chain = lambda x: F["g"](F["f"](x)); swap_of = lambda p, s: p[D.partner(s)] - p
    scale_g, scale_c = size_scale(D, swap_of(F["g"](D.get("fit", "zE")), "fit")), size_scale(D, swap_of(chain(D.get("fit", "zS")), "fit"))
    v = D.data["v"].astype(np.float64); R = {}
    for s in HELD:
        z, ze, m, y, pt, sg, dn = D.get(s, "zS"), D.get(s, "zE"), D.get(s, "m"), D.get(s, "y"), D.partner(s), D.sign(s)[:, None], D.name_difference(s)
        dE, dS = D.get(s, "zE_sw") - ze, z[pt] - z; rE, rS = 0.5 * (ze - ze[pt]), 0.5 * (z - z[pt]); fixed = sg * F["direction"]; pg, pc = F["g"](ze), chain(z); label = -D.sign(s)
        writes = {"swap": (z[pt], D.get(s, "m_sw")), "flip": (-z, D.get(s, "m_invS")), "pointer": (z + sg * v, D.get(s, "m_injPtr")), "names": (z + sg * dn, D.get(s, "m_injNm")), "mark": (z + sg * (v + dn), D.get(s, "m_injEq"))}
        g_between, g_mag = size_scores(D, s, swap_of(pg, s), scale_g); between, mag = size_scores(D, s, swap_of(pc, s), scale_c)
        R[s] = dict(role=share(rE, rS @ F["A"].T), change=share(dE, dS @ F["A"].T), change_shuffled=share(dE, dS @ F["A_shuffled"].T), change_direction=share(dE, fixed), beyond=beyond(dE, dS @ F["A"].T, fixed), change_swapfit=share(dE, dS @ F["A_swaps"].T), whole=r2(ze, F["f"](z)),
                    g_auc=auc(pg, y), g_between=g_between, g_mag=g_mag, auc=auc(pc, y), line_auc=auc(F["line"](z), y), name=agree(pc, m), between=between, mag=mag,
                    net_swap_auc=auc(writes["swap"][1], y), net_mark_auc=auc(writes["mark"][1], y), net_pointer_auc=auc(writes["pointer"][1], y), net_names_auc=auc(writes["names"][1], y),
                    **{f"same_{k}": agree(chain(x), mw) for k, (x, mw) in writes.items()}, **{f"line_{k}": agree(F["line"](x), mw) for k, (x, mw) in writes.items()}, **{f"ignore_{k}": agree(label, mw) for k, (x, mw) in writes.items()})
    return R


def mark_parts(D):
    """Mean change in the margin toward the other name, test and validation pooled: the full mark and each part alone."""
    out = {k: [] for k in ("m_injEq", "m_injPtr", "m_injNm")}
    for s in HELD:
        m = D.get(s, "m"); o = -(1 - 2 * D.get(s, "y")).astype(np.float64)
        for k in out:
            out[k].append((D.get(s, k) - m) * o)
    full, v, n = (float(np.concatenate(out[k]).mean()) for k in out)
    return dict(full=full, v=v, names=n, v_share=v / full, names_share=n / full, sum_share=(v + n) / full)


def between_share(D, s="test"):
    """The share of the variance of the swap's size that lies between sentences."""
    e = D.first(s); y, f = np.abs(D.swap(s)[e]), D.get(s, "frame")[e]
    return float(sum((f == k).sum() * (y[f == k].mean() - y.mean()) ** 2 for k in np.unique(f)) / ((y - y.mean()) ** 2).sum())


def sentence_agreement(data):
    """Per-sentence swap size over the sentences in all six models: mean correlation over model pairs, with token distances and length regressed out, and a shuffled null."""
    per = {}
    for model, D in data.items():
        per[model] = {}
        for s in SETS:
            e = D.first(s); fr = D.get(s, "frame")[e]; size = np.abs(D.swap(s))[e]; pos = D.get(s, "pos")[e]
            for f in np.unique(fr):
                q = fr == f; per[model][(s, int(f))] = (float(size[q].mean()), [float(x) for x in ((pos[q, 2] - pos[q, 1]).mean(), (pos[q, 3] - pos[q, 2]).mean(), (pos[q, 3] - pos[q, 0]).mean(), pos[q, 3].mean())])
    keys = sorted(set.intersection(*[set(r) for r in per.values()])); pairs = list(itertools.combinations(data, 2)); X, Xr = {}, {}
    for m in data:
        X[m] = np.array([per[m][k][0] for k in keys]); B = np.column_stack([np.ones(len(keys)), np.array([per[m][k][1] for k in keys])]); Xr[m] = X[m] - B @ np.linalg.lstsq(B, X[m], rcond=None)[0]
    g = np.random.default_rng(0)
    return dict(n=len(keys), raw=float(np.mean([corr(X[a], X[b]) for a, b in pairs])), removed=float(np.mean([corr(Xr[a], Xr[b]) for a, b in pairs])),
                null=float(np.mean([np.mean([corr(X[a], X[b][g.permutation(len(keys))]) for a, b in pairs]) for _ in range(200)])))


def shared_prompts(data):
    """For each model, per shared prompt pair (same sentence and names): role differences and swap changes, and the network's own swap effects."""
    rows = {}
    for model, D in data.items():
        tok = AutoTokenizer.from_pretrained(HF[model]); r = {}
        for s in HELD:
            e, pt, z, ze, m = D.first(s), D.partner(s), D.get(s, "zS"), D.get(s, "zE"), D.get(s, "m"); o = 1 - 2 * D.get(s, "y"); fr = D.get(s, "frame")[e]
            rS, rE, dS, dE = 0.5 * (z - z[pt])[e], 0.5 * (ze - ze[pt])[e], (z[pt] - z)[e], (D.get(s, "zE_sw") - ze)[e]
            full = (-(m[pt] - m) * o)[e]; toward = lambda k: (-(D.get(s, k) - m) * o)[e] / np.abs(full).mean(); s2, end, whole = toward("m_sw"), toward("m_pE"), full / np.abs(full).mean()
            for i, (f, (a, b)) in enumerate(zip(fr, D.get(s, "AB"))):
                key = (s, int(f), tok.decode([int(a)]).strip(), tok.decode([int(b)]).strip())
                if key not in r:
                    r[key] = dict(rS=rS[i], rE=rE[i], dS=dS[i], dE=dE[i], effects=[])
                r[key]["effects"].append((whole[i], s2[i], end[i]))
        rows[model] = r
    keys = sorted(set.intersection(*[set(r) for r in rows.values()]))
    return keys, {m: dict(**{k: np.stack([rows[m][key][k] for key in keys]) for k in ("rS", "rE", "dS", "dE")}, effects=np.stack([np.mean(rows[m][key]["effects"], 0) for key in keys])) for m in rows}


def across_models(fits, keys, arr):
    """A carried between models, and the agreement of the networks' own swap effects over the same prompts."""
    sent = np.array([f"{k[0]}{k[1]}" for k in keys]); uq = np.unique(sent); half = np.isin(sent, np.array_split(np.random.default_rng(0).permutation(uq), 2)[0])

    def carried(i, j, shuffled=False):
        a, b = arr[i], arr[j]; pred = np.zeros_like(b["dE"]); g = np.random.default_rng(1)
        for tr in (half, ~half):
            n = int(tr.sum()); p = g.permutation(n) if shuffled else np.arange(n); e1, e2 = g.choice([-1.0, 1.0], (2, n, 1)) if shuffled else (1.0, 1.0)
            pred[~tr] = b["dS"][~tr] @ ridge(b["rS"][tr], a["rS"][tr][p] * e1) @ fits[i]["A"].T @ ridge(a["rE"][tr], b["rE"][tr][p] * e2)
        return share(b["dE"], pred), beyond(b["dE"], pred, -fits[j]["direction"])   # the scored rows are first members, where s = -1

    pairs = list(itertools.permutations(fits, 2)); both = {f"{i}>{j}": carried(i, j) for i, j in pairs}
    out = dict(n_pairs=len(keys), n_sentences=len(uq), carried={k: v[0] for k, v in both.items()}, carried_beyond={k: v[1] for k, v in both.items()}, shuffled={f"{i}>{j}": carried(i, j, True)[0] for i, j in pairs},
               own={j: share(arr[j]["dE"], arr[j]["dS"] @ fits[j]["A"].T) for j in fits}, own_beyond={j: beyond(arr[j]["dE"], arr[j]["dS"] @ fits[j]["A"].T, -fits[j]["direction"]) for j in fits})

    def within(x):
        y = x.copy()
        for s_ in uq:
            y[sent == s_] -= y[sent == s_].mean()
        return y
    un = list(itertools.combinations(fits, 2)); E = lambda m, q: arr[m]["effects"][:, q]; names = ("whole input", "S2 core", "END core")
    out["network"] = {names[q]: dict(raw=float(np.mean([corr(E(a, q), E(b, q)) for a, b in un])), within=float(np.mean([corr(within(E(a, q)), within(E(b, q))) for a, b in un]))) for q in range(3)}
    out["s2_share"] = {m: float(E(m, 1).mean() / E(m, 0).mean()) for m in fits}
    return out


def report(J):
    R = J["models"]; rng = lambda k, s: f"{min(R[m][s][k] for m in MODELS):+.3f} to {max(R[m][s][k] for m in MODELS):+.3f}"
    for k in R[MODELS[0]]["test"]:
        print(f"{k:<18} test {rng(k, 'test')}   val {rng(k, 'val')}")
    X = J["across"]; mean_in = lambda d, j: float(np.mean([v for k, v in d.items() if k.endswith(f">{j}")])); row = lambda d: " ".join(f"{d[m]:.2f}" for m in MODELS)
    print(f"\nacross models, {X['n_pairs']} shared pairs, {X['n_sentences']} sentences: carried {np.mean(list(X['carried'].values())):.3f}, own {np.mean(list(X['own'].values())):.3f}, shuffled {np.mean(list(X['shuffled'].values())):.3f} ({min(X['shuffled'].values()):.3f} to {max(X['shuffled'].values()):.3f}); beyond one direction: carried {np.mean(list(X['carried_beyond'].values())):.3f}, own {np.mean(list(X['own_beyond'].values())):.3f}")
    print("  by target, carried:", " ".join(f"{mean_in(X['carried'], m):.2f}" for m in MODELS), "| own:", row(X["own"]), "| shuffled:", " ".join(f"{mean_in(X['shuffled'], m):.2f}" for m in MODELS))
    print("  from GPT-2 Small:", " ".join(f"{X['carried'][f'gpt2>{m}']:.2f}" for m in MODELS[1:]))
    print("  the networks' own swap effects, raw and within sentences:", {k: (round(v["raw"], 2), round(v["within"], 2)) for k, v in X["network"].items()})
    print("  S2 core's share of the swap:", row(X["s2_share"]))
    print("between-sentence share of the swap's size, test:", row(J["between_share"]))
    print("mark parts, share of the full mark's change in the margin:", {m: {k: round(J["mark"][m][k], 2) for k in ("v_share", "names_share", "sum_share")} for m in MODELS})
    print("sentence-level swap size across models:", {k: round(v, 3) for k, v in J["sentences"].items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", type=Path, default=Path("results/ioi/model"), help="the folder holding <model>/<model>_dump.npz")
    p.add_argument("--out", type=Path, default=Path("results/ioi/reduced_model.json"))
    args = p.parse_args()
    data = {m: Dump(args.dumps, m) for m in MODELS}; fits = {m: fit(data[m]) for m in MODELS}
    J = dict(models={m: score_model(data[m], fits[m]) for m in MODELS}, mark={m: mark_parts(data[m]) for m in MODELS}, between_share={m: between_share(data[m]) for m in MODELS}, sentences=sentence_agreement(data))
    keys, arr = shared_prompts(data); J["across"] = across_models(fits, keys, arr)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(J, indent=1)); report(J); print("wrote", args.out)


if __name__ == "__main__":
    main()
