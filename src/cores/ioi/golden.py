#!/usr/bin/env python3
"""
golden.py: the two-site IOI cores of one model: sweep, scan, test. Role-swap pairs, second moment of the pair differences, not centered.

Sweep    At every layer (--stride for larger models) and both sites: flip at rank max(16, 2.5% of D). Where flip works: removal at a fixed size, the other site at its
         reference layer. The lowest removal at each site gives the sweep's best layers.
Scan     At the reference pair and the sweep's best pair: control against rank, then the smallest core (S2 rank + END rank) that passes keep, remove and
         control, with control by flip or swap, at S2 or at both sites. Also the smallest core at one site alone.
Test     Each core scored on train and held-out prompts; keep and remove against rank for the figures.

  PYTHONPATH=src python -m cores.ioi.golden --model-name gpt2 --tag gpt2 --reference 7:10 --out-dir results/ioi/golden/gpt2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cores.ioi.common import (COL, CURVE, SEED, SITES, SwapHook, auc_ci, build_role_pairs, capture, ent, evaluate, get_blocks, get_dims, load_model, orth, role_factors, seed_all,
                              setup_data, stratified_pairs)

KEEP_TOL, REMOVE_MAX, CONTROL_MAX, FLOOR = 0.05, 0.55, 0.10, 0.40
RULES = {"flip at S2": ("flip", ("S2",)), "swap at S2": ("swap", ("S2",)), "flip at both sites": ("flip", SITES), "swap at both sites": ("swap", SITES)}
GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
LADDER = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
VAR = "orth_write"
res = lambda r: max(1, int(0.05 * r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--reference", required=True, help="S2:END layers the sweep starts from")
    p.add_argument("--stride", type=int, default=1, help="sweep every stride layers")
    p.add_argument("--scan-pairs", type=int, default=0, help="pairs for the sweep's cores (default 4000, or 8000 if D >= 2000)")
    p.add_argument("--n-pairs", type=int, default=12000, help="pairs for the final cores")
    p.add_argument("--per-frame", type=int, default=3, help="train pairs per extraction frame")
    p.add_argument("--heldout-per-frame", type=int, default=19)
    p.add_argument("--mem-gb", type=float, default=30, help="states and gradients held per extraction pass")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.batch_size -= args.batch_size % 2

    dev = torch.device(args.device); seed_all(SEED)
    tok, model = load_model(args.model_name, args.dtype, args.attn_impl, dev, sink=1)
    D, nl = get_dims(model)
    n_scan = args.scan_pairs or (4000 if D < 2000 else 8000)
    if min(n_scan, args.n_pairs) <= D:
        raise SystemExit(f"pairs ({n_scan}, {args.n_pairs}) must exceed D = {D}")
    S = setup_data(tok)
    train = stratified_pairs(tok, args.per_frame, S["seed"] + 301, S["fr_ext"], S["ext_names"])
    heldout = stratified_pairs(tok, args.heldout_per_frame, S["seed"] + 201, S["fr_test"], S["test_names"])
    refL = dict(zip(SITES, (int(x) for x in args.reference.split(":"))))
    print(f"{args.model_name} | {nl} layers, D={D} | reference S2 L{refL['S2']} + END L{refL['END']} | train {len(train[0]['y'])} prompts from {len(S['ext_names'])} names, held-out {len(heldout[0]['y'])} from {len(S['test_names'])} new names", flush=True)
    R = {"model": args.model_name, "tag": args.tag, "D": D, "n_layers": nl, "reference": [refL["S2"], refL["END"]], "n_pairs": args.n_pairs, "pairs": {}}
    path = args.out_dir / f"{args.tag}.json"
    save = lambda: path.write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else list(o)))

    def cores(n, seed, want):
        """want: {layer: sites}. Yields (layer, site, factors), holding at most --mem-gb of states and gradients at once."""
        pr, _ = build_role_pairs(tok, n, np.random.default_rng(S["seed"] + seed), S["fr_ext"], ent(S["ext_names"]), p_noun=0.0, p_pronoun=0.0)
        layers, per = list(want), max(1, int(args.mem_gb * 1e9 // (32 * n * D)))
        for i in range(0, len(layers), per):
            chunk = layers[i:i + per]
            H, G = capture(model, pr, chunk, n, args.grad_batch_size, dev)
            for l in chunk:
                for t in SITES:
                    h, g = H[l].pop(t), G[l].pop(t)
                    if t in want[l]:
                        yield l, t, role_factors(h, g, dev)
                    del h, g
            del H, G

    def score(data, sites=(), swap=None):
        """swap: (layer, column, Q) for a partner swap at one site."""
        d, pool = data
        hook = get_blocks(model)[swap[0] - 1].register_forward_hook(SwapHook(swap[2], d, swap[1], dev)) if swap else None
        try:
            return evaluate(model, d, pool, list(sites), args.batch_size, dev)
        finally:
            if hook:
                hook.remove()

    # ---- sweep
    mid = max(16, int(0.025 * D))
    rows, ref, size = {"S2": {}, "END": {}}, {}, {}

    def remove_with_reference(t, l, f, r):
        o = "END" if t == "S2" else "S2"
        return score(train, [(l, COL[t], orth(f["Tn"][:, :r]), f["mu"], "remove"), (refL[o], COL[o], orth(ref[o]["Tn"][:, :r]), ref[o]["mu"], "remove")])["auc"]

    def sweep(want):
        pending = []
        for l, t, f in cores(n_scan, 204, want):
            rows[t][l] = {"flip": score(train, [(l, COL[t], orth(f["Tn"][:, :mid]), f["mu"], "flip")])["auc"]}
            if l == refL[t]:
                ref[t] = f
            if rows[t][l]["flip"] <= CONTROL_MAX or l == refL[t]:
                pending.append((l, t, f))
            if len(ref) == 2:
                if not size:
                    hit = next((r for r in LADDER[1:9] if remove_with_reference("S2", refL["S2"], ref["S2"], r) <= REMOVE_MAX), 512)
                    size.update(probe=max(2, int(0.75 * hit)), check=2 * hit)
                for l_, t_, f_ in pending:
                    rows[t_][l_].update(remove=remove_with_reference(t_, l_, f_, size["probe"]), check=remove_with_reference(t_, l_, f_, size["check"]))
                pending.clear()

    def lowest(t):
        ok = [l for l, v in rows[t].items() if "remove" in v and v["flip"] <= CONTROL_MAX and v["check"] >= FLOOR]
        return min(ok, key=lambda l: rows[t][l]["remove"]) if ok else None

    want = {refL["S2"]: {"S2"}}
    want.setdefault(refL["END"], set()).add("END")
    for t in SITES:
        for l in range(1, nl, args.stride):
            want.setdefault(l, set()).add(t)
    sweep(want)
    if args.stride > 1:
        near = {}
        for t in SITES:
            b = lowest(t)
            for l in () if b is None else (b - 1, b + 1):
                if 1 <= l < nl and l not in rows[t]:
                    near.setdefault(l, set()).add(t)
        if near:
            sweep(near)

    pick = lambda t: refL[t] if lowest(t) is None else lowest(t)
    bS, bE = pick("S2"), pick("END")
    if bS >= bE:
        bE = refL["END"]
    R["sweep"] = {"flip_rank": mid, "removal_size": size["probe"], "check_size": size["check"], "n_pairs": n_scan, "S2": rows["S2"], "END": rows["END"], "best": [bS, bE]}
    print(f"\nSWEEP: flip at rank {mid}; removal at size {size['probe']} (check at {size['check']}), other site at its reference layer ({time.time() - t0:.0f}s)")
    for t in SITES:
        print(f"  {t} flip    " + " ".join(f"L{l}:{rows[t][l]['flip']:.2f}" for l in sorted(rows[t])))
        print(f"  {t} removal " + " ".join(f"L{l}:{rows[t][l]['remove']:.3f}" for l in sorted(rows[t]) if "remove" in rows[t][l]))
    print(f"  best layers: S2 L{bS}, END L{bE}", flush=True)
    save()
    ref.clear()

    # ---- final cores
    plan = [(refL["S2"], refL["END"])] + ([(bS, bE)] if (bS, bE) != (refL["S2"], refL["END"]) else [])
    want = {}
    for a, b in plan:
        want.setdefault(a, set()).add("S2")
        want.setdefault(b, set()).add("END")
    F = {}
    for l, t, f in cores(args.n_pairs, 205, want):
        F[t, l] = f
        np.savez(args.out_dir / f"{args.tag}_{t}_L{l}.npz", **{f"{VAR}_{t}_{k}": (f[k][:, :1024] if k in ("Tn", "Rn") else f[k]) for k in ("Tn", "Rn", "mu", "s", "Rg")}, **{f"{VAR}_{t}_layer": np.array(l)})
    print(f"cores from {args.n_pairs} pairs ({time.time() - t0:.0f}s)", flush=True)
    clean = {"train": score(train)["auc"], "held-out": score(heldout)["auc"]}
    R["clean"] = clean
    floor_keep = clean["train"] - KEEP_TOL

    class Scores:
        """Scores of one layer pair on one prompt set; nothing is measured twice."""

        def __init__(self, lS, lE, data):
            self.at = {"S2": (lS, F["S2", lS]), "END": (lE, F["END", lE])}
            self.data, self.memo = data, {}

        def spec(self, t, r, mode):
            l, f = self.at[t]
            return l, COL[t], orth(f["Tn"][:, :r]), f["mu"], mode

        def both(self, mode, a, b):
            """Keep or remove at both sites."""
            key = (mode, a, b)
            if key not in self.memo:
                self.memo[key] = score(self.data, [self.spec("S2", a, mode), self.spec("END", b, mode)])
            return self.memo[key]["auc"]

        def one(self, t, r, mode):
            """Keep, remove, flip or swap at one site."""
            key = (t, r, mode)
            if key not in self.memo:
                l, f = self.at[t]
                self.memo[key] = score(self.data, swap=(l, COL[t], orth(f["Tn"][:, :r]))) if mode == "swap" else score(self.data, [self.spec(t, r, mode)])
            return self.memo[key]["auc"]

    def scan(lS, lE):
        T, H = Scores(lS, lE, train), Scores(lS, lE, heldout)
        k = min(1024, F["S2", lS]["Tn"].shape[1], F["END", lE]["Tn"].shape[1])
        tiers = [r for r in LADDER if r <= min(k, 256)]
        control = {t: {m: {r: T.one(t, r, m) for r in tiers} for m in ("flip", "swap")} for t in SITES}

        def bisect(passes, lo, hi):
            while hi - lo > res(hi):
                c = max(1, (lo + hi) // 2)
                lo, hi = (lo, c) if passes(c) else (c, hi)
            return hi

        def onset(t, m):
            """Ranks where control holds: from the first passing rank to the last passing grid rank."""
            ok = [r for r in tiers if control[t][m][r] <= CONTROL_MAX]
            if not ok:
                return None
            i = tiers.index(ok[0])
            return bisect(lambda r: T.one(t, r, m) <= CONTROL_MAX, tiers[i - 1] if i else 0, ok[0]), ok[-1]
        on = {t: {m: onset(t, m) for m in ("flip", "swap")} for t in SITES}
        within = lambda v, r: v is not None and v[0] <= r <= v[1]

        def sufficient(t, m):
            """Smallest rank at which this site alone keeps and controls."""
            if on[t][m] is None:
                return None
            lo, hi = on[t][m]
            return next((r for r in [lo] + [g for g in GRID if lo < g <= hi] if T.one(t, r, "keep") >= floor_keep), None)

        def search(m, where):
            """Smallest pair of ranks passing remove and control, then keep. Equal ranks set the total to beat; S2 ranks that can't beat it are dropped after one test; END is bisected for the rest."""
            ctrl = lambda a, b: all(within(on[t][m], x) for t, x in (("S2", a), ("END", b)) if t in where)
            passing = []

            def note(a, b):
                if ctrl(a, b) and T.both("remove", a, b) <= REMOVE_MAX:
                    passing.append((a, b)); return True
                return False
            anchors = [r for r in LADDER[1:] if r <= k]
            hit = next((r for r in anchors if note(r, r)), None)
            if hit is None:
                return None, []
            i = anchors.index(hit)
            rd = bisect(lambda r: note(r, r), anchors[i - 1] if i else 0, hit)
            best = 2 * rd

            def try_a(a):
                nonlocal best
                ceil_ = min(best - a - 1, k)
                if ceil_ < 1 or not note(a, ceil_):
                    return
                best = min(best, a + bisect(lambda b: note(a, b), 0, ceil_))
            for a in sorted((r for r in GRID if r < best), key=lambda r: abs(r - rd / 2)):
                if a < best:
                    try_a(a)
            if passing:
                a0 = min(passing, key=sum)[0]
                for da in (-4, -2, -1, 1, 2, 4):
                    if a0 + da >= 1 and abs(da) <= max(2, 2 * res(a0)):
                        try_a(a0 + da)
            ordered = sorted(set(passing), key=sum)
            return next((q for q in ordered if T.both("keep", *q) >= floor_keep), None), ordered

        def test(a, b):
            out = {}
            for name, P in (("train", T), ("held-out", H)):
                v = {"keep": P.both("keep", a, b), "remove": P.both("remove", a, b), "remove_S2": P.one("S2", a, "remove"), "remove_END": P.one("END", b, "remove"),
                     "flip_S2": P.one("S2", a, "flip"), "flip_END": P.one("END", b, "flip"), "swap_S2": P.one("S2", a, "swap"), "swap_END": P.one("END", b, "swap")}
                y, m = P.data[0]["y"], P.memo["remove", a, b]
                v.update(top_answer=m["top_io"], top_subject=m["top_s"], remove_ci=auc_ci(v["remove"], int((y == 1).sum()), int((y == 0).sum())))
                out[name] = v
            return out

        def one_site(t):
            """Smallest core at this site alone that removes, with its keep and control."""
            ladder = [r for r in LADDER if r <= k]
            hit = next((r for r in ladder if T.one(t, r, "remove") <= REMOVE_MAX), None)
            if hit is None:
                return {"rank": None, "remove": T.one(t, ladder[-1], "remove"), "at": ladder[-1], "passes": False}
            i = ladder.index(hit)
            r = bisect(lambda x: T.one(t, x, "remove") <= REMOVE_MAX, ladder[i - 1] if i else 0, hit)
            v = {"rank": r, "remove": T.one(t, r, "remove"), "keep": T.one(t, r, "keep"), "flip": T.one(t, r, "flip"), "swap": T.one(t, r, "swap"), "heldout_remove": H.one(t, r, "remove")}
            v["passes"] = v["keep"] >= floor_keep and min(v["flip"], v["swap"]) <= CONTROL_MAX
            return v

        show = lambda v: (f"keep {v['keep']:.3f}, remove {v['remove']:.3f} (S2 alone {v['remove_S2']:.3f}, END alone {v['remove_END']:.3f}; top token answer {v['top_answer']:.2f}, subject {v['top_subject']:.2f}), "
                          f"flip S2 {v['flip_S2']:.3f}, END {v['flip_END']:.3f}, swap S2 {v['swap_S2']:.3f}, END {v['swap_END']:.3f}")
        print(f"\nSCAN: S2 L{lS} ({lS / nl:.2f}) + END L{lE} ({lE / nl:.2f})\n  control by rank: " + "; ".join(f"{t} {m} " + " ".join(f"{r}:{control[t][m][r]:.2f}" for r in tiers) for t in SITES for m in ("flip", "swap")), flush=True)
        suff = {m: {t: sufficient(t, m) for t in SITES} for m in ("flip", "swap")}
        print(f"  sufficient (each site alone keeps and controls): by flip S2 {suff['flip']['S2']} + END {suff['flip']['END']}, by swap S2 {suff['swap']['S2']} + END {suff['swap']['END']}", flush=True)
        core, tested = {}, {}
        for name, (m, where) in RULES.items():
            c, ordered = search(m, where)
            if c is None:
                core[name] = None; print(f"  smallest core, {name}: none", flush=True); continue
            new = c not in tested
            if new:
                tested[c] = test(*c)
            core[name] = {"ranks": list(c), "near": [q for q in ordered if sum(q) <= 1.2 * sum(c)], **tested[c]}
            print(f"  smallest core, {name}: S2 {c[0]} + END {c[1]} = {sum(c)} ({100 * sum(c) / D:.1f}% of D) ({time.time() - t0:.0f}s)" + (f"\n    train    {show(tested[c]['train'])}\n    held-out {show(tested[c]['held-out'])}" if new else ""), flush=True)
        single = {t: one_site(t) for t in SITES}
        for t, v in single.items():
            print(f"  {t} alone: " + (f"removal not reached ({v['remove']:.3f} at rank {v['at']})" if v["rank"] is None else f"rank {v['rank']}: remove {v['remove']:.3f}, keep {v['keep']:.3f}, flip {v['flip']:.3f}, swap {v['swap']:.3f}, held-out remove {v['heldout_remove']:.3f}" + (" PASSES AS ONE SITE" if v["passes"] else "")), flush=True)
        curve = {r: {"keep": T.both("keep", r, r), "remove": T.both("remove", r, r), "heldout_remove": H.both("remove", r, r)} for r in CURVE if r <= k}
        print("  rank: keep, remove, held-out remove (same rank at both sites): " + "; ".join(f"{r}: {v['keep']:.3f} {v['remove']:.3f} {v['heldout_remove']:.3f}" for r, v in curve.items()), flush=True)
        return {"layers": [lS, lE], "control": control, "control_range": on, "sufficient": suff, "core": core, "one_site": single, "curve": curve,
                "evaluated": [[*q, v["auc"]] for q, v in T.memo.items()], "evaluations": len(T.memo) + len(H.memo)}

    for a, b in plan:
        R["pairs"][f"{a}:{b}"] = scan(a, b)
        save()
    R["finished"] = True; save()
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
