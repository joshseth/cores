#!/usr/bin/env python3
"""
wang.py: the saved cores (golden.py) scored on the prompts of Wang et al. (2022): their 15 templates, places, objects and names,
as role-swap pairs. Forward passes only. Writes <dir>/<tag>_wang.json.

  PYTHONPATH=src python -m cores.ioi.wang --model-name gpt2 --tag gpt2 --dir results/ioi/golden/gpt2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cores.ioi.common import COL, CURVE, SITES, SwapHook, auc_ci, evaluate, get_blocks, load_model, orth, stratified_pairs

BABA_TEMPLATES = ['Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]', 'Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]', 'Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]', 'Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]', 'Then, [B] and [A] had a long argument, and afterwards [B] said to [A]', 'After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]', 'When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]', 'When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]', 'While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]', 'While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]', 'After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]', 'Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]', 'Then, [B] and [A] had a long argument. Afterwards [B] said to [A]', 'The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]', 'Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]']
NAMES = ['Michael', 'Christopher', 'Jessica', 'Matthew', 'Ashley', 'Jennifer', 'Joshua', 'Amanda', 'Daniel', 'David', 'James', 'Robert', 'John', 'Joseph', 'Andrew', 'Ryan', 'Brandon', 'Jason', 'Justin', 'Sarah', 'William', 'Jonathan', 'Stephanie', 'Brian', 'Nicole', 'Nicholas', 'Anthony', 'Heather', 'Eric', 'Elizabeth', 'Adam', 'Megan', 'Melissa', 'Kevin', 'Steven', 'Thomas', 'Timothy', 'Christina', 'Kyle', 'Rachel', 'Laura', 'Lauren', 'Amber', 'Brittany', 'Danielle', 'Richard', 'Kimberly', 'Jeffrey', 'Amy', 'Crystal', 'Michelle', 'Tiffany', 'Jeremy', 'Benjamin', 'Mark', 'Emily', 'Aaron', 'Charles', 'Rebecca', 'Jacob', 'Stephen', 'Patrick', 'Sean', 'Erin', 'Jamie', 'Kelly', 'Samantha', 'Nathan', 'Sara', 'Dustin', 'Paul', 'Angela', 'Tyler', 'Scott', 'Katherine', 'Andrea', 'Gregory', 'Erica', 'Mary', 'Travis', 'Lisa', 'Kenneth', 'Bryan', 'Lindsey', 'Kristen', 'Jose', 'Alexander', 'Jesse', 'Katie', 'Lindsay', 'Shannon', 'Vanessa', 'Courtney', 'Christine', 'Alicia', 'Cody', 'Allison', 'Bradley', 'Samuel']
PLACES = ['store', 'garden', 'restaurant', 'school', 'hospital', 'office', 'house', 'station']
OBJECTS = ['ring', 'kiss', 'bone', 'basketball', 'computer', 'necklace', 'drink', 'snack']


def frames(rng, per_template):
    """Each template with the answer dropped and the first mention, second mention and repeated name as {A}, {B}, {S}; random places and objects."""
    out = []
    for t in BABA_TEMPLATES:
        f = t[:t.rindex(" [A]")].replace("[B]", "{A}", 1).replace("[A]", "{B}", 1).replace("[B]", "{S}", 1)
        out += [f.replace("[PLACE]", str(rng.choice(PLACES))).replace("[OBJECT]", str(rng.choice(OBJECTS))) for _ in range(per_template)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dir", type=Path, required=True, help="the model's golden results")
    p.add_argument("--per-template", type=int, default=34)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--attn-impl", default=None)
    args = p.parse_args()
    args.batch_size -= args.batch_size % 2
    golden = json.loads((args.dir / f"{args.tag}.json").read_text())
    dev = torch.device(args.device)
    tok, model = load_model(args.model_name, args.dtype, args.attn_impl, dev, sink=1)
    data = stratified_pairs(tok, 1, 601, frames(np.random.default_rng(601), args.per_template), NAMES)
    y = data[0]["y"]; n1, n0 = int((y == 1).sum()), int((y == 0).sum())

    def score(sites=(), swap=None):
        hook = get_blocks(model)[swap[0] - 1].register_forward_hook(SwapHook(swap[2], data[0], swap[1], dev)) if swap else None
        try:
            return evaluate(model, data[0], data[1], list(sites), args.batch_size, dev)
        finally:
            if hook:
                hook.remove()
    clean = score()
    R = {"model": args.model_name, "prompts": len(y), "names": len(np.unique(data[1])), "clean": clean["auc"], "pairs": {}}
    print(f"{args.model_name} | Wang et al. prompts: {len(y)} prompts, 15 templates, {len(np.unique(data[1]))} names | clean {clean['auc']:.3f}", flush=True)
    path = args.dir / f"{args.tag}_wang.json"
    for q, v in golden["pairs"].items():
        at = {t: (l, np.load(args.dir / f"{args.tag}_{t}_L{l}.npz", allow_pickle=False)) for t, l in zip(SITES, v["layers"])}
        core = lambda t, r: orth(at[t][1][f"orth_write_{t}_Tn"][:, :r])
        spec = lambda t, r, m: (at[t][0], COL[t], core(t, r), at[t][1][f"orth_write_{t}_mu"], m)
        one = lambda t, r, m: score(swap=(at[t][0], COL[t], core(t, r))) if m == "swap" else score([spec(t, r, m)])
        both = lambda m, a, b: score([spec("S2", a, m), spec("END", b, m)])
        tested, single = {}, {}
        for a, b in sorted({tuple(c["ranks"]) for c in v["core"].values() if c}):
            rm = both("remove", a, b)
            row = {"keep": both("keep", a, b)["auc"], "remove": rm["auc"], "remove_ci": auc_ci(rm["auc"], n1, n0), "top_answer": rm["top_io"], "top_subject": rm["top_s"],
                   **{f"{m}_{t}": one(t, r, m)["auc"] for t, r in zip(SITES, (a, b)) for m in ("remove", "flip", "swap")}}
            tested[f"{a}+{b}"] = row
            print(f"  layers {q}, S2 {a} + END {b}: keep {row['keep']:.3f}, remove {row['remove']:.3f} (S2 alone {row['remove_S2']:.3f}, END alone {row['remove_END']:.3f}; top token answer {row['top_answer']:.2f}, subject {row['top_subject']:.2f}), flip S2 {row['flip_S2']:.3f}, END {row['flip_END']:.3f}, swap S2 {row['swap_S2']:.3f}, END {row['swap_END']:.3f}", flush=True)
        for t, x in v["one_site"].items():
            if x["rank"]:
                single[t] = {"rank": x["rank"], **{m: one(t, x["rank"], m)["auc"] for m in ("keep", "remove", "flip", "swap")}}
                print(f"  layers {q}, {t} alone, rank {x['rank']}: keep {single[t]['keep']:.3f}, remove {single[t]['remove']:.3f}, flip {single[t]['flip']:.3f}, swap {single[t]['swap']:.3f}", flush=True)
        k = min(at["S2"][1]["orth_write_S2_Tn"].shape[1], at["END"][1]["orth_write_END_Tn"].shape[1])
        curve = {r: both("remove", r, r)["auc"] for r in CURVE if r <= k}
        print(f"  layers {q}, removal at the same rank at both sites: " + " ".join(f"{r}:{a:.3f}" for r, a in curve.items()), flush=True)
        R["pairs"][q] = {"cores": tested, "one_site": single, "curve": curve}
        path.write_text(json.dumps(R, indent=1))
    R["finished"] = True; path.write_text(json.dumps(R, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
