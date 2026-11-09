#!/usr/bin/env python3
"""
wang_controls.py: the core against other directions on the prompts of Wang et al. (2022): keep and remove at both sites, flip at each site alone.

ACE          the saved core
PCA          the activations at each site, unpaired, mean subtracted, at the core's ranks
random       a random subspace at the core's ranks
mean diff    the mean of the role-swap pair differences at each site, rank 1
unembedding  each prompt's own W_U[A] - W_U[B], at both sites, rank 1

PCA and the mean difference come from the same pairs as the core. Forward passes only.

  PYTHONPATH=src python -m cores.ioi.wang_controls --model-name gpt2 --tag gpt2 --golden-dir results/ioi/golden/gpt2 --layers 6:9 --ranks 14:2 \
      --out results/ioi/wang_controls_gpt2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cores.ioi.common import COL, SEED, SITES, build_role_pairs, ent, evaluate, get_blocks, load_model, orth, seed_all, setup_data, stratified_pairs
from cores.ioi.controls import pca_dirs, states
from cores.ioi.wang import NAMES, frames


class PromptDirHook:
    """Keep, remove or flip along one unit direction per prompt. Batches come in evaluate()'s order."""

    def __init__(self, U, mu, data, col, mode, device):
        self.U, self.mu = U.to(device), torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.pos, self.mode, self.st = torch.as_tensor(data["pos"][:, col], device=device), mode, 0

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = h.shape[0]; b = torch.arange(n, device=h.device)
        p, u = self.pos[self.st:self.st + n], self.U[self.st:self.st + n]; self.st += n
        v = h[b, p].float() - self.mu; c = (v * u).sum(1, keepdim=True)
        new = self.mu + {"keep": c * u, "remove": v - c * u, "flip": v - 2 * c * u}[self.mode]
        h = h.clone(); h[b, p] = new.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--golden-dir", type=Path, required=True)
    p.add_argument("--layers", required=True, help="S2:END")
    p.add_argument("--ranks", required=True, help="S2:END")
    p.add_argument("--n-pairs", type=int, default=12000)
    p.add_argument("--per-template", type=int, default=34)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.batch_size -= args.batch_size % 2
    dev = torch.device(args.device); seed_all(SEED)
    tok, model = load_model(args.model_name, args.dtype, args.attn_impl, dev, sink=1)
    S = setup_data(tok)
    lay = dict(zip(SITES, (int(x) for x in args.layers.split(":")))); rank = dict(zip(SITES, (int(x) for x in args.ranks.split(":"))))

    pairs, _ = build_role_pairs(tok, args.n_pairs, np.random.default_rng(S["seed"] + 205), S["fr_ext"], ent(S["ext_names"]), p_noun=0.0, p_pronoun=0.0)
    H = states(model, pairs, lay, 2 * args.n_pairs, args.batch_size, dev)
    bases, mu, rng = {}, {}, np.random.default_rng(S["seed"] + 701)
    for t in SITES:
        z = np.load(args.golden_dir / f"{args.tag}_{t}_L{lay[t]}.npz", allow_pickle=False)
        mu[t] = z[f"orth_write_{t}_mu"]
        d = H[t][0::2].mean(0) - H[t][1::2].mean(0)
        bases[t] = {"ACE": orth(z[f"orth_write_{t}_Tn"][:, :rank[t]]), "PCA": pca_dirs(H[t], rank[t], dev),
                    "random": orth(rng.standard_normal((H[t].shape[1], rank[t]))), "mean diff": (d / np.linalg.norm(d))[:, None].astype(np.float32)}
    del H

    data = stratified_pairs(tok, 1, 601, frames(np.random.default_rng(601), args.per_template), NAMES)
    d, pool = data
    W = model.get_output_embeddings().weight.detach().float()
    U = W[torch.as_tensor(d["f"], device=W.device)] - W[torch.as_tensor(d["s"], device=W.device)]
    U = (U / U.norm(dim=1, keepdim=True)).cpu()
    clean = evaluate(model, d, pool, [], args.batch_size, dev)["auc"]

    def directions(mode, where):
        hooks = [get_blocks(model)[lay[t] - 1].register_forward_hook(PromptDirHook(U, mu[t], d, COL[t], mode, dev)) for t in where]
        try:
            return evaluate(model, d, pool, [], args.batch_size, dev)["auc"]
        finally:
            for h in hooks:
                h.remove()

    fixed = lambda b, mode, where: evaluate(model, d, pool, [(lay[t], COL[t], bases[t][b], mu[t], mode) for t in where], args.batch_size, dev)["auc"]
    R = {"model": args.model_name, "layers": [lay["S2"], lay["END"]], "ranks": [rank["S2"], rank["END"]], "prompts": len(d["y"]), "clean": clean, "rows": {}}
    for b, f in [(b, (lambda m, w, b=b: fixed(b, m, w))) for b in bases["S2"]] + [("unembedding", directions)]:
        R["rows"][b] = {"keep": f("keep", SITES), "remove": f("remove", SITES), "flip_S2": f("flip", ("S2",)), "flip_END": f("flip", ("END",))}
        v = R["rows"][b]
        print(f"  {b:<12} keep {v['keep']:.3f}, remove {v['remove']:.3f}, flip S2 {v['flip_S2']:.3f}, END {v['flip_END']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(R, indent=1))
    print(f"clean {clean:.3f}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
