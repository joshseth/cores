#!/usr/bin/env python3
"""
controls.py: at one saved core (golden.py): PCA of the activations, unpaired, and a random subspace at the core's ranks, against ACE, on the train prompts: keep and remove at both
sites, keep and flip at each site alone. PCA comes from the states of the same prompts as the core, one row per prompt, mean subtracted.

  PYTHONPATH=src python -m cores.ioi.controls --model-name gpt2 --tag gpt2 --golden-dir results/ioi/golden/gpt2 --layers 6:9 --ranks 14:2 \
      --out results/ioi/model/gpt2/gpt2_controls.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cores.ioi.common import COL, SEED, SITES, build_role_pairs, ent, evaluate, load_model, orth, seed_all, setup_data, stratified_pairs


def states(model, d, lay, n_prompts, bs, device):
    """The state at each site's token, one row per prompt."""
    H = {t: [] for t in SITES}
    with torch.no_grad():
        for st in range(0, n_prompts, bs):
            b = np.arange(st, min(st + bs, n_prompts))
            hs = model(input_ids=d["ids"][b].to(device), attention_mask=d["att"][b].to(device), output_hidden_states=True, use_cache=False).hidden_states
            s2, last, r = np.asarray(d["pos"][b, 2]), np.asarray(d["last"][b]), np.arange(len(b))
            H["S2"].append(hs[lay["S2"]].float().cpu().numpy()[r, s2]); H["END"].append(hs[lay["END"]].float().cpu().numpy()[r, last])
    return {t: np.concatenate(v) for t, v in H.items()}


def pca_dirs(A, r, device):
    """First r principal directions of the rows of A, mean subtracted."""
    dev = device if device.type == "cuda" else torch.device("cpu")
    A = torch.as_tensor(A, dtype=torch.float32, device=dev)
    R = torch.linalg.qr((A - A.mean(0)) / np.sqrt(len(A)), mode="r")[1]
    return torch.linalg.svd(R)[2][:r].T.float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--golden-dir", type=Path, required=True)
    p.add_argument("--layers", required=True, help="S2:END")
    p.add_argument("--ranks", required=True, help="S2:END")
    p.add_argument("--n-pairs", type=int, default=12000)
    p.add_argument("--per-frame", type=int, default=3)
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
    train = stratified_pairs(tok, args.per_frame, S["seed"] + 301, S["fr_ext"], S["ext_names"])
    auc = lambda sites: evaluate(model, train[0], train[1], sites, args.batch_size, dev)["auc"]
    clean = auc([])

    pairs, _ = build_role_pairs(tok, args.n_pairs, np.random.default_rng(S["seed"] + 205), S["fr_ext"], ent(S["ext_names"]), p_noun=0.0, p_pronoun=0.0)
    H = states(model, pairs, lay, 2 * args.n_pairs, args.batch_size, dev)
    bases, mu, rng = {}, {}, np.random.default_rng(S["seed"] + 701)
    for t in SITES:
        z = np.load(args.golden_dir / f"{args.tag}_{t}_L{lay[t]}.npz", allow_pickle=False)
        mu[t] = z[f"orth_write_{t}_mu"]
        bases[t] = {"ACE": orth(z[f"orth_write_{t}_Tn"][:, :rank[t]]), "PCA": pca_dirs(H[t], rank[t], dev), "random": orth(rng.standard_normal((H[t].shape[1], rank[t])))}
    del H
    spec = lambda t, b, m: (lay[t], COL[t], bases[t][b], mu[t], m)
    controls = {}
    for b in bases["S2"]:
        controls[b] = {"keep": auc([spec("S2", b, "keep"), spec("END", b, "keep")]), "remove": auc([spec("S2", b, "remove"), spec("END", b, "remove")]),
                       **{f"{m}_{t}": auc([spec(t, b, m)]) for t in SITES for m in ("keep", "flip")}}
        v = controls[b]
        print(f"  {b:<6} keep {v['keep']:.3f}, remove {v['remove']:.3f} | S2 alone keep {v['keep_S2']:.3f}, flip {v['flip_S2']:.3f} | END alone keep {v['keep_END']:.3f}, flip {v['flip_END']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model_name, "layers": [lay["S2"], lay["END"]], "ranks": [rank["S2"], rank["END"]], "clean": clean, "controls": controls}, indent=1))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
