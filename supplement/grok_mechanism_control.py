#!/usr/bin/env python3
"""
grok_mechanism_control.py: the grokking-delay law with the mechanism measured in every run.

Changing the modulus p also changes the training-set size (p^2 / 2 pairs), and data size can change the solution itself
(which Fourier frequencies, i.e. irreps of Z_p, the network uses). So data size is not partialled out; instead every run
records its mechanism, and the delay law can be read through it:
  key frequencies   per frequency k = 1..(p-1)/2, the power of the token embedding (and of the unembedding) along the
                    Fourier mode k over the token axis; the key set = the fewest frequencies holding 90% of the
                    non-constant power; also the share in the top 1, 3 and 5 frequencies and the top frequencies themselves
Runs (same model, optimizer, weight decay 1.0, full batch and thresholds as grok_sweep_two_layer.py; 3 seeds each;
training fraction 0.5 throughout):
  base   p = 17, 19, 23, 29, 31, 43, 53, 61, 79, 89, 101 (the modulus sweep itself)
  C      k = 2..4 independent copies of Z_31 (disjoint tokens) in one network: modulus and fraction fixed, distinct
         training pairs 480 k (the modulus sweep's sizes at p = 31, 43, 53, 61). If data size sets the delay, it falls
         toward the law at the larger p; if the modulus sets it, it stays at the value for p = 31. Frequencies logged per copy.
  D      Z_31 with a = 2, 3 token names per number (neighboring tokens collapsed onto one number), split over number
         pairs: group and fraction fixed, training examples 480 a^2, no new arithmetic information. With C this separates
         raw example count (D) from new information (C).
Max 30,000 full-batch steps per run; a run that has not grokked by then is recorded with "drift" (the delay) set to None.
Writes after every run (resumable). plot_grok_data_size.py draws the paper's figure from all three arms.

  python supplement/grok_mechanism_control.py --out supplement/grok_mechanism_control/results.json
  python supplement/grok_mechanism_control.py --quick ...     (code check: one short run)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import grok_sweep_two_layer as gs  # noqa: E402

CAP = {}


class Captured(gs.TwoLayerTransformerLM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        CAP["model"] = self


gs.TwoLayerTransformerLM = Captured
OMEGA, PCRIT = 4052.337393882202, 20.620330574611494


def fourier(W, p):
    """W: (p, d) rows indexed by residue. Power per frequency k = 1..(p-1)/2 (k and p - k combined)."""
    W = W - W.mean(0, keepdims=True)
    F = np.fft.fft(W, axis=0)                                  # (p, d)
    pw = (np.abs(F) ** 2).sum(1)
    K = (p - 1) // 2
    power = np.array([pw[k] + pw[p - k] for k in range(1, K + 1)])
    frac = power / power.sum()
    order = np.argsort(frac)[::-1]
    n90 = int(np.searchsorted(np.cumsum(frac[order]), 0.9) + 1)
    return {"n_key_90": n90, "top1": float(frac[order[:1]].sum()), "top3": float(frac[order[:3]].sum()),
            "top5": float(frac[order[:5]].sum()), "top_freqs": [int(k + 1) for k in order[:8]],
            "power_frac": [float(x) for x in frac]}


def train_on(X_tr, Y_tr, X_te, Y_te, cid, n_groups, vocab, seed, device, max_epochs, omega=1.0, lr=1e-3,
             patience_epochs=100, thr=0.99):
    """Full-batch training with the same model, optimizer and thresholds as grok_sweep_two_layer.train_run
    (memorization: train acc >= thr; grok: overall test acc >= thr); also the grok step of each group of test items."""
    torch.manual_seed(seed)
    model = Captured(vocab_size=vocab, d_model=128, n_head=4, d_ff=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=omega)
    loss_fn = torch.nn.CrossEntropyLoss()
    mem = grok = None; grok_g = [None] * n_groups; step = 0
    while step < max_epochs:
        model.train(); opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X_tr)[:, -1, :], Y_tr[:, -1]); loss.backward(); opt.step(); step += 1
        model.eval()
        with torch.no_grad():
            tr_acc = (model(X_tr)[:, -1, :].argmax(-1) == Y_tr[:, -1]).float().mean().item()
            ok = (model(X_te)[:, -1, :].argmax(-1) == Y_te[:, -1]).float()
            te_acc = ok.mean().item()
            for g in range(n_groups):
                if grok_g[g] is None and ok[cid == g].mean().item() >= thr:
                    grok_g[g] = step
        if mem is None and tr_acc >= thr:
            mem = step
        if grok is None and te_acc >= thr:
            grok = step
        if grok is not None and step > grok + patience_epochs:
            break
    return {"n_train": len(X_tr), "tau_mem_step": mem, "tau_grok_step": grok, "z_at_grok": None, "grok_per_group": grok_g,
            "tau_drift_step": (grok - mem) if (mem is not None and grok is not None and grok > mem) else None}


def train_copies(p, k, seed, device, max_epochs):
    """k independent copies of Z_p addition in one network: copy c uses tokens c*p .. c*p + p - 1, each copy split
    50/50 with its own permutation. Modulus and training fraction fixed; distinct training pairs = k * p^2 / 2, and each
    copy is new arithmetic information (k separate solutions to learn)."""
    tr, te = [], []
    for c in range(k):
        Xa, Ya, Xb, Yb = gs.build_data_tensors(p, device, frac=0.5, seed=42 + c)
        tr.append((Xa + c * p, Ya + c * p)); te.append((Xb + c * p, Yb + c * p))
    cid = torch.cat([torch.full((len(x),), c, device=device) for c, (x, _) in enumerate(te)])
    return train_on(torch.cat([x for x, _ in tr]), torch.cat([y for _, y in tr]), torch.cat([x for x, _ in te]),
                    torch.cat([y for _, y in te]), cid, k, k * p, seed, device, max_epochs)


def train_alias(m, a, seed, device, max_epochs):
    """Z_m addition where each number has a token names (neighboring tokens r*a .. r*a + a - 1 all mean r: the
    'collapsed neighbors' task). Output = the first name of (r1 + r2) mod m. The 50/50 split is over number pairs
    (as in grok_sweep_two_layer), and every alias combination of a number pair goes to the same side, so the group and
    its coverage are fixed while the distinct training examples grow by a^2 with no new arithmetic information."""
    Xa, Ya, Xb, Yb = gs.build_data_tensors(m, device, frac=0.5, seed=42)
    def expand(X, Y):
        i, j = torch.meshgrid(torch.arange(a, device=device), torch.arange(a, device=device), indexing="ij")
        i, j = i.flatten(), j.flatten()
        n = len(X)
        x0 = (X[:, 0:1] * a + i[None]).reshape(-1); x1 = (X[:, 1:2] * a + j[None]).reshape(-1)
        out = (Y[:, -1:] * a).expand(n, a * a).reshape(-1)
        return torch.stack([x0, x1], 1), torch.stack([x1, out], 1)
    X_tr, Y_tr = expand(Xa, Ya); X_te, Y_te = expand(Xb, Yb)
    return train_on(X_tr, Y_tr, X_te, Y_te, torch.zeros(len(X_te), dtype=torch.long, device=device), 1, a * m, seed,
                    device, max_epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "supplement" / "grok_mechanism_control" / "results.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", help="auto (cuda / mps / cpu) or cpu; the original sweep ran on cpu")
    ap.add_argument("--arms", default="", help="comma-separated subset of base,C_copies,D_alias")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    device = torch.device("cpu") if a.device == "cpu" else gs.get_device()[0]
    print("device", device, flush=True)
    settings = [("base", p, None) for p in (17, 19, 23, 29, 31, 43, 53, 61, 79, 89, 101)] + \
               [("C_copies", 31, k) for k in (2, 3, 4)] + [("D_alias", 31, a_) for a_ in (2, 3)]
    if a.arms:
        settings = [x for x in settings if x[0] in a.arms.split(",")]
    max_epochs, seeds = 30000, a.seeds
    if a.quick:
        settings, max_epochs, seeds = [("base", 31, None), ("C_copies", 31, 2), ("D_alias", 31, 2)], 300, 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(a.out.read_text())["runs"] if a.out.exists() and not a.quick else []
    done = {(r["arm"], r["p"], r["n_train"], r["seed"]) for r in rows}
    for arm, p, n in settings:
        copies = n if arm == "C_copies" else 1
        alias = n if arm == "D_alias" else 1
        frac = 0.5
        n_eff = copies * alias * alias * int(p * p * frac)
        for s in range(seeds):
            if (arm, p, n_eff, 42 + s) in done:
                continue
            t0 = time.time()
            if arm == "C_copies":
                out = train_copies(p, copies, 42 + s, device, max_epochs)
            elif arm == "D_alias":
                out = train_alias(p, alias, 42 + s, device, max_epochs)
            else:
                out = gs.train_run(p=p, device=device, d_model=128, n_head=4, d_ff=512, lr=1e-3, omega=1.0, frac=frac,
                                   batch_size=n_eff + 1, max_epochs=max_epochs, eval_every_steps=1, grok_thresh=0.99,
                                   mem_thresh=0.99, seed=42 + s, compile_mode="off")
            m = CAP["model"]
            with torch.no_grad():
                E = m.token_emb.weight.detach().float().cpu().numpy(); U = m.lm_head.weight.detach().float().cpu().numpy()
                if alias > 1:                                       # one row per number: mean of its names; output = first name
                    E = E[:alias * p].reshape(p, alias, -1).mean(1); U = U[:alias * p:alias]
                per = [(fourier(E[c * p:(c + 1) * p], p), fourier(U[c * p:(c + 1) * p], p)) for c in range(copies)]
                emb, unemb = per[0]
            r = {"arm": arm, "p": p, "n_train": out["n_train"], "frac": frac, "seed": 42 + s, "mem": out["tau_mem_step"],
                 "grok": out["tau_grok_step"], "drift": out["tau_drift_step"], "z_at_grok": out["z_at_grok"],
                 "ode_prediction_from_p": -OMEGA * math.log(1 - PCRIT / p) if p > PCRIT else None, "embedding": emb, "unembedding": unemb,
                 "copies": copies, "aliases": alias, "grok_per_copy": out.get("grok_per_group") if copies > 1 else None,
                 "per_copy": [{"embedding": e, "unembedding": u} for e, u in per] if copies > 1 else None,
                 "seconds": time.time() - t0}
            rows.append(r)
            print(f"{arm} p={p} n={out['n_train']} seed={42 + s}: mem={r['mem']} grok={r['grok']} drift={r['drift']} "
                  f"(law at this p {r['ode_prediction_from_p'] or float('nan'):.0f}) | key freqs (90%) emb {emb['n_key_90']}, unemb "
                  f"{unemb['n_key_90']}, top {emb['top_freqs'][:4]} | {r['seconds']:.0f}s", flush=True)
            a.out.write_text(json.dumps({"omega_constant": OMEGA, "pcrit": PCRIT, "runs": rows}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
