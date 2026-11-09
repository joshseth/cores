#!/usr/bin/env python3
r"""
model_dump.py: per-pair data for fitting reduced models of the IOI core offline (forward passes only).

Fit: extraction frames x extraction names; test / val: test / validation frames x 40 new names. For each set it saves, per
prompt (members of a pair adjacent; pair = same frame and names, only the repeated name differs):
  ids, pos (A, B, S2, END), last, f / s (answer / other name token), frame index, names AB per pair
  zS, zE, m              clean core coordinates at S2 and END, margin logit(f) - logit(s)
  lf, ls, lse            the two names' logits and the log-sum-exp over the vocabulary (for log-probabilities)
  nF                     norm of the last token's state entering the final norm (the readout's normalizer)
  hF                     that state itself (float16): with WU_names, norm_w / norm_b / norm_eps / norm_class and softcap,
                         the two names' logits can be recomputed offline under any normalization or clipping
  zE_sw, m_sw            after writing the partner's S2 core into S2 (the S2-core swap)
  m_pE                   after writing the partner's END core into END (the END-core swap)
  (every intervened run also saves its lf, ls, lse, nF and hF, e.g. lf_sw, hF_invE)
  zA, zB                 the two first mentions at the S2 layer, in S2 core coordinates
  hS, hE                 full residual states at (S2 layer, S2) and (END layer, END), float16
  nS, nE                 their norms (float32)
test and val sets only (v = the fit-set mean of s dz_S - dc_S, the only quantity fitted here):
  m_invS, m_invE, m_invSE   margins with the S2 core, the END core, or both inverted about the training mean (z -> -z;
                            for both, the END core's live state after the S2 inversion is inverted)
  m_injEq, m_injPtr, m_injNm  margins after adding s (v + dc_S), s v, or s dc_S to the S2 core (dc_S from isolated names)
and once: cS (each name alone after the start token, S2 layer, S2 core coordinates); hN_<k> (each name at the end of
neutral carrier k, S2 layer, full state, float16; carriers in `carriers`); the core bases, layers, ranks, v.
Nothing else is fitted here; the candidate models are fitted offline from this file.

  PYTHONPATH=src python -m cores.ioi.model_dump --model-name gpt2 --tag gpt2 --golden-dir results/ioi/golden/gpt2 --layers 6:9 --ranks 14:2 \
      --out results/ioi/model/gpt2/gpt2_dump.npz
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from cores.ioi.common import SEED, Site, build_role_pairs, core_basis, ent, get_blocks, get_device, load_model, seed_all, setup_data

CARRIERS = ["{n}", "The name was{n}", "Later that day,{n}", "I spoke with{n}", "This is a story about{n}"]   # name token keeps its leading space


def final_norm(model):
    """The norm module applied before the unembedding."""
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        m = model
        try:
            for a in path.split("."):
                m = getattr(m, a)
            return m
        except AttributeError:
            continue
    raise AttributeError("final norm not found")


def frame_index(tok, d, frames):
    """Frame of each prompt, recovered from its text; -1 if none matches."""
    pats = [re.compile("^" + re.escape(f).replace(re.escape("{A}"), r"\S+").replace(re.escape("{B}"), r"\S+")
                       .replace(re.escape("{S}"), r"\S+") + "$", re.IGNORECASE) for f in frames]
    out = []
    for i in range(len(d["y"])):
        t = tok.decode(d["ids"][i][: int(d["last"][i]) + 1], skip_special_tokens=True).strip()
        out.append(next((k for k, pt in enumerate(pats) if pt.match(t)), -1))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--tag", required=True)
    p.add_argument("--golden-dir", type=Path, required=True)
    p.add_argument("--layers", required=True, help="S2:END")
    p.add_argument("--ranks", required=True, help="S2:END")
    p.add_argument("--n-fit", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--attn-impl", default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lS, lE = (int(x) for x in args.layers.split(":")); rS, rE = (int(x) for x in args.ranks.split(":"))
    C = {t: np.load(args.golden_dir / f"{args.tag}_{t}_L{l}.npz", allow_pickle=False) for t, l in (("S2", lS), ("END", lE))}
    fac = {t: {k: C[t][f"orth_write_{t}_{k}"] for k in ("Tn", "Rn", "mu")} for t in ("S2", "END")}
    device = get_device()
    if device.type == "mps" and "gpt2" not in args.model_name:           # MPS corrupts large-model extraction; use CPU there
        device = torch.device("cpu")
    seed_all(SEED)
    tok, model = load_model(args.model_name, args.dtype, args.attn_impl, device, sink=1)
    S = setup_data(tok); rng = lambda s: np.random.default_rng(S["seed"] + s)
    sets = {"fit": (build_role_pairs(tok, args.n_fit, rng(405), S["fr_ext"], ent(S["ext_names"]), p_noun=0.0, p_pronoun=0.0), S["fr_ext"]),
            "test": (build_role_pairs(tok, args.n_test, rng(402), S["fr_test"], ent(S["test_names"]), p_noun=0.0, p_pronoun=0.0), S["fr_test"]),
            "val": (build_role_pairs(tok, args.n_test, rng(403), S["fr_val"], ent(S["test_names"]), p_noun=0.0, p_pronoun=0.0), S["fr_val"])}
    (QS, muS), (QE, muE) = core_basis(fac["S2"], rS, "orth_write"), core_basis(fac["END"], rE, "orth_write")

    ids_all = np.unique(np.concatenate([AB.ravel() for (_, AB), _ in sets.values()]))
    sink = [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id]
    cS = np.zeros((len(ids_all), rS), dtype=np.float32)
    with torch.no_grad():
        for st in range(0, len(ids_all), 64):
            ch = ids_all[st:st + 64]
            hs = model(input_ids=torch.tensor([sink + [int(i)] for i in ch], device=device), output_hidden_states=True, use_cache=False).hidden_states
            cS[st:st + len(ch)] = (hs[lS][:, -1].float().cpu().numpy() - muS) @ QS
    hN = {}
    with torch.no_grad():
        for k, car in enumerate(CARRIERS):
            pre = tok(car.split("{n}")[0], add_special_tokens=False)["input_ids"] if car != "{n}" else []
            out_k = []
            for st in range(0, len(ids_all), 64):
                ch = ids_all[st:st + 64]
                hs = model(input_ids=torch.tensor([sink + pre + [int(i)] for i in ch], device=device), output_hidden_states=True, use_cache=False).hidden_states
                out_k.append(hs[lS][:, -1].float().cpu().numpy().astype(np.float16))
            hN[f"hN_{k}"] = np.concatenate(out_k)

    hS, hE = Site(QS, muS, device), Site(QE, muE, device)
    blocks = get_blocks(model)
    PRE = {}
    hd = [blocks[lS - 1].register_forward_hook(hS), blocks[lE - 1].register_forward_hook(hE),
          final_norm(model).register_forward_hook(lambda mod, inp, out: PRE.__setitem__("x", inp[0]))]

    def run(d, zS=None, zE=None, full=False, invS=False, invE=False):
        n_ = len(d["y"]); out = {k: [] for k in ("zS", "zE", "m", "lf", "ls", "lse", "nF", "hF", "zA", "zB", "hS", "hE")}
        for st in range(0, n_, args.batch_size):
            b = np.arange(st, min(st + args.batch_size, n_)); P = torch.tensor(d["pos"][b], device=device)
            hS.pos, hE.pos = P[:, 2], P[:, 3]
            hS.z_new = None if zS is None else torch.tensor(zS[b], dtype=torch.float32, device=device)
            hE.z_new = None if zE is None else torch.tensor(zE[b], dtype=torch.float32, device=device)
            hS.invert, hE.invert = invS, invE
            with torch.no_grad():
                o = model(input_ids=d["ids"][b].to(device), attention_mask=d["att"][b].to(device), output_hidden_states=True, use_cache=False)
            bb = torch.arange(len(b), device=device); Lg = o.logits[bb, torch.tensor(d["last"][b], device=device)].float()
            lf_, ls_ = Lg[bb, torch.tensor(d["f"][b], device=device)], Lg[bb, torch.tensor(d["s"][b], device=device)]
            out["m"].append((lf_ - ls_).cpu().numpy()); out["lf"].append(lf_.cpu().numpy()); out["ls"].append(ls_.cpu().numpy())
            out["lse"].append(torch.logsumexp(Lg, dim=-1).cpu().numpy())
            xF = PRE["x"][bb, torch.tensor(d["last"][b], device=device)].float()
            out["nF"].append(xF.norm(dim=-1).cpu().numpy()); out["hF"].append(xF.cpu().numpy().astype(np.float16))
            xS, xE = o.hidden_states[lS][bb, P[:, 2]].float(), o.hidden_states[lE][bb, P[:, 3]].float()
            out["zS"].append(((xS - hS.mu) @ hS.Q).cpu().numpy()); out["zE"].append(((xE - hE.mu) @ hE.Q).cpu().numpy())
            if full:
                pA, pB = torch.minimum(P[:, 0], P[:, 1]), torch.maximum(P[:, 0], P[:, 1])
                out["zA"].append(((o.hidden_states[lS][bb, pA].float() - hS.mu) @ hS.Q).cpu().numpy())
                out["zB"].append(((o.hidden_states[lS][bb, pB].float() - hS.mu) @ hS.Q).cpu().numpy())
                out["hS"].append(xS.cpu().numpy().astype(np.float16)); out["hE"].append(xE.cpu().numpy().astype(np.float16))
            print(f"\r  {st + len(b)}/{n_}", end="", flush=True)
        print()
        hS.z_new = hE.z_new = None; hS.invert = hE.invert = False
        return {k: np.concatenate(v) for k, v in out.items() if v}

    partner = lambda n_: np.arange(n_) ^ 1
    rec = {"cS": cS, "cS_ids": ids_all, "QS": QS, "QE": QE, "muS": muS, "muE": muE, "layers": np.array([lS, lE]), "ranks": np.array([rS, rE]),
           "carriers": np.array(CARRIERS, dtype=object), **hN}
    fn, WU = final_norm(model), model.get_output_embeddings()
    rec.update({"WU_names": WU.weight[torch.as_tensor(ids_all, device=WU.weight.device)].detach().float().cpu().numpy(),
                "WU_bias": (WU.bias[torch.as_tensor(ids_all, device=WU.weight.device)].detach().float().cpu().numpy() if getattr(WU, "bias", None) is not None else np.zeros(len(ids_all), np.float32)),
                "norm_w": fn.weight.detach().float().cpu().numpy(),
                "norm_b": (fn.bias.detach().float().cpu().numpy() if getattr(fn, "bias", None) is not None else np.zeros_like(fn.weight.detach().float().cpu().numpy())),
                "norm_eps": float(getattr(fn, "eps", getattr(fn, "variance_epsilon", 1e-6))), "norm_class": type(fn).__name__,
                "softcap": float(getattr(model.config, "final_logit_softcapping", None) or 0.0)})
    row = {int(i): k for k, i in enumerate(ids_all)}
    dc = lambda AB: np.repeat(cS[[row[int(i)] for i in AB[:, 1]]] - cS[[row[int(i)] for i in AB[:, 0]]], 2, axis=0)   # per prompt: c(B) - c(A)
    sgn = lambda n_: np.where(np.arange(n_) % 2 == 0, -1.0, 1.0)[:, None]                                         # partner - self = s (v + dc)
    try:
        for name, (dd, frames) in sets.items():
            d, AB = dd
            print(f"[{name}] {len(d['y'])} prompts")
            base = run(d, full=True); pt = partner(len(base["m"]))
            sw = run(d, zS=base["zS"][pt]); pe = run(d, zE=base["zE"][pt])
            rec.update({f"{name}_{k}": v for k, v in base.items()})
            rec.update({f"{name}_{k}_{t}": r[k] for t, r in (("sw", sw), ("pE", pe)) for k in ("lf", "ls", "lse", "nF", "hF")})
            rec.update({f"{name}_zE_sw": sw["zE"], f"{name}_m_sw": sw["m"], f"{name}_m_pE": pe["m"], f"{name}_AB": AB,
                        f"{name}_pos": np.asarray(d["pos"]), f"{name}_last": np.asarray(d["last"]), f"{name}_f": np.asarray(d["f"]),
                        f"{name}_s": np.asarray(d["s"]), f"{name}_y": np.asarray(d["y"]), f"{name}_frame": frame_index(tok, d, frames),
                        f"{name}_nS": np.linalg.norm(base["hS"].astype(np.float32), axis=1), f"{name}_nE": np.linalg.norm(base["hE"].astype(np.float32), axis=1)})
            rec[f"{name}_frames"] = np.array(frames, dtype=object)
            sg, Cd = sgn(len(base["m"])), dc(AB)
            if name == "fit":
                v = (sg * (base["zS"][pt] - base["zS"]) - Cd).mean(0); rec["v"] = v
                continue
            for k_, kw in (("invS", dict(invS=True)), ("invE", dict(invE=True)), ("invSE", dict(invS=True, invE=True)),
                           ("injEq", dict(zS=base["zS"] + sg * (v + Cd))), ("injPtr", dict(zS=base["zS"] + sg * v)), ("injNm", dict(zS=base["zS"] + sg * Cd))):
                r_ = run(d, **kw)
                rec.update({f"{name}_{k}_{k_}": r_[k] for k in ("m", "lf", "ls", "lse", "nF", "hF")})
    finally:
        for h in hd:
            h.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **rec); print("wrote", args.out)


if __name__ == "__main__":
    main()
