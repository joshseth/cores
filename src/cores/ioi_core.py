#!/usr/bin/env python3
"""
ioi_core.py: single-site ACE analysis of indirect object identification
(IOI). Same recipe as the SVA analysis: extract per-layer cores,
run the keep/remove/flip battery, select the core layer, fit the affine
mechanism, and validate it against the flip intervention.

Task. Templates in ABB / BAB form:
  ABB: "When {A} and {B} went to the store, {B} gave a drink to" -> A
  BAB: "When {B} and {A} went to the store, {B} gave a drink to" -> A
Label y = position of the indirect object in the opening pair
(0 = first name, 1 = second name), balanced 50/50.
Margin m = logit(first name) - logit(second name) at the final token, so
correct behavior gives AUC(y, -m) ... sign fixed so that AUC(y, m_signed)
is ~1 at baseline and a successful flip drives it below 0.5.

Difference from SVA: the contrast tokens differ per prompt (each prompt has
its own two names), so the readout direction is per-prompt and the
constant-gradient condition under which rank-1 ACE equals the difference of
means does not hold; the reported spectra and the difference-of-means
baseline show how much this matters.

Usage (repo root):
  python src/cores/ioi_core.py --model-name gpt2 --n-samples 1200 --baselines --state-operator
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cores.extraction import ace_from_matrices
from cores.utils import seed_all, get_device


# ----------------------------- dataset --------------------------------------
CANDIDATE_NAMES = ["Mary", "John", "Tom", "James", "Dan", "Sarah", "Emma",
                   "Kevin", "Anna", "Mark", "Laura", "Peter", "Alice", "Jack",
                   "Rose", "Paul", "Kate", "David", "Lisa", "Ryan"]

TEMPLATES = [
    "When {X} and {Y} went to the store, {S} gave a drink to",
    "Then, {X} and {Y} went to the park, and {S} gave the ball to",
    "When {X} and {Y} got home from work, {S} handed a present to",
    "After {X} and {Y} left the bar, {S} gave a ride to",
    "While {X} and {Y} were at the office, {S} showed the report to",
]


def single_token_names(tok, restrict=None):
    """Names that tokenize to exactly one token (with leading space).
    `restrict`: optional explicit name list (for cross-model shared prompts);
    every restricted name must be a single token; otherwise the script stops,
    because a fallback would change the prompt set and break cross-model
    coordinate comparisons."""
    pool = restrict if restrict else CANDIDATE_NAMES
    keep, bad = [], []
    for n in pool:
        ids = tok(" " + n, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            keep.append((n, ids[0]))
        else:
            bad.append(n)
    if restrict and bad:
        raise ValueError(f"names not single-token for this tokenizer: {bad}. "
                         "Use --dump-names on every model and pass the intersection.")
    assert len(keep) >= 8, f"only {len(keep)} single-token names; extend the list"
    return keep


def build_ioi_dataset(n, rng, names):
    """Returns texts, y (IO position: 0=first, 1=second), first/second name ids."""
    texts, y, first_ids, second_ids = [], [], [], []
    for i in range(n):
        io, s = [names[int(j)] for j in rng.choice(len(names), size=2, replace=False)]
        tmpl = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        label = i % 2                      # 0: ABB (IO first) | 1: BAB (IO second)
        X, Yn = (io, s) if label == 0 else (s, io)
        texts.append(tmpl.format(X=X[0], Y=Yn[0], S=s[0]))
        y.append(label)
        first_ids.append(X[1]); second_ids.append(Yn[1])
    return texts, np.array(y, np.int64), np.array(first_ids), np.array(second_ids)


# ----------------------------- model plumbing --------------------------------
def get_blocks(model):
    """Decoder block list across HF families (GPT-2, LLaMA, Gemma, Qwen, ...)."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers",
                 "transformer.blocks", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"cannot locate decoder blocks on {type(model).__name__}")


def get_dims(model):
    cfg = model.config
    D = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    return int(D), int(n_layers)


class LastPosHook:
    """keep / remove / flip of a rank-r core (D, r) at a chosen position."""
    def __init__(self, Q, mu, device, mode):
        self.Q = torch.tensor(np.asarray(Q), dtype=torch.float32, device=device)
        if self.Q.dim() == 1:
            self.Q = self.Q[:, None]
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)
        self.mode = mode
        self.pos = None

    def set_pos(self, pos):
        self.pos = pos

    def set_targets(self, T):
        self.targets = torch.tensor(np.asarray(T), dtype=torch.float32,
                                    device=self.Q.device)

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        b = torch.arange(h.shape[0], device=h.device)
        v = h[b, self.pos].float() - self.mu
        c = v @ self.Q                      # (B, r) core coordinates
        proj = c @ self.Q.T                 # (B, D) component inside the core
        if self.mode == "keep":
            new = self.mu + proj
        elif self.mode == "remove":
            new = self.mu + v - proj
        elif self.mode == "flip":
            new = self.mu + v - 2.0 * proj
        elif self.mode == "set":
            new = self.mu + v + (self.targets - c) @ self.Q.T
        else:
            return out
        h = h.clone()
        h[b, self.pos] = new.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h


def margins_pass(model, ids, att, pos, fids, sids):
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    b = torch.arange(ids.shape[0], device=ids.device)
    L = out.logits[b, pos]
    return L[b, fids] - L[b, sids]        # m = logit(first) - logit(second)


def collect_margins_set(model, input_ids, attn_mask, idx, last_pos, fids, sids,
                        bs, device, Q, mu, layer, targets):
    hook = LastPosHook(Q, mu, device, "set")
    handle = get_blocks(model)[layer - 1].register_forward_hook(hook)
    ms = []
    with torch.no_grad():
        for st in range(0, len(idx), bs):
            b = idx[st:st + bs]
            pos = torch.tensor(last_pos[b], device=device)
            hook.set_pos(pos); hook.set_targets(targets[st:st + bs])
            m = margins_pass(model, input_ids[b].to(device), attn_mask[b].to(device),
                             pos, torch.tensor(fids[b], device=device),
                             torch.tensor(sids[b], device=device))
            ms.append(m.float().cpu().numpy())
    handle.remove()
    return np.concatenate(ms)


def collect_margins(model, input_ids, attn_mask, idx, last_pos, fids, sids,
                    bs, device, hook=None, layer=None):
    handle = None
    if hook is not None:
        handle = get_blocks(model)[layer - 1].register_forward_hook(hook)
    ms = []
    with torch.no_grad():
        for st in range(0, len(idx), bs):
            b = idx[st:st + bs]
            pos = torch.tensor(last_pos[b], device=device)
            if hook is not None:
                hook.set_pos(pos)
            m = margins_pass(model, input_ids[b].to(device), attn_mask[b].to(device),
                             pos, torch.tensor(fids[b], device=device),
                             torch.tensor(sids[b], device=device))
            ms.append(m.float().cpu().numpy())
    if handle is not None:
        handle.remove()
    return np.concatenate(ms)


def collect_H_and_G(model, input_ids, attn_mask, idx, last_pos, fids, sids,
                    n_layers, bs, device, n_jac):
    """H: last-position hidden states (per layer). G: margin gradients wrt each
    layer's hidden state at the last position (one backward per batch gives all
    layers)."""
    D, _ = get_dims(model)
    H = [np.zeros((len(idx), D), np.float32) for _ in range(n_layers + 1)]
    jac_sel = set(np.random.default_rng(0).choice(len(idx), size=min(n_jac, len(idx)),
                                                  replace=False).tolist())
    G = [[] for _ in range(n_layers + 1)]
    for st in range(0, len(idx), bs):
        b = idx[st:st + bs]
        pos = torch.tensor(last_pos[b], device=device)
        ids = input_ids[b].to(device); att = attn_mask[b].to(device)
        need_grad = any((st + j) in jac_sel for j in range(len(b)))
        with torch.set_grad_enabled(need_grad):
            out = model(input_ids=ids, attention_mask=att,
                        output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if need_grad:
                for h in hs:
                    h.retain_grad()
            bb = torch.arange(len(b), device=device)
            L = out.logits[bb, pos]
            m = (L[bb, torch.tensor(fids[b], device=device)]
                 - L[bb, torch.tensor(sids[b], device=device)])
            if need_grad:
                m.sum().backward()
        for li, h in enumerate(hs):
            H[li][st:st + len(b)] = h[bb, pos].detach().float().cpu().numpy()
            if need_grad and h.grad is not None:
                g = h.grad[bb, pos].detach().float().cpu().numpy()
                for j in range(len(b)):
                    if (st + j) in jac_sel:
                        G[li].append(g[j])
        model.zero_grad(set_to_none=True)
    G = [np.stack(g) if g else None for g in G]
    return H, G


def spectral_cliff(S, r_max, rel_floor=1e-6, min_ratio=1.5, start=1):
    """Rank at the largest separation of scales in the balanced spectrum.

    Balanced truncation keeps modes above a gap in the Hankel singular values;
    the rank is where consecutive values drop most sharply. Ranks whose
    singular value has fallen below `rel_floor` of the leading one are excluded
    so that numerical noise in the tail cannot manufacture a cliff. Returns
    (rank, ratio); ratio < min_ratio means no clear separation exists.
    `start` is the smallest rank considered: start=2 finds the second cliff, the
    separation after the leading mode, for spectra where one direction dominates
    and the modes that follow form a block.
    """
    S = np.asarray(S, dtype=float)
    n = min(len(S) - 1, r_max)
    best_k, best_ratio = start, 0.0
    for k in range(start, n + 1):                  # keep the first k modes
        if S[k - 1] < rel_floor * S[0]:
            break
        ratio = S[k - 1] / max(S[k], 1e-300)
        if ratio > best_ratio:
            best_k, best_ratio = k, ratio
    return best_k, best_ratio


def fit_affine(z, m):
    A = np.stack([z, np.ones_like(z)], 1)
    (a, b), *_ = np.linalg.lstsq(A, m, rcond=None)
    r2 = 1 - ((m - (a * z + b)) ** 2).sum() / ((m - m.mean()) ** 2).sum()
    return float(a), float(b), float(r2)


# ----------------------------- main ------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--n-jac", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--include-final-layer", action="store_true")
    p.add_argument("--rank-max", type=int, default=4)
    p.add_argument("--fast-sweep", action="store_true",
                   help="two-stage: cheap flip-only localization, then full "
                        "rank curves at the top-k layers")
    p.add_argument("--probe-rank", type=int, default=8)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--coarse-frac", type=float, default=0.5)
    p.add_argument("--rank-rule", type=str, default="causal",
                   choices=["causal", "cliff", "cliff2", "energy"],
                   help="cliff: largest separation of scales in the balanced "
                        "spectrum (balanced-truncation standard); energy: 99% "
                        "spectral energy")
    p.add_argument("--rank-fine", type=int, default=12,
                   help="every rank up to this value; coarser above")
    p.add_argument("--rank-stride", type=int, default=2)
    p.add_argument("--cascade-removal", action="store_true")
    p.add_argument("--causal-tol", type=float, default=0.02,
                   help="flip AUC must be within this of its floor")
    p.add_argument("--keep-tol", type=float, default=0.02,
                   help="keep-only AUC may fall at most this far below baseline")
    p.add_argument("--sweep-ranks", type=str, default="1",
                   help="comma-separated ranks for the joint layer x rank sweep, "
                        "e.g. 1,2,4,8")
    p.add_argument("--state-operator", action="store_true")
    p.add_argument("--names", type=str, default=None,
                   help="comma-separated shared name list (cross-model runs)")
    p.add_argument("--dump-names", action="store_true",
                   help="print single-token names for this tokenizer and exit")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", type=Path, default=Path("experiments/ioi/gpt2"))
    args = p.parse_args()

    device = get_device()
    seed_all(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"     # last_pos = attn_mask.sum(1)-1 assumes right pad
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dt).to(device)
    model.eval()
    print(f"device: {device} | model: {args.model_name} | dtype {args.dtype} "
          f"(gradients needed for the relevance term)")

    restrict = [x.strip() for x in args.names.split(",")] if args.names else None
    if args.dump_names:
        surv = [n for n, _ in single_token_names(tok)]
        print("SINGLE_TOKEN_NAMES: " + ",".join(surv))
        return
    rng = np.random.default_rng(args.seed)
    names = single_token_names(tok, restrict)
    print(f"names in use ({len(names)}): {', '.join(n for n, _ in names)}")
    texts, y, fids, sids = build_ioi_dataset(args.n_samples, rng, names)
    enc = tok(texts, return_tensors="pt", padding=True)
    input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
    last_pos = attn_mask.sum(1).numpy() - 1
    N = len(texts)
    perm = rng.permutation(N)
    tr, te = perm[:N // 2], perm[N // 2:]
    _, n_layers = get_dims(model)

    # baseline behavior: sign the margin so AUC(y, m_signed) ~ 1 when correct.
    m_base = collect_margins(model, input_ids, attn_mask, te, last_pos, fids, sids,
                             args.batch_size, device)
    # y=1 means IO is the SECOND name -> correct behavior: m = first-second < 0
    base_auc = roc_auc_score(y[te], -m_base)
    print(f"Base AUC (IO-position vs margin): {base_auc:.3f}")

    sweep_ranks = sorted({int(x) for x in args.sweep_ranks.split(",")} | {1})
    r_store = max(max(sweep_ranks), args.rank_max, 1)
    print(f"\n=== EXTRACTING CORES (ALL LAYERS, stored rank {r_store}) ===")
    H, G = collect_H_and_G(model, input_ids, attn_mask, tr, last_pos, fids, sids,
                           n_layers, args.batch_size, device, args.n_jac)
    Qs, mus, gaps, spectra = [], [], [], []
    for li in range(n_layers + 1):
        mu = H[li].mean(0)
        Xc = (H[li] - mu).astype(np.float64)
        Cov = torch.tensor(Xc.T @ Xc / len(Xc))
        g = G[li].astype(np.float64)
        Obs = torch.tensor(g.T @ g / len(g))
        Q, S = ace_from_matrices(Cov, Obs, r=r_store)
        S = S.numpy()
        gap = float((S[0] / max(S[1], 1e-30)) ** 2)
        cov1 = float(S[0] ** 2 / max((S ** 2).sum(), 1e-30))
        Qs.append(Q.numpy()); mus.append(mu); gaps.append(gap)
        spectra.append([float(v) for v in S[:64]])
        print(f"Layer {li:02d} | Top SVs: [{S[0]:.3e}, {S[1]:.3e}] | "
              f"Gap (S1/S2)² = {gap:.1f} | Coverage = {cov1:.3f}")

    # ---------------- causal battery helpers -----------------------------
    def eval_at(li, r, modes=("keep", "remove", "flip"), idx=None):
        idx = te if idx is None else idx
        out = {}
        for mode in modes:
            hook = LastPosHook(Qs[li][:, :r], mus[li], device, mode)
            m = collect_margins(model, input_ids, attn_mask, idx, last_pos, fids,
                                sids, args.batch_size, device, hook, li)
            out[mode] = roc_auc_score(y[idx], -m)
        return out

    layer_range = list(range(1, n_layers + 1)) if args.include_final_layer \
        else list(range(1, n_layers))
    rows, coarse = [], []

    if args.fast_sweep:
        # STAGE 1 -- localization. Flip only (selection statistic), at one probe
        # rank, on a prompt subsample. ~3x fewer passes than the full grid and
        # ~1/coarse_frac less data; the sufficiency guard is applied in stage 2
        # on the full eval set, so nothing is selected on the cheap pass alone.
        srng = np.random.default_rng(args.seed + 1)
        sub = te if args.coarse_frac >= 1.0 else srng.choice(
            te, size=max(100, int(len(te) * args.coarse_frac)), replace=False)
        print(f"\n=== STAGE 1: LOCALIZATION (flip only, rank {args.probe_rank}, "
              f"{len(sub)} prompts) ===")
        for li in layer_range:
            f = eval_at(li, args.probe_rank, ("flip",), sub)["flip"]
            coarse.append({"layer_idx": li, "rank": args.probe_rank, "auc_flip": f})
            print(f"layer={li:02d} | FLIP={f:.3f}")
        cand_layers = [c["layer_idx"] for c in
                       sorted(coarse, key=lambda c: c["auc_flip"])[:args.top_k]]
        rows = coarse
    else:
        print(f"\n=== FULL LAYER x RANK GRID (ranks {sweep_ranks}) ===")
        for li in layer_range:
            cells = []
            for r in sweep_ranks:
                a = eval_at(li, r)
                rows.append({"layer_idx": li, "rank": r,
                             **{f"auc_{k}": v for k, v in a.items()}})
                cells.append(f"r{r}: K={a['keep']:.2f} R={a['remove']:.2f} "
                             f"F={a['flip']:.3f}")
            print(f"layer={li:02d} | " + " | ".join(cells))
        by_layer = {}
        for r in rows:
            by_layer.setdefault(r["layer_idx"], []).append(r["auc_flip"])
        cand_layers = sorted(by_layer, key=lambda k: min(by_layer[k]))[:args.top_k]
    print(f"candidate layers (best flip): {cand_layers}")

    # STAGE 2 -- full keep/remove/flip rank curves at the candidate layers, on
    # the full eval set. Selection criterion: maximum flip disruption among
    # sites that still preserve baseline behavior under keep-only.
    print(f"\n=== STAGE 2: RANK CURVES AT CANDIDATE LAYERS "
          f"(ranks 1..{args.rank_max}, {len(te)} prompts) ===")
    rank_curves, energy_ranks = {}, {}
    for li in cand_layers:
        muL = mus[li]
        XcL = (H[li] - muL).astype(np.float64)
        CovL_i = torch.tensor(XcL.T @ XcL / len(XcL))
        gL = G[li].astype(np.float64)
        ObsL_i = torch.tensor(gL.T @ gL / len(gL))
        Qe, _ = ace_from_matrices(CovL_i, ObsL_i, r=None, energy_threshold=0.99)
        energy_ranks[li] = int(Qe.shape[1])
        fine = min(args.rank_fine, args.rank_max)
        rank_list = sorted(set(range(1, fine + 1))
                           | set(range(fine + 1, args.rank_max + 1, args.rank_stride))
                           | {args.rank_max}
                           | ({energy_ranks[li]} if energy_ranks[li] <= args.rank_max
                              else set()))
        curve = []
        for r in rank_list:
            a = eval_at(li, r)                    # slices the stored basis
            curve.append({"rank": r, **{f"auc_{k}": v for k, v in a.items()}})
            print(f"layer={li:02d} rank={r:02d} | KEEP={a['keep']:.3f} "
                  f"REMOVE={a['remove']:.3f} FLIP={a['flip']:.3f}"
                  + ("   <- 99%-energy rank" if r == energy_ranks[li] else ""))
        rank_curves[li] = curve

    # Selection:
    #   (i) the site is chosen by flip disruption at a rank held fixed across
    #       layers, since comparing layers at their own ranks would favor
    #       whichever site admits more dimensions;
    #   (ii) the rank at that site follows --rank-rule (default 'causal': the
    #        smallest rank whose flip AUC is within --causal-tol of the floor and
    #        whose keep AUC is within --keep-tol of baseline); the 99%-energy
    #        rank is reported for comparison.
    if args.fast_sweep:
        L = cand_layers[0]                       # stage 1: common probe rank
        probe_used = args.probe_rank
    else:
        probe_used = min(sweep_ranks, key=lambda r: abs(r - args.probe_rank))
        L = min([r for r in rows if r["rank"] == probe_used],
                key=lambda r: r["auc_flip"])["layer_idx"]
    print(f"\nSite: layer {L} (lowest flip AUC at the common probe rank "
          f"{probe_used})")
    r_cliff, cliff_ratio = spectral_cliff(spectra[L], args.rank_max)
    r_cliff2, cliff2_ratio = spectral_cliff(spectra[L], args.rank_max, start=2)
    Sl = np.asarray(spectra[L], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = Sl[:-1] / np.maximum(Sl[1:], 1e-300)
    print("  balanced singular values (top 16): "
          + ", ".join(f"{v:.3g}" for v in Sl[:16]))
    print("  consecutive ratios  S_k/S_k+1     : "
          + ", ".join(f"{v:.2f}" for v in rr[:15]))
    print(f"Balanced spectrum at layer {L}: first cliff rank {r_cliff} "
          f"(ratio {cliff_ratio:.1f}) | second cliff rank {r_cliff2} "
          f"(ratio {cliff2_ratio:.2f}) | 99%-energy rank r* = {energy_ranks[L]}")
    if args.rank_rule == "causal":
        floor_f = min(row["auc_flip"] for row in rank_curves[L])
        elig = [row["rank"] for row in rank_curves[L]
                if row["auc_flip"] <= floor_f + args.causal_tol
                and row["auc_keep"] >= base_auc - args.keep_tol]
        if elig:
            rule_rank = min(elig)
        else:
            print("  no rank meets both causal criteria; using the energy rank")
            rule_rank = energy_ranks[L]
    elif args.rank_rule == "cliff2":
        rule_rank = r_cliff2 if cliff2_ratio >= 1.5 else energy_ranks[L]
        if cliff2_ratio < 1.5:
            print("  no separation of scales beyond the leading mode "
                  f"(max ratio {cliff2_ratio:.2f} < 1.5); using the energy rank")
    elif args.rank_rule == "cliff" and cliff_ratio < 1.5:
        print("  no clear separation of scales (max ratio < 1.5); falling back "
              "to the energy criterion")
        rule_rank = energy_ranks[L]
    elif args.rank_rule == "cliff":
        rule_rank = r_cliff
    else:
        rule_rank = energy_ranks[L]
    r_sel = min(max(rule_rank, 1), args.rank_max)
    avail = [row["rank"] for row in rank_curves[L]]
    if r_sel not in avail:
        r_sel = min(avail, key=lambda r: abs(r - r_sel))
    best_row = [row for row in rank_curves[L] if row["rank"] == r_sel][0]
    print(f"Rank: {r_sel} by the '{args.rank_rule}' rule | for comparison: "
          f"first cliff {r_cliff}, second cliff {r_cliff2}, "
          f"99%-energy {energy_ranks[L]}")
    print(f"Validation at (layer {L}, rank {r_sel}): KEEP={best_row['auc_keep']:.3f} "
          f"(baseline {base_auc:.3f}) REMOVE={best_row['auc_remove']:.3f} "
          f"FLIP={best_row['auc_flip']:.3f}")
    if best_row["auc_keep"] < base_auc - args.keep_tol:
        print(f"  WARNING: keep-only falls more than {args.keep_tol} below "
              "baseline at the energy rank -- the selected subspace is not "
              "sufficient; report this rather than moving the rank.")
    r1 = [row for row in rank_curves[L] if row["rank"] == 1][0]
    print(f"  rank-1 at the same layer gives FLIP={r1['auc_flip']:.3f}: a "
          f"one-dimensional core does not control this task")
    # descriptive only (not the selection): smallest rank within tol of the floor
    floor = min(row["auc_flip"] for row in rank_curves[L])
    parsi = min(row["rank"] for row in rank_curves[L]
                if row["auc_flip"] <= floor + 0.02)
    print(f"  for reference, flip is within 0.02 of its floor from rank "
          f"{parsi} onward")

    mu_L = mus[L]
    XcL = (H[L] - mu_L).astype(np.float64)
    CovL = torch.tensor(XcL.T @ XcL / len(XcL))
    gL = G[L].astype(np.float64)
    ObsL = torch.tensor(gL.T @ gL / len(gL))
    r_energy = energy_ranks[L]
    results_rank = rank_curves[L]
    print(f"  99%-energy rank at layer {L}: r* = {r_energy} (reported for reference)")

    cascade_out = {}
    if args.cascade_removal:
        print("\n=== CASCADE REMOVAL (core removed at every layer from the site on) ===")
        print("  Single-site removal leaves the name tokens in context, and later")
        print("  heads can rebuild the summary (cf. backup name movers / self-repair).")
        print("  Removing the variable's whole carrier family tests necessity of the")
        print("  distributed realization rather than of one site.")
        for stop in ("exclusive", "inclusive"):
            hooks, handles = [], []
            last = n_layers if stop == "inclusive" else n_layers - 1
            for lly in range(L, last + 1):
                hk = LastPosHook(Qs[lly][:, :r_sel], mus[lly], device, "remove")
                hooks.append(hk)
                handles.append(get_blocks(model)[lly - 1].register_forward_hook(hk))
            ms = []
            with torch.no_grad():
                for st in range(0, len(te), args.batch_size):
                    b = te[st:st + args.batch_size]
                    pos = torch.tensor(last_pos[b], device=device)
                    for hk in hooks:
                        hk.set_pos(pos)
                    m = margins_pass(model, input_ids[b].to(device),
                                     attn_mask[b].to(device), pos,
                                     torch.tensor(fids[b], device=device),
                                     torch.tensor(sids[b], device=device))
                    ms.append(m.float().cpu().numpy())
            for h in handles:
                h.remove()
            auc_casc = roc_auc_score(y[te], -np.concatenate(ms))
            print(f"  layers {L}..{last} ({stop} of final layer): "
                  f"REMOVE-cascade AUC = {auc_casc:.3f} "
                  f"(single-site: {best_row['auc_remove']:.3f}; chance 0.5)")
            cascade_out[stop] = float(auc_casc)

    # knob fit + identify validation at the selected layer
    z = ((H[L] - mus[L]) @ Qs[L][:, 0])          # train-set coordinates unused;
    z_te_H, _ = collect_H_and_G(model, input_ids, attn_mask, te, last_pos, fids,
                                sids, n_layers, args.batch_size, device, n_jac=1)
    z_te = ((z_te_H[L] - mus[L]) @ Qs[L][:, 0])
    m_te = collect_margins(model, input_ids, attn_mask, te, last_pos, fids, sids,
                           args.batch_size, device)
    a, b, r2 = fit_affine(z_te, m_te)
    print(f"\nKnob fit (rank 1): a={a:.4f} b={b:.4f} R²={r2:.4f}")
    r_mech = min(max(r_sel, 1), args.rank_max)   # selected causal rank
    if r_mech > 1:
        Qm, _ = ace_from_matrices(CovL, ObsL, r=r_mech)
        Z_te = (z_te_H[L] - mu_L) @ Qm.numpy()          # (n, r) coordinates
        A_ = np.concatenate([Z_te, np.ones((len(Z_te), 1))], 1)
        coef, *_ = np.linalg.lstsq(A_, m_te, rcond=None)
        r2_multi = 1 - ((m_te - A_ @ coef) ** 2).sum() / ((m_te - m_te.mean()) ** 2).sum()
        print(f"Knob fit (selected rank {r_mech}): R²={r2_multi:.4f} "
              f"(vs rank-1 {r2:.4f})")

        # Multivariate identify validation: the rank-r law must predict the
        # margins observed under the rank-r flip, which negates all r coordinates.
        hook_r = LastPosHook(Qm.numpy(), mu_L, device, "flip")
        m_flip_r = collect_margins(model, input_ids, attn_mask, te, last_pos,
                                   fids, sids, args.batch_size, device, hook_r, L)
        m_pred_r = np.concatenate([-Z_te, np.ones((len(Z_te), 1))], 1) @ coef
        r2p_r = 1 - ((m_flip_r - m_pred_r) ** 2).sum() /                     ((m_flip_r - m_flip_r.mean()) ** 2).sum()
        sign_r = float((np.sign(m_pred_r) == np.sign(m_flip_r)).mean())
        corr_r = float(np.corrcoef(m_pred_r, m_flip_r)[0, 1])
        lam, c0 = np.polyfit(m_pred_r, m_flip_r, 1)
        print(f"Identify validation (rank {r_mech}): predicted-vs-observed "
              f"flipped margins R²={r2p_r:.3f} | sign agreement={sign_r:.3f}")
        print(f"  saturation diagnostics: corr²={corr_r**2:.3f} "
              f"(recalibrated ceiling) | attenuation λ={lam:.3f}")
    hook = LastPosHook(Qs[L], mus[L], device, "flip")
    m_flip = collect_margins(model, input_ids, attn_mask, te, last_pos, fids, sids,
                             args.batch_size, device, hook, L)
    m_pred = a * (-z_te) + b
    r2p = 1 - ((m_flip - m_pred) ** 2).sum() / ((m_flip - m_flip.mean()) ** 2).sum()
    sign = float((np.sign(m_pred) == np.sign(m_flip)).mean())
    print(f"Identify validation: predicted-vs-observed flipped margins "
          f"R²={r2p:.3f} | sign agreement={sign:.3f}")

    results = {"base_auc": float(base_auc), "sweep": rows,
               "cascade_removal": cascade_out,
               "coarse_sweep": coarse, "rank_curves": rank_curves,
               "energy_ranks": energy_ranks, "n_layers": int(n_layers),
               "candidate_layers": cand_layers, "model_name": args.model_name,
               "probe_rank_used": int(probe_used), "parsimonious_rank": int(parsi),
               "balanced_spectra": spectra, "rank_rule": args.rank_rule,
               "cliff_rank": int(r_cliff), "cliff_ratio": float(cliff_ratio),
               "cliff2_rank": int(r_cliff2), "cliff2_ratio": float(cliff2_ratio),
               "selected_rank": int(r_sel), "sweep_ranks": sweep_ranks,
               "rank_curve": results_rank, "r_energy": int(r_energy),
               "final_layer_idx": L, "spectral_gaps": gaps,
               "knob": {"a": a, "b": b, "r2": r2},
               "knob_multi": ({"rank": int(r_mech), "r2": float(r2_multi),
                               "r2_pred_flip": float(r2p_r),
                               "flip_sign_agreement": float(sign_r),
                               "corr2_recalibrated": float(corr_r**2),
                               "attenuation_lambda": float(lam)}
                              if r_mech > 1 else None),
               "identify_validation": {"r2_pred_flip": float(r2p),
                                        "flip_sign_agreement": sign}}

    if args.state_operator and r_mech >= 1:
        # State operator: instead of regressing the margin through the saturating
        # readout, fit the encoding map (core coordinates as an affine function of
        # the task variables: IO name, S name, IO position) and validate it by
        # writing the encoder's counterfactual prediction into the core and
        # checking that the behavior follows.
        print("\n=== STATE OPERATOR (encoder + transplant validation) ===")
        name_ix = {nid: k for k, (_, nid) in enumerate(names)}
        nn_ = len(names)

        def design(idx_, ylab):
            X = np.zeros((len(idx_), 2 * nn_ + 2))
            for j, i in enumerate(idx_):
                io_id = second_ids_g[i] if ylab[j] == 1 else first_ids_g[i]
                s_id = first_ids_g[i] if ylab[j] == 1 else second_ids_g[i]
                X[j, name_ix[int(io_id)]] = 1.0
                X[j, nn_ + name_ix[int(s_id)]] = 1.0
                X[j, 2 * nn_] = 2.0 * ylab[j] - 1.0
                X[j, 2 * nn_ + 1] = 1.0
            return X

        first_ids_g, second_ids_g = fids, sids
        Z_tr = (H[L] - mu_L) @ Qm.numpy()
        Xtr = design(tr, y[tr]); Xte = design(te, y[te])
        lam_r = 1e-3
        W = np.linalg.solve(Xtr.T @ Xtr + lam_r * np.eye(Xtr.shape[1]), Xtr.T @ Z_tr)
        Z_hat = Xte @ W
        r2_dim = 1 - ((Z_te - Z_hat) ** 2).sum(0) / ((Z_te - Z_te.mean(0)) ** 2).sum(0)
        # which coordinates encode position vs names: position-only refit
        Xp = np.stack([2.0 * y[tr] - 1.0, np.ones(len(tr))], 1)
        Wp = np.linalg.lstsq(Xp, Z_tr, rcond=None)[0]
        Zp = np.stack([2.0 * y[te] - 1.0, np.ones(len(te))], 1) @ Wp
        r2_pos = 1 - ((Z_te - Zp) ** 2).sum(0) / ((Z_te - Z_te.mean(0)) ** 2).sum(0)
        for d in range(r_mech):
            print(f"  coord {d}: held-out R²(full)={r2_dim[d]:.3f} | "
                  f"R²(position only)={r2_pos[d]:.3f}")

        # control transplant: write the factual predicted state (should preserve the behavior)
        m_fact = collect_margins_set(model, input_ids, attn_mask, te, last_pos,
                                     fids, sids, args.batch_size, device,
                                     Qm.numpy(), mu_L, L, Z_hat)
        auc_fact = roc_auc_score(y[te], -m_fact)
        # counterfactual transplant: same names, IO position flipped
        Xcf = design(te, 1 - y[te])
        Z_cf = Xcf @ W
        m_cf = collect_margins_set(model, input_ids, attn_mask, te, last_pos,
                                   fids, sids, args.batch_size, device,
                                   Qm.numpy(), mu_L, L, Z_cf)
        auc_cf = roc_auc_score(y[te], -m_cf)
        print(f"  transplant control (factual state written): AUC={auc_fact:.3f}  (expect ~1)")
        print(f"  counterfactual transplant (position flipped): AUC={auc_cf:.3f}  "
              f"(expect low; rank-{r_mech} flip reference: "
              f"{[q['auc_flip'] for q in results_rank if q['rank'] == r_mech][0]:.3f})")
        # Transplant vs. rank: the fitted operator's counterfactual control against
        # geometric reflection of the same subspace, at matched rank. A curve keeps
        # the result independent of the rank-selection rule; at high rank the
        # geometric flip saturates and the two become indistinguishable.
        print("\n  transplant vs geometric reflection, by rank:")
        print(f"    {'rank':>5s} {'geometric flip':>15s} {'counterfactual tx':>18s}")
        avail = [row["rank"] for row in rank_curves[L]]
        flip_at = {row["rank"]: row["auc_flip"] for row in rank_curves[L]}
        probe_set = sorted({1, 2, 4, 8, r_sel, energy_ranks[L], args.rank_max}
                           & set(avail))
        tx_curve = []
        for rr in probe_set:
            Qr = Qs[L][:, :rr]
            Ztr_r = (H[L] - mu_L) @ Qr
            Wr = np.linalg.solve(Xtr.T @ Xtr + lam_r * np.eye(Xtr.shape[1]),
                                 Xtr.T @ Ztr_r)
            m_cf_r = collect_margins_set(model, input_ids, attn_mask, te, last_pos,
                                         fids, sids, args.batch_size, device,
                                         Qr, mu_L, L, Xcf @ Wr)
            a_cf = roc_auc_score(y[te], -m_cf_r)
            tx_curve.append({"rank": rr, "auc_geometric_flip": flip_at[rr],
                             "auc_counterfactual_transplant": float(a_cf)})
            print(f"    {rr:5d} {flip_at[rr]:15.3f} {a_cf:18.3f}")
        results["transplant_curve"] = tx_curve

        results["state_operator"] = {
            "r2_per_coord_full": [float(v) for v in r2_dim],
            "r2_per_coord_position_only": [float(v) for v in r2_pos],
            "auc_factual_transplant": float(auc_fact),
            "auc_counterfactual_transplant": float(auc_cf)}

    if args.baselines:
        print("\n=== BASELINE AXES (same layer, same ablations) ===")
        mu = mus[L]; Xc = H[L] - mu
        pca1 = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)[2][0]
        s_lab = 2.0 * y[tr] - 1.0
        dm = (Xc[s_lab > 0].mean(0) - Xc[s_lab < 0].mean(0))
        results["baselines"] = {}
        for name, ax in (("pca1", pca1), ("diff_means", dm / np.linalg.norm(dm))):
            Q = ax.reshape(-1, 1).astype(np.float64)
            aucs = {}
            for mode in ("keep", "remove", "flip"):
                hook = LastPosHook(Q, mu, device, mode)
                m = collect_margins(model, input_ids, attn_mask, te, last_pos,
                                    fids, sids, args.batch_size, device, hook, L)
                aucs[mode] = roc_auc_score(y[te], -m)
            cosq = float(abs(ax @ Qs[L][:, 0]) / np.linalg.norm(ax))
            results["baselines"][name] = {**aucs, "cos_to_ace": cosq}
            print(f"  [{name:12s}] KEEP={aucs['keep']:.3f} REMOVE={aucs['remove']:.3f} "
                  f"FLIP={aucs['flip']:.3f} |cos(ACE)|={cosq:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    extra = {}
    if r_mech > 1:
        extra = {"encoder_W": W.astype(np.float32),
                 "Z_multi": Z_te.astype(np.float32),
                 "m_flip_multi": m_flip_r.astype(np.float32),
                 "m_pred_multi": m_pred_r.astype(np.float32),
                 "Q_multi": Qm.numpy().astype(np.float32)}
    np.savez(args.out_dir / "knob_fit_test.npz", z=z_te.astype(np.float32),
             m=m_te.astype(np.float32), y=y[te], m_flip=m_flip.astype(np.float32),
             q=Qs[L][:, 0].astype(np.float32),
             prompt_index=te.astype(np.int64),
             prompt_hash=np.array(  # stable across processes (hash() is salted)
                 [int(hashlib.md5(texts[int(i)].encode()).hexdigest()[:15], 16)
                  for i in te], np.int64),
             **extra)
    print(f"\nSaved: {args.out_dir}/results.json and knob_fit_test.npz")


if __name__ == "__main__":
    main()
