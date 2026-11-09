"""
common.py: shared by the two-site IOI scripts: the model, the prompts, states and gradients, ACE factors, interventions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cores.utils import get_device, seed_all  # noqa: F401

DATA = Path(__file__).resolve().parent / "data"
SEED = 0
SITES, COL = ("S2", "END"), {"S2": 2, "END": 3}
CURVE = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512)
orth = lambda A: np.linalg.qr(np.asarray(A, np.float64))[0].astype(np.float32)


# ---- model
def load_model(name, dtype, attn_impl, device, sink=False):
    """sink: every prompt starts with a start token, as LLaMA and Gemma do by default. For tokenizers without one
    (GPT-2, Qwen) the end-of-text token is prepended, and padding uses another token so position offsets stay right."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    if sink and (tok.bos_token_id is None or tok("x")["input_ids"][0] != tok.bos_token_id):
        from tokenizers.processors import TemplateProcessing
        tok.bos_token = tok.eos_token
        tok._tokenizer.post_processor = TemplateProcessing(single=f"{tok.eos_token} $A", pair=f"{tok.eos_token} $A $B",
                                                           special_tokens=[(tok.eos_token, tok.eos_token_id)])
        if tok.pad_token_id == tok.bos_token_id:
            tok.pad_token = tok.convert_ids_to_tokens(0)
    kw = {"attn_implementation": attn_impl} if attn_impl else {}
    if device.type == "cuda":
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype], **kw).to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)     # hidden states then carry gradients
    return tok, model


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


# ---- prompts
MALE_NAMES = ["John", "James", "David", "Michael", "Robert", "William", "Thomas", "Daniel", "Paul",
              "Mark", "Peter", "Kevin", "Brian", "George", "Edward", "Steven", "Andrew", "Ryan",
              "Jack", "Tom", "Dan", "Sam", "Ben", "Luke", "Adam", "Eric", "Scott", "Frank",
              "Henry", "Carl", "Nick", "Alex", "Jason", "Justin", "Kyle", "Sean", "Adam", "Joe",
              "Bill", "Matt"]
FEMALE_NAMES = ["Mary", "Sarah", "Emma", "Anna", "Lisa", "Kate", "Laura", "Rose", "Alice", "Jane",
                "Amy", "Julia", "Grace", "Linda", "Susan", "Karen", "Nancy", "Helen", "Rachel",
                "Emily", "Hannah", "Maria", "Diana", "Ruth", "Joan", "Megan", "Julie", "Anne",
                "Victoria", "Natalie", "Sophie", "Lucy", "Ellen", "Eva", "Jessica", "Ashley",
                "Kim", "Jill", "Beth", "Ann"]
_CAP = re.compile(r'(^|[.!?]\s+)(the|he|she) ')   # sentence starts only (not after a closing quote)


def _single(tok, w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def _forms(tok, *words):
    """Token ids of each word mid-sentence (" w") and at the very start of a text ("w").
    A text-initial word is a different token in BPE vocabularies such as GPT-2's."""
    out = set()
    for w in words:
        for v in (" " + w, w):
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return out


def _row_offset(row, tok):
    return 1 if (tok.bos_token_id is not None and tok.bos_token_id != tok.pad_token_id
                 and row[0] == tok.bos_token_id) else 0


def setup_data(tok, seed=SEED, val_hold=150):
    """Frames, name pools and splits."""
    ff = json.loads((DATA / "frames_v2_sink.json").read_text()); allf = [r["frame"] for r in ff["frames"]]; keep = ff["kept"]
    fr_ext = [allf[i] for j, i in enumerate(keep) if j % 4 != 3]
    fr_new = [allf[i] for j, i in enumerate(keep) if j % 4 == 3]
    fr_val, fr_test = fr_new[::2], fr_new[1::2]
    test_names = MALE_NAMES[1::2] + FEMALE_NAMES[1::2]
    pool = [x for x in open(DATA / "gpt2_single.txt").read().strip().split(",")
            if x not in test_names and _single(tok, x) is not None]
    pool = [pool[i] for i in np.random.default_rng(seed + 77).permutation(len(pool))]
    import nltk
    nltk.download("names", quiet=True)
    from nltk.corpus import names as nltk_names
    raw = sorted({n.strip() for n in nltk_names.words("male.txt") + nltk_names.words("female.txt")})
    raw = [n for n in raw if n.isalpha() and n[0].isupper() and len(n) >= 3]
    is_word = lambda w: len(tok(" " + w.lower(), add_special_tokens=False)["input_ids"]) == 1
    seen = set(pool) | set(test_names); extra = []
    for x in raw:
        if x not in seen and _single(tok, x) is not None and not is_word(x):
            extra.append(x); seen.add(x)
    extra = [extra[i] for i in np.random.default_rng(seed + 79).permutation(len(extra))]
    big = pool + extra
    val_names = [big[i] for i in np.random.default_rng(seed + 90).choice(len(big), val_hold, replace=False)]
    ext_names = [x for x in big if x not in set(val_names)]
    return dict(seed=seed, fr_ext=fr_ext, fr_val=fr_val, fr_test=fr_test, test_names=test_names,
                gpt2_pool=pool, nltk_extra=extra, val_names=val_names, ext_names=ext_names)


ent = lambda names: dict(males=names, females=[], nouns_m=[], nouns_f=[], nouns_n=[])


def build_role_pairs(tok, n_pairs, rng, frames, E, p_noun=0.4, p_pronoun=0.3):
    """Role-swap pairs. Returns a data dict (2*n_pairs prompts, members adjacent: x+, x-)
    plus per-pair token ids (A, B)."""
    keep = lambda ws: [(w, _single(tok, w)) for w in ws if _single(tok, w) is not None]
    M, F = keep(E["males"]), keep(E["females"])
    NM, NF, NN = keep(E["nouns_m"]), keep(E["nouns_f"]), keep(E["nouns_n"])
    pron_ids = _forms(tok, "he", "He", "she", "She")
    texts, y, fids, sids, pos_l, AB = [], [], [], [], [], []
    while len(AB) < n_pairs:
        noun = rng.random() < p_noun
        pron = rng.random() < p_pronoun
        fi = int(rng.integers(len(frames)))
        if pron:
            m_pool, f_pool = (NM, NF) if noun else (M, F)
            a = m_pool[int(rng.integers(len(m_pool)))]; b = f_pool[int(rng.integers(len(f_pool)))]
            male = {a[0]}
        else:
            pool = (NM + NF + NN) if noun else (M + F)
            j, k = rng.choice(len(pool), 2, replace=False); a, b = pool[int(j)], pool[int(k)]
            male = set()
        A, B = (a, b) if rng.random() < 0.5 else (b, a)          # A is mentioned first in both members
        surf = (lambda e: "the " + e[0]) if noun else (lambda e: e[0])
        members = []
        for S in (B, A):                                          # x+: B repeated (answer A); x-: A repeated (answer B)
            s2 = ("he" if S[0] in male else "she") if pron else surf(S)
            text = frames[fi].format(A=surf(A), B=surf(B), S=s2) + (" the" if noun else "")
            text = _CAP.sub(lambda m_: m_.group(1) + m_.group(2).capitalize() + " ", text)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            occ = lambda fset: [t for t, x in enumerate(ids) if x in fset]
            fA, fB = _forms(tok, A[0]), _forms(tok, B[0])
            oS, oO, oP = occ(fB if S is B else fA), occ(fA if S is B else fB), occ(pron_ids)
            if len(oO) != 1 or len(oS) != (1 if pron else 2) or len(oP) != (1 if pron else 0):
                break
            pa, pb = occ(fA)[0], occ(fB)[0]
            ps2 = oP[0] if pron else oS[1]
            if not (pa < pb < ps2 < len(ids) - 1):
                break
            # label convention of the pipeline: y=0 means IO is the first-mentioned entity
            members.append((text, 0 if S is B else 1, (pa, pb, ps2)))
        if len(members) != 2:
            continue
        for text, lab, pp in members:
            texts.append(text); y.append(lab); fids.append(A[1]); sids.append(B[1]); pos_l.append(pp)
        AB.append((A[1], B[1]))
    enc = tok(texts, return_tensors="pt", padding=True)
    ids, att = enc["input_ids"], enc["attention_mask"]
    pos = np.zeros((len(texts), 4), np.int64)
    for i, (pa, pb, ps2) in enumerate(pos_l):
        off = _row_offset(ids[i].tolist(), tok)
        io_p, s1_p = (pa, pb) if y[i] == 0 else (pb, pa)
        pos[i] = [io_p + off, s1_p + off, ps2 + off, int(att[i].sum()) - 1]
    return ({"ids": ids, "att": att, "pos": pos, "last": att.sum(1).numpy() - 1,
             "f": np.array(fids), "s": np.array(sids), "y": np.array(y, np.int64)}, np.array(AB))


def stratified_pairs(tok, per_frame, seed, frames, names):
    """per_frame role-swap pairs from every frame, members adjacent, collated into one set."""
    parts = [build_role_pairs(tok, per_frame, np.random.default_rng(seed + 1000 * i), [fr], ent(names), p_noun=0.0, p_pronoun=0.0) for i, fr in enumerate(frames)]
    T = max(p[0]["ids"].shape[1] for p in parts)
    pad = lambda x, v: torch.nn.functional.pad(x, (0, T - x.shape[1]), value=v)
    d = {"ids": torch.cat([pad(p[0]["ids"], tok.pad_token_id) for p in parts]), "att": torch.cat([pad(p[0]["att"], 0) for p in parts])}
    for k in ("pos", "last", "f", "s", "y"):
        d[k] = np.concatenate([p[0][k] for p in parts])
    return d, np.unique(np.concatenate([p[1].ravel() for p in parts]))


# ---- states, gradients, ACE factors
def capture(model, d, layers, n_pairs, bs, device):
    """Per layer: states and margin gradients at S2 and END; rows stay in pair order (x+, x-)."""
    H = {l: {"S2": [], "END": []} for l in layers}
    G = {l: {k: [] for k in H[l]} for l in layers}
    bs -= bs % 2
    for st in range(0, 2 * n_pairs, bs):
        b = np.arange(st, min(st + bs, 2 * n_pairs))
        bb = torch.arange(len(b), device=device)
        out = model(input_ids=d["ids"][b].to(device), attention_mask=d["att"][b].to(device), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for l in layers:
            hs[l].retain_grad()
        L = out.logits[bb, torch.tensor(d["last"][b], device=device)]
        (L[bb, torch.tensor(d["f"][b], device=device)] - L[bb, torch.tensor(d["s"][b], device=device)]).sum().backward()
        s2, last = np.asarray(d["pos"][b, 2]), np.asarray(d["last"][b])
        assert (np.asarray(d["pos"][b, 3]) == last).all(), "END is not the last token"
        assert (s2[0::2] == s2[1::2]).all() and (last[0::2] == last[1::2]).all(), "pair members are not aligned"
        for l in layers:
            h, g = hs[l].detach().float().cpu().numpy(), hs[l].grad.detach().float().cpu().numpy()
            r = np.arange(len(b))
            H[l]["S2"].append(h[r, s2]); G[l]["S2"].append(g[r, s2])
            H[l]["END"].append(h[r, last]); G[l]["END"].append(g[r, last])
        model.zero_grad(set_to_none=True)
    return H, G


def role_factors(H, G, device):
    """Write and read modes from the second moment of X = x+ - x- and the margin gradients, by QR of each and SVD of Rx Rg^T."""
    H, G = np.concatenate(H), np.concatenate(G)
    X = H[0::2] - H[1::2]
    nz = lambda A: A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
    if device.type == "cuda":
        try:
            Rx = torch.linalg.qr(torch.as_tensor(X, dtype=torch.float32, device=device) / np.sqrt(len(X)), mode="r")[1]
            Rg = torch.linalg.qr(torch.as_tensor(G, dtype=torch.float32, device=device) / np.sqrt(len(G)), mode="r")[1]
            U, s, Vh = torch.linalg.svd(Rx @ Rg.T)
            k = int((s > s[0] * 1e-6).sum())
            T, R = (Rx.T @ U[:, :k]).double().cpu().numpy(), (Rg.T @ Vh[:k].T).double().cpu().numpy()
            return {"mu": H.mean(0).astype(np.float32), "Tn": nz(T).astype(np.float32), "Rn": nz(R).astype(np.float32), "s": s[:k].double().cpu().numpy(), "Rg": Rg.float().cpu().numpy()}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print("  GPU out of memory in the factorization: CPU", flush=True)
    Rx = np.linalg.qr(np.asarray(X, np.float64) / np.sqrt(len(X)), mode="r"); Rg = np.linalg.qr(np.asarray(G, np.float64) / np.sqrt(len(G)), mode="r")
    U, s, Vt = np.linalg.svd(Rx @ Rg.T)
    k = int((s > s[0] * 1e-8).sum())
    T, R = Rx.T @ U[:, :k], Rg.T @ Vt[:k].T
    return {"mu": H.mean(0).astype(np.float32), "Tn": nz(T).astype(np.float32), "Rn": nz(R).astype(np.float32), "s": s[:k], "Rg": np.asarray(Rg, np.float32)}


def core_basis(fac, r, var):
    r = min(r, fac["Tn"].shape[1])
    A = fac["Tn"][:, :r] if var == "orth_write" else fac["Tn"][:, :r] + fac["Rn"][:, :r]
    return np.linalg.qr(A.astype(np.float64))[0].astype(np.float32), fac["mu"]


# ---- interventions and scoring
class SiteHook:
    __slots__ = ("Q", "mu", "mode", "pos", "col")

    def __init__(self, Q, mu, device, mode, col):
        self.Q = torch.as_tensor(np.asarray(Q), dtype=torch.float32, device=device)
        if self.Q.dim() == 1:
            self.Q = self.Q[:, None]
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.mode, self.col, self.pos = mode, col, None

    def set_pos(self, p):
        self.pos = p

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        b = torch.arange(h.shape[0], device=h.device)
        v = h[b, self.pos].float() - self.mu
        proj = (v @ self.Q) @ self.Q.T
        if self.mode == "flip":
            new = self.mu + v - 2.0 * proj
        elif self.mode == "remove":
            new = self.mu + v - proj
        elif self.mode == "keep":
            new = self.mu + proj
        else:
            return out
        h = h.clone(); h[b, self.pos] = new.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h


class SwapHook:
    """Writes the partner's core coordinates at one site. Batches hold whole pairs, in evaluate()'s order."""

    def __init__(self, Q, data, col, device):
        self.Q = torch.as_tensor(Q, device=device)
        self.pos, self.st = torch.as_tensor(data["pos"][:, col], device=device), 0

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = h.shape[0]
        p = self.pos[self.st:self.st + n]; self.st += n
        b = torch.arange(n, device=h.device)
        x = h[b, p].float()
        new = x + ((x[b ^ 1] - x) @ self.Q) @ self.Q.T
        h = h.clone(); h[b, p] = new.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h


class Site:
    """Core coordinates at one position per prompt: read, overwrite (z_new), or invert live (invert=True: z -> -z)."""
    def __init__(s, Q, mu, device):
        s.Q = torch.as_tensor(Q, dtype=torch.float32, device=device); s.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        s.pos, s.z_new, s.invert = None, None, False
    def __call__(s, mod, inp, out):
        if s.z_new is None and not s.invert:
            return out
        h = out[0] if isinstance(out, tuple) else out
        b = torch.arange(h.shape[0], device=h.device)
        x = h[b, s.pos].float(); z = (x - s.mu) @ s.Q
        zn = -z if s.invert else s.z_new
        h = h.clone(); h[b, s.pos] = (x + (zn - z) @ s.Q.T).to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h


@torch.no_grad()
def evaluate(model, data, pool, sites, bs, device):
    """sites: list of (layer, col, Q, mu, mode); all hooks active in one pass.
    Returns AUC plus the logit decomposition."""
    ids, att, pos, last_pos, fids, sids, y = (data[k] for k in
                                              ("ids", "att", "pos", "last", "f", "s", "y"))
    handles, hooks = [], []
    for (li, col, Q, mu, mode) in sites:
        h = SiteHook(Q, mu, device, mode, col)
        hooks.append(h)
        handles.append(get_blocks(model)[li - 1].register_forward_hook(h))
    pool_t = torch.as_tensor(pool, device=device)
    rec = {k: [] for k in ("m", "io", "s", "oth", "top")}
    idx = np.arange(len(y))
    try:
        for st in range(0, len(idx), bs):
            b = idx[st:st + bs]
            for h in hooks:
                h.set_pos(torch.tensor(pos[b, h.col], device=device))
            rp = torch.tensor(last_pos[b], device=device)
            out = model(input_ids=ids[b].to(device), attention_mask=att[b].to(device),
                        use_cache=False)
            bb = torch.arange(len(b), device=device)
            L = out.logits[bb, rp].float()
            f_t = torch.tensor(fids[b], device=device)
            s_t = torch.tensor(sids[b], device=device)
            yb = torch.tensor(y[b], device=device)
            io_t = torch.where(yb == 0, f_t, s_t)
            S_t = torch.where(yb == 0, s_t, f_t)
            Lp = L[:, pool_t]
            mask = (pool_t[None] != io_t[:, None]) & (pool_t[None] != S_t[:, None])
            oth = (Lp * mask).sum(1) / mask.sum(1).clamp(min=1)
            top = L.argmax(-1)
            cat = torch.full_like(top, 3)
            cat[torch.isin(top, pool_t)] = 2
            cat[top == S_t] = 1
            cat[top == io_t] = 0
            rec["m"].append((L[bb, f_t] - L[bb, s_t]).cpu())
            rec["io"].append(L[bb, io_t].cpu()); rec["s"].append(L[bb, S_t].cpu())
            rec["oth"].append(oth.cpu()); rec["top"].append(cat.cpu())
    finally:
        for h in handles:
            h.remove()
    R = {k: torch.cat(v).numpy() for k, v in rec.items()}
    return {"auc": float(roc_auc_score(y, -R["m"])),
            "io_rel": float(np.mean(R["io"] - R["oth"])),
            "s_rel": float(np.mean(R["s"] - R["oth"])),
            "top_io": float(np.mean(R["top"] == 0)), "top_s": float(np.mean(R["top"] == 1))}


def auc_ci(a, n_pos, n_neg):
    """95% interval for an AUC (Hanley & McNeil 1982)."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt(max(a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a), 0) / (n_pos * n_neg))
    return [float(max(0.0, a - 1.96 * se)), float(min(1.0, a + 1.96 * se))]
