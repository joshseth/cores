"""
Subject-verb agreement experiment (Universal: GPT-2 / LLaMA / Mistral).

Extracts a 1D algorithmic core at each layer via ACE, evaluates with
ablations (keep/remove/flip), and runs surgical generation demos.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

from cores.extraction import ace_from_matrices
from cores.utils import seed_all, get_device


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE AGNOSTIC HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_embed_dim(model):
    return getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))

def get_num_layers(model):
    return getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", None))

def get_blocks(model):
    """Dynamically fetch transformer blocks for GPT-2 or LLaMA/Mistral."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture")

def get_token_id(tok, text):
    """Safely extract single token IDs, warning if tokenizers split them."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > 1:
        print(f"Warning: '{text}' tokenized to multiple IDs {ids}. Using last token.")
    return ids[-1]

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SVAArgs:
    model_name: str = "gpt2"
    seed: int = 0
    n_samples: int = 1200
    max_len: int = 64
    batch_size: int = 16
    layer_idx: int = -1          
    run_layer_sweep: bool = False
    run_flip_demo: bool = True
    run_generation_demo: bool = True
    gen_max_new_tokens: int = 50
    gen_temperature: float = 1.0
    gen_top_k: int = 50


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

def build_sva_dataset(n_total, rng):
    sing_heads = ["key", "label", "author", "pilot", "teacher", "child", "man", "woman", "mouse"]
    plur_heads = ["keys", "labels", "authors", "pilots", "teachers", "children", "men", "women", "mice"]
    sing_attract = ["cabinet", "drawer", "door", "folder", "box", "room", "student", "guard", "desk"]
    plur_attract = ["cabinets", "drawers", "doors", "folders", "boxes", "rooms", "students", "guards", "desks"]
    connectors = ["to the", "of the", "near the", "behind the", "next to the", "alongside the"]
    front_pads = ["In this ancient kingdom,", "Long ago in a distant land,",
                  "In a busy laboratory,", "Deep under the old castle,"]
    back_pads = ["in the old kingdom", "in this strange world", "in the present tense", "on most days"]
    rel_chunks = ["that the", "that guards the", "that surrounds the", "that hides the"]
    time_prefixes = ["", "In the past,"]
    styles = ["base", "front", "back", "there", "rel"]

    choice = lambda xs: xs[int(rng.integers(0, len(xs)))]
    assert n_total % 2 == 0

    texts, y, heads = [], [], []

    def add(head, attr, label):
        conn, fp, bp, rc, tp = choice(connectors), choice(front_pads), choice(back_pads), choice(rel_chunks), choice(time_prefixes)
        st = choice(styles)
        s = {"base": f"The {head} {conn} {attr}",
             "front": f"{fp} the {head} {conn} {attr}",
             "back": f"The {head} {conn} {attr} {bp}",
             "there": f"There {head} {conn} {attr}",
             "rel": f"The {head} {rc} {attr}"}[st]
        if tp: s = f"{tp} {s}"
        texts.append(s); y.append(label); heads.append(head)

    for _ in range(n_total // 2): add(choice(sing_heads), choice(plur_attract), 0)
    for _ in range(n_total // 2): add(choice(plur_heads), choice(sing_attract), 1)

    idx = np.arange(len(texts)); rng.shuffle(idx)
    return [texts[i] for i in idx], np.array([y[i] for i in idx], dtype=np.int64), [heads[i] for i in idx]

def find_head_positions(tok, input_ids, attn_mask, heads):
    """Find head-noun and last-token positions for each example (Padding Agnostic)."""
    N = input_ids.shape[0]
    head_pos, last_pos = np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64)
    for i in range(N):
        non_pad_indices = torch.nonzero(attn_mask[i] == 1).flatten()
        lp = int(non_pad_indices[-1].item())
        last_pos[i] = lp
        
        seq = input_ids[i].tolist()
        head_ids = tok.encode(" " + heads[i], add_special_tokens=False)
        head_pos[i] = lp
        
        for st in range(len(seq) - len(head_ids) + 1):
            if seq[st:st + len(head_ids)] == head_ids:
                head_pos[i] = st + len(head_ids) - 1
                break
    return head_pos, last_pos


# ═══════════════════════════════════════════════════════════════════════
# HIDDEN STATE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_states(model, input_ids, attn_mask, layer_idx, positions, batch_size, device):
    N, D = input_ids.shape[0], get_embed_dim(model)
    H = np.zeros((N, D), dtype=np.float32)
    for st in range(0, N, batch_size):
        en = min(N, st + batch_size)
        ids = input_ids[st:en].to(device)
        att = attn_mask[st:en].to(device)
        pos = torch.tensor(positions[st:en], device=device, dtype=torch.long)
        out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer_idx]
        H[st:en] = h[torch.arange(en - st, device=device), pos].float().cpu().numpy()
    return H

def extract_all_cores(model, input_ids, attn_mask, tr_idx, last_pos, n_layers,
                      is_id, are_id, was_id, were_id, batch_size, device):
    N = len(tr_idx)
    D = get_embed_dim(model)
    num_states = n_layers + 1

    H_all = np.zeros((num_states, N, D), dtype=np.float32)
    Obs_all = [torch.zeros(D, D, dtype=torch.float64, device="cpu") for _ in range(num_states)]
    pos_t = torch.tensor(last_pos[tr_idx], device=device, dtype=torch.long)
    
    for st in range(0, N, batch_size):
        en = min(N, st + batch_size)
        b = tr_idx[st:en]
        ids = input_ids[b].to(device)
        att = attn_mask[b].to(device)
        pos = pos_t[st:en]
        B, SeqLen = ids.shape
        pos_mask = torch.nn.functional.one_hot(pos, num_classes=SeqLen).to(model.dtype)
        
        with torch.enable_grad():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
            all_h = out.hidden_states 
            margin_all = margin(out.logits, is_id, are_id, was_id, were_id)
            score = (margin_all * pos_mask).sum()
            all_grads = torch.autograd.grad(score, all_h, retain_graph=False, create_graph=False)
            
            for li in range(num_states):
                h_val = all_h[li][torch.arange(B, device=device), pos].detach().float().cpu().numpy()
                H_all[li, st:en] = h_val
                g = all_grads[li]
                g_pos = (g * pos_mask.unsqueeze(-1)).sum(dim=1)
                g_cpu = g_pos.float().cpu().to(torch.float64)
                Obs_all[li] += g_cpu.T @ g_cpu

    Q_cores, mus = [], []
    for li in range(num_states):
        H = H_all[li]
        mu = H.mean(axis=0).astype(np.float32)
        Ht = torch.tensor(H, dtype=torch.float64)
        Hc = Ht - Ht.mean(0, keepdim=True)
        Cov = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)
        Obs = Obs_all[li] / max(1, N)
        
        Q_full, S = ace_from_matrices(Cov, Obs, r=2)
        
        # Compute the spectral gap: (S_1 / S_2)^2
        s1, s2 = float(S[0]), float(S[1])
        gap = (s1 / s2) ** 2 if s2 > 1e-9 else float('inf')
        
        print(f"Layer {li:02d} | Top SVs: [{s1:.3e}, {s2:.3e}] | Spectral Gap (S1/S2)² = {gap:.1f}")
        
        Q_core = Q_full[:, 0:1].numpy().astype(np.float32)
        Q_cores.append(Q_core)
        mus.append(mu)
        
    return Q_cores, mus


# ═══════════════════════════════════════════════════════════════════════
# MARGIN + OBS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def margin(L, is_id, are_id, was_id, were_id):
    return L[..., are_id] + L[..., were_id] - L[..., is_id] - L[..., was_id]

@torch.no_grad()
def margin_batch(model, ids, att, pos, is_id, are_id, was_id, were_id, device):
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    L = out.logits[torch.arange(ids.shape[0], device=device), pos]
    return margin(L, is_id, are_id, was_id, were_id).float().cpu().numpy()

def compute_obs(model, input_ids, attn_mask, last_pos, layer_idx,
                is_id, are_id, was_id, were_id, batch_size, device):
    D = get_embed_dim(model)
    Obs = torch.zeros(D, D, dtype=torch.float64, device="cpu")
    total = 0
    pos_t = torch.tensor(last_pos, device=device, dtype=torch.long)
    N = input_ids.shape[0]

    for st in range(0, N, batch_size):
        en = min(N, st + batch_size)
        ids = input_ids[st:en].to(device)
        att = attn_mask[st:en].to(device)
        pos = pos_t[st:en]
        B, SeqLen = ids.shape
        pos_mask = torch.nn.functional.one_hot(pos, num_classes=SeqLen).to(model.dtype)

        with torch.enable_grad():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer_idx]
            margin_all = margin(out.logits, is_id, are_id, was_id, were_id) 
            score = (margin_all * pos_mask).sum()
            g = torch.autograd.grad(score, h, retain_graph=False, create_graph=False)[0]
            g_pos = (g * pos_mask.unsqueeze(-1)).sum(dim=1)
            g_cpu = g_pos.float().cpu().to(torch.float64)

        Obs += g_cpu.T @ g_cpu
        total += B

    return Obs / max(1, total)

# ═══════════════════════════════════════════════════════════════════════
# CORE EXTRACTION 
# ═══════════════════════════════════════════════════════════════════════

def build_core(model, input_ids, attn_mask, tr_idx, last_pos, layer_idx,
               is_id, are_id, was_id, were_id, batch_size, device):
    H = extract_states(model, input_ids[tr_idx], attn_mask[tr_idx], layer_idx, last_pos[tr_idx], batch_size, device)
    mu = H.mean(axis=0).astype(np.float32)

    Ht = torch.tensor(H, dtype=torch.float64)
    Hc = Ht - Ht.mean(0, keepdim=True)
    Cov = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)

    Obs = compute_obs(model, input_ids[tr_idx], attn_mask[tr_idx], last_pos[tr_idx], layer_idx,
                      is_id, are_id, was_id, were_id, batch_size, device)

    Q, _ = ace_from_matrices(Cov, Obs, r=1)
    return Q.numpy().reshape(-1, 1).astype(np.float32), mu

# ═══════════════════════════════════════════════════════════════════════
# INTERVENTION HOOKS
# ═══════════════════════════════════════════════════════════════════════

def _qr(Q_np, device):
    Q, _ = np.linalg.qr(np.asarray(Q_np, np.float32))
    return torch.tensor(Q, device=device, dtype=torch.float32)


def _extract_hidden(out):
    """
    Robustly extract the hidden-states tensor from a decoder layer's forward-hook output.
    Returns (h, is_tuple) where h is the (B, SeqLen, D) tensor and is_tuple indicates
    whether the output was wrapped in a tuple/dataclass.
    """
    if torch.is_tensor(out):
        return out, False
    if isinstance(out, tuple):
        return out[0], True
    if hasattr(out, '__getitem__'):
        return out[0], True
    return out, False


def _repack(out, h, is_tuple):
    """Re-wrap the (possibly modified) hidden-states tensor back into the original format."""
    if not is_tuple:
        return h
    if isinstance(out, tuple):
        return (h,) + out[1:]
    return out

class PositionHook:
    def __init__(self, Q_np, mu_np, device, mode):
        assert mode in ("clean", "keep", "remove", "flip")
        self.mode, self.Q = mode, _qr(Q_np, device)
        self.mu = torch.tensor(mu_np.ravel(), device=device, dtype=torch.float32).view(1, -1)
        self.pos = None

    def set_pos(self, pos): self.pos = pos

    def __call__(self, module, inp, out):
        if self.mode == "clean" or self.pos is None: return out
        h, is_tuple = _extract_hidden(out)
        
        B = h.shape[0]
        x = h[torch.arange(B, device=h.device), self.pos].float() - self.mu
        xp = (x @ self.Q) @ self.Q.T
        if self.mode == "keep": x_new = xp
        elif self.mode == "remove": x_new = x - xp
        else: x_new = x - 2.0 * xp  
        h[torch.arange(B, device=h.device), self.pos] = (x_new + self.mu).to(h.dtype)
        
        return _repack(out, h, is_tuple)


@torch.no_grad()
def auc_all_interventions(model, input_ids, attn_mask, y, idx, last_pos, layer_idx,
                          Q_core, mu, is_id, are_id, was_id, were_id, batch_size, device):
    
    class MultiModeHook:
        def __init__(self, Q_np, mu_np, device):
            self.Q = _qr(Q_np, device)
            self.mu = torch.tensor(mu_np.ravel(), device=device, dtype=torch.float32).view(1, -1)
            self.pos = None

        def set_pos(self, pos): self.pos = pos

        def __call__(self, module, inp, out):
            if self.pos is None: return out
            h, is_tuple = _extract_hidden(out)
            
            B3 = h.shape[0]
            B = B3 // 3  
            pos3 = self.pos.repeat(3)
            
            x = h[torch.arange(B3, device=h.device), pos3[:B3]].float() - self.mu
            xp = (x @ self.Q) @ self.Q.T
            
            x_new = torch.cat([
                xp[:B],                             
                x[B:2*B] - xp[B:2*B],               
                x[2*B:] - 2.0 * xp[2*B:]            
            ], dim=0)
            
            h[torch.arange(B3, device=h.device), pos3[:B3]] = (x_new + self.mu).to(h.dtype)
            
            return _repack(out, h, is_tuple)

    hook = MultiModeHook(Q_core, mu, device)
    blocks = get_blocks(model)
    handle = blocks[layer_idx - 1].register_forward_hook(hook)
    
    scores_k, scores_r, scores_f = [], [], []
    
    for st in range(0, len(idx), batch_size):
        b = idx[st:st + batch_size]
        B = len(b)
        
        ids = input_ids[b].to(device).repeat(3, 1)
        att = attn_mask[b].to(device).repeat(3, 1)
        pos = torch.tensor(last_pos[b], device=device, dtype=torch.long)
        
        hook.set_pos(pos)
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        pos3 = pos.repeat(3)
        L3 = out.logits[torch.arange(B * 3, device=device), pos3]
        m3 = margin(L3, is_id, are_id, was_id, were_id).float().cpu().numpy()
        
        scores_k.append(m3[:B])
        scores_r.append(m3[B:2*B])
        scores_f.append(m3[2*B:])
        
    handle.remove()
    return (
        float(roc_auc_score(y[idx], np.concatenate(scores_k))),
        float(roc_auc_score(y[idx], np.concatenate(scores_r))),
        float(roc_auc_score(y[idx], np.concatenate(scores_f)))
    )

class GenerationHook:
    def __init__(self, Q_np, mu_np, device, mode):
        assert mode in ("clean", "remove", "flip")
        self.mode, self.Q = mode, _qr(Q_np, device)
        self.mu = torch.tensor(mu_np.ravel(), device=device, dtype=torch.float32).view(1, -1)
        self.strength = 1.0

    def __call__(self, module, inp, out):
        if self.mode == "clean" or self.strength <= 0: return out
        h, is_tuple = _extract_hidden(out)
        pos = h.shape[1] - 1
        
        x = h[:, pos].float() - self.mu
        xp = (x @ self.Q) @ self.Q.T
        if self.mode == "remove": x_full = x - xp
        else: x_full = x - 2.0 * xp
        h[:, pos] = (x + self.strength * (x_full - x) + self.mu).to(h.dtype)
        
        return _repack(out, h, is_tuple)

# ═══════════════════════════════════════════════════════════════════════
# FLIP DEMO
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def verb_probs(prompt, model, tok, device, is_id, are_id, was_id, were_id,
               layer_idx, Q_core, mu, mode):
    enc = tok(prompt, return_tensors="pt")
    ids, att = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    pos = (att.sum(1) - 1).long()
    if mode == "clean" or layer_idx == 0:
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
    else:
        hook = PositionHook(Q_core, mu, device, mode)
        blocks = get_blocks(model)
        handle = blocks[layer_idx - 1].register_forward_hook(hook)
        hook.set_pos(pos)
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        handle.remove()
    p = softmax(out.logits[0, pos.item()], dim=-1).float().cpu().numpy()
    return {"is": float(p[is_id]), "are": float(p[are_id]),
            "was": float(p[was_id]), "were": float(p[were_id])}

# ═══════════════════════════════════════════════════════════════════════
# SURGICAL GENERATION
# ═══════════════════════════════════════════════════════════════════════

class GenerationHook:
    def __init__(self, Q_np, mu_np, device, mode):
        assert mode in ("clean", "remove", "flip")
        self.mode, self.Q = mode, _qr(Q_np, device)
        self.mu = torch.tensor(mu_np.ravel(), device=device, dtype=torch.float32)
        self.strength = 1.0

    def __call__(self, module, inp, out):
        if self.mode == "clean" or self.strength <= 0: return out
        h, is_tuple = _extract_hidden(out)
        pos = h.shape[1] - 1
        x = h[:, pos].float() - self.mu
        xp = (x @ self.Q) @ self.Q.T
        if self.mode == "remove": x_full = x - xp
        else: x_full = x - 2.0 * xp
        h[:, pos] = (x + self.strength * (x_full - x) + self.mu).to(h.dtype)
        return _repack(out, h, is_tuple)


def copula_margin_lse(logits, is_id, are_id, was_id, were_id):
    sP = torch.logsumexp(logits[:, [are_id, were_id]], dim=-1)
    sS = torch.logsumexp(logits[:, [is_id, was_id]], dim=-1)
    return float((sP - sS).item())


@torch.no_grad()
def generate_surgical(prompt, model, tok, device, seed, layer_idx,
                      Q_core, mu, is_id, are_id, was_id, were_id, *,
                      max_new_tokens=50, temperature=1.0, top_k=50,
                      eps=1.0, s0=0.20, s_cap=50.0):
    seed_all(seed)
    enc = tok(prompt, return_tensors="pt")
    ids, att = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    hook = GenerationHook(Q_core, mu, device, "flip")
    blocks = get_blocks(model)
    handle = blocks[layer_idx - 1].register_forward_hook(hook) if layer_idx > 0 else None

    for _ in range(max_new_tokens):
        if hook and handle:
            hook.strength = 0.0
            lp0 = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, -1]
            m0 = copula_margin_lse(lp0, is_id, are_id, was_id, were_id)

            hook.strength = s0
            lp1 = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, -1]
            m1 = copula_margin_lse(lp1, is_id, are_id, was_id, were_id)

            g = (m1 - m0) / max(s0, 1e-8)
            m_tgt = -eps if m0 > 0 else eps
            s_star = float(np.clip((m_tgt - m0) / g, -s_cap, s_cap)) if abs(g) > 1e-8 else 0.0
            hook.strength = s_star

        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        logits = out.logits[:, -1] / max(temperature, 1e-5)
        if top_k > 0:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        att = torch.cat([att, torch.ones_like(nxt)], dim=1)

    if handle: handle.remove()
    return tok.decode(ids[0].cpu(), skip_special_tokens=True)


@torch.no_grad()
def generate_clean(prompt, model, tok, device, seed, *,
                   max_new_tokens=50, temperature=1.0, top_k=50):
    seed_all(seed)
    enc = tok(prompt, return_tensors="pt")
    ids, att = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    
    for _ in range(max_new_tokens):
        # Dropped past_key_values and set use_cache=False for universal stability
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        logits = out.logits[:, -1] / max(temperature, 1e-5)
        
        if top_k > 0:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        att = torch.cat([att, torch.ones_like(nxt)], dim=1)
        
    return tok.decode(ids[0].cpu(), skip_special_tokens=True)

# ═══════════════════════════════════════════════════════════════════════
# KNOB FIT 
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_z_and_margin(model, input_ids, attn_mask, idx, last_pos, layer_idx,
                         Q_core, mu, is_id, are_id, was_id, were_id, batch_size, device):
    q = torch.tensor(Q_core[:, 0], device=device, dtype=torch.float32)
    mu_t = torch.tensor(mu, device=device, dtype=torch.float32)
    zs, ms = [], []
    for st in range(0, len(idx), batch_size):
        b = idx[st:st + batch_size]
        ids = input_ids[b].to(device)
        att = attn_mask[b].to(device)
        pos = torch.tensor(last_pos[b], device=device, dtype=torch.long)
        out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer_idx]
        B = ids.shape[0]
        x = h[torch.arange(B, device=device), pos].float()
        zs.append(((x - mu_t) @ q).cpu().numpy())
        L = out.logits[torch.arange(B, device=device), pos]
        ms.append(margin(L, is_id, are_id, was_id, were_id).float().cpu().numpy())
    return np.concatenate(zs).astype(np.float64), np.concatenate(ms).astype(np.float64)

def fit_affine(x, y):
    X = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - a*x - b)**2) / (np.sum((y - y.mean())**2) + 1e-12)
    return float(a), float(b), float(r2)

# ═══════════════════════════════════════════════════════════════════════
# LAYER SWEEP
# ═══════════════════════════════════════════════════════════════════════

def layer_sweep(model, input_ids, attn_mask, y, te, last_pos,
                n_layers, Q_cores, mus, is_id, are_id, was_id, were_id, batch_size, device):
    scores_te = []
    for st in range(0, len(te), batch_size):
        b = te[st:st + batch_size]
        pos = torch.tensor(last_pos[b], device=device, dtype=torch.long)
        scores_te.append(margin_batch(model, input_ids[b].to(device), attn_mask[b].to(device),
                                      pos, is_id, are_id, was_id, were_id, device))
    base_auc = float(roc_auc_score(y[te], np.concatenate(scores_te)))
    print(f"Base AUC: {base_auc:.3f}")

    rows = []
    for li in range(n_layers + 1):
        if li == 0:
            a_keep, a_rem, a_flip = base_auc, base_auc, base_auc
        else:
            a_keep, a_rem, a_flip = auc_all_interventions(
                model, input_ids, attn_mask, y, te, last_pos, li, 
                Q_cores[li], mus[li], is_id, are_id, was_id, were_id, batch_size, device
            )
        print(f"layer={li:02d} | BASE={base_auc:.3f} KEEP={a_keep:.3f} REMOVE={a_rem:.3f} FLIP={a_flip:.3f}")
        rows.append(dict(layer_idx=li, behavior_auc_base=base_auc, behavior_auc_keep=a_keep, behavior_auc_remove=a_rem, behavior_auc_flip=a_flip))
    return rows

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))

def load_model(model_name, device, dtype="auto"):
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "auto":
        # Default to bfloat16 for modern models if on MPS/CUDA
        torch_dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    else:
        raise ValueError(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    model.eval()
    print("dtype:", next(model.parameters()).dtype)
    return model


def run_experiment(run_dir: Path, args: SVAArgs):
    seed_all(args.seed)
    device = get_device()
    print(f"device: {device} | model: {args.model_name}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # Modern models default to bfloat16 for better numerical stability 
    dtype_to_use = "bfloat16" if any(x in args.model_name.lower() for x in ["llama", "mistral", "gemma"]) else "float16"
    model = load_model(args.model_name, device, dtype=dtype_to_use)

    n_layers = get_num_layers(model)
    
    is_id = get_token_id(tok, " is")
    are_id = get_token_id(tok, " are")
    was_id = get_token_id(tok, " was")
    were_id = get_token_id(tok, " were")

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", {"args": asdict(args)})

    rng = np.random.default_rng(args.seed)
    texts, y, heads = build_sva_dataset(args.n_samples, rng)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
    input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
    N = len(texts)
    perm = rng.permutation(N)
    tr, te = perm[:N//2], perm[N//2:]
    
    head_pos, last_pos = find_head_positions(tok, input_ids, attn_mask, heads)

    sweep_results = None
    if args.run_layer_sweep:
        print("\n=== EXTRACTING CORES (ALL LAYERS) ===")
        Q_cores, mus = extract_all_cores(model, input_ids, attn_mask, tr, last_pos, n_layers,
                                         is_id, are_id, was_id, were_id, args.batch_size, device)
        
        print("\n=== LAYER SWEEP ABLATIONS ===")
        sweep_results = layer_sweep(model, input_ids, attn_mask, y, te, last_pos,
                                    n_layers, Q_cores, mus, is_id, are_id, was_id, were_id, args.batch_size, device)
        save_json(run_dir / "layer_sweep.json", {"args": asdict(args), "results": sweep_results})

        if args.layer_idx == -1:
            cands = [r for r in sweep_results if np.isfinite(r.get("behavior_auc_flip", np.nan))]
            if cands:
                best = min(cands, key=lambda r: r["behavior_auc_flip"])
                args = replace(args, layer_idx=best["layer_idx"])
                print(f"Selected layer {args.layer_idx} (FLIP AUC={best['behavior_auc_flip']:.3f})")
                
        Q_core = Q_cores[args.layer_idx]
        mu = mus[args.layer_idx]

    else:
        if args.layer_idx == -1:
            cache = run_dir / "layer_sweep.json"
            if cache.exists():
                sweep_results = json.loads(cache.read_text()).get("results", [])
                cands = [r for r in sweep_results if np.isfinite(r.get("behavior_auc_flip", np.nan))]
                if cands:
                    best = min(cands, key=lambda r: r["behavior_auc_flip"])
                    args = replace(args, layer_idx=best["layer_idx"])
                    print(f"Selected layer {args.layer_idx} from cached sweep")
                    
        assert args.layer_idx >= 0, "No layer selected. Run with --run_layer_sweep first to auto-select."

        print(f"\n=== EXTRACTING CORE @ layer {args.layer_idx} ===")
        Q_core, mu = build_core(model, input_ids, attn_mask, tr, last_pos, args.layer_idx,
                                is_id, are_id, was_id, were_id, args.batch_size, device)

    print(f"Q_core shape: {Q_core.shape}")

    A_keep, A_rem, A_flip = auc_all_interventions(
        model=model, input_ids=input_ids, attn_mask=attn_mask, y=y, idx=te,
        last_pos=last_pos, layer_idx=args.layer_idx, Q_core=Q_core, mu=mu,
        is_id=is_id, are_id=are_id, was_id=was_id, were_id=were_id,
        batch_size=args.batch_size, device=device
    )
    print(f"KEEP={A_keep:.3f}  REMOVE={A_rem:.3f}  FLIP={A_flip:.3f}")

    flip_rows = []
    if args.run_flip_demo:
        templates = [
            "The key next to the cabinets", "The keys next to the cabinet",
            "The children near the guard", "In this ancient kingdom, the key to the cabinets",
            "There key near the boxes", "There keys near the box",
        ]
        print("\n=== FLIP DEMO ===")
        for t in templates:
            print(f"\n  {t!r}")
            for m in ["clean", "remove", "flip"]:
                p = verb_probs(t, model, tok, device, is_id, are_id, was_id, were_id,
                               args.layer_idx, Q_core, mu, m)
                best = max(p, key=p.get)
                print(f"    [{m:7s}] is={p['is']:.3f} are={p['are']:.3f} was={p['was']:.3f} were={p['were']:.3f} → {best}")
                flip_rows.append({"prompt": t, "mode": m, **p, "best": best})

    gated_rows = []
    if args.run_generation_demo:
        prompts = [
            "As a new field of research, artificial intelligence",
            "We hold these truths to be self-evident:",
            "Scientific research",
        ]
        print("\n=== GENERATION DEMO ===")
        for ptxt in prompts:
            clean = generate_clean(ptxt, model, tok, device, args.seed,
                                   max_new_tokens=args.gen_max_new_tokens,
                                   temperature=args.gen_temperature, top_k=args.gen_top_k)
            steered = generate_surgical(ptxt, model, tok, device, args.seed, args.layer_idx,
                                        Q_core, mu, is_id, are_id, was_id, were_id,
                                        max_new_tokens=args.gen_max_new_tokens,
                                        temperature=args.gen_temperature, top_k=args.gen_top_k)
            print(f"\n  PROMPT: {ptxt!r}")
            print(f"  CLEAN:   {clean}")
            print(f"  STEERED: {steered}")
            gated_rows.append({"prompt": ptxt, "clean": clean, "steered": steered})

        gen_path = run_dir / "generation_outputs.txt"
        with gen_path.open("w", encoding="utf-8") as f:
            for row in gated_rows:
                f.write(f"PROMPT: {row['prompt']!r}\n")
                f.write(f"CLEAN:\n{row['clean']}\n")
                f.write(f"STEERED:\n{row['steered']}\n")
                f.write("\n" + "=" * 80 + "\n\n")
        print(f"Saved: {gen_path}")

    z_te, m_te = collect_z_and_margin(model, input_ids, attn_mask, te, last_pos, args.layer_idx,
                                      Q_core, mu, is_id, are_id, was_id, were_id, args.batch_size, device)
    a, b, r2 = fit_affine(z_te, m_te)
    print(f"\nKnob fit: a={a:.4f} b={b:.4f} R²={r2:.4f}")

    save_json(run_dir / "results.json", {
        "final_layer_idx": args.layer_idx,
        "causal": {"keep_auc": A_keep, "remove_auc": A_rem, "flip_auc": A_flip},
        "flip_demo": flip_rows,
        "gated_demo": gated_rows,
        "layer_sweep": sweep_results,
    })

    payload_dir = run_dir / "plot_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    np.savez(payload_dir / "knob_fit_test.npz",
             z=z_te.astype(np.float32), m=m_te.astype(np.float32),
             y=y[te].astype(np.int64), layer_idx=np.int64(args.layer_idx),
             model_name=np.array([args.model_name], dtype=object),
             fit_r2=np.float32(r2))
    print(f"Saved: {payload_dir / 'knob_fit_test.npz'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--layer-idx", type=int, default=-1)
    p.add_argument("--run-layer-sweep", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--gen-temperature", type=float, default=1.0)
    p.add_argument("--gen-top-k", type=int, default=50)
    cli = p.parse_args()

    name = cli.run_name or cli.model_name.split("/")[-1].replace("-", "")
    run_dir = Path(f"experiments/sva/{name}")

    args = SVAArgs(
        model_name=cli.model_name, seed=cli.seed,
        n_samples=cli.n_samples, batch_size=cli.batch_size,
        layer_idx=cli.layer_idx,
        run_layer_sweep=cli.run_layer_sweep,
        gen_temperature=cli.gen_temperature, gen_top_k=cli.gen_top_k,
    )
    run_experiment(run_dir, args)