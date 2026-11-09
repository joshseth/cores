"""GPT-2 SVA robustness to the number of Jacobian samples.

This experiment isolates the finite-sample approximation in ACE's sensitivity
Gram matrix.  It holds fixed:

  * the 1,200-prompt paper dataset and its 600/600 train/test split,
  * the full 600-prompt activity covariance and centering mean,
  * GPT-2 Small, layer 11, rank 1, and the held-out test prompts.

Only the number of per-prompt Jacobians used to estimate Obs = E[J^T J]
changes.  Subsets are nested within each replicate and independently resampled
across replicates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
ANALYSIS_REPO = HERE.parent
SRC = ANALYSIS_REPO / "src"
sys.path.insert(0, str(SRC))

from cores.SVA_uni import (  # noqa: E402
    auc_all_interventions,
    build_sva_dataset,
    find_head_positions,
    get_embed_dim,
    get_token_id,
    margin,
    margin_batch,
)
from cores.extraction import ace_from_matrices  # noqa: E402


DEFAULT_OUT = HERE / "sva_jacobian_robustness"


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_gpt2(device: torch.device):
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tok


def reproduce_dataset(tok, seed: int = 0, n_total: int = 1200):
    rng = np.random.default_rng(seed)
    texts, y, heads = build_sva_dataset(n_total, rng)
    y_attr = 1 - y  # the attractor always has the opposite number
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    perm = rng.permutation(n_total)
    tr = perm[: n_total // 2]
    te = perm[n_total // 2 :]
    _, last_pos = find_head_positions(tok, input_ids, attn_mask, heads)
    return input_ids, attn_mask, y, y_attr, tr, te, last_pos


def collect_hidden_and_jacobians(
    model,
    input_ids,
    attn_mask,
    tr,
    last_pos,
    layer_idx,
    token_ids,
    batch_size,
    device,
):
    """Collect one hidden vector and exact per-prompt margin Jacobian per row."""
    n = len(tr)
    d = get_embed_dim(model)
    hidden = np.zeros((n, d), dtype=np.float32)
    jac = np.zeros((n, d), dtype=np.float32)
    pos_all = torch.tensor(last_pos[tr], device=device, dtype=torch.long)
    is_id, are_id, was_id, were_id = token_ids

    started = time.time()
    for st in range(0, n, batch_size):
        en = min(n, st + batch_size)
        b = tr[st:en]
        ids = input_ids[b].to(device)
        att = attn_mask[b].to(device)
        pos = pos_all[st:en]
        seq_len = ids.shape[1]
        pos_mask = torch.nn.functional.one_hot(
            pos, num_classes=seq_len
        ).to(model.dtype)

        with torch.enable_grad():
            out = model(
                input_ids=ids,
                attention_mask=att,
                output_hidden_states=True,
                use_cache=False,
            )
            h = out.hidden_states[layer_idx]
            margin_all = margin(
                out.logits, is_id, are_id, was_id, were_id
            )
            score = (margin_all * pos_mask).sum()
            g = torch.autograd.grad(
                score, h, retain_graph=False, create_graph=False
            )[0]

        row = torch.arange(en - st, device=device)
        h_pos = h[row, pos].detach().float().cpu().numpy()
        g_pos = (g * pos_mask.unsqueeze(-1)).sum(dim=1)
        hidden[st:en] = h_pos
        jac[st:en] = g_pos.detach().float().cpu().numpy()

        elapsed = time.time() - started
        print(
            f"jacobians {en:3d}/{n}  elapsed={elapsed:7.1f}s",
            flush=True,
        )

    return hidden, jac


@torch.no_grad()
def clean_auc(
    model,
    input_ids,
    attn_mask,
    y,
    te,
    last_pos,
    token_ids,
    batch_size,
    device,
):
    is_id, are_id, was_id, were_id = token_ids
    scores = []
    for st in range(0, len(te), batch_size):
        b = te[st : st + batch_size]
        pos = torch.tensor(last_pos[b], device=device, dtype=torch.long)
        scores.append(
            margin_batch(
                model,
                input_ids[b].to(device),
                attn_mask[b].to(device),
                pos,
                is_id,
                are_id,
                was_id,
                were_id,
                device,
            )
        )
    return float(roc_auc_score(y[te], np.concatenate(scores)))


def covariance_from_hidden(hidden: np.ndarray):
    h = torch.tensor(hidden, dtype=torch.float64)
    hc = h - h.mean(0, keepdim=True)
    return (hc.T @ hc) / max(1, len(h) - 1)


def core_from_jacobians(cov, jac_subset: np.ndarray):
    g = torch.tensor(jac_subset, dtype=torch.float64)
    obs = (g.T @ g) / max(1, len(g))
    q, spectrum = ace_from_matrices(cov, obs, r=1)
    return q.numpy().astype(np.float32), spectrum.numpy()


def evaluate_core(
    model,
    input_ids,
    attn_mask,
    y,
    te,
    last_pos,
    layer_idx,
    q,
    mu,
    token_ids,
    batch_size,
    device,
):
    is_id, are_id, was_id, were_id = token_ids
    return auc_all_interventions(
        model=model,
        input_ids=input_ids,
        attn_mask=attn_mask,
        y=y,
        idx=te,
        last_pos=last_pos,
        layer_idx=layer_idx,
        Q_core=q,
        mu=mu,
        is_id=is_id,
        are_id=are_id,
        was_id=was_id,
        were_id=were_id,
        batch_size=batch_size,
        device=device,
    )


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    device = select_device()
    print(f"device={device} threads={torch.get_num_threads()}", flush=True)
    model, tok = load_gpt2(device)
    print(f"dtype={next(model.parameters()).dtype}", flush=True)

    (
        input_ids,
        attn_mask,
        y,
        y_attr,
        tr,
        te,
        last_pos,
    ) = reproduce_dataset(tok, seed=args.seed)
    token_ids = (
        get_token_id(tok, " is"),
        get_token_id(tok, " are"),
        get_token_id(tok, " was"),
        get_token_id(tok, " were"),
    )

    cache_path = args.out_dir / "gpt2_layer11_hidden_jacobians.npz"
    if cache_path.exists() and not args.recompute:
        cache = np.load(cache_path)
        hidden = cache["hidden"]
        jac = cache["jacobians"]
        print(f"loaded cache: {cache_path}", flush=True)
    else:
        hidden, jac = collect_hidden_and_jacobians(
            model,
            input_ids,
            attn_mask,
            tr,
            last_pos,
            args.layer_idx,
            token_ids,
            args.batch_size,
            device,
        )
        np.savez_compressed(
            cache_path,
            hidden=hidden,
            jacobians=jac,
            train_indices=tr,
            test_indices=te,
            last_positions=last_pos,
            labels=y,
            attractor_labels=y_attr,
        )
        print(f"saved cache: {cache_path}", flush=True)

    n_train = len(jac)
    sample_sizes = sorted(set(args.sample_sizes))
    if sample_sizes[-1] > n_train:
        raise ValueError(
            f"largest sample size {sample_sizes[-1]} exceeds {n_train}"
        )

    cov = covariance_from_hidden(hidden)
    mu = hidden.mean(axis=0).astype(np.float32)
    q_full, spectrum_full = core_from_jacobians(cov, jac)
    gap_full = float(spectrum_full[0] / spectrum_full[1])
    baseline = clean_auc(
        model,
        input_ids,
        attn_mask,
        y,
        te,
        last_pos,
        token_ids,
        args.batch_size,
        device,
    )
    print(
        f"baseline_auc={baseline:.6f} full_spectral_gap={gap_full:.1f}",
        flush=True,
    )

    results_path = args.out_dir / "results.json"
    if results_path.exists() and not args.recompute:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
    else:
        rows = []

    existing = {
        (int(row["jacobian_samples"]), int(row["replicate"])) for row in rows
    }
    permutations = [
        np.random.default_rng(args.subset_seed + rep).permutation(n_train)
        for rep in range(args.repeats)
    ]

    for n_jac in sample_sizes:
        reps = [0] if n_jac == n_train else range(args.repeats)
        for rep in reps:
            key = (n_jac, rep)
            if key in existing:
                print(f"skip completed n={n_jac} rep={rep}", flush=True)
                continue

            subset = np.arange(n_train) if n_jac == n_train else permutations[rep][:n_jac]
            q, spectrum = core_from_jacobians(cov, jac[subset])
            alignment = float(abs(q[:, 0] @ q_full[:, 0]))
            started = time.time()
            keep_auc, remove_auc, flip_auc = evaluate_core(
                model,
                input_ids,
                attn_mask,
                y,
                te,
                last_pos,
                args.layer_idx,
                q,
                mu,
                token_ids,
                args.batch_size,
                device,
            )
            elapsed = time.time() - started
            row = {
                "jacobian_samples": int(n_jac),
                "replicate": int(rep),
                "subset_seed": int(args.subset_seed + rep),
                "alignment_to_full": alignment,
                "keep_auc": float(keep_auc),
                "remove_auc": float(remove_auc),
                "flip_auc": float(flip_auc),
                "top_weight": float(spectrum[0]),
                "second_weight": float(spectrum[1]),
                "spectral_gap": float(spectrum[0] / spectrum[1]),
                "eval_seconds": float(elapsed),
            }
            rows.append(row)
            existing.add(key)
            payload = {
                "protocol": {
                    "model": "gpt2",
                    "dataset": "paper SVA; opposite-number attractors",
                    "n_total": 1200,
                    "n_activity_covariance": 600,
                    "n_test": 600,
                    "layer_idx": args.layer_idx,
                    "rank": 1,
                    "sample_sizes": sample_sizes,
                    "repeats": args.repeats,
                    "seed": args.seed,
                    "subset_seed": args.subset_seed,
                    "device": str(device),
                    "dtype": str(next(model.parameters()).dtype),
                },
                "baseline_auc": baseline,
                "full_spectral_gap": gap_full,
                "rows": sorted(
                    rows,
                    key=lambda x: (
                        x["jacobian_samples"],
                        x["replicate"],
                    ),
                ),
            }
            save_json(results_path, payload)
            print(
                f"n={n_jac:3d} rep={rep} align={alignment:.4f} "
                f"keep={keep_auc:.4f} remove={remove_auc:.4f} "
                f"flip={flip_auc:.4f} eval={elapsed:.1f}s",
                flush=True,
            )

    print(f"saved results: {results_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layer-idx", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=2401)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 600],
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
