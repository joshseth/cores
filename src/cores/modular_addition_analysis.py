"""
Mod-add grokking: analysis script.

"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from cores.utils import get_device, list_ckpt_epochs
from cores.models import TwoLayerTransformer, build_modadd_dataloader, eval_accuracy, collect_last_token_h2
from cores.extraction import ace_ablate
from cores.ablations import ablate
from cores.infer_mechanism import token_core_reps, fit_operator_svd


def _find_ckpt(ep: int, *run_dirs: Path) -> Path:
    """Find checkpoint for epoch ep, searching directories in order."""
    name = f"epoch_{ep:06d}.pt"
    for rd in run_dirs:
        p = Path(rd) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint epoch {ep} not found in {run_dirs}")


def load_model(ep: int, m_ix: int, device, *run_dirs: Path,
               vocab_size=53, d_model=128):
    ckpt = _find_ckpt(ep, *run_dirs)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = TwoLayerTransformer(vocab_size=vocab_size, d_model=d_model).to(device)
    model.load_state_dict(payload["model_state_dicts"][m_ix])
    model.eval()
    return model

# ═══════════════════════════════════════════════════════════════════════
# CYCLIC OPERATOR FIT + SPECTRUM
# ═══════════════════════════════════════════════════════════════════════

def ring_metrics(evals, eps=0.03):
    """Fraction of eigenvalues near unit circle, mean and p90 radius."""
    r = np.abs(evals)
    if r.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(np.abs(r - 1) <= eps)), float(np.mean(r)), float(np.quantile(r, 0.9))


def mode_bin_counts(evals, P, ring_eps):
    """
    Count how many near-unit eigenvalues fall in each of the P//2+1 frequency bins.
    Returns (present, counts) arrays of length K = P//2 + 1.
    """
    K = P // 2 + 1
    present = np.zeros(K, dtype=bool)
    counts = np.zeros(K, dtype=int)

    ev = np.asarray(evals)
    if ev.size == 0:
        return present, counts

    near = np.abs(np.abs(ev) - 1.0) <= ring_eps
    ev = ev[near]
    if ev.size == 0:
        return present, counts

    ang = np.angle(ev)
    k = np.mod(np.rint((ang * P) / (2 * np.pi)).astype(int), P)
    k_folded = np.minimum(k, P - k)

    counts = np.bincount(k_folded, minlength=K).astype(int)
    present = counts > 0
    return present, counts


# ═══════════════════════════════════════════════════════════════════════
# PER-EPOCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_epoch(*run_dirs, ep, m_ix, test_loader, device,
                  vocab_size=53, d_model=128, max_seqs=3000,
                  sample_size=3000, ring_eps=None, k_max=52,
                  svd_tol=1e-3, ridge=0.0,
                  holdout_style="none", holdout_frac=0.075,
                  holdout_seed=0):
    """
    Full per-epoch analysis: load model, extract core, fit operator, get spectrum.
    Returns dict with acc, r_nec, r_nec_strict, r_suf, r_suf_strict,
    keep_acc, remove_acc, evals, r2, r2_test, k, frac_near, present, counts.
    """
    if ring_eps is None:
        ring_eps = np.sin(np.pi / vocab_size)

    model = load_model(ep, m_ix, device, *run_dirs,
                       vocab_size=vocab_size, d_model=d_model)

    H_last = collect_last_token_h2(model, test_loader, device, max_seqs=max_seqs)

    Q_full, r_energy, r_nec_strict, r_suf_strict, base_acc = ace_ablate(
        model, H_last, device, test_loader,
        vocab_size=vocab_size, d_model=d_model, sample_size=sample_size)

    r_nec = max(r_energy, r_nec_strict)   # operational
    r_suf = max(r_energy, r_suf_strict)   # operational

    U_nec = Q_full[:, :r_nec].contiguous()
    keep_acc = ablate(model, model.layer2, U_nec, "keep", test_loader, device, last_token=True)
    remove_acc = ablate(model, model.layer2, U_nec, "remove", test_loader, device, last_token=True)

    reps = token_core_reps(model, U_nec, test_loader, device, P=vocab_size)
    A, r2, r2_test, k, evals = fit_operator_svd(
        reps, k_max=k_max, svd_tol=svd_tol, ridge=ridge,
        holdout_style=holdout_style, holdout_frac=holdout_frac,
        holdout_seed=holdout_seed)

    frac_near, _, _ = ring_metrics(evals, eps=ring_eps)
    present, counts = mode_bin_counts(evals, vocab_size, ring_eps)

    return dict(
        acc=base_acc,
        r_nec=r_nec, r_nec_strict=r_nec_strict,
        r_suf=r_suf, r_suf_strict=r_suf_strict,
        keep_acc=keep_acc, remove_acc=remove_acc,
        evals=evals, r2=r2, r2_test=r2_test, k=k, frac_near=frac_near,
        present=present, counts=counts,
    )


# ═══════════════════════════════════════════════════════════════════════
# GROK DETECTION
# ═══════════════════════════════════════════════════════════════════════

def find_grok_epoch(*run_dirs, test_loader, device, m_ixs=(0, 1, 2),
                    vocab_size=53, d_model=128, acc_thr=0.99, stride=1):
    epochs = list_ckpt_epochs(*run_dirs)
    scan = epochs[::max(1, stride)]
    for ep in scan:
        if all(eval_accuracy(
                load_model(ep, m, device, *run_dirs, vocab_size=vocab_size, d_model=d_model),
                test_loader, device, last_token=True) >= acc_thr
               for m in m_ixs):
            return ep
    return scan[-1]


# ═══════════════════════════════════════════════════════════════════════
# EPOCH SCHEDULE
# ═══════════════════════════════════════════════════════════════════════

def epoch_schedule(all_epochs, *, step=None, max_epoch=None):
    """Simple uniform subsampling of available epochs."""
    e = np.array(sorted(set(map(int, all_epochs))), dtype=int)
    if max_epoch is not None:
        e = e[e <= int(max_epoch)]
    if step is not None:
        e = e[(e % int(step)) == 0]
    return e.tolist()


# ═══════════════════════════════════════════════════════════════════════
# SWEEP: CORE FORMATION (Figure 1 A,B data + Figure 2 A,B data)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoreFormationResult:
    """Per-run results for core formation plots."""
    grok_ep: int
    epochs: np.ndarray         # (T,)
    test_acc: np.ndarray       # (S, T)
    r_nec: np.ndarray          # (S, T)  
    r_nec_strict: np.ndarray   # (S, T)  
    r_suf: np.ndarray          # (S, T)  
    r_suf_strict: np.ndarray   # (S, T)  
    keep_acc: np.ndarray       # (S, T)
    remove_acc: np.ndarray     # (S, T)


def sweep_core_formation(
    run_dir: Path, test_loader, device, *,
    prefix_dir: Path | None = None,
    m_ixs=(0, 1, 2), vocab_size=53, d_model=128,
    max_seqs=3000, sample_size=3000,
    max_epoch=None, epoch_step=None, grok_stride=1,
) -> CoreFormationResult:
    run_dir = Path(run_dir)
    dirs = (run_dir, prefix_dir) if prefix_dir else (run_dir,)

    all_epochs = list_ckpt_epochs(*dirs)
    epochs = epoch_schedule(all_epochs, step=epoch_step, max_epoch=max_epoch)

    grok_ep = find_grok_epoch(*dirs, test_loader=test_loader, device=device,
                              m_ixs=m_ixs, vocab_size=vocab_size, d_model=d_model,
                              stride=grok_stride)

    T, S = len(epochs), len(m_ixs)
    test_acc     = np.full((S, T), np.nan)
    r_nec        = np.full((S, T), np.nan)
    r_nec_strict = np.full((S, T), np.nan)
    r_suf        = np.full((S, T), np.nan)
    r_suf_strict = np.full((S, T), np.nan)
    keep_acc     = np.full((S, T), np.nan)
    remove_acc   = np.full((S, T), np.nan)

    for s, m_ix in enumerate(m_ixs):
        print(f"Core sweep: model {m_ix}")
        for t, ep in enumerate(epochs):
            out = analyze_epoch(
                *dirs, ep=ep, m_ix=m_ix, test_loader=test_loader, device=device,
                vocab_size=vocab_size, d_model=d_model,
                max_seqs=max_seqs, sample_size=sample_size)
            test_acc[s, t]     = out["acc"]
            r_nec[s, t]        = out["r_nec"]
            r_nec_strict[s, t] = out["r_nec_strict"]
            r_suf[s, t]        = out["r_suf"]
            r_suf_strict[s, t] = out["r_suf_strict"]
            keep_acc[s, t]     = out["keep_acc"]
            remove_acc[s, t]   = out["remove_acc"]
            if ep % 2000 == 0 or ep == epochs[-1] or ep == 0:
                print(f"  ep={ep:6d}  acc={out['acc']:.3f}"
                      f"  r_nec={out['r_nec']:3d}"
                      f"  r_suf={out['r_suf']:3d}"
                      f"  keep={out['keep_acc']:.3f}  rem={out['remove_acc']:.3f}")

    return CoreFormationResult(
        grok_ep=grok_ep, epochs=np.array(epochs, dtype=int),
        test_acc=test_acc,
        r_nec=r_nec, r_nec_strict=r_nec_strict,
        r_suf=r_suf, r_suf_strict=r_suf_strict,
        keep_acc=keep_acc, remove_acc=remove_acc)


# ═══════════════════════════════════════════════════════════════════════
# SWEEP: MODE SPREAD (Figure 2 D data)
# ═══════════════════════════════════════════════════════════════════════
def sweep_mode_spread(
    run_dir: Path, test_loader, device, *,
    prefix_dir: Path | None = None,
    m_ixs=(0, 1, 2), vocab_size=53, d_model=128,
    max_seqs=3000, sample_size=3000,
    ring_eps=None, k_max=52, svd_tol=1e-3, ridge=0.0,
    holdout_style="none", holdout_frac=0.075, holdout_seed=0,
    max_epoch=None, epoch_step=None, grok_stride=1,
) -> dict:
    if ring_eps is None:
        ring_eps = np.sin(np.pi / vocab_size)

    run_dir = Path(run_dir)
    dirs = (run_dir, prefix_dir) if prefix_dir else (run_dir,)

    all_epochs = list_ckpt_epochs(*dirs)
    epochs = epoch_schedule(all_epochs, step=epoch_step, max_epoch=max_epoch)

    grok_ep = find_grok_epoch(*dirs, test_loader=test_loader, device=device,
                              m_ixs=m_ixs, vocab_size=vocab_size, d_model=d_model,
                              stride=grok_stride)

    K = vocab_size // 2 + 1
    T, S = len(epochs), len(m_ixs)
    present_list = []
    frac_near = np.full((S, T), np.nan)
    r2 = np.full((S, T), np.nan)

    for s, m_ix in enumerate(m_ixs):
        print(f"Mode sweep: model {m_ix}")
        present_st = np.zeros((K, T), dtype=bool)
        for t, ep in enumerate(epochs):
            out = analyze_epoch(
                *dirs, ep=ep, m_ix=m_ix, test_loader=test_loader, device=device,
                vocab_size=vocab_size, d_model=d_model,
                max_seqs=max_seqs, sample_size=sample_size,
                ring_eps=ring_eps, k_max=k_max, svd_tol=svd_tol, ridge=ridge,
                holdout_style=holdout_style, holdout_frac=holdout_frac,
                holdout_seed=holdout_seed)
            present_st[:, t] = out["present"]
            frac_near[s, t] = out["frac_near"]
            r2[s, t] = out["r2"]
        present_list.append(present_st)

    return dict(epochs=np.array(epochs, dtype=int), grok_ep_plot=grok_ep, K_norm=K,
                per_model=dict(present=present_list, frac_near=frac_near, r2=r2))


# ═══════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════

def save_core_result(path, result: CoreFormationResult):
    np.savez_compressed(path,
                        grok_ep=result.grok_ep,
                        epochs=result.epochs,
                        test_acc=result.test_acc,
                        r_nec=result.r_nec,
                        r_nec_strict=result.r_nec_strict,
                        r_suf=result.r_suf,
                        r_suf_strict=result.r_suf_strict,
                        keep_acc=result.keep_acc,
                        remove_acc=result.remove_acc)


def load_core_result(path) -> CoreFormationResult:
    d = np.load(path)
    r_nec_strict = d["r_nec_strict"] if "r_nec_strict" in d.files else d["r_nec"]
    r_suf_strict = d["r_suf_strict"] if "r_suf_strict" in d.files else d["r_suf"]
    return CoreFormationResult(
        grok_ep=int(d["grok_ep"]),
        epochs=d["epochs"],
        test_acc=d["test_acc"],
        r_nec=d["r_nec"],
        r_nec_strict=r_nec_strict,
        r_suf=d["r_suf"],
        r_suf_strict=r_suf_strict,
        keep_acc=d["keep_acc"],
        remove_acc=d["remove_acc"])


def save_mode_result(path, result: dict):
    present_stack = np.stack(result["per_model"]["present"])  # (S, K, T)
    np.savez_compressed(path,
                        epochs=result["epochs"],
                        grok_ep=result["grok_ep_plot"],
                        K_norm=result["K_norm"],
                        present=present_stack,
                        frac_near=result["per_model"]["frac_near"],
                        r2=result["per_model"]["r2"])


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--prefix-dir", type=Path, default=None,
                   help="Directory with earlier checkpoints (pre-branch)")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--epoch-step", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=53)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--analysis", choices=["core", "mode", "both"], default="both")
    cli = p.parse_args()

    device = get_device()
    test_loader = build_modadd_dataloader(train=False, batch_size=512)
    cli.out.mkdir(parents=True, exist_ok=True)

    kw = dict(test_loader=test_loader, device=device, prefix_dir=cli.prefix_dir,
              vocab_size=cli.vocab_size, d_model=cli.d_model,
              max_epoch=cli.max_epoch, epoch_step=cli.epoch_step)

    if cli.analysis in ("core", "both"):
        result = sweep_core_formation(cli.run_dir, **kw)
        save_core_result(cli.out / "core_formation.npz", result)
        print(f"Saved: {cli.out / 'core_formation.npz'}")

    if cli.analysis in ("mode", "both"):
        mode = sweep_mode_spread(cli.run_dir, **kw)
        save_mode_result(cli.out / "mode_spread.npz", mode)
        print(f"Saved: {cli.out / 'mode_spread.npz'}")
