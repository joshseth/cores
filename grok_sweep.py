"""Grokking sweep reproduction script.

Usage:
    python grok_sweep.py        # full paper-style run, 12 reps
    python grok_sweep.py quick  # same settings, but 3 reps

Outputs:
    experiments/grok_sweep/drift_vs_p_results.npz
    experiments/grok_sweep/drift_vs_omega_results.npz
    experiments/grok_sweep/drift_two_panel_results.npz
    experiments/grok_sweep/grok_sweep_two_panel.pdf
    experiments/grok_sweep/grok_sweep_two_panel.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, ScalarFormatter
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "grok_sweep"


# ----------------------------
# Utilities
# ----------------------------
def save_npz_with_meta(path: Path, **arrays) -> None:
    """Save numpy arrays plus small JSON-serializable metadata into one .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    for k, v in arrays.items():
        if isinstance(v, (np.ndarray, np.number)) or np.isscalar(v):
            payload[k] = np.asarray(v)
        elif isinstance(v, (list, tuple)):
            payload[k] = np.asarray(v)
        else:
            payload[f"{k}__json"] = np.array(json.dumps(v), dtype=object)

    payload["saved_at__json"] = np.array(
        json.dumps({"saved_at": datetime.now().isoformat()}), dtype=object
    )
    np.savez_compressed(path, **payload)
    print(f"Saved: {path.resolve()}")


def get_device():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda"), "reduce-overhead"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "off"
    return torch.device("cpu"), "off"


def power_fit(x, y):
    lx, ly = np.log(x), np.log(y)
    b, lc = np.polyfit(lx, ly, 1)
    r2 = 1 - np.sum((ly - (b * lx + lc)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(b), float(np.exp(lc)), float(r2)


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def exact_ode_p(p, A, p_c):
    inner = np.clip(1 - p_c / np.asarray(p, dtype=float), 1e-12, 1.0)
    return -A * np.log(inner)


def exact_ode_omega(omega_vals, K):
    return K / np.asarray(omega_vals, dtype=float)


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


# ----------------------------
# Model and data
# ----------------------------
class OneLayerTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, max_len=64, d_ff=1024):
        super().__init__()
        assert d_model % n_head == 0, f"d_model={d_model} must be divisible by n_head={n_head}"
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layer1 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, batch_first=True
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        _, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.layer1(h, src_mask=mask)
        return self.lm_head(h)


def build_data_tensors(p, device, frac=0.5, seed=42):
    a = torch.arange(p, device=device)
    b = torch.arange(p, device=device)
    A, B_grid = torch.meshgrid(a, b, indexing="ij")
    C = (A + B_grid) % p
    data = torch.stack([A.flatten(), B_grid.flatten(), C.flatten()], dim=1)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(len(data), generator=g, device=device)
    data = data[perm]

    split = int(len(data) * frac)
    X_tr, Y_tr = data[:split, :-1].contiguous(), data[:split, 1:].contiguous()
    X_te, Y_te = data[split:, :-1].contiguous(), data[split:, 1:].contiguous()
    return X_tr, Y_tr, X_te, Y_te


@torch.no_grad()
def eval_acc_and_margin_stats(model, X, Y):
    model.eval()
    logits = model(X)[:, -1, :]
    y_true = Y[:, -1]

    pred = logits.argmax(-1)
    acc = (pred == y_true).float().mean().item()

    true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    logits = logits.clone()
    logits.scatter_(1, y_true.unsqueeze(1), float("-inf"))
    max_other = logits.max(dim=1).values

    margins = true_logits - max_other
    mean_margin = margins.mean().item()
    std_margin = margins.std(unbiased=False).item()
    return acc, mean_margin, std_margin


# ----------------------------
# Training
# ----------------------------
def train_run(
    p: int,
    device,
    d_model=128,
    n_head=4,
    d_ff=512,
    lr=1e-3,
    omega=1.0,
    frac=0.5,
    batch_size=512,
    max_epochs=8000,
    eval_every_steps=1,
    grok_thresh=0.99,
    mem_thresh=0.99,
    patience_epochs=100,
    seed=42,
    compile_mode="off",
):
    torch.manual_seed(seed)

    X_tr, Y_tr, X_te, Y_te = build_data_tensors(p, device, frac=frac, seed=42)
    n_train = len(X_tr)

    model = OneLayerTransformerLM(
        vocab_size=p, d_model=d_model, n_head=n_head, d_ff=d_ff
    ).to(device)

    if device.type == "cuda" and compile_mode != "off":
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            print(f"[!] torch.compile failed, using eager: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=omega)
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = math.ceil(n_train / batch_size)
    max_steps = max_epochs * steps_per_epoch
    patience_steps = patience_epochs * steps_per_epoch

    tau_mem_step = None
    tau_grok_step = None
    z_at_grok = None

    is_full_batch = batch_size >= n_train
    if not is_full_batch:
        perm = torch.randperm(n_train, device=device)
        idx = 0

    model.train()
    global_step = 0

    while global_step < max_steps:
        if is_full_batch:
            x, y = X_tr, Y_tr
        else:
            if idx + batch_size > n_train:
                perm = torch.randperm(n_train, device=device)
                idx = 0
            batch_idx = perm[idx:idx + batch_size]
            idx += batch_size
            x, y = X_tr[batch_idx], Y_tr[batch_idx]

        opt.zero_grad(set_to_none=True)
        logits = model(x)[:, -1, :]
        loss = loss_fn(logits, y[:, -1])
        loss.backward()
        opt.step()
        global_step += 1

        if (global_step % eval_every_steps == 0) or (global_step == 1):
            tr_acc, _, _ = eval_acc_and_margin_stats(model, X_tr, Y_tr)
            te_acc, te_m, te_s = eval_acc_and_margin_stats(model, X_te, Y_te)
            te_z = te_m / max(1e-12, te_s)

            if tau_mem_step is None and tr_acc >= mem_thresh:
                tau_mem_step = global_step
            if tau_grok_step is None and te_acc >= grok_thresh:
                tau_grok_step = global_step
                z_at_grok = te_z

            if (tau_grok_step is not None) and (global_step > tau_grok_step + patience_steps):
                break

            model.train()

    if tau_mem_step is not None and tau_grok_step is not None and tau_grok_step > tau_mem_step:
        tau_drift_step = tau_grok_step - tau_mem_step
    else:
        tau_drift_step = None

    return dict(
        p=p,
        omega=omega,
        n_train=n_train,
        steps_per_epoch=steps_per_epoch,
        batch_size=min(batch_size, n_train),
        tau_mem_step=tau_mem_step,
        tau_grok_step=tau_grok_step,
        tau_drift_step=tau_drift_step,
        z_at_grok=z_at_grok,
    )


# ----------------------------
# Sweeps
# ----------------------------
def run_p_sweep(*, seeds: int, device, compile_mode: str):
    primes = [31, 43, 53, 61, 79, 89, 101]
    omega = 1.0
    frac = 0.5
    batch_size = 512
    full_batch = True
    d_model = 128
    n_head = 4
    d_ff = 512
    lr = 1e-3
    max_epochs = 30000
    eval_every_steps = 1
    grok_thresh = 0.99
    mem_thresh = 0.99

    rows = []
    for p in primes:
        effective_bs = (int(frac * p * p) + 1) if full_batch else batch_size
        print(f"\n--- p={p} (B={effective_bs}{' full' if full_batch else ''}) ---")
        for s in range(seeds):
            t0 = time.time()
            out = train_run(
                p=p,
                device=device,
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                lr=lr,
                omega=omega,
                frac=frac,
                batch_size=effective_bs,
                max_epochs=max_epochs,
                eval_every_steps=eval_every_steps,
                grok_thresh=grok_thresh,
                mem_thresh=mem_thresh,
                seed=42 + s,
                compile_mode=compile_mode,
            )
            dt = time.time() - t0
            print(
                f"  s={s}: mem={out['tau_mem_step']} grok={out['tau_grok_step']} "
                f"drift={out['tau_drift_step']} z={None if out['z_at_grok'] is None else round(out['z_at_grok'], 2)} "
                f"({dt:.1f}s)"
            )
            rows.append(out)

    pv, tau_mean, tau_sd = [], [], []
    for p in primes:
        vals = [r["tau_drift_step"] for r in rows if r["p"] == p]
        m, s = mean_sd(vals)
        if m is not None:
            pv.append(p)
            tau_mean.append(m)
            tau_sd.append(s)

    pv = np.array(pv, dtype=float)
    tau_mean = np.array(tau_mean, dtype=float)
    tau_sd = np.array(tau_sd, dtype=float)

    beta = C_pl = R2_pl = np.nan
    A_fit = B_fit = r2_ode = np.nan
    if len(pv) >= 3:
        beta, C_pl, R2_pl = power_fit(pv, tau_mean)
        try:
            popt, _ = curve_fit(
                exact_ode_p,
                pv,
                tau_mean,
                p0=[10000, 10],
                bounds=(0, [np.inf, np.min(pv) - 0.1]),
                maxfev=20000,
            )
            A_fit, B_fit = map(float, popt)
            ss_res = np.sum((tau_mean - exact_ode_p(pv, A_fit, B_fit)) ** 2)
            ss_tot = np.sum((tau_mean - np.mean(tau_mean)) ** 2)
            r2_ode = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
        except Exception as e:
            print(f"ODE fit failed: {e}")

    save_npz_with_meta(
        OUT_DIR / "drift_vs_p_results.npz",
        pv=pv,
        td_p_mean_steps=tau_mean,
        td_p_sd_steps=tau_sd,
        rows=np.array(rows, dtype=object),
        fit_p_power=np.array([beta, C_pl, R2_pl], dtype=float),
        fit_p_exact_ode=np.array([A_fit, B_fit, r2_ode], dtype=float),
        meta=dict(
            sweep="p",
            omega=omega,
            frac=frac,
            full_batch=full_batch,
            batch_size=batch_size,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            lr=lr,
            max_epochs=max_epochs,
            eval_every_steps=eval_every_steps,
            grok_thresh=grok_thresh,
            mem_thresh=mem_thresh,
            seeds=seeds,
            primes=primes,
            device=str(device),
        ),
    )


def run_omega_sweep(*, seeds: int, device, compile_mode: str):
    omegas = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    p = 53
    frac = 0.5
    batch_size = 512
    full_batch = False
    d_model = 128
    n_head = 4
    d_ff = 512
    lr = 1e-3
    max_epochs = 20000
    eval_every_steps = 1
    grok_thresh = 0.99
    mem_thresh = 0.99

    rows = []
    effective_bs = (int(frac * p * p) + 1) if full_batch else batch_size
    print(f"\n--- Fixed p={p} (B={effective_bs}{' full' if full_batch else ''}) ---")

    for om in omegas:
        print(f"\nTesting omega = {om}")
        for s in range(seeds):
            t0 = time.time()
            out = train_run(
                p=p,
                device=device,
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                lr=lr,
                omega=om,
                frac=frac,
                batch_size=effective_bs,
                max_epochs=max_epochs,
                eval_every_steps=eval_every_steps,
                grok_thresh=grok_thresh,
                mem_thresh=mem_thresh,
                seed=42 + s,
                compile_mode=compile_mode,
            )
            dt = time.time() - t0
            print(
                f"  s={s}: mem={out['tau_mem_step']} grok={out['tau_grok_step']} "
                f"drift={out['tau_drift_step']} z={None if out['z_at_grok'] is None else round(out['z_at_grok'], 2)} "
                f"({dt:.1f}s)"
            )
            out["omega"] = om
            rows.append(out)

    ov, tau_mean, tau_sd = [], [], []
    for om in omegas:
        vals = [r["tau_drift_step"] for r in rows if r["omega"] == om]
        m, s = mean_sd(vals)
        if m is not None:
            ov.append(om)
            tau_mean.append(m)
            tau_sd.append(s)

    ov = np.asarray(ov, dtype=float)
    tau_mean = np.asarray(tau_mean, dtype=float)
    tau_sd = np.asarray(tau_sd, dtype=float)

    beta = C_pl = R2_pl = np.nan
    K_fit = r2_ode = np.nan

    if len(ov) >= 3:
        beta, C_pl, R2_pl = power_fit(ov, tau_mean)
        try:
            popt, _ = curve_fit(exact_ode_omega, ov, tau_mean, p0=[2000])
            K_fit = float(popt[0])
            ss_res = np.sum((tau_mean - exact_ode_omega(ov, K_fit)) ** 2)
            ss_tot = np.sum((tau_mean - np.mean(tau_mean)) ** 2)
            r2_ode = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
        except Exception as e:
            print(f"Exact ODE fit failed: {e}")

    save_npz_with_meta(
        OUT_DIR / "drift_vs_omega_results.npz",
        om_v=ov,
        td_om_mean_steps=tau_mean,
        td_om_sd_steps=tau_sd,
        rows=np.array(rows, dtype=object),
        fit_om_power=np.array([beta, C_pl, R2_pl], dtype=float),
        fit_om_exact=np.array([K_fit, r2_ode], dtype=float),
        meta=dict(
            sweep="omega",
            p=p,
            frac=frac,
            full_batch=full_batch,
            batch_size=batch_size,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            lr=lr,
            max_epochs=max_epochs,
            eval_every_steps=eval_every_steps,
            grok_thresh=grok_thresh,
            mem_thresh=mem_thresh,
            seeds=seeds,
            omegas=omegas,
            device=str(device),
        ),
    )


def combine_results():
    pdat = np.load(OUT_DIR / "drift_vs_p_results.npz", allow_pickle=True)
    omdat = np.load(OUT_DIR / "drift_vs_omega_results.npz", allow_pickle=True)

    save_npz_with_meta(
        OUT_DIR / "drift_two_panel_results.npz",
        om_v=omdat["om_v"],
        td_om_mean_steps=omdat["td_om_mean_steps"],
        td_om_sd_steps=omdat["td_om_sd_steps"],
        pv=pdat["pv"],
        td_p_mean_steps=pdat["td_p_mean_steps"],
        td_p_sd_steps=pdat["td_p_sd_steps"],
        meta=dict(note="Two-panel drift results combined from p/omega sweep files"),
    )


# ----------------------------
# Final two-panel plot
# ----------------------------
def _apply_clean_log_ticks_y(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1, 5) * 0.1, numticks=4))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 5) * 0.1, numticks=4))
    ax.yaxis.set_minor_formatter(ScalarFormatter())


def _set_plot_style():
    # Keep LaTeX off so this runs on machines without a TeX install.
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 6,
        "font.style": "normal",
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
    })


def plot_two_panel():
    _set_plot_style()

    data = np.load(OUT_DIR / "drift_two_panel_results.npz", allow_pickle=True)

    om_v = data["om_v"] if "om_v" in data.files else np.array([])
    td_om = data["td_om_mean_steps"] if "td_om_mean_steps" in data.files else np.array([])
    sd_om = data["td_om_sd_steps"] if "td_om_sd_steps" in data.files else np.array([])

    pv = data["pv"] if "pv" in data.files else np.array([])
    td_p = data["td_p_mean_steps"] if "td_p_mean_steps" in data.files else np.array([])
    sd_p = data["td_p_sd_steps"] if "td_p_sd_steps" in data.files else np.array([])

    om_v, td_om = _clean_xy(om_v, td_om)
    pv, td_p = _clean_xy(pv, td_p)

    sd_om = np.asarray(sd_om, dtype=float) if sd_om is not None and len(sd_om) else sd_om
    sd_p = np.asarray(sd_p, dtype=float) if sd_p is not None and len(sd_p) else sd_p

    has_om = len(om_v) >= 2
    has_p = len(pv) >= 2

    beta_om = C_om = R2_om = np.nan
    if has_om and len(om_v) >= 3:
        beta_om, C_om, R2_om = power_fit(om_v, td_om)

    A_p = pcrit_p = R2_p_ode = np.nan
    if has_p and len(pv) >= 3:
        try:
            p0_A = max(np.median(td_p) * np.median(pv), 1.0)
            p0_pc = max(min(pv) * 0.5, 1.0)
            popt, _ = curve_fit(
                exact_ode_p,
                pv,
                td_p,
                p0=[p0_A, p0_pc],
                bounds=([0.0, 0.0], [np.inf, float(np.min(pv) - 1e-6)]),
                maxfev=20000,
            )
            A_p, pcrit_p = map(float, popt)
            yhat = exact_ode_p(pv, A_p, pcrit_p)
            ss_res = np.sum((td_p - yhat) ** 2)
            ss_tot = np.sum((td_p - np.mean(td_p)) ** 2)
            R2_p_ode = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

            print("\n=== ODE Fit Results (Panel B) ===")
            print(f"Omega (Optimizer Constant)    = {A_p:.2f}")
            print(f"p_crit (Architectural Limit)  = {pcrit_p:.2f}")
            print(f"R^2                           = {R2_p_ode:.4f}\n")
        except Exception as e:
            print(f"ODE fit for p-panel failed: {e}")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.75, 1.5),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.1},
    )

    # Panel A: tau vs omega
    ax = axes[0]
    if has_om:
        if sd_om is not None and len(sd_om) == len(om_v):
            ax.errorbar(
                om_v,
                td_om,
                yerr=sd_om,
                fmt="o",
                color="k",
                ms=2,
                capsize=2,
                zorder=5,
                elinewidth=0.5,
                capthick=0.5,
            )
        else:
            ax.plot(om_v, td_om, "o", color="k", ms=3, zorder=5)

        xf = np.linspace(om_v.min() * 0.85, om_v.max() * 1.15, 300)

        if np.isfinite(beta_om):
            ax.plot(xf, C_om * xf**beta_om, "-", color="red", lw=1.0,
                    label=rf"fit: $\omega^{{{beta_om:.2f}}}$")

        exp = -1.0
        lc = np.mean(np.log(td_om) - exp * np.log(om_v))
        line, = ax.plot(
            xf,
            np.exp(lc + exp * np.log(xf)),
            ":",
            color="grey",
            lw=1.0,
            alpha=1.0,
            label=r"theory: $\omega^{-1}$",
        )
        line.set_dashes([1.0, 0.5])

        ax.set_xscale("log")
        ax.set_yscale("log")

        default_om_ticks = [0.3, 0.5, 1.0, 2.0, 3.0]
        if (om_v.min() <= min(default_om_ticks)) and (om_v.max() >= max(default_om_ticks)) and (len(om_v) <= 8):
            ax.set_xticks(default_om_ticks)
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())

        _apply_clean_log_ticks_y(ax)
        ax.legend(loc="upper right", handlelength=0.25, labelspacing=0.0, handletextpad=0.25)

        if np.isfinite(R2_om):
            ax.text(0.05, 0.06, rf"$R^2={R2_om:.3f}$", transform=ax.transAxes, fontsize=7.0)

    ax.set_xlabel(r"Weight decay $\omega$", size=8)
    ax.set_ylabel(r"Delay $\tau_{\mathrm{grok}}$ (steps)", size=8)
    ax.set_title("Grok Time vs. Weight Decay", size=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # Panel B: tau vs p
    ax = axes[1]
    if has_p:
        if sd_p is not None and len(sd_p) == len(pv):
            ax.errorbar(
                pv,
                td_p,
                yerr=sd_p,
                fmt="o",
                color="k",
                ms=2,
                capsize=2,
                zorder=5,
                elinewidth=0.5,
                capthick=0.5,
            )
        else:
            ax.plot(pv, td_p, "o", color="k", ms=3, zorder=5)

        x_lo = max(pv.min() * 0.9, pcrit_p * 1.03) if np.isfinite(pcrit_p) else pv.min() * 0.9
        xf = np.linspace(x_lo, pv.max() * 1.1, 300)

        if np.isfinite(A_p) and np.isfinite(pcrit_p):
            ax.plot(xf, exact_ode_p(xf, A_p, pcrit_p), "-", color="green", lw=1.0, label="theory")

        ax.text(
            0.7,
            0.65,
            r"$-\Omega \,\log\!\left(1-\frac{p_{\mathrm{crit}}}{p}\right)$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([31, 53, 79, 101])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        _apply_clean_log_ticks_y(ax)
        ax.legend(loc="best", handlelength=0.25, labelspacing=0.0, handletextpad=0.25)

        if np.isfinite(R2_p_ode):
            ax.text(0.05, 0.06, rf"$R^2={R2_p_ode:.3f}$", transform=ax.transAxes, fontsize=7.0)

    ax.set_xlabel(r"Modulus $p$", size=8)
    ax.set_ylabel(r"Delay $\tau_{\mathrm{grok}}$ (steps)", size=8)
    ax.set_title("Grok Time vs. Redundancy", size=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    pdf_path = OUT_DIR / "grok_sweep_two_panel.pdf"
    png_path = OUT_DIR / "grok_sweep_two_panel.png"
    fig.savefig(pdf_path, dpi=500, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, dpi=500, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved figure to: {pdf_path.resolve()}")
    print(f"Saved figure to: {png_path.resolve()}")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["full", "quick", "plot-only"],
        default="full",
        help="Default is full. quick uses 3 reps instead of 12. plot-only rebuilds the final plot from saved npz files.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "plot-only":
        combine_results()
        plot_two_panel()
        return

    seeds = 3 if args.mode == "quick" else 12
    device, compile_mode = get_device()
    print(f"Using device: {device}")
    print(f"Repetitions per setting: {seeds}")

    run_p_sweep(seeds=seeds, device=device, compile_mode=compile_mode)
    run_omega_sweep(seeds=seeds, device=device, compile_mode=compile_mode)
    combine_results()
    plot_two_panel()


if __name__ == "__main__":
    main()
