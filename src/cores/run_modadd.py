"""
Mod-add grokking: full pipeline runner.

Usage:
  python run_modadd.py                              # train + analyze + plot
  python run_modadd.py --skip-train                 # analyze + plot
  python run_modadd.py --skip-train --skip-analyze  # plot only
  python run_modadd.py --only train                 # train only
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────
SEED        = 1031
K_MODELS    = 3
VOCAB_SIZE  = 53
D_MODEL     = 128
BATCH_SIZE  = 512
LR          = 1e-3

WD            = 1.0       # weight decay for the un-branched arm
TOTAL_EPOCHS  = 20000      # bump to 20000 for the paper run
BRANCH_EPOCH  = 900       # where the WD=0 branch forks from WD=1
DENSE_SAVE    = 100       # save cadence before BRANCH_EPOCH
SPARSE_SAVE   = 500       # save cadence after BRANCH_EPOCH (use 500 for paper run)
LATE_AFTER    = 2000

# ── Paths ────────────────────────────────────────────────────────────
BASE     = Path("experiments/modular_addition")
WD1_DIR  = BASE / "ckpts" / "wd1"
WD0_DIR  = BASE / "ckpts" / "wd0"
RES_WD1  = BASE / "results" / "wd1_res"
RES_WD0  = BASE / "results" / "wd0_res"
FIG_DIR  = BASE

# -- Testing -------------

def apply_quick_config():
    """Small smoke test config.

    This is not the paper setting. 
    """
    global K_MODELS, TOTAL_EPOCHS, BRANCH_EPOCH, DENSE_SAVE, SPARSE_SAVE, LATE_AFTER

    K_MODELS = 2
    TOTAL_EPOCHS = 1000
    BRANCH_EPOCH = 800
    DENSE_SAVE = 100
    SPARSE_SAVE = 100
    LATE_AFTER = 800
# ----------------------------

def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


# ── Train ────────────────────────────────────────────────────────────
def run_train():
    from cores.utils import seed_all
    from cores.modular_addition_experiment import train, E2Args

    assert BRANCH_EPOCH % DENSE_SAVE == 0, \
        f"BRANCH_EPOCH={BRANCH_EPOCH} not divisible by DENSE_SAVE={DENSE_SAVE}"
    assert BRANCH_EPOCH < TOTAL_EPOCHS, \
        f"BRANCH_EPOCH={BRANCH_EPOCH} must be < TOTAL_EPOCHS={TOTAL_EPOCHS}"

    common = dict(lr=LR, batch_size=BATCH_SIZE, k_models=K_MODELS, seed=SEED)

    # Phase 1: WD={WD} from 0 → TOTAL_EPOCHS, cadence flips at BRANCH_EPOCH
    _banner(f"PHASE 1: WD={WD}, Epochs: 0 to {TOTAL_EPOCHS}")
    seed_all(SEED)
    train(WD1_DIR, E2Args(
        **common,
        epochs=TOTAL_EPOCHS,
        weight_decay=WD,
        save_every=DENSE_SAVE,
        save_every_late=SPARSE_SAVE,
        late_after=LATE_AFTER,
    ), vocab_size=VOCAB_SIZE)

    # Phase 2: WD=0 fork from BRANCH_EPOCH → TOTAL_EPOCHS, sparse cadence
    branch_ckpt = WD1_DIR / f"epoch_{BRANCH_EPOCH:06d}.pt"
    assert branch_ckpt.exists(), f"missing branch checkpoint: {branch_ckpt}"

    _banner(f"PHASE 2: WD=0.0, Epochs: {BRANCH_EPOCH} to {TOTAL_EPOCHS}")
    seed_all(SEED)
    train(WD0_DIR, E2Args(
        **common,
        epochs=TOTAL_EPOCHS,
        weight_decay=0.0,
        save_every=DENSE_SAVE,
        resume_from=str(branch_ckpt),
        save_every_late=SPARSE_SAVE,
        late_after=LATE_AFTER,
        reset_optimizer=True,
    ), vocab_size=VOCAB_SIZE)


# ── Analyze ──────────────────────────────────────────────────────────
def run_analyze():
    from cores.utils import get_device
    from cores.models import build_modadd_dataloader
    from cores.modular_addition_analysis import (
        sweep_core_formation, sweep_mode_spread,
        save_core_result, save_mode_result,
    )

    device = get_device()
    test_loader = build_modadd_dataloader(train=False, batch_size=BATCH_SIZE)

    arms = [
        # label,  ckpt_dir, out_dir, prefix_dir (for resolving epochs that live in WD=1)
        ("WD=1",  WD1_DIR,  RES_WD1, None),
        ("WD=0",  WD0_DIR,  RES_WD0, WD1_DIR),
    ]
    for label, ckpt_dir, out_dir, prefix_dir in arms:
        out_dir.mkdir(parents=True, exist_ok=True)
        _banner(f"ANALYZE: {label}")

        kw = dict(
            test_loader=test_loader,
            device=device,
            prefix_dir=prefix_dir,
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            m_ixs=tuple(range(K_MODELS)),
        )

        cf = out_dir / "core_formation.npz"
        save_core_result(cf, sweep_core_formation(ckpt_dir, **kw))
        print(f"Saved: {cf}")

        ms = out_dir / "mode_spread.npz"
        save_mode_result(ms, sweep_mode_spread(ckpt_dir, **kw))
        print(f"Saved: {ms}")


# ── Plot ─────────────────────────────────────────────────────────────
def run_plot():
    import numpy as np
    from cores.utils import get_device
    from cores.models import build_modadd_dataloader
    from cores.modular_addition_analysis import load_core_result
    from cores.plot_mod_add import figure1, figure2

    device = get_device()
    test_loader = build_modadd_dataloader(train=False, batch_size=BATCH_SIZE)
    ring_eps = np.sin(np.pi / VOCAB_SIZE)

    wd1 = load_core_result(RES_WD1 / "core_formation.npz")
    wd0 = load_core_result(RES_WD0 / "core_formation.npz")

    if TOTAL_EPOCHS <= 1000:
        fig1_clock_epochs = (0, 300, BRANCH_EPOCH, 900, TOTAL_EPOCHS)
        fig1_max_epoch = TOTAL_EPOCHS
    else:
        fig1_clock_epochs = (0, 300, 800, 900, 2000)
        fig1_max_epoch = 2000

    _banner("PLOT: Figure 1")
    figure1(
        wd1,
        run_dir=WD1_DIR,
        test_loader=test_loader,
        device=device,
        ring_eps=ring_eps,
        clock_epochs=fig1_clock_epochs,
        max_epoch=fig1_max_epoch,
        save_path=str(FIG_DIR / "mod_add_fig1.pdf"),
    )

    _banner("PLOT: Figure 2")
    ms_wd1 = dict(np.load(RES_WD1 / "mode_spread.npz", allow_pickle=False))
    ms_wd0 = dict(np.load(RES_WD0 / "mode_spread.npz", allow_pickle=False))
    for ms in (ms_wd1, ms_wd0):
        pr = ms.pop("present")
        ms["per_model"] = {
            "present":   [pr[s] for s in range(pr.shape[0])],
            "frac_near": ms.pop("frac_near"),
            "r2":        ms.pop("r2"),
        }
        ms["K_norm"] = int(ms["K_norm"])
        ms["grok_ep_plot"] = int(ms.get("grok_ep", wd1.grok_ep))

    figure2(wd1, wd0, ms_wd1, ms_wd0,
            run_dir_wd1=WD1_DIR, run_dir_wd0=WD0_DIR,
            test_loader=test_loader, device=device, ring_eps=ring_eps,
            clock_epoch=TOTAL_EPOCHS,
            save_path=str(FIG_DIR / "mod_add_fig2.pdf"))


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["train", "analyze", "plot"], default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller smoke-test version: 2 models, 1000 epochs, branch at 800.",
    )
    args = parser.parse_args()

    if args.quick:
        apply_quick_config()
        _banner(
            "QUICK MODE: 2 models, 1000 epochs, branch at 800. "
            "This is not the paper setting."
        )

    if args.only:
        {"train": run_train, "analyze": run_analyze, "plot": run_plot}[args.only]()
    else:
        if not args.skip_train:
            run_train()
        if not args.skip_analyze:
            run_analyze()
        run_plot()


if __name__ == "__main__":
    main()