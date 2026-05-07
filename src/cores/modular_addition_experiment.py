"""
Mod-add grokking: train k models, save checkpoints.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import torch
from torch import nn

from cores.utils import get_device, seed_all
from cores.models import TwoLayerTransformer, build_modadd_dataloader, eval_accuracy


@dataclass(frozen=True)
class E2Args:
    d_model: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    batch_size: int = 512
    k_models: int = 2
    seed: int = 1031
    weight_decay: float = 1.0
    save_every: int = 100          # cadence for early epochs
    save_every_late: int = 500     # cadence after late_after
    late_after: int = 2000         # switch cadence at this epoch
    resume_from: str | None = None
    reset_optimizer: bool = False


def _should_save(ep, args):
    if ep == args.epochs:
        return True
    cadence = args.save_every if ep <= args.late_after else args.save_every_late
    return ep % cadence == 0


def train(run_dir: Path, args: E2Args, vocab_size: int = 53):
    device = get_device()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    models = [TwoLayerTransformer(vocab_size=vocab_size, d_model=args.d_model).to(device)
              for _ in range(args.k_models)]
    opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for m in models]

    train_loaders = [build_modadd_dataloader(train=True, seed=k, batch_size=args.batch_size)
                     for k in range(args.k_models)]
    test_loader = build_modadd_dataloader(train=False, batch_size=args.batch_size)

    # Resume from checkpoint (explicit path, or own latest.pt)
    start_ep = 0
    ckpt_path = None
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
    elif (ckpt_dir / "latest.pt").exists():
        ckpt_path = ckpt_dir / "latest.pt"

    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        for m, sd in zip(models, ckpt["model_state_dicts"]):
            m.load_state_dict(sd)
        if args.reset_optimizer:
            opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    for m in models]
            print("Reset optimizer state.")
        else:
            for o, osd in zip(opts, ckpt["opt_state_dicts"]):
                o.load_state_dict(osd)
        for o in opts:
            for pg in o.param_groups:
                pg["weight_decay"] = float(args.weight_decay)
        start_ep = ckpt["epoch"] + 1
        print(f"Resumed from {ckpt_path} (epoch {start_ep - 1})")

    # Train
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        for k, m in enumerate(models):
            m.train()
            for x, y in train_loaders[k]:
                x, y = x.to(device), y.to(device)
                opts[k].zero_grad()
                loss = loss_fn(m(x)[:, -1, :], y[:, -1])
                loss.backward()
                opts[k].step()

        if _should_save(ep, args):
            accs = [eval_accuracy(m, test_loader, device, last_token=True) for m in models]
            print(f"Epoch {ep:5d}  " + "  ".join(f"M{i}={a:.4f}" for i, a in enumerate(accs))
                  + f"  [{time.time()-t0:.0f}s]")
            payload = {
                "epoch": ep,
                "args": asdict(args),
                "model_state_dicts": [m.state_dict() for m in models],
                "opt_state_dicts": [o.state_dict() for o in opts],
            }
            torch.save(payload, ckpt_dir / f"epoch_{ep:06d}.pt")
            torch.save(payload, ckpt_dir / "latest.pt")

    print(f"Done. {time.time()-t0:.1f}s")
    return models


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("experiments/modular_addition/default"))
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1.0)
    p.add_argument("--k-models", type=int, default=2)
    p.add_argument("--seed", type=int, default=1031)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-every-late", type=int, default=500)
    p.add_argument("--late-after", type=int, default=2000)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--reset-opt", action="store_true")
    p.add_argument("--vocab-size", type=int, default=53)
    cli = p.parse_args()

    args = E2Args(
        epochs=cli.epochs, lr=cli.lr, batch_size=512,
        k_models=cli.k_models, seed=cli.seed,
        weight_decay=cli.wd,
        save_every=cli.save_every, save_every_late=cli.save_every_late,
        late_after=cli.late_after,
        resume_from=cli.resume_from, reset_optimizer=cli.reset_opt,
    )
    seed_all(args.seed)
    train(cli.run_dir, args, vocab_size=cli.vocab_size)