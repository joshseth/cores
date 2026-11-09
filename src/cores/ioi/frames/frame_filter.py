#!/usr/bin/env python3
"""
frame_filter.py: one behavior filter for every frame (original FRAMES + EXTRA_FRAMES): GPT-2 Large accuracy on
names-only prompts with the repeated name at S2 (IO logit > S logit), --n prompts per frame. Writes the per-frame
accuracy and the kept frames (accuracy >= --min-acc) used by the IOI scripts (data/frames_v2_sink.json).

  PYTHONPATH=src python -m cores.ioi.frames.frame_filter --out frames_v2_sink.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cores.utils import get_device
from cores.ioi.common import load_model
from cores.ioi.frames.frames_extra import EXTRA_FRAMES
from cores.ioi.frames import natural_prompts as N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="gpt2-large")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--min-acc", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sink", type=int, default=1, help="prompts start with a start token (as in the final run)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = get_device()
    tok, model = load_model(args.model_name, "float32", None, device, sink=bool(args.sink))
    frames = list(N.FRAMES) + list(EXTRA_FRAMES)
    names = dict(males=N.MALE_NAMES, females=N.FEMALE_NAMES, nouns_m=[], nouns_f=[], nouns_n=[])
    rec = []
    for i, fr in enumerate(frames):
        try:
            d, texts, _, _ = N.build_natural(tok, args.n, np.random.default_rng(args.seed + i), [fr], **names, p_noun=0.0, p_pronoun=0.0)
        except RuntimeError:
            rec.append({"i": i, "frame": fr, "acc": None, "note": "no valid prompt"}); continue
        with torch.no_grad():
            L = model(input_ids=d["ids"].to(device), attention_mask=d["att"].to(device), use_cache=False).logits
        b = torch.arange(len(d["y"]))
        Lg = L[b, torch.as_tensor(d["last"])].float().cpu().numpy()
        m = Lg[np.arange(len(d["y"])), d["f"]] - Lg[np.arange(len(d["y"])), d["s"]]       # logit(A) - logit(B)
        correct = (m > 0) == (d["y"] == 0)                                                   # y = 0: IO mentioned first
        rec.append({"i": i, "frame": fr, "acc": float(correct.mean()), "logit_diff": float(np.where(d["y"] == 0, m, -m).mean())})
    kept = [r["i"] for r in rec if r["acc"] is not None and r["acc"] >= args.min_acc]
    out = {"model": args.model_name, "sink": bool(args.sink), "n_per_frame": args.n, "min_acc": args.min_acc, "n_frames": len(frames),
           "n_original": len(N.FRAMES), "kept": kept, "frames": rec}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=1))
    orig = [r for r in rec if r["i"] < len(N.FRAMES)]; new = [r for r in rec if r["i"] >= len(N.FRAMES)]
    ok = lambda rs: sum(1 for r in rs if r["acc"] is not None and r["acc"] >= args.min_acc)
    print(f"{args.model_name}: {len(frames)} frames ({len(orig)} original, {len(new)} new); kept {len(kept)} "
          f"(original {ok(orig)}, new {ok(new)}) at accuracy >= {args.min_acc}")
    for r in rec:
        if r["acc"] is None or r["acc"] < args.min_acc:
            print(f"  dropped {r['i']:>3} acc {r['acc'] if r['acc'] is None else round(r['acc'], 2)}: {r['frame'][:90]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
