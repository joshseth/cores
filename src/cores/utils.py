import numpy as np
import torch
import os
import random
from pathlib import Path
import re

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # optional: Apple MPS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def fmt_eig(z):
    if abs(z.imag) < 1e-9:
        return f"{z.real:.2f}"
    return f"{z.real:.2f}{z.imag:+.2f}j"

def list_ckpt_epochs(*run_dirs: Path) -> list[int]:
    out = set()
    for rd in run_dirs:
        d = Path(rd) 
        if not d.exists():
            continue
        for p in d.glob("epoch_*.pt"):
            m = re.search(r"epoch_(\d+)\.pt$", p.name)
            if m:
                out.add(int(m.group(1)))
    return sorted(out)