"""Subject-verb agreement experiment wrapper.

Usage:
    python sva.py run --model-name gpt2 --run-name gpt2small --seed 0
    python sva.py run --model-name gpt2-medium --run-name gpt2med --seed 0
    python sva.py plot experiments/sva/gpt2small experiments/sva/gpt2med
    python sva.py quick
"""

from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

RUN_SCRIPT = SRC / "cores" / "SVA_uni.py"
PLOT_SCRIPT = SRC / "cores" / "plot_sva0.py"

LOG_DIR = ROOT / "experiments" / "sva" / "logs"


class Tee:
    """Write stdout/stderr to both terminal and a log file."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        pass


def _run_script(path: Path, argv: list[str], *, log_path: Path | None = None):
    print(f"\n{path.relative_to(ROOT)} {' '.join(argv)}")

    saved_argv = sys.argv[:]
    saved_cwd = Path.cwd()

    sys.argv = [str(path)] + argv
    sys.path.insert(0, str(SRC))

    try:
        os.chdir(ROOT)

        if log_path is None:
            runpy.run_path(str(path), run_name="__main__")
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                tee_out = Tee(sys.stdout, f)
                tee_err = Tee(sys.stderr, f)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    runpy.run_path(str(path), run_name="__main__")

    finally:
        os.chdir(saved_cwd)
        sys.path.pop(0)
        sys.argv = saved_argv


def run_sva(args):
    run_name = args.run_name or args.model_name.split("/")[-1].replace("-", "")
    log_path = LOG_DIR / f"{run_name}_run.log"

    child_argv = [
        "--model-name", args.model_name,
        "--run-name", run_name,
        "--seed", str(args.seed),
        "--n-samples", str(args.n_samples),
        "--batch-size", str(args.batch_size),
    ]

    if args.layer_idx is not None:
        child_argv += ["--layer-idx", str(args.layer_idx)]

    if args.run_layer_sweep:
        child_argv += ["--run-layer-sweep"]

    _run_script(RUN_SCRIPT, child_argv, log_path=log_path)
    print(f"\nSaved run log to: {log_path}")


def plot_sva(dirs: list[str]):
    log_path = LOG_DIR / "plot.log"
    _run_script(PLOT_SCRIPT, dirs, log_path=log_path)
    print(f"\nSaved plot log to: {log_path}")


def quick_sva(args):
    runs = [
        ("gpt2", "gpt2small"),
        ("gpt2-medium", "gpt2med"),
        ("gpt2-large", "gpt2large"),
    ]

    output_dirs = []

    for model_name, run_name in runs:
        _run_script(
            RUN_SCRIPT,
            [
                "--model-name", model_name,
                "--run-name", run_name,
                "--seed", "0",
                "--run-layer-sweep",
            ],
            log_path=LOG_DIR / f"{run_name}_run.log",
        )
        output_dirs.append(str(ROOT / "experiments" / "sva" / run_name))

    _run_script(
        PLOT_SCRIPT,
        output_dirs,
        log_path=LOG_DIR / "plot.log",
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--model-name", type=str, required=True)
    run_p.add_argument("--run-name", type=str, default=None)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--n-samples", type=int, default=1200)
    run_p.add_argument("--batch-size", type=int, default=16)
    run_p.add_argument("--layer-idx", type=int, default=None)
    run_p.add_argument("--run-layer-sweep", action="store_true")
    run_p.set_defaults(func=run_sva)

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("dirs", nargs="+")
    plot_p.set_defaults(func=lambda args: plot_sva(args.dirs))

    quick_p = sub.add_parser("quick")
    quick_p.set_defaults(func=quick_sva)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()