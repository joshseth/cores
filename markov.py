"""Markov experiment wrapper.

Usage:
    python markov.py              # run experiment, then plot
    python markov.py --only run   # just run src/cores/markov.py
    python markov.py --only plot  # just run src/cores/plot_markov.py
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

RUN_SCRIPT = SRC / "cores" / "markov.py"
PLOT_SCRIPT = SRC / "cores" / "plot_markov.py"


def _exec(path: Path):
    print(f"\n{path.relative_to(ROOT)}")

    saved_argv = sys.argv[:]
    saved_cwd = Path.cwd()

    # Hide wrapper flags from the child script.
    sys.argv = [str(path)]

    # Make `from cores.foo import ...` work even before/without pip install -e .
    sys.path.insert(0, str(SRC))

    try:
        os.chdir(ROOT)
        runpy.run_path(str(path), run_name="__main__")
    finally:
        os.chdir(saved_cwd)
        sys.path.pop(0)
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["run", "plot"], default=None)
    args = parser.parse_args()

    if args.only in (None, "run"):
        _exec(RUN_SCRIPT)

    if args.only in (None, "plot"):
        _exec(PLOT_SCRIPT)


if __name__ == "__main__":
    main()