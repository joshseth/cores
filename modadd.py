"""Modular addition experiment wrapper.

Usage:
    python modadd.py                    # paper-scale train + analyze + plot
    python modadd.py --quick            # smaller smoke test
    python modadd.py --only train
    python modadd.py --only analyze
    python modadd.py --only plot
    python modadd.py --skip-train
    python modadd.py --skip-train --skip-analyze
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RUN_SCRIPT = SRC / "cores" / "run_modadd.py"


def main():
    saved_argv = sys.argv[:]
    saved_cwd = Path.cwd()

    sys.argv = [str(RUN_SCRIPT)] + sys.argv[1:]
    sys.path.insert(0, str(SRC))

    try:
        os.chdir(ROOT)
        runpy.run_path(str(RUN_SCRIPT), run_name="__main__")
    finally:
        os.chdir(saved_cwd)
        sys.path.pop(0)
        sys.argv = saved_argv


if __name__ == "__main__":
    main()