"""Append the full-state control to the Markov null-ladder evidence table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
BASE = OUT / "19_markov_core_evidence.csv"
FULL = HERE / "markov_full_model_control" / "results.json"
OUTPUT = OUT / "27_markov_model_control_evidence.csv"


def row(section, condition, unit, metric, value, notes):
    return {
        "section": section,
        "condition": condition,
        "unit": unit,
        "metric": metric,
        "value": float(value),
        "notes": notes,
    }


def main():
    with BASE.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    full = json.loads(FULL.read_text(encoding="utf-8"))
    condition = "Full state (64D)"

    for record in full["cca_profiles"]:
        for index, value in enumerate(record["cca_corrs"], start=1):
            rows.append(
                row(
                    "cca",
                    condition,
                    record["pair"],
                    f"CC{index}",
                    value,
                    "top three CCA values from paired full 64D hidden states",
                )
            )

    for record in full["evaluations"]:
        unit = f"seed-{record['seed']}"
        for metric in ("keep", "remove"):
            rows.append(
                row(
                    "causal",
                    condition,
                    unit,
                    metric,
                    record[metric],
                    "full 64D state; keep/remove is intentionally non-compact",
                )
            )
        rows.append(
            row(
                "spectrum",
                condition,
                unit,
                "normalized_error",
                record["spectral_distance"],
                (
                    "best match of three true modes into the 64D fitted "
                    "spectrum; extra eigenvalues are unpenalized"
                ),
            )
        )
        rows.append(
            row(
                "overlap",
                condition,
                unit,
                "projector_overlap_with_ace",
                record["ace_coverage"],
                "full state contains ACE-3 by definition",
            )
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "condition", "unit", "metric", "value", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
