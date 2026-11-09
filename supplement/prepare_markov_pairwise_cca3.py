"""Append the pairwise full-state CCA-3 control to figure-19 evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
BASE = OUT / "19_markov_core_evidence.csv"
CCA3 = HERE / "markov_pairwise_cca3_control" / "results.json"
OUTPUT = OUT / "28_markov_pairwise_cca3_evidence.csv"


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
    results = json.loads(CCA3.read_text(encoding="utf-8"))
    condition = "Full-state CCA-3"

    for record in results["cca_profiles"]:
        for index, value in enumerate(record["cca_corrs"], start=1):
            rows.append(
                row(
                    "cca",
                    condition,
                    record["pair"],
                    f"CC{index}",
                    value,
                    (
                        "top CCs from full 64D states; model pair always uses "
                        "different training seeds"
                    ),
                )
            )

    for record in results["per_model"]:
        unit = f"seed-{record['seed']}"
        for metric in ("keep", "remove"):
            rows.append(
                row(
                    "causal",
                    condition,
                    unit,
                    metric,
                    record[metric],
                    "rank 3; mean over 11 partner-derived canonical subspaces",
                )
            )
        rows.append(
            row(
                "spectrum",
                condition,
                unit,
                "normalized_error",
                record["spectral_distance"],
                "rank 3; mean over 11 partner-derived canonical subspaces",
            )
        )
        rows.append(
            row(
                "overlap",
                condition,
                unit,
                "projector_overlap_with_ace",
                record["overlap_with_ace"],
                "rank 3; mean over 11 partner-derived canonical subspaces",
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
