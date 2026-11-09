"""Prepare the evidence table for the Markov null-ladder figure."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
NULLS = HERE / "markov_null_ladder_12seed" / "results.json"
OVERLAP = HERE / "markov_overlap_12seed" / "results.json"


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add(rows, section, condition, unit, metric, value, notes=""):
    rows.append(
        {
            "section": section,
            "condition": condition,
            "unit": unit,
            "metric": metric,
            "value": float(value),
            "notes": notes,
        }
    )


def main():
    nulls = load(NULLS)
    overlap = load(OVERLAP)
    rows = []

    cca_map = [
        ("ACE-3 (trained)", "ACE-3"),
        ("Untrained ACE-3", "Untrained ACE-3"),
        ("PCA-3 (trained)", "PCA-3"),
    ]
    for display, key in cca_map:
        for record in nulls["cca_profiles"][key]:
            for index, value in enumerate(record["cca_corrs"], start=1):
                add(
                    rows,
                    "cca",
                    display,
                    record["pair"],
                    f"CC{index}",
                    value,
                    "rank 3; unit is a model pair",
                )

    random_by_draw = defaultdict(list)
    for record in nulls["ambient_random_cca_draws"]:
        random_by_draw[record["draw"]].append(record["cca_corrs"])
    for draw, spectra in sorted(random_by_draw.items()):
        mean_spectrum = np.asarray(spectra, dtype=float).mean(axis=0)
        for index, value in enumerate(mean_spectrum, start=1):
            add(
                rows,
                "cca",
                "Random-3",
                f"draw-{draw}",
                f"CC{index}",
                value,
                "rank 3; each draw is averaged over all 66 model pairs",
            )

    for record in nulls["prefix_rows"]:
        for metric in ("keep", "remove"):
            add(
                rows,
                "prefix",
                f"rank-{record['rank']}",
                f"seed-{record['seed']}",
                metric,
                record[metric],
                "ordered ACE prefix",
            )

    causal_display = {
        "ACE-3": "ACE-3",
        "PCA-3": "PCA-3",
        "Random-3": "Random-3",
    }
    for record in nulls["evaluations"]:
        if record["subspace"] not in causal_display:
            continue
        display = causal_display[record["subspace"]]
        for metric in ("keep", "remove"):
            add(
                rows,
                "causal",
                display,
                f"seed-{record['seed']}",
                metric,
                record[metric],
                "trained model; rank 3",
            )
        add(
            rows,
            "spectrum",
            display,
            f"seed-{record['seed']}",
            "normalized_error",
            record["spectral_distance"],
            "minimum one-to-one eigenvalue matching error",
        )

    for record in overlap["records"]:
        if record["condition"] == "Centered readout-3":
            continue
        draw = record["draw"]
        unit = f"seed-{record['seed']}"
        if draw is not None:
            unit += f"-draw-{draw}"
        condition = (
            "Random-3"
            if record["condition"] == "Ambient random-3"
            else record["condition"]
        )
        add(
            rows,
            "overlap",
            condition,
            unit,
            "projector_overlap_with_ace",
            record["projector_overlap"],
            (
                "rank-three normalized projector overlap; untrained comparison "
                "is raw ambient-coordinate overlap"
            ),
        )

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "19_markov_core_evidence.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "condition", "unit", "metric", "value", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
