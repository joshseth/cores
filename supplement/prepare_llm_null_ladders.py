"""Prepare SVA/IOI CCA-control and causal-comparator evidence.

The CCA controls and causal comparators answer different questions and are
therefore kept as separate evidence types:

* CCA controls ask whether cross-model alignment survives removing shared task
  variables and exceeds a prompt-shuffled, rank-matched chance level.
* Causal comparators ask whether alternative axes/subspaces reproduce the
  intervention effects of ACE.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CORE_ROOT = HERE.parent
SVA_ROOT = CORE_ROOT / "experiments" / "sva"
IOI_ROOT = CORE_ROOT / "src" / "cores" / "experiments" / "ioi_all"

SVA_RUNS = [
    ("gpt2_mixed", "GPT-2 S"),
    ("gpt2medium_mixed", "GPT-2 M"),
    ("gpt2large_mixed", "GPT-2 L"),
    ("Llama3.18B_mixed", "LLaMA"),
    ("gemma29b_mixed", "Gemma"),
    ("Qwen2.532B_mixed", "Qwen"),
]
IOI_RUNS = [
    ("gpt2", "GPT-2 S"),
    ("gpt2medium", "GPT-2 M"),
    ("gpt2large", "GPT-2 L"),
    ("llama318b", "LLaMA"),
    ("gemma29b", "Gemma"),
    ("qwen2532b", "Qwen"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def corr(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first, second)[0, 1])


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return (values - values.mean()) / values.std()


def residualize(values: np.ndarray, labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    design = np.stack([np.ones_like(values), *labels], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ coefficients
    return values - predicted, predicted


def curve_at_selected(run: dict) -> list[dict]:
    layer = run["final_layer_idx"]
    curves = run.get("rank_curves") or {}
    return curves.get(str(layer)) or curves.get(layer) or run["rank_curve"]


def append_row(
    rows: list[dict],
    *,
    task: str,
    evidence_type: str,
    unit: str,
    method: str,
    metric: str,
    value: float,
    sd: float | None = None,
) -> None:
    rows.append(
        {
            "task": task,
            "evidence_type": evidence_type,
            "unit": unit,
            "method": method,
            "metric": metric,
            "value": float(value),
            "sd": "" if sd is None else float(sd),
        }
    )


def prepare_sva(rows: list[dict]) -> None:
    loaded = []
    for run_dir, label in SVA_RUNS:
        payload = np.load(
            SVA_ROOT / run_dir / "plot_payload" / "knob_fit_test.npz",
            allow_pickle=True,
        )
        coordinate = np.asarray(payload["z"], dtype=float).reshape(-1)
        subject = np.asarray(payload["y"], dtype=int).reshape(-1)
        attractor = np.asarray(payload["y_attr"], dtype=int).reshape(-1)
        if corr(coordinate, 2 * subject - 1) < 0:
            coordinate = -coordinate
        coordinate = standardized(coordinate)
        residual, predicted = residualize(
            coordinate,
            [2.0 * subject - 1.0, 2.0 * attractor - 1.0],
        )
        loaded.append(
            {
                "run_dir": run_dir,
                "label": label,
                "z": coordinate,
                "residual": residual,
                "predicted": predicted,
                "subject": subject,
                "attractor": attractor,
            }
        )

    first = loaded[0]
    for run in loaded[1:]:
        if not np.array_equal(first["subject"], run["subject"]) or not np.array_equal(
            first["attractor"], run["attractor"]
        ):
            raise ValueError("SVA payloads are not prompt-aligned")

    for pair_index, (first_run, second_run) in enumerate(combinations(loaded, 2)):
        pair = f"{first_run['label']} × {second_run['label']}"
        raw = abs(corr(first_run["z"], second_run["z"]))
        label_only = abs(
            float(np.mean(first_run["predicted"] * second_run["predicted"]))
        )
        labels_removed = abs(
            corr(first_run["residual"], second_run["residual"])
        )
        rng = np.random.default_rng(1729 + pair_index)
        shuffled = np.asarray(
            [
                abs(
                    corr(
                        first_run["residual"],
                        second_run["residual"][
                            rng.permutation(len(second_run["residual"]))
                        ],
                    )
                )
                for _ in range(1000)
            ]
        )
        for method, value, sd in [
            ("Raw ACE", raw, None),
            ("Label-only expectation", label_only, None),
            ("Labels removed", labels_removed, None),
            ("Prompt-shuffled", shuffled.mean(), shuffled.std(ddof=1)),
        ]:
            append_row(
                rows,
                task="SVA",
                evidence_type="CCA",
                unit=pair,
                method=method,
                metric="CC1",
                value=value,
                sd=sd,
            )

    for run_dir, label in SVA_RUNS:
        result = load_json(SVA_ROOT / run_dir / "results.json")
        methods = {
            "ACE": result["causal"],
            "PCA-1": result["baselines"]["pca1"],
            "Difference of means": result["baselines"]["diff_means"],
            "Unembedding": result["baselines"]["unembed_margin"],
        }
        for method, metrics in methods.items():
            for source_key, metric in [
                ("keep_auc", "keep_auc"),
                ("remove_auc", "remove_auc"),
                ("flip_auc", "flip_auc"),
            ]:
                append_row(
                    rows,
                    task="SVA",
                    evidence_type="Causal",
                    unit=label,
                    method=method,
                    metric=metric,
                    value=metrics[source_key],
                )


def prepare_ioi(rows: list[dict]) -> None:
    alignment = load_json(IOI_ROOT / "cross_model_alignment.json")
    for record in alignment:
        if "cca_raw" not in record:
            continue
        pair = record["pair"]
        controls = {
            "Raw ACE": record["cca_raw"],
            "Answer label removed": record["cca_label_removed"],
            "Task covariates removed": record["cca_all_task_vars_removed"],
            "Prompt-shuffled": record["cca_all_task_vars_null_mean"],
        }
        null_sds = record.get("cca_shuffled_null_sd")
        for method, spectrum in controls.items():
            for dimension, value in enumerate(spectrum[:3], start=1):
                sd = None
                if method == "Prompt-shuffled" and null_sds is not None:
                    sd = null_sds[dimension - 1]
                append_row(
                    rows,
                    task="IOI",
                    evidence_type="CCA",
                    unit=pair,
                    method=method,
                    metric=f"CC{dimension}",
                    value=value,
                    sd=sd,
                )

    for run_dir, label in IOI_RUNS:
        result = load_json(IOI_ROOT / run_dir / "results.json")
        curve = curve_at_selected(result)
        selected = next(
            row for row in curve if row["rank"] == result["selected_rank"]
        )
        rank_one = next(row for row in curve if row["rank"] == 1)
        methods = {
            "ACE selected": {
                "keep_auc": selected["auc_keep"],
                "remove_auc": selected["auc_remove"],
                "flip_auc": selected["auc_flip"],
            },
            "ACE rank 1": {
                "keep_auc": rank_one["auc_keep"],
                "remove_auc": rank_one["auc_remove"],
                "flip_auc": rank_one["auc_flip"],
            },
            "PCA-1": {
                "keep_auc": result["baselines"]["pca1"]["keep"],
                "remove_auc": result["baselines"]["pca1"]["remove"],
                "flip_auc": result["baselines"]["pca1"]["flip"],
            },
            "Difference of means": {
                "keep_auc": result["baselines"]["diff_means"]["keep"],
                "remove_auc": result["baselines"]["diff_means"]["remove"],
                "flip_auc": result["baselines"]["diff_means"]["flip"],
            },
        }
        for method, metrics in methods.items():
            for metric, value in metrics.items():
                append_row(
                    rows,
                    task="IOI",
                    evidence_type="Causal",
                    unit=label,
                    method=method,
                    metric=metric,
                    value=value,
                )


def main() -> None:
    rows: list[dict] = []
    prepare_sva(rows)
    prepare_ioi(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "14_llm_null_ladders_evidence.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "evidence_type",
                "unit",
                "method",
                "metric",
                "value",
                "sd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
