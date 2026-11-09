"""Prepare the SVA and IOI intervention and geometry tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CORE_ROOT = HERE.parent
SVA_ROOT = CORE_ROOT / "experiments" / "sva"
IOI_ROOT = CORE_ROOT / "src" / "cores" / "experiments" / "ioi_all"

SVA_RUNS = [
    ("gpt2_mixed", "GPT-2 Small"),
    ("gpt2medium_mixed", "GPT-2 Medium"),
    ("gpt2large_mixed", "GPT-2 Large"),
    ("Llama3.18B_mixed", "LLaMA-3.1-8B"),
    ("gemma29b_mixed", "Gemma-2-9B"),
    ("Qwen2.532B_mixed", "Qwen-2.5-32B"),
]
IOI_RUNS = [
    ("gpt2", "GPT-2 Small"),
    ("gpt2medium", "GPT-2 Medium"),
    ("gpt2large", "GPT-2 Large"),
    ("llama318b", "LLaMA-3.1-8B"),
    ("gemma29b", "Gemma-2-9B"),
    ("qwen2532b", "Qwen-2.5-32B"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def curve_at_selected(result: dict) -> list[dict]:
    layer = result["final_layer_idx"]
    curves = result.get("rank_curves") or {}
    return curves.get(str(layer)) or curves.get(layer) or result["rank_curve"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(
    rows: list[dict],
    *,
    task: str,
    model: str,
    method: str,
    metrics: dict,
    overlap: float | None,
    selected_rank: int | None,
    notes: str,
) -> None:
    for source, metric in [
        ("keep", "Keep"),
        ("remove", "Remove"),
        ("flip", "Flip"),
    ]:
        rows.append(
            {
                "task": task,
                "model": model,
                "method": method,
                "metric": metric,
                "value": metrics[source],
                "selected_rank": (
                    "" if selected_rank is None else selected_rank
                ),
                "notes": notes,
            }
        )
    if overlap is not None:
        rows.append(
            {
                "task": task,
                "model": model,
                "method": method,
                "metric": "Overlap",
                "value": overlap,
                "selected_rank": (
                    "" if selected_rank is None else selected_rank
                ),
                "notes": (
                    "absolute cosine with the leading ACE direction; "
                    + notes
                ),
            }
        )


def prepare_sva() -> list[dict]:
    rows: list[dict] = []
    for run_dir, model in SVA_RUNS:
        result = load_json(SVA_ROOT / run_dir / "results.json")
        append_metrics(
            rows,
            task="SVA",
            model=model,
            method="ACE",
            metrics={
                "keep": result["causal"]["keep_auc"],
                "remove": result["causal"]["remove_auc"],
                "flip": result["causal"]["flip_auc"],
            },
            overlap=1.0,
            selected_rank=1,
            notes="rank-one ACE at the selected intermediate layer",
        )
        for key, label, note in [
            ("pca1", "PCA-1", "top activation principal component"),
            (
                "diff_means",
                "Difference of means",
                "supervised subject-number contrast",
            ),
            (
                "unembed_margin",
                "Unembedding",
                "fixed singular/plural unembedding-margin direction",
            ),
        ]:
            baseline = result["baselines"][key]
            append_metrics(
                rows,
                task="SVA",
                model=model,
                method=label,
                metrics={
                    "keep": baseline["keep_auc"],
                    "remove": baseline["remove_auc"],
                    "flip": baseline["flip_auc"],
                },
                overlap=baseline["cos_to_ace"],
                selected_rank=1,
                notes=note,
            )
    return rows


def prepare_ioi() -> list[dict]:
    rows: list[dict] = []
    for run_dir, model in IOI_RUNS:
        result = load_json(IOI_ROOT / run_dir / "results.json")
        curve = curve_at_selected(result)
        rank = int(result["selected_rank"])
        selected = next(row for row in curve if row["rank"] == rank)
        rank_one = next(row for row in curve if row["rank"] == 1)
        append_metrics(
            rows,
            task="IOI",
            model=model,
            method="ACE selected",
            metrics={
                "keep": selected["auc_keep"],
                "remove": selected["auc_remove"],
                "flip": selected["auc_flip"],
            },
            overlap=None,
            selected_rank=rank,
            notes="selected multidimensional ACE subspace",
        )
        append_metrics(
            rows,
            task="IOI",
            model=model,
            method="ACE rank 1",
            metrics={
                "keep": rank_one["auc_keep"],
                "remove": rank_one["auc_remove"],
                "flip": rank_one["auc_flip"],
            },
            overlap=1.0,
            selected_rank=1,
            notes="leading direction of the selected ACE ordering",
        )
        for key, label, note in [
            ("pca1", "PCA-1", "top activation principal component"),
            (
                "diff_means",
                "Difference of means",
                "fixed supervised answer-position contrast",
            ),
        ]:
            baseline = result["baselines"][key]
            append_metrics(
                rows,
                task="IOI",
                model=model,
                method=label,
                metrics={
                    "keep": baseline["keep"],
                    "remove": baseline["remove"],
                    "flip": baseline["flip"],
                },
                overlap=baseline["cos_to_ace"],
                selected_rank=1,
                notes=note,
            )
    return rows


def markdown_table(task: str, rows: list[dict]) -> str:
    by_key = {}
    for row in rows:
        by_key.setdefault((row["model"], row["method"]), {})[
            row["metric"]
        ] = float(row["value"])
        by_key[(row["model"], row["method"])]["rank"] = row[
            "selected_rank"
        ]
    lines = [
        f"# {task} causal controls and ACE overlap",
        "",
        "| Model | Method | Rank | Keep AUC | Remove AUC | Flip AUC | "
        "Abs. cosine with ACE-1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (model, method), values in by_key.items():
        overlap = values.get("Overlap")
        overlap_text = "—" if overlap is None else f"{overlap:.3f}"
        lines.append(
            f"| {model} | {method} | {values['rank']} | "
            f"{values['Keep']:.3f} | {values['Remove']:.3f} | "
            f"{values['Flip']:.3f} | {overlap_text} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sva = prepare_sva()
    ioi = prepare_ioi()
    write_csv(OUT / "21_sva_controls_evidence.csv", sva)
    write_csv(OUT / "22_ioi_controls_evidence.csv", ioi)
    (OUT / "21_sva_controls_table.md").write_text(
        markdown_table("SVA", sva),
        encoding="utf-8",
    )
    (OUT / "22_ioi_controls_table.md").write_text(
        markdown_table("IOI", ioi),
        encoding="utf-8",
    )
    print(f"Wrote {len(sva)} SVA and {len(ioi)} IOI rows")


if __name__ == "__main__":
    main()
