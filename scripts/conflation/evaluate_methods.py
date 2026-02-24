"""Evaluate rule-based and ML decisions against manual golden labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.utils.io import ANALYSIS_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_dir, write_csv


def _load_label_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Golden labels file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict] = []
    for record in payload:
        record_id = str(record.get("id", ""))
        labels = record.get("labels", {}) or {}
        for attr, label in labels.items():
            normalized = str(label).strip().lower()
            if normalized in {"current", "base"}:
                rows.append({"id": record_id, "attribute": attr, "gold_label": normalized})

    if not rows:
        return pd.DataFrame(columns=["id", "attribute", "gold_label"])
    return pd.DataFrame(rows)


def _eval_single(decisions: pd.DataFrame, labels: pd.DataFrame, method_name: str) -> pd.DataFrame:
    merged = decisions.merge(labels, on=["id", "attribute"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "method": method_name,
                    "attribute": "ALL",
                    "labeled_rows": 0,
                    "accuracy": 0.0,
                }
            ]
        )

    merged["correct"] = merged["winner"] == merged["gold_label"]

    overall = pd.DataFrame(
        [
            {
                "method": method_name,
                "attribute": "ALL",
                "labeled_rows": int(len(merged)),
                "accuracy": float(merged["correct"].mean()),
            }
        ]
    )

    per_attr = (
        merged.groupby("attribute")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "labeled_rows"})
    )
    per_attr.insert(0, "method", method_name)

    return pd.concat([overall, per_attr], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conflation methods against manual labels")
    parser.add_argument(
        "--golden-labels",
        type=Path,
        default=ANALYSIS_DIR / "golden" / "golden_dataset_template.json",
    )
    parser.add_argument(
        "--rule-decisions",
        type=Path,
        default=REPORTS_DIR / "conflation" / "rule_attribute_decisions.csv",
    )
    parser.add_argument(
        "--ml-decisions",
        type=Path,
        default=REPORTS_DIR / "conflation" / "ml_attribute_decisions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "conflation",
    )
    args = parser.parse_args()

    labels = _load_label_table(args.golden_labels)

    tables: list[pd.DataFrame] = []

    if args.rule_decisions.exists():
        rule_df = pd.read_csv(args.rule_decisions, dtype={"id": str})
        tables.append(_eval_single(rule_df, labels, "rule"))

    if args.ml_decisions.exists():
        ml_df = pd.read_csv(args.ml_decisions, dtype={"id": str})
        tables.append(_eval_single(ml_df, labels, "ml"))

    if not tables:
        raise RuntimeError("No decisions files found for evaluation.")

    result = pd.concat(tables, ignore_index=True)
    out_dir = ensure_dir(args.output_dir)
    out_path = out_dir / "method_evaluation_against_golden.csv"
    write_csv(out_path, result)

    print(f"Evaluation complete: {out_path}")


if __name__ == "__main__":
    main()
