"""Evaluate rule-based and ML decisions against manual golden labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy, precision, recall, and F1 for binary (current vs base) predictions."""
    labels = ["base", "current"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="current", labels=labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="current", labels=labels, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="current", labels=labels, zero_division=0)),
    }


def _eval_single(decisions: pd.DataFrame, labels: pd.DataFrame, method_name: str) -> pd.DataFrame:
    merged = decisions.merge(labels, on=["id", "attribute"], how="inner")
    merged = merged[merged["winner"].isin(["current", "base"]) & merged["gold_label"].isin(["current", "base"])]
    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "method": method_name,
                    "attribute": "ALL",
                    "labeled_rows": 0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            ]
        )

    overall_metrics = _compute_metrics(merged["gold_label"], merged["winner"])
    overall = pd.DataFrame(
        [
            {
                "method": method_name,
                "attribute": "ALL",
                "labeled_rows": int(len(merged)),
                **overall_metrics,
            }
        ]
    )

    per_attr_rows: list[dict] = []
    for attr, group in merged.groupby("attribute"):
        attr_metrics = _compute_metrics(group["gold_label"], group["winner"])
        per_attr_rows.append({
            "method": method_name,
            "attribute": attr,
            "labeled_rows": int(len(group)),
            **attr_metrics,
        })
    per_attr = pd.DataFrame(per_attr_rows)

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

    print(f"\nEvaluation complete: {out_path}\n")

    for method in result["method"].unique():
        method_df = result[result["method"] == method]
        print(f"=== {method.upper()} ===")
        for _, row in method_df.iterrows():
            print(
                f"  {row['attribute']:>12s}  |  "
                f"F1={row['f1']:.4f}  "
                f"Precision={row['precision']:.4f}  "
                f"Recall={row['recall']:.4f}  "
                f"Accuracy={row['accuracy']:.4f}  "
                f"(n={int(row['labeled_rows'])})"
            )
        print()


if __name__ == "__main__":
    main()
