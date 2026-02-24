"""Run rule-based attribute selection and export conflated outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from scripts.utils.conflation import CORE_ATTRIBUTES, decide_rule_based
from scripts.utils.io import DEFAULT_DATA_PATH, REPORTS_DIR, ensure_dir, load_parquet_duckdb, write_csv


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based conflation baseline")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "conflation",
        help="Directory for rule-based output files",
    )
    args = parser.parse_args()

    df = load_parquet_duckdb(args.input)
    attrs = [a for a in CORE_ATTRIBUTES if a in df.columns and f"base_{a}" in df.columns]
    output_dir = ensure_dir(args.output_dir)

    decision_rows: list[dict] = []
    selected_rows: list[dict] = []

    for _, row in df.iterrows():
        selected_record = {
            "id": row.get("id"),
            "base_id": row.get("base_id"),
        }

        for attr in attrs:
            decision = decide_rule_based(
                attr=attr,
                current_value=row.get(attr),
                base_value=row.get(f"base_{attr}"),
                confidence=_safe_float(row.get("confidence")),
                base_confidence=_safe_float(row.get("base_confidence")),
                current_sources=row.get("sources"),
                base_sources=row.get("base_sources"),
            )

            if decision.winner == "current":
                selected_value = row.get(attr)
            elif decision.winner == "base":
                selected_value = row.get(f"base_{attr}")
            else:
                selected_value = row.get(attr)

            selected_record[f"selected_{attr}"] = selected_value
            selected_record[f"selected_from_{attr}"] = decision.winner

            decision_rows.append(
                {
                    "id": row.get("id"),
                    "base_id": row.get("base_id"),
                    "attribute": attr,
                    "winner": decision.winner,
                    "score_current": decision.score_current,
                    "score_base": decision.score_base,
                    "decision_reason": decision.reason,
                }
            )

        selected_rows.append(selected_record)

    decisions_df = pd.DataFrame(decision_rows)
    selected_df = pd.DataFrame(selected_rows)

    summary_df = (
        decisions_df.groupby(["attribute", "winner"]).size().reset_index(name="count")
        .sort_values(["attribute", "winner"])
    )

    decisions_path = output_dir / "rule_attribute_decisions.csv"
    selected_path = output_dir / "rule_selected_records.csv"
    summary_path = output_dir / "rule_summary.csv"

    write_csv(decisions_path, decisions_df)
    write_csv(selected_path, selected_df)
    write_csv(summary_path, summary_df)

    print("Rule-based conflation complete. Wrote:")
    print(f"  {decisions_path}")
    print(f"  {selected_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
