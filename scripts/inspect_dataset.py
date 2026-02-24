"""Dataset inspection for Project A places attribute conflation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.utils.conflation import CORE_ATTRIBUTES
from scripts.utils.io import ANALYSIS_DIR, DEFAULT_DATA_PATH, ensure_dir, load_parquet_duckdb, write_csv, write_json, write_jsonl
from scripts.utils.parsing import is_missing, value_signature


def _attribute_pairs(columns: set[str]) -> list[str]:
    return [attr for attr in CORE_ATTRIBUTES if attr in columns and f"base_{attr}" in columns]


def _print_overview(df: pd.DataFrame, attrs: list[str]) -> None:
    print("=" * 72)
    print("FusePlace Dataset Inspection")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Attribute pairs found: {attrs}")

    if "id" in df.columns:
        print(f"Unique id: {df['id'].nunique():,}")
    if "base_id" in df.columns:
        print(f"Unique base_id: {df['base_id'].nunique():,}")

    print("\nTop missing columns:")
    missing = df.isna().sum().sort_values(ascending=False).head(12)
    for col, count in missing.items():
        pct = 100 * count / len(df)
        print(f"  {col:<20} {count:>6} ({pct:5.1f}%)")


def _build_side_by_side(df: pd.DataFrame, attrs: list[str], sample_n: int) -> pd.DataFrame:
    cols = [c for c in ["id", "base_id", "confidence", "base_confidence", "sources", "base_sources"] if c in df.columns]
    for attr in attrs:
        cols.extend([f"base_{attr}", attr])

    sample = df[cols].sample(min(sample_n, len(df)), random_state=42)
    return sample.reset_index(drop=True)


def _row_disagreement_count(row: pd.Series, attrs: list[str]) -> int:
    score = 0
    for attr in attrs:
        current = row.get(attr)
        base = row.get(f"base_{attr}")
        if is_missing(current) and is_missing(base):
            continue
        if is_missing(current) != is_missing(base):
            score += 1
            continue
        if value_signature(attr, current) != value_signature(attr, base):
            score += 1
    return score


def _build_golden_template(df: pd.DataFrame, attrs: list[str], golden_n: int) -> list[dict]:
    work = df.copy()
    work["__disagreement_count"] = work.apply(lambda row: _row_disagreement_count(row, attrs), axis=1)
    work = work.sort_values("__disagreement_count", ascending=False).head(golden_n).reset_index(drop=True)

    records: list[dict] = []
    for idx, row in work.iterrows():
        current_payload = {
            attr: row.get(attr)
            for attr in attrs
        }
        base_payload = {
            attr: row.get(f"base_{attr}")
            for attr in attrs
        }

        if "confidence" in row:
            current_payload["confidence"] = row.get("confidence")
        if "base_confidence" in row:
            base_payload["confidence"] = row.get("base_confidence")
        if "sources" in row:
            current_payload["sources"] = row.get("sources")
        if "base_sources" in row:
            base_payload["sources"] = row.get("base_sources")

        records.append(
            {
                "record_index": int(idx),
                "id": str(row.get("id", idx)),
                "base_id": str(row.get("base_id", "")),
                "labels": {attr: "" for attr in attrs},
                "notes": "",
                "current": current_payload,
                "base": base_payload,
            }
        )

    return records


def _attribute_disagreement_report(df: pd.DataFrame, attrs: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for attr in attrs:
        current_col = attr
        base_col = f"base_{attr}"

        current_missing = df[current_col].map(is_missing)
        base_missing = df[base_col].map(is_missing)

        both_present = ~(current_missing | base_missing)
        comparable = int(both_present.sum())

        disagreements = 0
        if comparable:
            signatures_current = df.loc[both_present, current_col].map(lambda x: value_signature(attr, x))
            signatures_base = df.loc[both_present, base_col].map(lambda x: value_signature(attr, x))
            disagreements = int((signatures_current != signatures_base).sum())

        rows.append(
            {
                "attribute": attr,
                "rows_with_current": int((~current_missing).sum()),
                "rows_with_base": int((~base_missing).sum()),
                "comparable_rows": comparable,
                "disagreements": disagreements,
                "disagreement_rate": (disagreements / comparable) if comparable else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("disagreement_rate", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Project A parquet dataset and export analysis artifacts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Path to parquet file")
    parser.add_argument("--sample-size", type=int, default=40, help="Rows for side-by-side sample")
    parser.add_argument("--golden-size", type=int, default=200, help="Rows for golden labeling template")
    args = parser.parse_args()

    df = load_parquet_duckdb(args.input)
    columns = set(df.columns)
    attrs = _attribute_pairs(columns)

    inspection_dir = ensure_dir(ANALYSIS_DIR)
    side_by_side_dir = ensure_dir(inspection_dir / "side_by_side")
    golden_dir = ensure_dir(inspection_dir / "golden")

    _print_overview(df, attrs)

    sample_df = _build_side_by_side(df, attrs, args.sample_size)
    side_csv = side_by_side_dir / "side_by_side_sample.csv"
    side_jsonl = side_by_side_dir / "side_by_side_sample.jsonl"
    write_csv(side_csv, sample_df)
    write_jsonl(side_jsonl, sample_df.to_dict(orient="records"))

    golden_records = _build_golden_template(df, attrs, args.golden_size)
    golden_json = golden_dir / "golden_dataset_template.json"
    write_json(golden_json, golden_records, indent=2)

    disagreement_df = _attribute_disagreement_report(df, attrs)
    disagreement_csv = inspection_dir / "attribute_disagreement_rates.csv"
    write_csv(disagreement_csv, disagreement_df)

    print("\nWrote artifacts:")
    print(f"  {side_csv}")
    print(f"  {side_jsonl}")
    print(f"  {golden_json}")
    print(f"  {disagreement_csv}")


if __name__ == "__main__":
    main()
