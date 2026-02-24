"""Shared attribute-pair inspection for base_<attr> vs <attr>."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.utils.io import ANALYSIS_DIR, DEFAULT_DATA_PATH, ensure_dir, load_parquet_duckdb, write_csv, write_json
from scripts.utils.parsing import is_missing, value_signature


def _validate_attr(df: pd.DataFrame, attr: str) -> tuple[str, str]:
    base_attr = f"base_{attr}"
    if attr not in df.columns or base_attr not in df.columns:
        raise ValueError(f"Missing required columns: {attr} and/or {base_attr}")
    return attr, base_attr


def run_attr_analysis(attr: str, input_path: Path, sample_n: int = 60) -> None:
    df = load_parquet_duckdb(input_path)
    current_col, base_col = _validate_attr(df, attr)

    total_rows = len(df)
    current_missing = df[current_col].map(is_missing)
    base_missing = df[base_col].map(is_missing)

    comparable_mask = ~(current_missing | base_missing)
    comparable_rows = int(comparable_mask.sum())

    if comparable_rows:
        current_sig = df.loc[comparable_mask, current_col].map(lambda x: value_signature(attr, x))
        base_sig = df.loc[comparable_mask, base_col].map(lambda x: value_signature(attr, x))
        disagreements = int((current_sig != base_sig).sum())
    else:
        disagreements = 0

    out_dir = ensure_dir(ANALYSIS_DIR / "attributes")

    sample_df = df.loc[~(current_missing & base_missing), ["id", "base_id", base_col, current_col]].head(sample_n)
    output_json = out_dir / f"{attr}_pair_sample.json"
    output_csv = out_dir / f"{attr}_pair_sample.csv"

    write_json(output_json, sample_df.to_dict(orient="records"), indent=2)
    write_csv(output_csv, sample_df)

    print("=" * 64)
    print(f"Attribute inspection: base_{attr} vs {attr}")
    print("=" * 64)
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with {attr}: {(~current_missing).sum():,}")
    print(f"Rows with base_{attr}: {(~base_missing).sum():,}")
    print(f"Comparable rows: {comparable_rows:,}")
    print(f"Disagreements: {disagreements:,}")
    print(f"Disagreement rate: {(disagreements / comparable_rows * 100) if comparable_rows else 0:.2f}%")
    print(f"\nWrote:\n  {output_json}\n  {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one attribute pair")
    parser.add_argument("--attr", required=True, help="Attribute name, e.g., phones")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-size", type=int, default=60)
    args = parser.parse_args()

    run_attr_analysis(args.attr, args.input, sample_n=args.sample_size)


if __name__ == "__main__":
    main()
