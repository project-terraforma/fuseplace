"""Comprehensive audit of Project A parquet data quality and conflicts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.utils.conflation import CORE_ATTRIBUTES
from scripts.utils.io import DEFAULT_DATA_PATH, REPORTS_DIR, ensure_dir, load_parquet_duckdb, write_csv
from scripts.utils.parsing import extract_tokens, is_missing


def _pair_columns(df: pd.DataFrame) -> list[str]:
    cols = set(df.columns)
    return [attr for attr in CORE_ATTRIBUTES if attr in cols and f"base_{attr}" in cols]


def _missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(df)
    for col in df.columns:
        count = int(df[col].map(is_missing).sum())
        rows.append({"column": col, "missing_count": count, "missing_rate": count / n})
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def _conflict_table(df: pd.DataFrame, attrs: list[str]) -> pd.DataFrame:
    rows: list[dict] = []

    for attr in attrs:
        current_col = attr
        base_col = f"base_{attr}"

        sig_current = df[current_col].map(lambda x: tuple(sorted(extract_tokens(attr, x))))
        sig_base = df[base_col].map(lambda x: tuple(sorted(extract_tokens(attr, x))))

        current_missing = df[current_col].map(is_missing)
        base_missing = df[base_col].map(is_missing)

        both_missing = current_missing & base_missing
        one_missing = current_missing ^ base_missing
        both_present = ~(both_missing | one_missing)

        same = both_present & (sig_current == sig_base)
        conflict = both_present & (sig_current != sig_base)

        rows.append(
            {
                "attribute": attr,
                "both_missing_pct": both_missing.mean(),
                "one_missing_pct": one_missing.mean(),
                "same_pct": same.mean(),
                "conflict_pct": conflict.mean(),
                "comparable_rows": int(both_present.sum()),
            }
        )

        conflicts_df = df.loc[conflict, ["id", "base_id", current_col, base_col]].head(100)
        write_csv(REPORTS_DIR / "audit" / f"audit_conflicts_{attr}.csv", conflicts_df)

    return pd.DataFrame(rows).sort_values("conflict_pct", ascending=False)


def _confidence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for col in ["confidence", "base_confidence"]:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        rows.extend(
            [
                {"column": col, "metric": "min", "value": float(numeric.min())},
                {"column": col, "metric": "max", "value": float(numeric.max())},
                {"column": col, "metric": "mean", "value": float(numeric.mean())},
                {"column": col, "metric": "std", "value": float(numeric.std())},
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data quality and conflict rates")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()

    df = load_parquet_duckdb(args.input)

    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)

    audit_dir = ensure_dir(REPORTS_DIR / "audit")

    missing_df = _missingness_table(df)
    conflict_df = _conflict_table(df, _pair_columns(df))
    confidence_df = _confidence_table(df)

    write_csv(audit_dir / "audit_missingness.csv", missing_df)
    write_csv(audit_dir / "audit_conflict_rates.csv", conflict_df)
    write_csv(audit_dir / "audit_confidence_stats.csv", confidence_df)

    quality_checks = pd.DataFrame(
        [
            {"check": "rows", "value": len(df)},
            {"check": "columns", "value": len(df.columns)},
            {"check": "duplicate_id_rows", "value": int(df.duplicated(subset=["id"]).sum()) if "id" in df.columns else 0},
            {
                "check": "duplicate_base_id_rows",
                "value": int(df.duplicated(subset=["base_id"]).sum()) if "base_id" in df.columns else 0,
            },
        ]
    )
    write_csv(audit_dir / "audit_quality_checks.csv", quality_checks)

    print("Audit complete. Wrote:")
    print(f"  {audit_dir / 'audit_missingness.csv'}")
    print(f"  {audit_dir / 'audit_conflict_rates.csv'}")
    print(f"  {audit_dir / 'audit_confidence_stats.csv'}")
    print(f"  {audit_dir / 'audit_quality_checks.csv'}")


if __name__ == "__main__":
    main()
