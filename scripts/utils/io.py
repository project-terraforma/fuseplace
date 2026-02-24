"""Shared I/O helpers for loading parquet and exporting artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "inspection"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_parquet_duckdb(path: Path | str) -> pd.DataFrame:
    """Load a parquet file via DuckDB to avoid pyarrow runtime dependency."""
    import duckdb

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    safe = parquet_path.as_posix().replace("'", "''")
    return duckdb.query(f"SELECT * FROM '{safe}'").df()


def write_json(path: Path | str, payload: object, indent: int = 2) -> None:
    out_path = Path(path)
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=str)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    out_path = Path(path)
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def write_csv(path: Path | str, df: pd.DataFrame) -> None:
    out_path = Path(path)
    ensure_parent(out_path)
    df.to_csv(out_path, index=False)

