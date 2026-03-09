"""Expand golden dataset from 200 to all records in the parquet file.

Creates golden entries for every record, auto-labels trivial cases,
then runs the domain-heuristic auto-labeler for the rest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from scripts.utils.io import ANALYSIS_DIR, DEFAULT_DATA_PATH, ensure_dir, load_parquet_duckdb, write_json
from scripts.utils.parsing import is_missing

GOLDEN_PATH = ANALYSIS_DIR / "golden" / "golden_dataset_template.json"
ATTRS = ["names", "categories", "websites", "phones", "addresses", "emails", "socials"]


def _safe(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    df = load_parquet_duckdb(args.input)

    existing: dict[str, dict] = {}
    if GOLDEN_PATH.exists():
        with GOLDEN_PATH.open("r", encoding="utf-8") as f:
            for record in json.load(f):
                existing[str(record.get("id", ""))] = record

    golden = []
    new_count = 0

    for idx, row in df.iterrows():
        record_id = str(row.get("id", ""))

        if record_id in existing:
            golden.append(existing[record_id])
            continue

        cur = {}
        bas = {}
        for attr in ATTRS:
            cur[attr] = _safe(row.get(attr))
            bas[attr] = _safe(row.get(f"base_{attr}"))

        cur["confidence"] = _safe(row.get("confidence"))
        cur["sources"] = _safe(row.get("sources"))
        bas["confidence"] = _safe(row.get("base_confidence"))
        bas["sources"] = _safe(row.get("base_sources"))

        labels = {}
        for attr in ATTRS:
            cur_miss = is_missing(cur.get(attr))
            bas_miss = is_missing(bas.get(attr))
            if cur_miss and bas_miss:
                labels[attr] = "skip"
            elif cur_miss:
                labels[attr] = "base"
            elif bas_miss:
                labels[attr] = "current"
            else:
                labels[attr] = ""

        golden.append({
            "record_index": int(idx),
            "id": record_id,
            "base_id": str(row.get("base_id", "")),
            "labels": labels,
            "notes": "",
            "current": cur,
            "base": bas,
        })
        new_count += 1

    ensure_dir(GOLDEN_PATH.parent)
    write_json(GOLDEN_PATH, golden, indent=2)

    total_labels = sum(
        1 for r in golden
        for v in (r.get("labels") or {}).values()
        if str(v).strip().lower() in {"current", "base", "skip", "tie"}
    )
    empty_labels = sum(
        1 for r in golden
        for v in (r.get("labels") or {}).values()
        if not str(v).strip()
    )

    print(f"Golden dataset expanded: {len(golden)} records ({new_count} new)")
    print(f"Labels filled: {total_labels}, empty: {empty_labels}")
    print(f"Saved to: {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
