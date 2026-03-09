"""Prepare golden dataset for manual labeling.

Steps:
1. Auto-label trivial cases (one side missing → pick the other, both missing → skip).
2. Export ambiguous pairs (both values present) to a human-readable CSV.
3. After manual review of the CSV, run with --import to merge labels back into the golden JSON.

Usage:
    # Step 1: Export for labeling
    python3 -m scripts.label_golden

    # Step 2: Open the CSV, fill in the 'label' column with 'current' or 'base'

    # Step 3: Import labels back
    python3 -m scripts.label_golden --import
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scripts.utils.io import ANALYSIS_DIR, ensure_dir
from scripts.utils.parsing import (
    extract_primary_text,
    is_missing,
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_url,
)

GOLDEN_PATH = ANALYSIS_DIR / "golden" / "golden_dataset_template.json"
LABEL_CSV_PATH = ANALYSIS_DIR / "golden" / "labeling_worksheet.csv"

ATTRS = ["names", "categories", "websites", "phones", "addresses", "emails", "socials"]


def _readable(attr: str, raw_value: object) -> str:
    """Convert raw JSON-encoded attribute value to a human-readable string."""
    if is_missing(raw_value):
        return "<MISSING>"
    if attr == "names":
        return normalize_name(raw_value) or str(raw_value)
    if attr == "categories":
        return normalize_category(raw_value) or str(raw_value)
    if attr == "addresses":
        return normalize_address(raw_value) or str(raw_value)
    if attr == "phones":
        return normalize_phone(raw_value) or str(raw_value)
    if attr == "websites":
        return normalize_url(raw_value) or str(raw_value)
    return extract_primary_text(raw_value) or str(raw_value)


def _export(golden_path: Path, csv_path: Path) -> None:
    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    auto_labeled = 0
    ambiguous_rows: list[dict] = []

    for record in data:
        record_id = str(record.get("id", ""))
        labels = record.get("labels", {}) or {}
        cur = record.get("current", {})
        bas = record.get("base", {})

        cur_conf = cur.get("confidence")
        bas_conf = bas.get("confidence")

        for attr in ATTRS:
            if labels.get(attr, "").strip():
                continue

            cur_val = cur.get(attr)
            bas_val = bas.get(attr)
            cur_miss = is_missing(cur_val)
            bas_miss = is_missing(bas_val)

            if cur_miss and bas_miss:
                labels[attr] = "skip"
                auto_labeled += 1
            elif cur_miss:
                labels[attr] = "base"
                auto_labeled += 1
            elif bas_miss:
                labels[attr] = "current"
                auto_labeled += 1
            else:
                ambiguous_rows.append({
                    "record_index": record.get("record_index", ""),
                    "id": record_id,
                    "attribute": attr,
                    "current_value": _readable(attr, cur_val),
                    "base_value": _readable(attr, bas_val),
                    "current_confidence": cur_conf,
                    "base_confidence": bas_conf,
                    "label": "",
                })

        record["labels"] = labels

    with golden_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    ensure_dir(csv_path.parent)
    pd.DataFrame(ambiguous_rows).to_csv(csv_path, index=False)

    print(f"Auto-labeled {auto_labeled} trivial cases (saved to golden JSON).")
    print(f"Exported {len(ambiguous_rows)} ambiguous pairs to: {csv_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {csv_path}")
    print("  2. For each row, fill the 'label' column with 'current' or 'base'")
    print("     (leave blank or write 'skip' if you truly can't decide)")
    print("  3. Run: python3 -m scripts.label_golden --import")


def _import_labels(golden_path: Path, csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Labeling CSV not found: {csv_path}")

    csv_df = pd.read_csv(csv_path, dtype=str).fillna("")
    label_map: dict[tuple[str, str], str] = {}
    for _, row in csv_df.iterrows():
        label = row.get("label", "").strip().lower()
        if label in {"current", "base", "skip", "tie"}:
            label_map[(str(row["id"]), str(row["attribute"]))] = label

    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    imported = 0
    for record in data:
        record_id = str(record.get("id", ""))
        labels = record.get("labels", {}) or {}
        for attr in ATTRS:
            key = (record_id, attr)
            if key in label_map:
                labels[attr] = label_map[key]
                imported += 1
        record["labels"] = labels

    with golden_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    filled = sum(
        1 for r in data
        for v in (r.get("labels") or {}).values()
        if str(v).strip().lower() in {"current", "base", "skip", "tie"}
    )
    total = len(data) * len(ATTRS)

    print(f"Imported {imported} labels from CSV into golden JSON.")
    print(f"Total labels filled: {filled}/{total}")
    print()
    print("Now rerun the ML pipeline:")
    print("  python3 -m scripts.conflation.ml_selection")
    print("  python3 -m scripts.conflation.evaluate_methods")


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden dataset labeling helper")
    parser.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="Import labels from CSV back into golden JSON",
    )
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--csv", type=Path, default=LABEL_CSV_PATH)
    args = parser.parse_args()

    if args.do_import:
        _import_labels(args.golden, args.csv)
    else:
        _export(args.golden, args.csv)


if __name__ == "__main__":
    main()
