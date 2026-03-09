"""Verify and label golden dataset using the Yelp Fusion API as ground truth.

For each place in the golden dataset:
1. Look up the business on Yelp by phone number (most reliable match).
2. Compare the Yelp-verified data against both 'current' and 'base' sources.
3. Label each attribute based on which source is closer to Yelp's data.

Usage:
    export YELP_API_KEY="your_key_here"
    python3 -m scripts.yelp_verify

    # Or pass it directly:
    python3 -m scripts.yelp_verify --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd

from scripts.utils.io import ANALYSIS_DIR, DEFAULT_DATA_PATH, ensure_dir, load_parquet_duckdb, write_json
from scripts.utils.parsing import (
    is_missing,
    normalize_address,
    normalize_name,
    normalize_phone,
    normalize_url,
    parse_maybe_json,
)

GOLDEN_PATH = ANALYSIS_DIR / "golden" / "golden_dataset_template.json"
YELP_CACHE_PATH = ANALYSIS_DIR / "golden" / "yelp_lookups.json"
YELP_SEARCH_URL = "https://api.yelp.com/v3/businesses/search/phone"
YELP_SEARCH_BY_NAME_URL = "https://api.yelp.com/v3/businesses/search"

ATTRS = ["names", "categories", "websites", "phones", "addresses"]


def _yelp_search_by_phone(phone: str, api_key: str) -> dict | None:
    """Search Yelp by phone number, return best match or None."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 10:
        return None

    if not digits.startswith("+"):
        if len(digits) == 10:
            digits = "+1" + digits
        elif len(digits) == 11 and digits.startswith("1"):
            digits = "+" + digits
        else:
            digits = "+" + digits

    resp = requests.get(
        YELP_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        params={"phone": digits},
        timeout=10,
    )
    if resp.status_code != 200:
        return None

    data = resp.json()
    businesses = data.get("businesses", [])
    if not businesses:
        return None

    return businesses[0]


def _yelp_search_by_name(name: str, location: str, api_key: str) -> dict | None:
    """Fallback: search Yelp by business name + location."""
    resp = requests.get(
        YELP_SEARCH_BY_NAME_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        params={"term": name, "location": location, "limit": 1},
        timeout=10,
    )
    if resp.status_code != 200:
        return None

    data = resp.json()
    businesses = data.get("businesses", [])
    if not businesses:
        return None

    return businesses[0]


def _extract_location_string(addr_raw: object) -> str:
    """Extract a location string from raw address data for Yelp search."""
    parsed = parse_maybe_json(addr_raw)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if isinstance(parsed, dict):
        parts = [
            str(parsed.get("freeform", "")),
            str(parsed.get("locality", "")),
            str(parsed.get("region", "")),
            str(parsed.get("country", "")),
        ]
        return ", ".join(p for p in parts if p.strip())
    return str(addr_raw) if addr_raw else ""


def _name_similarity(a: str, b: str) -> float:
    """Simple normalized edit-distance-based similarity."""
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0
    intersection = a_words & b_words
    union = a_words | b_words
    return len(intersection) / len(union)


def _phone_similarity(a: str, b: str) -> float:
    a_digits = re.sub(r"\D", "", a)
    b_digits = re.sub(r"\D", "", b)
    if not a_digits or not b_digits:
        return 0.0
    if a_digits == b_digits:
        return 1.0
    if a_digits.endswith(b_digits) or b_digits.endswith(a_digits):
        return 0.9
    return 0.0


def _category_similarity(categories_raw: object, yelp_categories: list[dict]) -> float:
    """Compare source categories against Yelp categories."""
    parsed = parse_maybe_json(categories_raw)
    if not parsed or not yelp_categories:
        return 0.0

    source_cats = set()
    if isinstance(parsed, dict):
        if parsed.get("primary"):
            source_cats.add(str(parsed["primary"]).lower().replace("_", " "))
        for alt in (parsed.get("alternate") or []):
            source_cats.add(str(alt).lower().replace("_", " "))
    elif isinstance(parsed, str):
        source_cats.add(parsed.lower().replace("_", " "))

    yelp_cats = set()
    for yc in yelp_categories:
        alias = yc.get("alias", "").lower().replace("_", " ")
        title = yc.get("title", "").lower()
        if alias:
            yelp_cats.add(alias)
        if title:
            yelp_cats.add(title)

    if not source_cats or not yelp_cats:
        return 0.0

    best = 0.0
    for sc in source_cats:
        for yc in yelp_cats:
            if sc == yc:
                best = max(best, 1.0)
            elif sc in yc or yc in sc:
                best = max(best, 0.7)
            else:
                sc_words = set(sc.split())
                yc_words = set(yc.split())
                overlap = sc_words & yc_words
                if overlap:
                    best = max(best, len(overlap) / max(len(sc_words), len(yc_words)))

    return best


def _address_similarity(addr_raw: object, yelp_location: dict) -> float:
    """Compare source address against Yelp location."""
    source_addr = normalize_address(addr_raw).lower()
    if not source_addr or not yelp_location:
        return 0.0

    yelp_parts = []
    for field in ["address1", "address2", "address3"]:
        val = yelp_location.get(field, "")
        if val:
            yelp_parts.append(val.lower())
    yelp_addr = " ".join(yelp_parts)

    if not yelp_addr:
        return 0.0

    source_words = set(source_addr.split())
    yelp_words = set(yelp_addr.split())
    if not source_words or not yelp_words:
        return 0.0

    intersection = source_words & yelp_words
    union = source_words | yelp_words
    return len(intersection) / len(union)


def _label_from_yelp(
    attr: str,
    cur_raw: object,
    bas_raw: object,
    yelp_biz: dict,
) -> str | None:
    """Compare current and base values against Yelp data; return 'current', 'base', or None."""

    if attr == "names":
        yelp_name = yelp_biz.get("name", "")
        cur_sim = _name_similarity(normalize_name(cur_raw), yelp_name)
        bas_sim = _name_similarity(normalize_name(bas_raw), yelp_name)

    elif attr == "phones":
        yelp_phone = yelp_biz.get("phone", "")
        cur_sim = _phone_similarity(normalize_phone(cur_raw), yelp_phone)
        bas_sim = _phone_similarity(normalize_phone(bas_raw), yelp_phone)

    elif attr == "categories":
        yelp_cats = yelp_biz.get("categories", [])
        cur_sim = _category_similarity(cur_raw, yelp_cats)
        bas_sim = _category_similarity(bas_raw, yelp_cats)

    elif attr == "addresses":
        yelp_loc = yelp_biz.get("location", {})
        cur_sim = _address_similarity(cur_raw, yelp_loc)
        bas_sim = _address_similarity(bas_raw, yelp_loc)

    elif attr == "websites":
        yelp_url = yelp_biz.get("url", "")
        cur_sim = 0.5
        bas_sim = 0.5

    else:
        return None

    if abs(cur_sim - bas_sim) < 0.05:
        return None

    return "current" if cur_sim > bas_sim else "base"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify golden dataset against Yelp Fusion API")
    parser.add_argument("--api-key", type=str, default=os.environ.get("YELP_API_KEY", ""))
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--limit", type=int, default=200, help="Max records to look up")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between API calls")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No Yelp API key provided.")
        print("  export YELP_API_KEY='your_key'")
        print("  OR: python3 -m scripts.yelp_verify --api-key YOUR_KEY")
        print()
        print("Get a free key at: https://www.yelp.com/developers/v3/manage_app")
        return

    with args.golden.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    # Load cache if exists
    cache: dict[str, dict] = {}
    if YELP_CACHE_PATH.exists():
        with YELP_CACHE_PATH.open("r", encoding="utf-8") as f:
            cache = json.load(f)

    records_to_process = golden[: args.limit]
    total_verified = 0
    total_labeled = 0
    yelp_found = 0
    attr_stats: dict[str, dict[str, int]] = {a: {"current": 0, "base": 0, "skip": 0} for a in ATTRS}

    for idx, record in enumerate(records_to_process):
        record_id = str(record.get("id", ""))
        cur = record.get("current", {})
        bas = record.get("base", {})
        labels = record.get("labels", {}) or {}

        # Try to find on Yelp (phone first, then name+location)
        yelp_biz = cache.get(record_id)

        if yelp_biz is None:
            cur_phone = normalize_phone(cur.get("phones", ""))
            bas_phone = normalize_phone(bas.get("phones", ""))
            phone_to_try = cur_phone or bas_phone

            if phone_to_try:
                yelp_biz = _yelp_search_by_phone(phone_to_try, args.api_key)
                time.sleep(args.delay)

            if yelp_biz is None:
                cur_name = normalize_name(cur.get("names", ""))
                location = _extract_location_string(cur.get("addresses")) or _extract_location_string(bas.get("addresses"))
                if cur_name and location:
                    yelp_biz = _yelp_search_by_name(cur_name, location, args.api_key)
                    time.sleep(args.delay)

            if yelp_biz is not None:
                cache[record_id] = yelp_biz
                # Save cache incrementally
                if idx % 20 == 0:
                    ensure_dir(YELP_CACHE_PATH.parent)
                    write_json(YELP_CACHE_PATH, cache)

        if yelp_biz is None:
            continue

        yelp_found += 1
        yelp_name = yelp_biz.get("name", "unknown")

        for attr in ATTRS:
            cur_val = cur.get(attr)
            bas_val = bas.get(attr)

            if is_missing(cur_val) or is_missing(bas_val):
                continue

            label = _label_from_yelp(attr, cur_val, bas_val, yelp_biz)
            if label:
                labels[attr] = label
                attr_stats[attr][label] += 1
                total_labeled += 1
            else:
                attr_stats[attr]["skip"] += 1

        record["labels"] = labels
        total_verified += 1

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(records_to_process)} records ({yelp_found} found on Yelp)")

    # Save final cache
    ensure_dir(YELP_CACHE_PATH.parent)
    write_json(YELP_CACHE_PATH, cache)

    # Save updated golden dataset
    with args.golden.open("w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, default=str)

    print(f"\n=== Yelp Verification Complete ===")
    print(f"  Records processed: {total_verified}")
    print(f"  Found on Yelp: {yelp_found}/{len(records_to_process)}")
    print(f"  Attributes labeled: {total_labeled}")
    print(f"\n  Per-attribute breakdown:")
    for attr in ATTRS:
        s = attr_stats[attr]
        total = s["current"] + s["base"] + s["skip"]
        if total > 0:
            print(f"    {attr:>12s}: current={s['current']}  base={s['base']}  undecided={s['skip']}")

    print(f"\n  Golden dataset updated: {args.golden}")
    print(f"  Yelp cache saved: {YELP_CACHE_PATH}")
    print(f"\n  Next steps:")
    print(f"    python3 -m scripts.conflation.ml_selection --no-proxy")
    print(f"    python3 -m scripts.conflation.evaluate_methods")


if __name__ == "__main__":
    main()
