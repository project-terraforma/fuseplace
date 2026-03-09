"""Build genuine conflation pairs using Yelp as a second data source.

For each record in the original parquet that has a Yelp match:
- One side = Overture Maps data (the "current" or "base" source)
- Other side = Yelp data (converted to matching schema)

This creates real conflation pairs with genuine attribute differences,
since Yelp and Overture often have different names, addresses, phones, etc.

Ground truth labels are determined by comparing both sources against the
Yelp-verified data (independent third-party ground truth).

Usage:
    python3 -m scripts.build_yelp_pairs
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scripts.utils.io import (
    ANALYSIS_DIR,
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    ensure_dir,
    load_parquet_duckdb,
    write_json,
)
from scripts.utils.parsing import is_missing, normalize_name, normalize_phone, normalize_address

YELP_CACHE_PATH = ANALYSIS_DIR / "golden" / "yelp_lookups.json"
GOLDEN_PATH = ANALYSIS_DIR / "golden" / "golden_dataset_template.json"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "yelp_verified_dataset.parquet"

ATTRS = ["names", "categories", "websites", "phones", "addresses", "emails", "socials"]


def _yelp_to_overture_names(biz: dict) -> str | None:
    name = biz.get("name")
    if not name:
        return None
    return json.dumps({"primary": name})


def _yelp_to_overture_categories(biz: dict) -> str | None:
    cats = biz.get("categories", [])
    if not cats:
        return None
    primary = cats[0].get("alias", "").replace("-", "_") if cats else None
    alternate = [c.get("alias", "").replace("-", "_") for c in cats[1:]] if len(cats) > 1 else []
    return json.dumps({"primary": primary, "alternate": alternate})


def _yelp_to_overture_phones(biz: dict) -> str | None:
    phone = biz.get("phone", "")
    if not phone:
        return None
    return json.dumps([phone])


def _yelp_to_overture_addresses(biz: dict) -> str | None:
    loc = biz.get("location", {})
    if not loc:
        return None
    parts = []
    for f in ["address1", "address2", "address3"]:
        if loc.get(f):
            parts.append(loc[f])
    freeform = " ".join(parts)
    return json.dumps([{
        "freeform": freeform or None,
        "locality": loc.get("city"),
        "postcode": loc.get("zip_code"),
        "region": loc.get("state"),
        "country": loc.get("country"),
    }])


def _yelp_to_overture_websites(biz: dict) -> str | None:
    url = biz.get("url", "")
    if not url:
        return None
    return json.dumps([url])


def _name_sim(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_w, b_w = set(a.split()), set(b.split())
    if not a_w or not b_w:
        return 0.0
    return len(a_w & b_w) / len(a_w | b_w)


def _phone_sim(a: str, b: str) -> float:
    ad = re.sub(r"\D", "", a)
    bd = re.sub(r"\D", "", b)
    if not ad or not bd:
        return 0.0
    if ad == bd:
        return 1.0
    if ad.endswith(bd[-10:]) or bd.endswith(ad[-10:]):
        return 0.9
    return 0.0


def _label_attr_from_yelp(
    attr: str,
    overture_val: object,
    yelp_val: object,
    yelp_biz: dict,
) -> str:
    """Decide which value is closer to Yelp ground truth.

    Returns 'current' if overture_val is closer, 'base' if yelp_val is closer.
    In the pair schema, 'current' = overture side, 'base' = yelp side.
    """
    if is_missing(overture_val) and is_missing(yelp_val):
        return "skip"
    if is_missing(overture_val):
        return "base"
    if is_missing(yelp_val):
        return "current"

    if attr == "names":
        yelp_name = yelp_biz.get("name", "")
        ov_sim = _name_sim(normalize_name(overture_val), yelp_name)
        yp_sim = _name_sim(normalize_name(yelp_val), yelp_name)
        if abs(ov_sim - yp_sim) < 0.05:
            return "current" if ov_sim >= yp_sim else "base"
        return "current" if ov_sim > yp_sim else "base"

    elif attr == "phones":
        yelp_phone = yelp_biz.get("phone", "")
        ov_sim = _phone_sim(normalize_phone(overture_val), yelp_phone)
        yp_sim = _phone_sim(normalize_phone(yelp_val), yelp_phone)
        if abs(ov_sim - yp_sim) < 0.05:
            return "current" if ov_sim >= yp_sim else "base"
        return "current" if ov_sim > yp_sim else "base"

    elif attr == "addresses":
        yelp_loc = yelp_biz.get("location", {})
        yelp_addr = " ".join(
            (yelp_loc.get(f) or "") for f in ["address1", "address2", "city", "state", "zip_code"]
        ).lower().strip()
        ov_addr = normalize_address(overture_val).lower()
        yp_addr = normalize_address(yelp_val).lower()
        ov_w = set(ov_addr.split())
        yp_w = set(yp_addr.split())
        yl_w = set(yelp_addr.split())
        ov_sim = len(ov_w & yl_w) / len(ov_w | yl_w) if (ov_w | yl_w) else 0
        yp_sim = len(yp_w & yl_w) / len(yp_w | yl_w) if (yp_w | yl_w) else 0
        return "current" if ov_sim >= yp_sim else "base"

    # For other attributes, prefer whichever has more content
    ov_str = str(overture_val) if overture_val else ""
    yp_str = str(yelp_val) if yelp_val else ""
    return "current" if len(ov_str) >= len(yp_str) else "base"


def main() -> None:
    df = load_parquet_duckdb(DEFAULT_DATA_PATH)

    if not YELP_CACHE_PATH.exists():
        print("No Yelp cache found. Run scripts.yelp_verify first.")
        return

    with YELP_CACHE_PATH.open("r", encoding="utf-8") as f:
        yelp_cache = json.load(f)

    print(f"Original parquet: {len(df)} records")
    print(f"Yelp cache: {len(yelp_cache)} businesses")

    # Build pairs: Overture data as "current", Yelp data as "base"
    yelp_pairs = []
    golden_records = []

    for _, row in df.iterrows():
        record_id = str(row.get("id", ""))
        yelp_biz = yelp_cache.get(record_id)

        if yelp_biz is None:
            continue

        pair = {
            "id": record_id,
            "base_id": f"yelp_{yelp_biz.get('id', '')}",
            "sources": row.get("sources"),
            "confidence": row.get("confidence"),
            # Overture side (current)
            "names": row.get("names"),
            "categories": row.get("categories"),
            "websites": row.get("websites"),
            "phones": row.get("phones"),
            "addresses": row.get("addresses"),
            "emails": row.get("emails"),
            "socials": row.get("socials"),
            "brand": row.get("brand"),
            # Yelp side (base)
            "base_sources": json.dumps([{"dataset": "yelp", "record_id": yelp_biz.get("id", "")}]),
            "base_confidence": 0.85,
            "base_names": _yelp_to_overture_names(yelp_biz),
            "base_categories": _yelp_to_overture_categories(yelp_biz),
            "base_websites": _yelp_to_overture_websites(yelp_biz),
            "base_phones": _yelp_to_overture_phones(yelp_biz),
            "base_addresses": _yelp_to_overture_addresses(yelp_biz),
            "base_emails": None,
            "base_socials": None,
            "base_brand": None,
        }
        yelp_pairs.append(pair)

        # Build golden label for this pair
        labels = {}
        cur_data = {}
        bas_data = {}
        for attr in ATTRS:
            ov_val = pair.get(attr)
            yp_val = pair.get(f"base_{attr}")
            cur_data[attr] = ov_val
            bas_data[attr] = yp_val
            labels[attr] = _label_attr_from_yelp(attr, ov_val, yp_val, yelp_biz)

        cur_data["confidence"] = pair["confidence"]
        cur_data["sources"] = pair["sources"]
        bas_data["confidence"] = pair["base_confidence"]
        bas_data["sources"] = pair["base_sources"]

        golden_records.append({
            "id": record_id,
            "base_id": pair["base_id"],
            "labels": labels,
            "notes": "yelp_verified",
            "current": cur_data,
            "base": bas_data,
        })

    # Also include the original pairs (Overture vs Overture) with Yelp-verified labels
    existing_golden: dict[str, dict] = {}
    if GOLDEN_PATH.exists():
        with GOLDEN_PATH.open("r", encoding="utf-8") as f:
            for r in json.load(f):
                existing_golden[str(r.get("id", ""))] = r

    original_golden = []
    for _, row in df.iterrows():
        record_id = str(row.get("id", ""))
        yelp_biz = yelp_cache.get(record_id)

        if record_id in existing_golden:
            existing = existing_golden[record_id]
            if yelp_biz is not None:
                # Re-label using Yelp ground truth
                labels = existing.get("labels", {})
                cur = existing.get("current", {})
                bas = existing.get("base", {})
                for attr in ATTRS:
                    cur_val = cur.get(attr)
                    bas_val = bas.get(attr)
                    if is_missing(cur_val) and is_missing(bas_val):
                        labels[attr] = "skip"
                    elif is_missing(cur_val):
                        labels[attr] = "base"
                    elif is_missing(bas_val):
                        labels[attr] = "current"
                    else:
                        from scripts.yelp_verify import _label_from_yelp
                        yelp_label = _label_from_yelp(attr, cur_val, bas_val, yelp_biz)
                        if yelp_label:
                            labels[attr] = yelp_label
                        # If Yelp can't decide, keep existing label
                existing["labels"] = labels
                existing["notes"] = "yelp_verified"
            original_golden.append(existing)
        else:
            # Non-Yelp records: label trivial cases, leave hard ones empty
            cur_data = {}
            bas_data = {}
            labels = {}
            for attr in ATTRS:
                cv = row.get(attr)
                bv = row.get(f"base_{attr}")
                cur_data[attr] = cv if not isinstance(cv, float) else None
                bas_data[attr] = bv if not isinstance(bv, float) else None
                if is_missing(cv) and is_missing(bv):
                    labels[attr] = "skip"
                elif is_missing(cv):
                    labels[attr] = "base"
                elif is_missing(bv):
                    labels[attr] = "current"
                else:
                    labels[attr] = ""  # Not labeled - no ground truth

            original_golden.append({
                "id": record_id,
                "base_id": str(row.get("base_id", "")),
                "labels": labels,
                "notes": "",
                "current": cur_data,
                "base": bas_data,
            })

    # Save the Yelp pair dataset
    yelp_df = pd.DataFrame(yelp_pairs)
    ensure_dir(OUTPUT_PARQUET.parent)
    yelp_df.to_parquet(OUTPUT_PARQUET, index=False)

    # Combine original + yelp pairs into merged dataset
    orig_df = pd.read_parquet(DEFAULT_DATA_PATH)
    common_cols = [c for c in orig_df.columns if c in yelp_df.columns]
    for col in common_cols:
        if col not in yelp_df.columns:
            yelp_df[col] = None
    merged = pd.concat([orig_df[common_cols], yelp_df[common_cols]], ignore_index=True)
    merged_path = PROJECT_ROOT / "data" / "merged_dataset.parquet"
    merged.to_parquet(merged_path, index=False)

    # Save combined golden dataset (original + yelp pairs)
    combined_golden = original_golden + golden_records
    ensure_dir(GOLDEN_PATH.parent)
    write_json(GOLDEN_PATH, combined_golden, indent=2)

    # Stats
    yelp_verified = sum(1 for g in combined_golden if g.get("notes") == "yelp_verified")
    total_labels = sum(
        1 for g in combined_golden
        for v in (g.get("labels") or {}).values()
        if str(v).strip().lower() in {"current", "base", "skip", "tie"}
    )
    empty_labels = sum(
        1 for g in combined_golden
        for v in (g.get("labels") or {}).values()
        if not str(v).strip()
    )

    print(f"\n=== Results ===")
    print(f"  Yelp-verified pairs created: {len(yelp_pairs)}")
    print(f"  Yelp pair dataset: {OUTPUT_PARQUET}")
    print(f"  Merged dataset: {merged_path} ({len(merged)} total rows)")
    print(f"  Golden dataset: {GOLDEN_PATH}")
    print(f"    Total records: {len(combined_golden)}")
    print(f"    Yelp-verified: {yelp_verified}")
    print(f"    Labels filled: {total_labels}")
    print(f"    Labels empty: {empty_labels}")
    print(f"\n  Next: retrain on this dataset:")
    print(f"    python3 -m scripts.conflation.ml_selection --input {merged_path}")


if __name__ == "__main__":
    main()
