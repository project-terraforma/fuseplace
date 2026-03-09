"""Intelligent auto-labeling using domain-specific content analysis.

Unlike the simple proxy_label (which just uses confidence + a basic quality score),
this labeler actually parses and compares attribute content to decide which source
is more complete, more specific, and better formatted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scripts.utils.io import ANALYSIS_DIR, ensure_dir
from scripts.utils.parsing import (
    _flatten_raw_values,
    extract_primary_text,
    is_missing,
    normalize_address,
    normalize_name,
    normalize_phone,
    normalize_url,
    parse_maybe_json,
)

GOLDEN_PATH = ANALYSIS_DIR / "golden" / "golden_dataset_template.json"
LABEL_CSV_PATH = ANALYSIS_DIR / "golden" / "labeling_worksheet.csv"
ATTRS = ["names", "categories", "websites", "phones", "addresses", "emails", "socials"]


def _count_address_fields(raw: object) -> int:
    """Count how many address components are filled (freeform, locality, postcode, region, country)."""
    parsed = parse_maybe_json(raw)
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        return 0
    fields = ["freeform", "locality", "postcode", "region", "country", "house_number", "road"]
    return sum(1 for f in fields if parsed.get(f) and str(parsed[f]).strip())


def _postcode_specificity(raw: object) -> int:
    """More specific postcodes (e.g., 39051-4222 vs 39051) score higher."""
    parsed = parse_maybe_json(raw)
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        return 0
    pc = str(parsed.get("postcode", "")).strip()
    if not pc:
        return 0
    return len(pc)


def _count_categories(raw: object) -> int:
    """Count total categories (primary + alternates)."""
    parsed = parse_maybe_json(raw)
    if parsed is None:
        return 0
    if isinstance(parsed, dict):
        count = 1 if parsed.get("primary") else 0
        alt = parsed.get("alternate", [])
        if isinstance(alt, list):
            count += len(alt)
        return count
    return 0


def _category_specificity(raw: object) -> float:
    """More specific categories (longer, more descriptive) score higher."""
    parsed = parse_maybe_json(raw)
    if parsed is None:
        return 0.0
    if isinstance(parsed, dict):
        primary = str(parsed.get("primary", "")).replace("_", " ").strip()
        return len(primary)
    return 0.0


def _phone_has_country_code(raw: object) -> bool:
    parsed = parse_maybe_json(raw)
    if isinstance(parsed, list):
        for item in parsed:
            s = str(item).strip()
            if s.startswith("+"):
                return True
    if isinstance(parsed, str):
        return parsed.strip().startswith("+")
    return False


def _phone_digit_count(raw: object) -> int:
    phone = normalize_phone(raw)
    return len(re.sub(r"\D", "", phone))


def _website_quality(raw: object) -> float:
    """Score website quality: penalize empty, YouTube, generic pages; reward specific paths."""
    urls = _flatten_raw_values(raw)
    if not urls:
        return 0.0

    best_score = 0.0
    for url_str in urls:
        url = str(url_str).strip().lower()
        if not url:
            continue

        score = 1.0
        normalized = normalize_url(url)
        if not normalized:
            continue

        # Penalize non-business URLs
        if "youtube.com" in normalized or "facebook.com" in normalized:
            score *= 0.3
        if "twitter.com" in normalized or "x.com" in normalized:
            score *= 0.3

        # Reward HTTPS
        if url.startswith("https"):
            score += 0.2

        # Reward specific paths (not just root)
        path_parts = normalized.split("/")
        if len(path_parts) > 1 and path_parts[-1]:
            score += 0.3

        # Penalize URLs with tracking params
        if "utm_" in url:
            score -= 0.3

        best_score = max(best_score, score)

    return best_score


def _name_quality(raw: object) -> float:
    """Score name quality: longer, more complete names are better."""
    name = normalize_name(raw)
    if not name:
        return 0.0

    score = len(name) / 50.0

    # Penalize truncated-looking names (ending with abbreviations or fragments)
    if name.endswith(".") or name.endswith("-"):
        score -= 0.2

    # Penalize garbled text (high ratio of special chars)
    alpha_count = sum(1 for c in name if c.isalpha() or c.isspace())
    if len(name) > 0 and alpha_count / len(name) < 0.7:
        score -= 0.3

    return score


def label_names(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_score = _name_quality(cur_raw)
    bas_score = _name_quality(bas_raw)

    cur_name = normalize_name(cur_raw)
    bas_name = normalize_name(bas_raw)

    # If names are essentially the same, prefer higher confidence
    name_diff = abs(len(cur_name) - len(bas_name))
    if name_diff <= 2 and cur_name.replace("-", "").replace(" ", "") == bas_name.replace("-", "").replace(" ", ""):
        return "current" if cur_conf >= bas_conf else "base"

    # Prefer longer / more descriptive name
    if abs(cur_score - bas_score) > 0.1:
        return "current" if cur_score > bas_score else "base"

    return "current" if cur_conf >= bas_conf else "base"


def label_categories(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_count = _count_categories(cur_raw)
    bas_count = _count_categories(bas_raw)
    cur_spec = _category_specificity(cur_raw)
    bas_spec = _category_specificity(bas_raw)

    # More categories + more specific primary = better
    cur_score = cur_count * 0.4 + cur_spec * 0.3 + cur_conf * 0.3
    bas_score = bas_count * 0.4 + bas_spec * 0.3 + bas_conf * 0.3

    if abs(cur_score - bas_score) < 0.05:
        return "current" if cur_conf >= bas_conf else "base"

    return "current" if cur_score > bas_score else "base"


def label_websites(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_q = _website_quality(cur_raw)
    bas_q = _website_quality(bas_raw)

    if abs(cur_q - bas_q) < 0.1:
        return "current" if cur_conf >= bas_conf else "base"

    return "current" if cur_q > bas_q else "base"


def label_phones(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_has_cc = _phone_has_country_code(cur_raw)
    bas_has_cc = _phone_has_country_code(bas_raw)
    cur_digits = _phone_digit_count(cur_raw)
    bas_digits = _phone_digit_count(bas_raw)

    # Prefer phone with country code
    if cur_has_cc and not bas_has_cc:
        return "current"
    if bas_has_cc and not cur_has_cc:
        return "base"

    # Prefer more digits (more complete)
    if cur_digits != bas_digits:
        return "current" if cur_digits > bas_digits else "base"

    return "current" if cur_conf >= bas_conf else "base"


def label_addresses(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_fields = _count_address_fields(cur_raw)
    bas_fields = _count_address_fields(bas_raw)
    cur_pc = _postcode_specificity(cur_raw)
    bas_pc = _postcode_specificity(bas_raw)

    cur_addr = normalize_address(cur_raw)
    bas_addr = normalize_address(bas_raw)

    # Score: field completeness + postcode specificity + address length
    cur_score = cur_fields * 2.0 + cur_pc * 0.5 + len(cur_addr) * 0.1
    bas_score = bas_fields * 2.0 + bas_pc * 0.5 + len(bas_addr) * 0.1

    # Penalize garbled addresses
    if cur_addr:
        alpha_ratio = sum(1 for c in cur_addr if c.isalpha() or c.isspace()) / len(cur_addr)
        if alpha_ratio < 0.5:
            cur_score *= 0.5
    if bas_addr:
        alpha_ratio = sum(1 for c in bas_addr if c.isalpha() or c.isspace()) / len(bas_addr)
        if alpha_ratio < 0.5:
            bas_score *= 0.5

    if abs(cur_score - bas_score) < 0.5:
        return "current" if cur_conf >= bas_conf else "base"

    return "current" if cur_score > bas_score else "base"


def label_emails(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_vals = _flatten_raw_values(cur_raw)
    bas_vals = _flatten_raw_values(bas_raw)
    cur_valid = [v for v in cur_vals if "@" in str(v)]
    bas_valid = [v for v in bas_vals if "@" in str(v)]

    if len(cur_valid) != len(bas_valid):
        return "current" if len(cur_valid) > len(bas_valid) else "base"

    return "current" if cur_conf >= bas_conf else "base"


def label_socials(cur_raw: object, bas_raw: object, cur_conf: float, bas_conf: float) -> str:
    cur_vals = [v for v in _flatten_raw_values(cur_raw) if str(v).strip()]
    bas_vals = [v for v in _flatten_raw_values(bas_raw) if str(v).strip()]

    if len(cur_vals) != len(bas_vals):
        return "current" if len(cur_vals) > len(bas_vals) else "base"

    return "current" if cur_conf >= bas_conf else "base"


LABELERS = {
    "names": label_names,
    "categories": label_categories,
    "websites": label_websites,
    "phones": label_phones,
    "addresses": label_addresses,
    "emails": label_emails,
    "socials": label_socials,
}


def main() -> None:
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.read_csv(LABEL_CSV_PATH, dtype=str).fillna("")

    label_results: dict[tuple[str, str], str] = {}
    stats: dict[str, dict[str, int]] = {a: {"current": 0, "base": 0} for a in ATTRS}

    for record in data:
        record_id = str(record.get("id", ""))
        cur = record.get("current", {})
        bas = record.get("base", {})
        cur_conf = float(cur.get("confidence", 0) or 0)
        bas_conf = float(bas.get("confidence", 0) or 0)

        for attr in ATTRS:
            cur_val = cur.get(attr)
            bas_val = bas.get(attr)

            if is_missing(cur_val) or is_missing(bas_val):
                continue

            labeler = LABELERS.get(attr)
            if labeler is None:
                continue

            label = labeler(cur_val, bas_val, cur_conf, bas_conf)
            label_results[(record_id, attr)] = label
            stats[attr][label] += 1

    labeled_count = 0
    for idx, row in df.iterrows():
        key = (str(row["id"]), str(row["attribute"]))
        if key in label_results:
            df.at[idx, "label"] = label_results[key]
            labeled_count += 1

    df.to_csv(LABEL_CSV_PATH, index=False)

    print(f"Auto-labeled {labeled_count} ambiguous pairs.")
    print("\nLabel distribution per attribute:")
    for attr in ATTRS:
        cur = stats[attr]["current"]
        bas = stats[attr]["base"]
        total = cur + bas
        if total > 0:
            print(f"  {attr:>12s}: current={cur} ({cur/total:.0%})  base={bas} ({bas/total:.0%})")

    print(f"\nLabeled CSV saved to: {LABEL_CSV_PATH}")
    print("\nNext: import labels and retrain:")
    print("  python3 -m scripts.label_golden --import")
    print("  python3 -m scripts.conflation.ml_selection")
    print("  python3 -m scripts.conflation.evaluate_methods")


if __name__ == "__main__":
    main()
