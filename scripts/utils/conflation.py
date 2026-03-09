"""Rule and feature utilities for attribute selection baselines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz

from scripts.utils.parsing import (
    _flatten_raw_values,
    extract_tokens,
    is_missing,
    jaccard_overlap,
    normalize_address,
    normalize_name,
    normalize_phone,
    normalize_url,
    parse_maybe_json,
    token_count,
)

CORE_ATTRIBUTES = [
    "names",
    "categories",
    "websites",
    "phones",
    "addresses",
    "emails",
    "socials",
]


@dataclass
class RuleDecision:
    winner: str
    score_current: float
    score_base: float
    reason: str


def source_count(value: Any) -> int:
    """Count source records from raw list/dict/string JSON fields."""
    parsed = parse_maybe_json(value)
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        return len(parsed)
    if isinstance(parsed, str):
        return 1 if parsed.strip() else 0
    return 1


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def pair_similarity(attr: str, current_value: Any, base_value: Any) -> float:
    """Attribute-aware similarity between current and base values."""
    left = extract_tokens(attr, current_value)
    right = extract_tokens(attr, base_value)

    if attr in {"names", "categories", "addresses"}:
        left_text = " ".join(sorted(left))
        right_text = " ".join(sorted(right))
        return _text_similarity(left_text, right_text)

    return jaccard_overlap(left, right)


def _validity_score(attr: str, value: Any) -> float:
    if is_missing(value):
        return 0.0

    tokens = extract_tokens(attr, value)
    if not tokens:
        return 0.0

    # Quality prior: more normalized candidates can indicate richer evidence,
    # but we cap contribution to avoid over-favoring very long token sets.
    return min(len(tokens), 4) / 4.0


def _length_score(attr: str, value: Any) -> float:
    if is_missing(value):
        return 0.0

    if attr == "names":
        return min(len(normalize_name(value)), 60) / 60.0
    if attr == "addresses":
        return min(len(normalize_address(value)), 80) / 80.0
    return min(token_count(value), 5) / 5.0


def attribute_quality(attr: str, value: Any) -> float:
    """Content-only quality score in [0, 1]."""
    validity = _validity_score(attr, value)
    length = _length_score(attr, value)
    return 0.7 * validity + 0.3 * length


def decide_rule_based(
    attr: str,
    current_value: Any,
    base_value: Any,
    confidence: float | None,
    base_confidence: float | None,
    current_sources: Any,
    base_sources: Any,
) -> RuleDecision:
    """Decide winner for one attribute using confidence + quality heuristics."""
    current_missing = is_missing(current_value)
    base_missing = is_missing(base_value)

    if current_missing and base_missing:
        return RuleDecision("tie", 0.0, 0.0, "both_missing")
    if current_missing:
        return RuleDecision("base", 0.0, 1.0, "current_missing")
    if base_missing:
        return RuleDecision("current", 1.0, 0.0, "base_missing")

    current_quality = attribute_quality(attr, current_value)
    base_quality = attribute_quality(attr, base_value)

    conf = float(confidence) if confidence is not None else 0.0
    base_conf = float(base_confidence) if base_confidence is not None else 0.0

    source_bonus_current = min(source_count(current_sources), 5) / 5.0
    source_bonus_base = min(source_count(base_sources), 5) / 5.0

    score_current = 0.45 * conf + 0.45 * current_quality + 0.10 * source_bonus_current
    score_base = 0.45 * base_conf + 0.45 * base_quality + 0.10 * source_bonus_base

    margin = score_current - score_base
    if abs(margin) <= 0.03:
        winner = "current" if current_quality >= base_quality else "base"
        reason = "quality_tiebreak"
    elif margin > 0:
        winner = "current"
        reason = "higher_rule_score"
    else:
        winner = "base"
        reason = "higher_rule_score"

    return RuleDecision(winner, round(score_current, 4), round(score_base, 4), reason)


def proxy_label(
    attr: str,
    current_value: Any,
    base_value: Any,
    confidence: float | None,
    base_confidence: float | None,
) -> str:
    """Weak supervision label used when a manual golden set is unavailable."""
    current_missing = is_missing(current_value)
    base_missing = is_missing(base_value)
    if current_missing and base_missing:
        return "skip"
    if current_missing:
        return "base"
    if base_missing:
        return "current"

    current_quality = attribute_quality(attr, current_value)
    base_quality = attribute_quality(attr, base_value)

    conf = float(confidence) if confidence is not None else 0.0
    base_conf = float(base_confidence) if base_confidence is not None else 0.0

    current_score = 0.55 * conf + 0.45 * current_quality
    base_score = 0.55 * base_conf + 0.45 * base_quality

    if abs(current_score - base_score) <= 0.02:
        return "skip"
    return "current" if current_score > base_score else "base"


def _char_trigrams(text: str) -> set[str]:
    text = text.lower().strip()
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    ta = _char_trigrams(a)
    tb = _char_trigrams(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _source_freshness(sources_raw: Any) -> float:
    """Extract most recent source timestamp as fractional year since 2020."""
    parsed = parse_maybe_json(sources_raw)
    if not isinstance(parsed, list):
        return 0.0

    latest = 0.0
    for src in parsed:
        if not isinstance(src, dict):
            continue
        ts = src.get("update_time", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            years_since_2020 = (dt.year - 2020) + dt.month / 12.0
            latest = max(latest, years_since_2020)
        except (ValueError, TypeError):
            pass
    return latest


def _unique_datasets(sources_raw: Any) -> int:
    parsed = parse_maybe_json(sources_raw)
    if not isinstance(parsed, list):
        return 0
    datasets = {str(s.get("dataset", "")).lower() for s in parsed if isinstance(s, dict)}
    return len(datasets - {""})


def _address_field_count(value: Any) -> int:
    parsed = parse_maybe_json(value)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return 0
    fields = ["freeform", "locality", "postcode", "region", "country", "house_number", "road"]
    return sum(1 for f in fields if parsed.get(f) and str(parsed[f]).strip())


def _phone_has_country_code(value: Any) -> float:
    parsed = parse_maybe_json(value)
    if isinstance(parsed, list):
        for item in parsed:
            if str(item).strip().startswith("+"):
                return 1.0
    if isinstance(parsed, str) and parsed.strip().startswith("+"):
        return 1.0
    return 0.0


def _phone_digit_count(value: Any) -> int:
    phone = normalize_phone(value)
    return len(re.sub(r"\D", "", phone))


def _category_count(value: Any) -> int:
    parsed = parse_maybe_json(value)
    if not isinstance(parsed, dict):
        return 0
    count = 1 if parsed.get("primary") else 0
    alt = parsed.get("alternate", [])
    if isinstance(alt, list):
        count += len(alt)
    return count


def _is_https(value: Any) -> float:
    for raw in _flatten_raw_values(value):
        if str(raw).strip().lower().startswith("https"):
            return 1.0
    return 0.0


def _url_has_path(value: Any) -> float:
    for raw in _flatten_raw_values(value):
        url = normalize_url(raw)
        if url and "/" in url.split("//", 1)[-1]:
            return 1.0
    return 0.0


def _is_social_url(value: Any) -> float:
    social_domains = {"facebook.com", "youtube.com", "twitter.com", "instagram.com", "x.com"}
    for raw in _flatten_raw_values(value):
        url = str(raw).lower()
        if any(d in url for d in social_domains):
            return 1.0
    return 0.0


def _char_count(attr: str, value: Any) -> int:
    if is_missing(value):
        return 0
    if attr == "names":
        return len(normalize_name(value))
    if attr == "addresses":
        return len(normalize_address(value))
    flat = _flatten_raw_values(value)
    return sum(len(str(v)) for v in flat)


def _word_count(attr: str, value: Any) -> int:
    if is_missing(value):
        return 0
    if attr == "names":
        return len(normalize_name(value).split())
    if attr == "addresses":
        return len(normalize_address(value).split())
    return token_count(value)


def feature_vector(
    attr: str,
    current_value: Any,
    base_value: Any,
    confidence: float | None,
    base_confidence: float | None,
    current_sources: Any,
    base_sources: Any,
) -> dict[str, float]:
    """Build rich model features for a single attribute decision."""
    current_miss = 1.0 if is_missing(current_value) else 0.0
    base_miss = 1.0 if is_missing(base_value) else 0.0
    both_present = 1.0 if (current_miss == 0.0 and base_miss == 0.0) else 0.0

    current_quality = attribute_quality(attr, current_value)
    base_quality = attribute_quality(attr, base_value)

    conf = float(confidence) if confidence is not None else 0.0
    base_conf = float(base_confidence) if base_confidence is not None else 0.0
    conf_sum = conf + base_conf
    conf_ratio = conf / conf_sum if conf_sum > 0 else 0.5

    current_tokens = extract_tokens(attr, current_value)
    base_tokens = extract_tokens(attr, base_value)

    src_cur = source_count(current_sources)
    src_bas = source_count(base_sources)

    # Edit distance (rapidfuzz)
    cur_text = " ".join(sorted(current_tokens))
    bas_text = " ".join(sorted(base_tokens))
    edit_ratio = fuzz.ratio(cur_text, bas_text) / 100.0 if cur_text or bas_text else 1.0
    partial_ratio = fuzz.partial_ratio(cur_text, bas_text) / 100.0 if cur_text or bas_text else 1.0
    token_sort_ratio = fuzz.token_sort_ratio(cur_text, bas_text) / 100.0 if cur_text or bas_text else 1.0

    # Trigram overlap
    trigram_sim = _trigram_jaccard(cur_text, bas_text)

    # Source freshness
    fresh_cur = _source_freshness(current_sources)
    fresh_bas = _source_freshness(base_sources)

    # Source diversity
    datasets_cur = float(_unique_datasets(current_sources))
    datasets_bas = float(_unique_datasets(base_sources))

    # Character and word counts
    chars_cur = float(_char_count(attr, current_value))
    chars_bas = float(_char_count(attr, base_value))
    words_cur = float(_word_count(attr, current_value))
    words_bas = float(_word_count(attr, base_value))

    features: dict[str, float] = {
        # --- core ---
        "current_missing": current_miss,
        "base_missing": base_miss,
        "both_present": both_present,
        "conf_current": conf,
        "conf_base": base_conf,
        "conf_delta": conf - base_conf,
        "conf_ratio": conf_ratio,
        "quality_current": current_quality,
        "quality_base": base_quality,
        "quality_delta": current_quality - base_quality,
        "source_count_current": float(src_cur),
        "source_count_base": float(src_bas),
        "source_count_delta": float(src_cur - src_bas),
        "token_count_current": float(len(current_tokens)),
        "token_count_base": float(len(base_tokens)),
        "token_count_delta": float(len(current_tokens) - len(base_tokens)),
        "pair_similarity": pair_similarity(attr, current_value, base_value),
        # --- edit distance ---
        "edit_ratio": edit_ratio,
        "partial_ratio": partial_ratio,
        "token_sort_ratio": token_sort_ratio,
        "trigram_similarity": trigram_sim,
        # --- recency ---
        "freshness_current": fresh_cur,
        "freshness_base": fresh_bas,
        "freshness_delta": fresh_cur - fresh_bas,
        # --- source diversity ---
        "datasets_current": datasets_cur,
        "datasets_base": datasets_bas,
        "datasets_delta": datasets_cur - datasets_bas,
        # --- length ---
        "char_count_current": chars_cur,
        "char_count_base": chars_bas,
        "char_count_delta": chars_cur - chars_bas,
        "word_count_current": words_cur,
        "word_count_base": words_bas,
        "word_count_delta": words_cur - words_bas,
    }

    # --- attribute-specific structural features ---
    if attr == "addresses":
        features["addr_fields_current"] = float(_address_field_count(current_value))
        features["addr_fields_base"] = float(_address_field_count(base_value))
        features["addr_fields_delta"] = features["addr_fields_current"] - features["addr_fields_base"]

    if attr == "phones":
        features["phone_cc_current"] = _phone_has_country_code(current_value)
        features["phone_cc_base"] = _phone_has_country_code(base_value)
        features["phone_digits_current"] = float(_phone_digit_count(current_value))
        features["phone_digits_base"] = float(_phone_digit_count(base_value))
        features["phone_digits_delta"] = features["phone_digits_current"] - features["phone_digits_base"]

    if attr == "categories":
        features["cat_count_current"] = float(_category_count(current_value))
        features["cat_count_base"] = float(_category_count(base_value))
        features["cat_count_delta"] = features["cat_count_current"] - features["cat_count_base"]

    if attr == "websites":
        features["https_current"] = _is_https(current_value)
        features["https_base"] = _is_https(base_value)
        features["url_path_current"] = _url_has_path(current_value)
        features["url_path_base"] = _url_has_path(base_value)
        features["is_social_current"] = _is_social_url(current_value)
        features["is_social_base"] = _is_social_url(base_value)

    if attr == "names":
        features["name_chars_current"] = chars_cur
        features["name_chars_base"] = chars_bas
        features["name_words_current"] = words_cur
        features["name_words_base"] = words_bas

    return features
