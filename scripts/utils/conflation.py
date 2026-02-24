"""Rule and feature utilities for attribute selection baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from scripts.utils.parsing import (
    extract_tokens,
    is_missing,
    jaccard_overlap,
    normalize_address,
    normalize_name,
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


def feature_vector(
    attr: str,
    current_value: Any,
    base_value: Any,
    confidence: float | None,
    base_confidence: float | None,
    current_sources: Any,
    base_sources: Any,
) -> dict[str, float]:
    """Build model features for a single attribute decision."""
    current_missing = 1.0 if is_missing(current_value) else 0.0
    base_missing = 1.0 if is_missing(base_value) else 0.0

    current_quality = attribute_quality(attr, current_value)
    base_quality = attribute_quality(attr, base_value)

    conf = float(confidence) if confidence is not None else 0.0
    base_conf = float(base_confidence) if base_confidence is not None else 0.0

    current_tokens = extract_tokens(attr, current_value)
    base_tokens = extract_tokens(attr, base_value)

    return {
        "current_missing": current_missing,
        "base_missing": base_missing,
        "conf_current": conf,
        "conf_base": base_conf,
        "conf_delta": conf - base_conf,
        "quality_current": current_quality,
        "quality_base": base_quality,
        "quality_delta": current_quality - base_quality,
        "source_count_current": float(source_count(current_sources)),
        "source_count_base": float(source_count(base_sources)),
        "source_count_delta": float(source_count(current_sources) - source_count(base_sources)),
        "token_count_current": float(len(current_tokens)),
        "token_count_base": float(len(base_tokens)),
        "token_count_delta": float(len(current_tokens) - len(base_tokens)),
        "pair_similarity": pair_similarity(attr, current_value, base_value),
    }
