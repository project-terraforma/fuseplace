"""Parsing and normalization helpers for place attribute conflation."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Common address abbreviations for basic address normalization.
ADDRESS_ABBR_MAP: dict[str, str] = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bpkwy\b": "parkway",
    r"\bcir\b": "circle",
    r"\bhwy\b": "highway",
}


def parse_maybe_json(value: Any) -> Any:
    """Parse JSON-like strings and return native Python objects when possible."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        json_like = (stripped[:1] in "[{" and stripped[-1:] in "]}") or stripped.lower() in {
            "null",
            "true",
            "false",
        }
        if json_like:
            try:
                return json.loads(stripped)
            except Exception:
                return value
        return value
    return value


def is_missing(value: Any) -> bool:
    """Treat null-like values (None/NaN/empty) as missing."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def normalize_text(value: str | None) -> str:
    """Lowercase and collapse internal whitespace."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return _WS_RE.sub(" ", text)


def _flatten_raw_values(value: Any) -> list[str]:
    """Convert nested value structures into a flat list of string candidates."""
    parsed = parse_maybe_json(value)
    if is_missing(parsed):
        return []

    if isinstance(parsed, str):
        return [parsed]

    if isinstance(parsed, list):
        out: list[str] = []
        for item in parsed:
            out.extend(_flatten_raw_values(item))
        return out

    if isinstance(parsed, dict):
        out: list[str] = []
        if "primary" in parsed:
            out.extend(_flatten_raw_values(parsed.get("primary")))
        alt = parsed.get("alternate")
        if isinstance(alt, list):
            out.extend(_flatten_raw_values(alt))
        elif alt is not None:
            out.extend(_flatten_raw_values(alt))

        for key, item in parsed.items():
            if key in {"primary", "alternate"}:
                continue
            if isinstance(item, (str, list, dict, tuple)):
                out.extend(_flatten_raw_values(item))
        return out

    return [str(parsed)]


def extract_primary_text(value: Any) -> str:
    """Extract a single representative text value from nested inputs."""
    parsed = parse_maybe_json(value)
    if is_missing(parsed):
        return ""

    if isinstance(parsed, str):
        return normalize_text(parsed)

    if isinstance(parsed, list):
        for item in parsed:
            primary = extract_primary_text(item)
            if primary:
                return primary
        return ""

    if isinstance(parsed, dict):
        for key in ("primary", "freeform", "name"):
            candidate = parsed.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return normalize_text(candidate)

        ordered = [
            "house_number",
            "road",
            "locality",
            "region",
            "postcode",
            "country",
        ]
        pieces = [normalize_text(parsed.get(k)) for k in ordered if isinstance(parsed.get(k), str)]
        pieces = [p for p in pieces if p]
        if pieces:
            return " ".join(pieces)

        # Fallback to first usable value.
        for item in parsed.values():
            primary = extract_primary_text(item)
            if primary:
                return primary

    return ""


def normalize_address(value: Any) -> str:
    """Normalize addresses into a comparable text form."""
    address = extract_primary_text(value)
    if not address:
        return ""
    normalized = address
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        normalized = re.sub(pattern, replacement, normalized)
    return normalize_text(normalized)


def normalize_name(value: Any) -> str:
    return normalize_text(extract_primary_text(value))


def normalize_category(value: Any) -> str:
    text = extract_primary_text(value).replace("_", " ")
    return normalize_text(text)


def normalize_url(value: Any) -> str:
    """Normalize URL-like values to host+path without scheme/query."""
    if is_missing(value):
        return ""
    text = normalize_text(str(value))
    if not text:
        return ""

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = f"http://{text}"

    try:
        parsed = urlparse(text)
    except Exception:
        return ""

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/").lower()

    if host.startswith("www."):
        host = host[4:]

    if not host:
        return ""

    return f"{host}{path}"


def normalize_phone(value: Any) -> str:
    """Normalize phone numbers by stripping non-digits and keeping optional leading +."""
    if is_missing(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    keep_plus = text.startswith("+")
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return f"+{digits}" if keep_plus else digits


def extract_email_tokens(value: Any) -> set[str]:
    """Extract normalized emails from nested structures."""
    tokens: set[str] = set()
    for raw in _flatten_raw_values(value):
        for match in EMAIL_RE.findall(str(raw)):
            tokens.add(match.lower())
    return tokens


def extract_social_tokens(value: Any) -> set[str]:
    """Extract handles and URLs from social attributes."""
    tokens: set[str] = set()
    parsed = parse_maybe_json(value)

    def add_one(item: Any) -> None:
        if is_missing(item):
            return
        text = str(item).strip()
        if not text:
            return
        if text.startswith("@"):
            tokens.add(text.lower())
            return
        normalized_url = normalize_url(text)
        if normalized_url:
            tokens.add(normalized_url)
            return
        tokens.add(normalize_text(text))

    if isinstance(parsed, dict):
        for k, v in parsed.items():
            add_one(k)
            add_one(v)
    elif isinstance(parsed, list):
        for v in parsed:
            add_one(v)
    else:
        add_one(parsed)

    return {t for t in tokens if t}


def extract_tokens(attr: str, value: Any) -> set[str]:
    """Convert attribute value into normalized comparison tokens."""
    attr = attr.lower()

    if attr == "names":
        text = normalize_name(value)
        return {text} if text else set()

    if attr == "categories":
        text = normalize_category(value)
        return {text} if text else set()

    if attr == "addresses":
        text = normalize_address(value)
        return {text} if text else set()

    if attr == "websites":
        return {normalize_url(v) for v in _flatten_raw_values(value) if normalize_url(v)}

    if attr == "phones":
        return {normalize_phone(v) for v in _flatten_raw_values(value) if normalize_phone(v)}

    if attr == "emails":
        return extract_email_tokens(value)

    if attr == "socials":
        return extract_social_tokens(value)

    text = extract_primary_text(value)
    return {text} if text else set()


def jaccard_overlap(left: set[str], right: set[str]) -> float:
    """Jaccard similarity over two token sets."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def value_signature(attr: str, value: Any) -> tuple[str, ...]:
    """Stable signature for conflict detection."""
    return tuple(sorted(extract_tokens(attr, value)))


def token_count(value: Any) -> int:
    """Approximate count of atomic values represented in an attribute."""
    return len([x for x in _flatten_raw_values(value) if normalize_text(str(x))])

