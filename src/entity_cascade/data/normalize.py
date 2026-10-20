"""Text normalization, tokenization and model-code / brand extraction."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# Symbols that NFKC would turn into letters ("™" -> "TM"); drop them before normalizing.
_STRIP_SYMBOLS = str.maketrans("", "", "®™©℠")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = dict.fromkeys(map(ord, "″“”„˝"), '"')

_NUM = r"(\d+(?:\.\d+)?)"
# 10", 10'' , 10-inch, 10 inches -> 10in
_INCH_RE = re.compile(_NUM + r"\s*(?:\"|''|-?\s*inch(?:es)?\b)")
_UNIT_ALIASES = {
    "megapixel": "mp",
    "megapixels": "mp",
    "mega pixel": "mp",
    "mega pixels": "mp",
    "watt": "w",
    "watts": "w",
    "volt": "v",
    "volts": "v",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
}
_UNIT_RE = re.compile(
    _NUM
    + r"[\s\-]*(gb|mb|tb|kb|ghz|mhz|khz|hz|mp|megapixels?|mega pixels?|watts?|w|lbs?|pounds?|oz"
    r"|mm|cm|ft|mah|ma|volts?|v)\b"
)
_SPACES_RE = re.compile(r"\s+")

# A plain unit token like "4gb" or "720p" is not a model code.
_UNIT_TOKEN_RE = re.compile(
    r"\d+(?:gb|mb|tb|kb|ghz|mhz|khz|hz|mp|w|v|lb|oz|mm|cm|ft|in|mah|ma|p|i|x|k|g|m)"
)
_CODE_RE = re.compile(r"[a-z0-9]+(?:[\-/][a-z0-9]+)*")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-/.][a-z0-9]+)*")
_COMPANY_SUFFIXES = frozenset({"inc", "corp", "corporation", "co", "ltd", "llc", "company"})


def normalize_text(text: str) -> str:
    """Lowercase, unicode-normalize, unify dashes/quotes and measurement units."""
    s = text.translate(_STRIP_SYMBOLS)
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.translate(_DASHES).translate(_QUOTES)
    s = _INCH_RE.sub(r"\1in", s)

    def _unit(m: re.Match[str]) -> str:
        unit = m.group(2)
        unit = _UNIT_ALIASES.get(re.sub(r"\s+", " ", unit), unit)
        return f"{m.group(1)}{unit}"

    s = _UNIT_RE.sub(_unit, s)
    return _SPACES_RE.sub(" ", s).strip()


def tokenize(text: str) -> list[str]:
    """Tokens for BM25. Hyphen/slash-joined codes are emitted both split and joined."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(normalize_text(text)):
        tok = m.group(0)
        parts = [p for p in re.split(r"[\-/.]", tok) if p]
        out.extend(parts)
        if len(parts) > 1:
            joined = "".join(parts)
            if any(c.isdigit() for c in joined) and any(c.isalpha() for c in joined):
                out.append(joined)
    return out


def canonical_code(code: str) -> str:
    """Model code in canonical form: lowercase, without hyphens and slashes."""
    return re.sub(r"[\-/]", "", code.casefold())


def extract_codes(text: str, min_len: int = 4) -> list[str]:
    """Model numbers / part numbers: mixed letters+digits tokens, unit tokens excluded.

    Returned in canonical form ("DMC-FZ18" -> "dmcfz18"), in order of first appearance.
    """
    seen: dict[str, None] = {}
    for m in _CODE_RE.finditer(normalize_text(text)):
        canon = canonical_code(m.group(0))
        if len(canon) < min_len:
            continue
        if not (any(c.isdigit() for c in canon) and any(c.isalpha() for c in canon)):
            continue
        if _UNIT_TOKEN_RE.fullmatch(canon):
            continue
        seen.setdefault(canon, None)
    return list(seen)


def normalize_brand(brand: str) -> str:
    """Brand in comparable form: normalized, punctuation and company suffixes removed."""
    words: list[str] = re.findall(r"[a-z0-9]+", normalize_text(brand))
    while words and words[-1] in _COMPANY_SUFFIXES:
        words.pop()
    return " ".join(words)


def build_brand_vocab(brands: Iterable[str | None]) -> frozenset[str]:
    out = {normalize_brand(b) for b in brands if b and b.strip()}
    out.discard("")
    return frozenset(out)


def extract_brand(name: str, vocab: frozenset[str]) -> str | None:
    """Find a known brand in a product name: two-word prefix, first word, then any word."""
    words: list[str] = re.findall(r"[a-z0-9]+", normalize_text(name))
    if len(words) >= 2 and f"{words[0]} {words[1]}" in vocab:
        return f"{words[0]} {words[1]}"
    if words and words[0] in vocab:
        return words[0]
    for w in words:
        if w in vocab:
            return w
    return None
