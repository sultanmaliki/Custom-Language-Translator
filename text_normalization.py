"""Light, deterministic text cleaning for both sides of the corpus."""
import re
import sys
import unicodedata
from pathlib import Path

import config

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
_QUOTES = {
    ord("‘"): "'", ord("’"): "'", ord("ʼ"): "'",
    ord("“"): '"', ord("”"): '"',
}
_SPACES = re.compile(r"\s+")
_REPEATS = re.compile(r"(.)\1{2,}")
# a word = letters, optionally joined by apostrophes (keeps "ka'im" together)
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def use_utf8_console() -> None:
    """Windows consoles default to a legacy code page that raises an error on characters
    like Devanagari or accented letters. Call this at the start of any script that prints text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_ZERO_WIDTH).translate(_QUOTES)
    return _SPACES.sub(" ", text).strip()


def clean_raw(text: str) -> str:
    """Minimal cleanup used when *storing* a pair: whitespace only, so the raw
    corpus keeps what the contributor typed. Tabs/newlines would break the TSV."""
    return _SPACES.sub(" ", str(text)).strip()


def load_variants(path: Path = config.VARIANTS_FILE) -> dict[str, str]:
    """Read `variant<TAB>canonical` lines (case-insensitive). Missing file = {}."""
    variants: dict[str, str] = {}
    if not Path(path).exists():
        return variants
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            variants[parts[0].strip().lower()] = parts[1].strip().lower()
    return variants


def normalize_english(text: str) -> str:
    return _clean(text)


def normalize_nawayathi(text: str, variants: dict[str, str] | None = None) -> str:
    text = _clean(text)
    if config.LOWERCASE_NWY:
        text = text.lower()
    if config.COLLAPSE_REPEATS_NWY:
        text = _REPEATS.sub(r"\1\1", text)
    if variants:
        text = _WORD.sub(lambda m: variants.get(m.group(0).lower(), m.group(0)), text)
    return text
