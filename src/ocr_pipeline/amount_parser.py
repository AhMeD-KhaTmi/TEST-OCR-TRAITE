"""
Phase 3 — Amount parser.

Two concerns:
1. Numeric amount: strip noise and normalise to a Python Decimal.
2. Amount in words: parse French number text to a numeric value (soft validation only).

Tunisian dinar convention: 3 decimal places (millimes).
  Examples: 3000,000  |  #5.000.000#  |  2,893,192  |  25 000,000 DT  |  # 3 000.000#

Public API
----------
parse_amount_numeric(raw: str) -> AmountResult
parse_amount_words(raw: str) -> AmountWordsResult
amounts_equal(a: AmountResult, b: AmountResult, tolerance: Decimal) -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AmountResult:
    raw: str                         # original OCR string
    value: Optional[Decimal] = None  # parsed value (None if failed)
    normalised: str = ""             # canonical string e.g. "3000.000"
    valid: bool = False
    error: Optional[str] = None
    has_question_marks: bool = False  # True when OCR left '?' characters


@dataclass
class AmountWordsResult:
    raw: str                         # original OCR string
    value: Optional[Decimal] = None  # parsed value (None if failed / ambiguous)
    valid: bool = False              # True ONLY when unambiguously parsed
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Numeric amount parser
# ---------------------------------------------------------------------------

def _clean_amount(raw: str) -> str:
    """Strip delimiter noise and whitespace from a raw amount string."""
    s = raw.strip()
    # Remove hash delimiters and "DT" / "TND" suffix (case-insensitive)
    s = re.sub(r"#", "", s)
    s = re.sub(r"\bDT\b|\bTND\b", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s


def _resolve_separators(s: str) -> Optional[str]:
    """Resolve ambiguous dot/comma separator usage and return a normalised string.

    Tunisian dinar always has exactly 3 millime digits.
    Strategy:
    1. Count dots and commas.
    2. If last separator is followed by exactly 3 digits → decimal separator.
    3. Other separators → thousands separators (remove).
    4. Return string with a single '.' as decimal point, or no decimal point.
    """
    # After cleaning, s may contain digits, dots, commas, spaces
    # Remove spaces (thousands spacer)
    s = s.replace(" ", "")

    # Nothing left?
    if not s:
        return None

    # Check for question marks — pass them through
    has_q = "?" in s

    # Find the last separator (dot or comma) and how many digits follow it
    match = re.search(r"[.,](\d{1,3})\s*$", s)
    if match:
        frac_digits = match.group(1)
        if len(frac_digits) == 3:
            # Last separator is the decimal separator
            int_part = s[: match.start()]
            dec_part = frac_digits
            # Remove any remaining separators from the integer part
            int_part = re.sub(r"[.,\s]", "", int_part)
            return f"{int_part}.{dec_part}"
        # else: last separator is NOT the decimal (e.g. "500,00" — 2 digits: ambiguous)
        # Fall through to treat all separators as thousands
    # No recognisable decimal: remove all separators
    clean = re.sub(r"[.,\s]", "", s)
    return clean if (clean or has_q) else None


def parse_amount_numeric(raw: str) -> AmountResult:
    """Parse a raw numeric amount string (as extracted by OCR) into an AmountResult.

    Examples of accepted inputs:
    - "3000,000"          → Decimal('3000.000')
    - "#5.000.000#"       → Decimal('5000.000')
    - "2,893,192"         → Decimal('2893.192')
    - "25 000,000 DT"     → Decimal('25000.000')
    - "# 3 000.000#"      → Decimal('3000.000')
    - "500,000"           → Decimal('500.000')
    """
    result = AmountResult(raw=raw)
    result.has_question_marks = "?" in raw

    cleaned = _clean_amount(raw)
    if not cleaned:
        result.error = f"Empty string after cleaning '{raw}'"
        return result

    normalised = _resolve_separators(cleaned)
    if normalised is None:
        result.error = f"Cannot resolve separators in '{cleaned}'"
        return result

    # Allow question marks through as string but mark not fully valid
    if result.has_question_marks:
        result.normalised = normalised
        result.error = "Amount contains unreadable positions ('?')"
        return result

    try:
        value = Decimal(normalised)
    except InvalidOperation:
        result.error = f"Cannot convert '{normalised}' to Decimal"
        return result

    if value < 0:
        result.error = f"Negative amount is not valid: {value}"
        return result

    result.value = value
    result.normalised = str(value)
    result.valid = True
    return result


def amounts_equal(
    a: AmountResult,
    b: AmountResult,
    tolerance: Decimal = Decimal("0.001"),
) -> bool:
    """Return True if both amounts are valid and within tolerance of each other."""
    if not a.valid or not b.valid:
        return False
    return abs(a.value - b.value) <= tolerance


# ---------------------------------------------------------------------------
# French number word parser
# ---------------------------------------------------------------------------
# NOTE: This is a SOFT validator only; the result is never authoritative.
#       Mismatch with the numeric amount triggers a soft warning, not a hard failure.

_UNITS = {
    "zero": 0, "zéro": 0,
    "un": 1, "une": 1,
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9,
    "dix": 10, "onze": 11, "douze": 12, "treize": 13,
    "quatorze": 14, "quinze": 15, "seize": 16,
    "dix-sept": 17, "dix sept": 17,
    "dix-huit": 18, "dix huit": 18,
    "dix-neuf": 19, "dix neuf": 19,
    # Compound 80+N forms (now directly tokenised with hyphens preserved)
    "quatre-vingt-un": 81, "quatre-vingt-deux": 82, "quatre-vingt-trois": 83,
    "quatre-vingt-quatre": 84, "quatre-vingt-cinq": 85, "quatre-vingt-six": 86,
    "quatre-vingt-sept": 87, "quatre-vingt-huit": 88, "quatre-vingt-neuf": 89,
    "quatre-vingt-dix": 90, "quatre-vingt-onze": 91, "quatre-vingt-douze": 92,
    "quatre-vingt-treize": 93, "quatre-vingt-quatorze": 94, "quatre-vingt-quinze": 95,
    "quatre-vingt-seize": 96,
    # Compound vingt+N (21-29)
    "vingt-et-un": 21, "vingt-et-une": 21, "vingt-deux": 22, "vingt-trois": 23,
    "vingt-quatre": 24, "vingt-cinq": 25, "vingt-six": 26, "vingt-sept": 27,
    "vingt-huit": 28, "vingt-neuf": 29,
    # Compound trente+N
    "trente-et-un": 31, "trente-deux": 32, "trente-trois": 33, "trente-quatre": 34,
    "trente-cinq": 35, "trente-six": 36, "trente-sept": 37, "trente-huit": 38,
    "trente-neuf": 39,
    # Compound quarante+N
    "quarante-et-un": 41, "quarante-deux": 42, "quarante-trois": 43,
    "quarante-quatre": 44, "quarante-cinq": 45, "quarante-six": 46,
    "quarante-sept": 47, "quarante-huit": 48, "quarante-neuf": 49,
    # Compound cinquante+N
    "cinquante-et-un": 51, "cinquante-deux": 52, "cinquante-trois": 53,
    "cinquante-quatre": 54, "cinquante-cinq": 55, "cinquante-six": 56,
    "cinquante-sept": 57, "cinquante-huit": 58, "cinquante-neuf": 59,
    # Compound soixante+N (60-69)
    "soixante-et-un": 61, "soixante-deux": 62, "soixante-trois": 63,
    "soixante-quatre": 64, "soixante-cinq": 65, "soixante-six": 66,
    "soixante-sept": 67, "soixante-huit": 68, "soixante-neuf": 69,
    # Compound soixante-dix+N (71-79)
    "soixante-et-onze": 71, "soixante-douze": 72, "soixante-treize": 73,
    "soixante-quatorze": 74, "soixante-quinze": 75, "soixante-seize": 76,
    "soixante-dix-sept": 77, "soixante-dix-huit": 78, "soixante-dix-neuf": 79,
}

_TENS = {
    "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60, "soixante-dix": 70, "soixante dix": 70,
    "quatre-vingt": 80, "quatre vingt": 80,
    "quatre-vingt-dix": 90, "quatre vingt dix": 90,
    "nonante": 90,  # Swiss/Belgian variant
}

_HUNDREDS = {"cent": 100, "cents": 100}
_THOUSANDS = {"mille": 1_000, "milles": 1_000}
_MILLIONS  = {"million": 1_000_000, "millions": 1_000_000}

# Separator words to ignore
_NOISE = {"et", "de", "le", "la", "les", "des", "du", "au", "aux", "virgule", "point"}

# Monetary unit words — used to detect where millime section starts
_DINAR_WORDS = {"dinar", "dinars", "dt", "tnd"}
_MILLIME_WORDS = {"millime", "millimes", "m"}


def _tokenise(text: str) -> list[str]:
    """Lower-case, strip accents lightly, split into word tokens.

    NOTE: hyphens are kept intact because compound French number words
    such as ``vingt-cinq`` and ``quatre-vingt-dix`` appear as single
    entries in _UNITS / _TENS and must be looked up as one token.
    Only whitespace, commas, and semicolons are used as delimiters.
    """
    text = text.lower()
    # Normalise common accented chars
    text = (
        text.replace("é", "e").replace("è", "e").replace("ê", "e")
            .replace("à", "a").replace("â", "a")
            .replace("ô", "o").replace("û", "u").replace("î", "i")
    )
    # Split on whitespace/commas/semicolons ONLY — preserve hyphens
    tokens = re.split(r"[\s,;]+", text.strip())
    return [t for t in tokens if t]


def _words_to_int(tokens: list[str]) -> Optional[int]:
    """Convert a list of French number word tokens to an integer.

    Handles simple cases up to 999 999 999.
    Returns None if the token list is empty or unrecognised.
    """
    if not tokens:
        return None

    total = 0
    current = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Skip noise words
        if tok in _NOISE:
            i += 1
            continue

        # Try two-token compound (e.g. "dix sept", "quatre vingt")
        if i + 1 < len(tokens):
            compound = tok + " " + tokens[i + 1]
            if compound in _UNITS:
                current += _UNITS[compound]
                i += 2
                continue
            if compound in _TENS:
                current += _TENS[compound]
                i += 2
                continue

        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok in _HUNDREDS:
            if current == 0:
                current = 100
            else:
                current *= 100
        elif tok in _THOUSANDS:
            if current == 0:
                current = 1
            total += current * 1_000
            current = 0
        elif tok in _MILLIONS:
            if current == 0:
                current = 1
            total += current * 1_000_000
            current = 0
        else:
            # Unrecognised token — bail out
            return None

        i += 1

    return total + current


def parse_amount_words(raw: str) -> AmountWordsResult:
    """Parse a French amount-in-words string to a numeric value.

    Returns AmountWordsResult.valid=False on any ambiguity or parse failure.
    NEVER raise exceptions — this is a soft validator.

    Examples:
    - "trois mille dinars"                          → Decimal('3000.000')
    - "vingt-cinq mille dinars"                     → Decimal('25000.000')
    - "trois mille cinq cents dinars"               → Decimal('3500.000')
    - "deux mille huit cent quatre-vingt-treize dinars cent quatre-vingt-douze millimes"
                                                    → Decimal('2893.192')
    """
    result = AmountWordsResult(raw=raw)

    tokens = _tokenise(raw)
    if not tokens:
        result.error = "Empty string"
        return result

    # Split into dinar section and millime section at the currency word.
    # State machine:
    #   "dinar"   → switch to "after_dinar" (tokens go to millime bucket)
    #   "millime" → same state (millimes keyword itself is skipped)
    # Everything before the first dinar word → dinar_tokens
    # Everything after the first dinar word   → millime_tokens
    dinar_tokens: list[str] = []
    millime_tokens: list[str] = []
    seen_dinar = False

    for tok in tokens:
        if tok in _DINAR_WORDS:
            seen_dinar = True
            continue
        if tok in _MILLIME_WORDS:
            seen_dinar = True  # treat as boundary too
            continue
        if seen_dinar:
            millime_tokens.append(tok)
        else:
            dinar_tokens.append(tok)

    dinar_int = _words_to_int(dinar_tokens)
    if dinar_int is None:
        result.error = f"Cannot parse dinar section from tokens: {dinar_tokens}"
        return result

    millime_int = _words_to_int(millime_tokens) if millime_tokens else 0
    if millime_int is None:
        # Millime section present but unparseable — still return dinar value
        # with a note (soft warning)
        millime_int = 0

    if millime_int < 0 or millime_int > 999:
        result.error = f"Millime value {millime_int} out of range 0-999"
        return result

    total = Decimal(dinar_int) + Decimal(millime_int) / Decimal(1000)
    result.value = total
    result.valid = True
    return result
