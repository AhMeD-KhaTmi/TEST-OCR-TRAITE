"""
Phase 3 — Name / address parser.

Handles company names and personal names as extracted from Tunisian Lettre de Change
fields: Tireur (R07), Bénéficiaire (R08), Nom et adresse du Tiré (R15).

Rules (per plan section 3.5):
- Minimal transformation only — do NOT correct spelling
- Trim whitespace, collapse runs of spaces
- Uppercase everything
- Detect and annotate company names (contains legal suffix) vs personal names
- Strip [STAMP] markers left by Qwen when stamp occlusion is detected

Public API
----------
parse_name(raw: str) -> NameResult
names_match(a: NameResult, b: NameResult, threshold: float) -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Company legal suffix indicators
# ---------------------------------------------------------------------------

_COMPANY_SUFFIXES = re.compile(
    r"\b(SARL|SA|SUARL|STE|SNC|SC|SEP|GIE|ETS|ETS\.|CIE|LLC|SPA|NIF|MF|"
    r"SOCIETE|ENTREPRISE|GROUPE|CORP|CO\.|COMPANY)\b",
    re.IGNORECASE,
)

# Marker inserted by Qwen when a stamp blocks the field
_STAMP_MARKER = re.compile(r"\[STAMP\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NameResult:
    raw: str                          # original OCR string
    value: str = ""                   # cleaned, uppercased name/address text
    is_company: bool = False          # True when legal suffix detected
    has_stamp_occlusion: bool = False # True when '[STAMP]' marker was present
    valid: bool = False               # True when non-empty result produced
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_whitespace(s: str) -> str:
    """Collapse multiple whitespace chars (including newlines) to a single space."""
    return re.sub(r"\s+", " ", s).strip()


def _strip_common_ocr_noise(s: str) -> str:
    """Remove stray OCR characters unlikely to be part of a name."""
    # Remove lone punctuation runs that are clearly OCR noise (not part of abbreviation)
    s = re.sub(r"(?<!\w)[|\\]{1,3}(?!\w)", " ", s)
    return s


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_name(raw: str) -> NameResult:
    """Parse a raw OCR name/address string into a NameResult.

    Accepts multi-line strings (newlines treated as spaces).
    Does NOT attempt spelling correction.
    """
    result = NameResult(raw=raw)

    # Detect stamp occlusion marker before any other processing
    result.has_stamp_occlusion = bool(_STAMP_MARKER.search(raw))
    # Remove the [STAMP] marker itself before further processing
    cleaned = _STAMP_MARKER.sub(" ", raw)

    cleaned = _strip_common_ocr_noise(cleaned)
    cleaned = _normalise_whitespace(cleaned)
    cleaned = cleaned.upper()

    if not cleaned:
        result.error = "Empty string after cleaning"
        return result

    result.value = cleaned
    result.is_company = bool(_COMPANY_SUFFIXES.search(cleaned))
    result.valid = True
    return result


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def names_match(a: NameResult, b: NameResult, threshold: float = 0.8) -> bool:
    """Return True if the two name values are 'close enough'.

    Uses a simple normalised character overlap (Jaccard on word sets) for
    robustness against OCR differences and abbreviations.
    Does NOT raise exceptions.
    """
    if not a.valid or not b.valid:
        return False

    words_a = set(re.findall(r"\w+", a.value))
    words_b = set(re.findall(r"\w+", b.value))

    if not words_a and not words_b:
        return True
    if not words_a or not words_b:
        return False

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return (intersection / union) >= threshold
