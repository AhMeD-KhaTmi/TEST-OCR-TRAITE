"""
Phase 3 — Date parser.

Parses dates extracted from Tunisian Lettre de Change documents.

Expected format: DD/MM/YYYY  (Tunisian standard)
Accepted alternative separators: - . (space)

Public API
----------
parse_date(raw: str) -> DateResult
dates_equal(a: DateResult, b: DateResult) -> bool
date_a_after_b(a: DateResult, b: DateResult) -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reasonable year range for Tunisian bills of exchange in service today
_YEAR_MIN = 2000
_YEAR_MAX = 2035


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DateResult:
    raw: str                      # original OCR string
    value: Optional[date] = None  # parsed date object (None if parsing failed)
    normalised: str = ""          # canonical DD/MM/YYYY string (empty if failed)
    valid: bool = False           # True when date is parseable and within bounds
    error: Optional[str] = None   # description of first failure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Splits on /, -, ., spaces, or runs of non-digits
_SEP = re.compile(r"[\s/\-\.]+")

def _split_parts(raw: str) -> list[str]:
    """Split a raw date string into numeric parts, ignoring separators."""
    parts = _SEP.split(raw.strip())
    return [p for p in parts if p]  # drop empty strings


def _ocr_fix(s: str) -> str:
    """Fix common OCR char confusions in digit strings."""
    return (
        s.replace("O", "0").replace("o", "0")
         .replace("l", "1").replace("I", "1")
         .replace("S", "5").replace("Z", "2")
         .replace("B", "8").replace("G", "6")
    )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_date(raw: str) -> DateResult:
    """Parse a raw OCR date string into a DateResult.

    Handles:
    - "15/06/2025"
    - "15-06-2025"
    - "15.06.2025"
    - "15 06 2025"
    - "1/6/2025"        (no zero-padding)
    - "2025/06/15"      (ISO order — rare but observed)
    - "15/06/25"        (2-digit year)
    - OCR noise: '0' vs 'O', '1' vs 'l', '5' vs 'S'
    """
    result = DateResult(raw=raw)

    parts = _split_parts(raw)
    if len(parts) != 3:
        result.error = f"Expected 3 date parts, got {len(parts)} in '{raw}'"
        return result

    fixed = [_ocr_fix(p) for p in parts]

    if not all(p.isdigit() for p in fixed):
        result.error = f"Non-digit characters remain after OCR fix: {fixed}"
        return result

    nums = [int(p) for p in fixed]

    # Detect ISO order (YYYY/MM/DD): first part looks like a 4-digit year
    if len(fixed[0]) == 4 or nums[0] > 1000:
        year, month, day = nums[0], nums[1], nums[2]
    else:
        day, month, year = nums[0], nums[1], nums[2]

    # 2-digit year expansion
    if year < 100:
        year += 2000

    # Bounds check
    if not (1 <= month <= 12):
        result.error = f"Invalid month {month} in '{raw}'"
        return result
    if not (1 <= day <= 31):
        result.error = f"Invalid day {day} in '{raw}'"
        return result
    if not (_YEAR_MIN <= year <= _YEAR_MAX):
        result.error = f"Year {year} outside expected range {_YEAR_MIN}-{_YEAR_MAX} in '{raw}'"
        return result

    try:
        parsed = date(year, month, day)
    except ValueError as exc:
        result.error = f"Invalid date: {exc}"
        return result

    result.value = parsed
    result.normalised = parsed.strftime("%d/%m/%Y")
    result.valid = True
    return result


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def dates_equal(a: DateResult, b: DateResult) -> bool:
    """Return True if both DateResults represent the same calendar date."""
    if not a.valid or not b.valid:
        return False
    return a.value == b.value


def date_a_after_b(a: DateResult, b: DateResult) -> bool:
    """Return True if date a is strictly after date b."""
    if not a.valid or not b.valid:
        return False
    return a.value > b.value


def date_a_on_or_after_b(a: DateResult, b: DateResult) -> bool:
    """Return True if date a is on or after date b."""
    if not a.valid or not b.valid:
        return False
    return a.value >= b.value
