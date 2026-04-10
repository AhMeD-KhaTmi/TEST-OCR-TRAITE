"""
Phase 3 — RIB parser.

Tunisian RIB structure: BB AAA CCCCCCCCCCCCC KK
  BB  = 2-digit bank code
  AAA = 3-digit branch code
  CCC…= 13-digit account number
  KK  = 2-digit check key (mod 97)

Public API
----------
parse_rib(raw: str) -> RIBResult
verify_rib_key(bank, branch, account, key) -> bool
bank_name_for_code(code: str) -> str | None
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bank code table
# ---------------------------------------------------------------------------

_BANK_CODES_PATH = Path(__file__).parent.parent.parent / "config" / "bank_codes.json"

def _load_bank_codes() -> dict[str, str]:
    try:
        with open(_BANK_CODES_PATH, encoding="utf-8") as fh:
            return json.load(fh).get("bank_codes", {})
    except (OSError, json.JSONDecodeError):
        return {}

_BANK_CODES: dict[str, str] = _load_bank_codes()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RIBResult:
    raw: str                          # original OCR string before cleaning
    digits: str = ""                  # all 20 digits (empty string if parsing failed)
    bank_code: str = ""               # 2-digit bank code
    branch_code: str = ""             # 3-digit branch code
    account_number: str = ""          # 13-digit account number
    key: str = ""                     # 2-digit check key
    key_valid: bool = False           # True when mod-97 check passes
    bank_name: Optional[str] = None   # human-readable bank name (None if unknown)
    valid: bool = False               # True when RIB is fully parseable and key checks out
    error: Optional[str] = None       # description of first failure
    has_question_marks: bool = False  # True when OCR left ? for unreadable positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_rib_string(raw: str) -> str:
    """Remove all non-digit, non-question-mark characters."""
    return re.sub(r"[^\d?]", "", raw)


def verify_rib_key(bank: str, branch: str, account: str, key: str) -> bool:
    """Compute Tunisian RIB check key and compare to supplied key.

    Formula: key = 97 - ((N * 100) mod 97)
    where N is the 18-digit integer formed by concatenating bank + branch + account.
    Returns False if any argument contains non-digit characters.
    """
    combined = bank + branch + account
    if not re.fullmatch(r"\d{18}", combined):
        return False
    if not re.fullmatch(r"\d{2}", key):
        return False
    n = int(combined)
    computed_key = 97 - ((n * 100) % 97)
    # computed_key can be 97 → normalise to 00 (rare edge case)
    if computed_key == 97:
        computed_key = 0
    return computed_key == int(key)


def bank_name_for_code(code: str) -> Optional[str]:
    """Return the bank name for a 2-digit code, or None if unknown."""
    return _BANK_CODES.get(code)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_rib(raw: str) -> RIBResult:
    """Parse a raw OCR string into a structured RIBResult.

    Accepts many dirty formats:
    - "08 006 0110510000870 41"
    - "08006011051000087041"
    - "08-006-0110510000870-41"
    - "08 006 01105?0000870 41"   (OCR question marks preserved)
    - "0B 006 0110510000870 41"   (OCR confusion — B stripped, treated as ?)

    Confidence degrades gracefully: parse as much as possible.
    """
    result = RIBResult(raw=raw)

    cleaned = _clean_rib_string(raw)
    result.has_question_marks = "?" in cleaned

    if len(cleaned) != 20:
        result.error = (
            f"Expected 20 digit/? characters after cleaning, got {len(cleaned)}: '{cleaned}'"
        )
        return result

    result.digits       = cleaned
    result.bank_code    = cleaned[0:2]
    result.branch_code  = cleaned[2:5]
    result.account_number = cleaned[5:18]
    result.key          = cleaned[18:20]

    result.bank_name = bank_name_for_code(result.bank_code)

    # Can only verify key when there are no question marks
    if result.has_question_marks:
        result.valid = False
        result.error = "RIB contains unreadable positions ('?') — key cannot be verified"
        return result

    result.key_valid = verify_rib_key(
        result.bank_code, result.branch_code, result.account_number, result.key
    )
    result.valid = result.key_valid
    if not result.key_valid:
        result.error = (
            f"RIB check key mismatch: extracted key={result.key}, "
            f"computed from bank={result.bank_code} branch={result.branch_code} "
            f"account={result.account_number}"
        )

    return result
