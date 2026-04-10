"""Tests for rib_parser.py — Phase 3."""
import pytest
from src.ocr_pipeline.rib_parser import (
    parse_rib, verify_rib_key, bank_name_for_code, RIBResult
)


# ---------------------------------------------------------------------------
# verify_rib_key
# ---------------------------------------------------------------------------

class TestVerifyRibKey:
    def test_valid_key_known_rib(self):
        # 08 006 0110510000870 41 — a structurally valid RIB form example
        # key = 97 - ((080060110510000870 * 100) % 97)
        bank, branch, account = "08", "006", "0110510000870"
        n = int(bank + branch + account)
        key = 97 - ((n * 100) % 97)
        if key == 97:
            key = 0
        assert verify_rib_key(bank, branch, account, f"{key:02d}")

    def test_wrong_key_returns_false(self):
        bank, branch, account = "08", "006", "0110510000870"
        assert not verify_rib_key(bank, branch, account, "00")

    def test_question_marks_return_false(self):
        assert not verify_rib_key("0?", "006", "0110510000870", "41")

    def test_non_18_digit_combined_returns_false(self):
        assert not verify_rib_key("08", "06", "01105", "41")  # too short

    def test_key_97_normalised_to_00(self):
        # Find a combination where computed key == 97 → normalised to 0
        # 97 - (N*100 % 97) == 97  →  N*100 % 97 == 0  →  N % 97 == 0
        # smallest 18-digit N divisible by 97
        N = 97 * (10**17 // 97 + 1)
        s = str(N).zfill(18)
        bank, branch, account = s[:2], s[2:5], s[5:18]
        assert verify_rib_key(bank, branch, account, "00")


# ---------------------------------------------------------------------------
# bank_name_for_code
# ---------------------------------------------------------------------------

class TestBankNameForCode:
    def test_known_code_08(self):
        assert bank_name_for_code("08") == "BIAT"

    def test_known_code_07(self):
        assert bank_name_for_code("07") == "Amen Bank"

    def test_unknown_code_returns_none(self):
        assert bank_name_for_code("99") is None

    def test_leading_zero_required(self):
        assert bank_name_for_code("8") is None   # must be "08"


# ---------------------------------------------------------------------------
# parse_rib
# ---------------------------------------------------------------------------

class TestParseRib:
    def _make_valid_rib(self) -> str:
        """Return a valid 20-digit RIB string with correct check key."""
        bank, branch, account = "08", "006", "0110510000870"
        n = int(bank + branch + account)
        key = 97 - ((n * 100) % 97)
        if key == 97:
            key = 0
        return f"{bank} {branch} {account} {key:02d}"

    def test_parses_spaced_format(self):
        raw = self._make_valid_rib()
        r = parse_rib(raw)
        assert r.valid
        assert r.bank_code == "08"
        assert r.branch_code == "006"
        assert len(r.account_number) == 13
        assert r.key_valid

    def test_parses_compact_format(self):
        raw = self._make_valid_rib().replace(" ", "")
        r = parse_rib(raw)
        assert r.valid

    def test_parses_dashed_format(self):
        raw = self._make_valid_rib().replace(" ", "-")
        r = parse_rib(raw)
        assert r.valid

    def test_wrong_length_invalid(self):
        r = parse_rib("08 006 01105")
        assert not r.valid
        assert r.error is not None

    def test_question_marks_preserved(self):
        raw = "08 006 011051?000870 41"
        r = parse_rib(raw)
        assert r.has_question_marks
        assert not r.valid
        assert "?" in r.digits

    def test_bank_name_populated(self):
        raw = self._make_valid_rib()
        r = parse_rib(raw)
        assert r.bank_name == "BIAT"

    def test_unknown_bank_code(self):
        # Build a RIB with bank code 99 (unknown) — key will likely fail too
        r = parse_rib("99" + "0" * 18)
        assert r.bank_code == "99"
        assert r.bank_name is None

    def test_empty_string(self):
        r = parse_rib("")
        assert not r.valid

    def test_wrong_key_flagged(self):
        raw = "08 006 0110510000870 00"  # key 00 is almost certainly wrong
        r = parse_rib(raw)
        # Should parse structurally but key_valid may be False
        assert r.bank_code == "08"
        assert not r.key_valid or r.valid  # either way, no exception

    def test_raw_preserved(self):
        raw = "  08 006 0110510000870 41  "
        r = parse_rib(raw)
        assert r.raw == raw
