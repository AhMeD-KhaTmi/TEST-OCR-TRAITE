"""Tests for amount_parser.py — Phase 3."""
from decimal import Decimal
import pytest

from src.ocr_pipeline.amount_parser import (
    parse_amount_numeric,
    parse_amount_words,
    amounts_equal,
    AmountResult,
    AmountWordsResult,
)


# ---------------------------------------------------------------------------
# parse_amount_numeric
# ---------------------------------------------------------------------------

class TestParseAmountNumeric:
    def test_simple_comma_decimal(self):
        r = parse_amount_numeric("3000,000")
        assert r.valid
        assert r.value == Decimal("3000.000")

    def test_hash_delimiters_stripped(self):
        r = parse_amount_numeric("#5.000.000#")
        assert r.valid
        assert r.value == Decimal("5000.000")

    def test_multiple_commas_last_is_decimal(self):
        r = parse_amount_numeric("2,893,192")
        assert r.valid
        assert r.value == Decimal("2893.192")

    def test_space_thousands_comma_decimal(self):
        r = parse_amount_numeric("25 000,000 DT")
        assert r.valid
        assert r.value == Decimal("25000.000")

    def test_hash_space_dot_decimal(self):
        r = parse_amount_numeric("# 3 000.000#")
        assert r.valid
        assert r.value == Decimal("3000.000")

    def test_simple_500_comma(self):
        r = parse_amount_numeric("500,000")
        assert r.valid
        assert r.value == Decimal("500.000")

    def test_dt_suffix_stripped(self):
        r = parse_amount_numeric("1500.000 DT")
        assert r.valid
        assert r.value == Decimal("1500.000")

    def test_tnd_suffix_stripped(self):
        r = parse_amount_numeric("750.500 TND")
        assert r.valid
        assert r.value == Decimal("750.500")

    def test_integer_only_no_decimal(self):
        r = parse_amount_numeric("3000")
        assert r.valid
        assert r.value == Decimal("3000")

    def test_question_marks_preserved(self):
        r = parse_amount_numeric("3?00,000")
        assert r.has_question_marks
        assert not r.valid
        assert r.error is not None

    def test_empty_string_invalid(self):
        r = parse_amount_numeric("")
        assert not r.valid
        assert r.error is not None

    def test_raw_preserved(self):
        raw = "  3000,000  "
        r = parse_amount_numeric(raw)
        assert r.raw == raw

    def test_zero_value_valid(self):
        r = parse_amount_numeric("0,000")
        assert r.valid
        assert r.value == Decimal("0.000")

    def test_negative_value_invalid(self):
        # Negative amounts don't exist on a Lettre de Change
        r = parse_amount_numeric("-100,000")
        assert not r.valid

    def test_million_value(self):
        r = parse_amount_numeric("1.000.000,000")
        assert r.valid
        assert r.value == Decimal("1000000.000")

    def test_only_hash_invalid(self):
        r = parse_amount_numeric("##")
        assert not r.valid


# ---------------------------------------------------------------------------
# amounts_equal
# ---------------------------------------------------------------------------

class TestAmountsEqual:
    def test_equal_values(self):
        a = parse_amount_numeric("3000,000")
        b = parse_amount_numeric("3 000.000")
        assert amounts_equal(a, b)

    def test_unequal_values(self):
        a = parse_amount_numeric("3000,000")
        b = parse_amount_numeric("3001,000")
        assert not amounts_equal(a, b)

    def test_one_invalid(self):
        a = parse_amount_numeric("3000,000")
        b = parse_amount_numeric("")
        assert not amounts_equal(a, b)

    def test_tolerance_boundary(self):
        a = parse_amount_numeric("3000,000")
        b = parse_amount_numeric("3000,001")
        # tolerance default is 0.001 — exactly at the boundary → equal
        assert amounts_equal(a, b, tolerance=Decimal("0.001"))

    def test_tolerance_exceeded(self):
        a = parse_amount_numeric("3000,000")
        b = parse_amount_numeric("3000,002")
        assert not amounts_equal(a, b, tolerance=Decimal("0.001"))


# ---------------------------------------------------------------------------
# parse_amount_words
# ---------------------------------------------------------------------------

class TestParseAmountWords:
    def test_trois_mille_dinars(self):
        r = parse_amount_words("trois mille dinars")
        assert r.valid
        assert r.value == Decimal("3000")

    def test_vingt_cinq_mille_dinars(self):
        r = parse_amount_words("vingt-cinq mille dinars")
        assert r.valid
        assert r.value == Decimal("25000")

    def test_with_millimes(self):
        r = parse_amount_words(
            "deux mille huit cent quatre-vingt-treize dinars "
            "cent quatre-vingt-douze millimes"
        )
        assert r.valid
        assert r.value == Decimal("2893.192")

    def test_cinq_cents_dinars(self):
        r = parse_amount_words("cinq cents dinars")
        assert r.valid
        assert r.value == Decimal("500")

    def test_un_dinar(self):
        r = parse_amount_words("un dinar")
        assert r.valid
        assert r.value == Decimal("1")

    def test_empty_string_invalid(self):
        r = parse_amount_words("")
        assert not r.valid
        assert r.error is not None

    def test_unknown_token_invalid(self):
        r = parse_amount_words("blah blah")
        assert not r.valid

    def test_raw_preserved(self):
        raw = "trois mille dinars"
        r = parse_amount_words(raw)
        assert r.raw == raw

    def test_mille_alone(self):
        r = parse_amount_words("mille dinars")
        assert r.valid
        assert r.value == Decimal("1000")

    def test_accent_normalisation(self):
        # "zéro" with accent should parse as 0
        r = parse_amount_words("zéro dinars")
        assert r.valid
        assert r.value == Decimal("0")
