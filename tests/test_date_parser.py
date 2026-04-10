"""Tests for date_parser.py — Phase 3."""
from datetime import date
import pytest

from src.ocr_pipeline.date_parser import (
    parse_date,
    dates_equal,
    date_a_after_b,
    date_a_on_or_after_b,
    DateResult,
)


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_slash_separator(self):
        r = parse_date("15/06/2025")
        assert r.valid
        assert r.value == date(2025, 6, 15)
        assert r.normalised == "15/06/2025"

    def test_dash_separator(self):
        r = parse_date("15-06-2025")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_dot_separator(self):
        r = parse_date("15.06.2025")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_space_separator(self):
        r = parse_date("15 06 2025")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_no_zero_padding(self):
        r = parse_date("1/6/2025")
        assert r.valid
        assert r.value == date(2025, 6, 1)

    def test_iso_order_yyyy_mm_dd(self):
        r = parse_date("2025/06/15")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_two_digit_year(self):
        r = parse_date("15/06/25")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_ocr_o_to_zero(self):
        r = parse_date("15/O6/2025")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_ocr_l_to_one(self):
        r = parse_date("l5/06/2025")
        assert r.valid
        assert r.value == date(2025, 6, 15)

    def test_invalid_month(self):
        r = parse_date("15/13/2025")
        assert not r.valid
        assert "month" in r.error.lower()

    def test_invalid_day(self):
        r = parse_date("32/06/2025")
        assert not r.valid
        assert "day" in r.error.lower()

    def test_invalid_calendar_date(self):
        r = parse_date("30/02/2025")
        assert not r.valid  # Feb 30 doesn't exist

    def test_year_out_of_range_low(self):
        r = parse_date("15/06/1990")
        assert not r.valid
        assert "year" in r.error.lower()

    def test_year_out_of_range_high(self):
        r = parse_date("15/06/2040")
        assert not r.valid

    def test_empty_string(self):
        r = parse_date("")
        assert not r.valid

    def test_too_few_parts(self):
        r = parse_date("15/06")
        assert not r.valid

    def test_too_many_parts(self):
        r = parse_date("15/06/2025/extra")
        assert not r.valid

    def test_raw_preserved(self):
        raw = "  15/06/2025  "
        r = parse_date(raw)
        assert r.raw == raw

    def test_normalised_format(self):
        r = parse_date("5/1/2025")
        assert r.normalised == "05/01/2025"

    def test_non_digit_after_fix(self):
        r = parse_date("XX/06/2025")
        assert not r.valid


# ---------------------------------------------------------------------------
# Date comparison helpers
# ---------------------------------------------------------------------------

class TestDatesEqual:
    def test_same_dates_equal(self):
        a = parse_date("15/06/2025")
        b = parse_date("15/06/2025")
        assert dates_equal(a, b)

    def test_different_dates_not_equal(self):
        a = parse_date("15/06/2025")
        b = parse_date("16/06/2025")
        assert not dates_equal(a, b)

    def test_one_invalid(self):
        a = parse_date("15/06/2025")
        b = parse_date("")
        assert not dates_equal(a, b)


class TestDateAfter:
    def test_a_after_b(self):
        a = parse_date("16/06/2025")
        b = parse_date("15/06/2025")
        assert date_a_after_b(a, b)

    def test_a_not_after_b_same(self):
        a = parse_date("15/06/2025")
        b = parse_date("15/06/2025")
        assert not date_a_after_b(a, b)

    def test_a_not_after_b_earlier(self):
        a = parse_date("14/06/2025")
        b = parse_date("15/06/2025")
        assert not date_a_after_b(a, b)

    def test_on_or_after_same(self):
        a = parse_date("15/06/2025")
        b = parse_date("15/06/2025")
        assert date_a_on_or_after_b(a, b)

    def test_on_or_after_earlier(self):
        a = parse_date("14/06/2025")
        b = parse_date("15/06/2025")
        assert not date_a_on_or_after_b(a, b)
