"""Tests for name_parser.py — Phase 3."""
import pytest

from src.ocr_pipeline.name_parser import parse_name, names_match, NameResult


# ---------------------------------------------------------------------------
# parse_name
# ---------------------------------------------------------------------------

class TestParseName:
    def test_simple_name_uppercase(self):
        r = parse_name("ahmed ben ali")
        assert r.valid
        assert r.value == "AHMED BEN ALI"

    def test_already_uppercase(self):
        r = parse_name("STE DELTA SARL")
        assert r.valid
        assert r.value == "STE DELTA SARL"

    def test_mixed_case(self):
        r = parse_name("Société Moderne")
        assert r.valid

    def test_collapses_multiple_spaces(self):
        r = parse_name("AHMED   BEN   ALI")
        assert r.value == "AHMED BEN ALI"

    def test_strips_leading_trailing_whitespace(self):
        r = parse_name("  Ahmed Ben Ali  ")
        assert r.value == "AHMED BEN ALI"

    def test_newline_treated_as_space(self):
        r = parse_name("AHMED\nBEN\nALI")
        assert r.value == "AHMED BEN ALI"

    def test_company_sarl_detected(self):
        r = parse_name("STE DELTA ALGÉRIE SARL")
        assert r.is_company

    def test_company_sa_detected(self):
        r = parse_name("BIAT SA")
        assert r.is_company

    def test_personal_name_not_company(self):
        r = parse_name("Ahmed Ben Ali")
        assert not r.is_company

    def test_stamp_marker_detected(self):
        r = parse_name("STE EXAMPLE [STAMP] SARL")
        assert r.has_stamp_occlusion

    def test_stamp_marker_removed_from_value(self):
        r = parse_name("STE EXAMPLE [STAMP] SARL")
        assert "[STAMP]" not in r.value
        assert "STE EXAMPLE" in r.value

    def test_empty_string_invalid(self):
        r = parse_name("")
        assert not r.valid
        assert r.error is not None

    def test_only_spaces_invalid(self):
        r = parse_name("   ")
        assert not r.valid

    def test_raw_preserved(self):
        raw = "Ahmed Ben Ali"
        r = parse_name(raw)
        assert r.raw == raw

    def test_ocr_noise_pipe_stripped(self):
        r = parse_name("ABD| ALLAH")
        # Pipe stripped → "ABD ALLAH"
        assert "|" not in r.value

    def test_ets_suffix_detected(self):
        r = parse_name("ETS SLIM FRERES")
        assert r.is_company

    def test_groupe_detected(self):
        r = parse_name("GROUPE POULINA")
        assert r.is_company


# ---------------------------------------------------------------------------
# names_match
# ---------------------------------------------------------------------------

class TestNamesMatch:
    def test_identical_names_match(self):
        a = parse_name("STE DELTA SARL")
        b = parse_name("STE DELTA SARL")
        assert names_match(a, b)

    def test_slightly_different_match(self):
        a = parse_name("STE DELTA ALGERIE SARL")
        b = parse_name("STE DELTA SARL")
        # 3 words out of 5 in union → low overlap; should NOT match at 0.8
        assert not names_match(a, b, threshold=0.8)

    def test_completely_different_no_match(self):
        a = parse_name("AHMED BEN ALI")
        b = parse_name("FATMA BOUGHDIRI")
        assert not names_match(a, b)

    def test_one_invalid_no_match(self):
        a = parse_name("AHMED BEN ALI")
        b = parse_name("")
        assert not names_match(a, b)

    def test_same_words_different_order_match(self):
        a = parse_name("ALI AHMED")
        b = parse_name("AHMED ALI")
        # both have same 2 words → Jaccard = 1.0 → match
        assert names_match(a, b)
