"""
Tests for Phase 2 — tesseract_ocr.py

Coverage:
  - TESSERACT_AVAILABLE flag
  - _apply_digit_confusion
  - _apply_date_confusion
  - _normalize_whitespace
  - run_tesseract (graceful degradation when Tess unavailable)
  - ocr_digits / ocr_amounts / ocr_date / ocr_text
  - ocr_from_roi with different whitelists
  - Result type and field completeness
"""

from __future__ import annotations

import re
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from ocr_pipeline.tesseract_ocr import (
    TESSERACT_AVAILABLE,
    TesseractResult,
    _apply_digit_confusion,
    _apply_date_confusion,
    _normalize_whitespace,
    run_tesseract,
    ocr_digits,
    ocr_amounts,
    ocr_date,
    ocr_text,
    ocr_from_roi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def white_patch(h: int = 80, w: int = 200) -> np.ndarray:
    """Create a plain white BGRimage — Tesseract returns empty string, not exception."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def grey_patch(h: int = 80, w: int = 200) -> np.ndarray:
    return np.full((h, w), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_tesseract_available_is_bool(self):
        assert isinstance(TESSERACT_AVAILABLE, bool)

    def test_availability_logged(self, capsys):
        # Just ensure importing doesn't crash; availability may be True or False.
        assert TESSERACT_AVAILABLE in (True, False)


# ---------------------------------------------------------------------------
# 2. Digit confusion correction
# ---------------------------------------------------------------------------

class TestDigitConfusion:
    def test_O_becomes_zero(self):
        assert _apply_digit_confusion("OO8") == "008"

    def test_l_becomes_one(self):
        assert _apply_digit_confusion("l23") == "123"

    def test_I_becomes_one(self):
        assert _apply_digit_confusion("I00") == "100"

    def test_S_becomes_five(self):
        assert _apply_digit_confusion("S12") == "512"

    def test_B_becomes_eight(self):
        assert _apply_digit_confusion("B00") == "800"

    def test_Z_becomes_two(self):
        assert _apply_digit_confusion("Z0") == "20"

    def test_already_correct_unchanged(self):
        assert _apply_digit_confusion("12345678900") == "12345678900"

    def test_empty_string(self):
        assert _apply_digit_confusion("") == ""

    def test_mixed_string(self):
        result = _apply_digit_confusion("O8 OO6 Ol10510000870 41")
        assert "O" not in result  # all O → 0


# ---------------------------------------------------------------------------
# 3. Date confusion correction
# ---------------------------------------------------------------------------

class TestDateConfusion:
    def test_O_to_zero(self):
        assert _apply_date_confusion("O5/O6/2O25") == "05/06/2025"

    def test_I_to_one(self):
        assert _apply_date_confusion("I5/06/2025") == "15/06/2025"

    def test_separators_preserved(self):
        result = _apply_date_confusion("15-06-2025")
        assert "-" in result

    def test_plain_digits_unchanged(self):
        assert _apply_date_confusion("15/06/2025") == "15/06/2025"


# ---------------------------------------------------------------------------
# 4. Whitespace normalization
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_strips_leading_trailing(self):
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert _normalize_whitespace("hello   world") == "hello world"

    def test_removes_newlines(self):
        assert _normalize_whitespace("hello\nworld") == "hello world"

    def test_empty_string(self):
        assert _normalize_whitespace("") == ""

    def test_only_whitespace(self):
        assert _normalize_whitespace("   \n\t  ") == ""


# ---------------------------------------------------------------------------
# 5. run_tesseract — structural tests
# ---------------------------------------------------------------------------

class TestRunTesseract:
    def test_returns_TesseractResult(self):
        crop = white_patch()
        result = run_tesseract(crop)
        assert isinstance(result, TesseractResult)

    def test_result_has_all_fields(self):
        result = run_tesseract(white_patch())
        assert hasattr(result, "text")
        assert hasattr(result, "raw_text")
        assert hasattr(result, "confidence")
        assert hasattr(result, "engine_available")
        assert hasattr(result, "error")

    def test_engine_available_matches_flag(self):
        result = run_tesseract(white_patch())
        assert result.engine_available == TESSERACT_AVAILABLE

    def test_no_exception_on_white_image(self):
        run_tesseract(white_patch())  # must not raise

    def test_no_exception_on_greyscale_image(self):
        run_tesseract(grey_patch())  # must not raise

    def test_confidence_in_range(self):
        result = run_tesseract(white_patch())
        assert 0.0 <= result.confidence <= 1.0

    def test_text_is_string(self):
        result = run_tesseract(white_patch())
        assert isinstance(result.text, str)

    def test_no_exception_on_tiny_image(self):
        tiny = np.full((10, 30, 3), 255, dtype=np.uint8)
        run_tesseract(tiny)


# ---------------------------------------------------------------------------
# 6. Specialised OCR entry points — structural + basic contract
# ---------------------------------------------------------------------------

class TestOcrDigits:
    def test_returns_TesseractResult(self):
        assert isinstance(ocr_digits(white_patch()), TesseractResult)

    def test_output_is_digits_only(self):
        result = ocr_digits(white_patch())
        # Any returned text should be digits only
        assert re.fullmatch(r"\d*", result.text), f"Non-digit chars in: {result.text!r}"

    def test_no_letters_in_output(self):
        result = ocr_digits(white_patch())
        assert not re.search(r"[a-zA-Z]", result.text)


class TestOcrAmounts:
    def test_returns_TesseractResult(self):
        assert isinstance(ocr_amounts(white_patch()), TesseractResult)

    def test_only_digits_commas_dots(self):
        result = ocr_amounts(white_patch())
        assert re.fullmatch(r"[0-9,.]*", result.text), f"Bad chars: {result.text!r}"


class TestOcrDate:
    def test_returns_TesseractResult(self):
        assert isinstance(ocr_date(white_patch()), TesseractResult)

    def test_no_exception(self):
        ocr_date(white_patch())


class TestOcrText:
    def test_returns_TesseractResult(self):
        assert isinstance(ocr_text(white_patch()), TesseractResult)

    def test_no_exception(self):
        ocr_text(white_patch())


# ---------------------------------------------------------------------------
# 7. ocr_from_roi
# ---------------------------------------------------------------------------

class TestOcrFromRoi:
    def test_digit_whitelist_config(self):
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        result = ocr_from_roi(white_patch(), config)
        assert isinstance(result, TesseractResult)
        assert re.fullmatch(r"\d*", result.text)

    def test_amount_whitelist_config(self):
        config = "--psm 7 -c tessedit_char_whitelist=0123456789,."
        result = ocr_from_roi(white_patch(), config)
        assert re.fullmatch(r"[0-9,.]*", result.text)

    def test_plain_psm7_config(self):
        config = "--psm 7"
        result = ocr_from_roi(white_patch(), config)
        assert isinstance(result, TesseractResult)

    def test_no_exception_on_various_configs(self):
        for config in ["--psm 6", "--psm 7", "--psm 8", "--psm 13"]:
            ocr_from_roi(white_patch(), config)
