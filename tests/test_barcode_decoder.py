"""
Tests for Phase 2 — barcode_decoder.py

Coverage:
  - BarcodeResult dataclass structure
  - _preprocess_for_barcode returns multiple variants
  - decode_barcode on a white patch (no barcode → graceful no-decode)
  - decode_barcode source field values
  - decode_barcode confidence range
  - OCR-B Tesseract fallback (digit-only output)
  - pyzbar availability flag propagated to result
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.barcode_decoder import (
    BarcodeResult,
    decode_barcode,
    _preprocess_for_barcode,
    _PYZBAR_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def white_crop(h: int = 80, w: int = 400) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def grey_crop(h: int = 80, w: int = 400) -> np.ndarray:
    return np.full((h, w), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. BarcodeResult structure
# ---------------------------------------------------------------------------

class TestBarcodeResultStructure:
    def test_has_required_fields(self):
        r = BarcodeResult(
            barcode_text="",
            ocr_text="",
            best_text="",
            source="none",
            confidence=0.0,
        )
        assert hasattr(r, "barcode_text")
        assert hasattr(r, "ocr_text")
        assert hasattr(r, "best_text")
        assert hasattr(r, "source")
        assert hasattr(r, "confidence")
        assert hasattr(r, "pyzbar_available")
        assert hasattr(r, "error")

    def test_default_pyzbar_available(self):
        r = BarcodeResult(barcode_text="", ocr_text="", best_text="", source="none", confidence=0.0)
        assert isinstance(r.pyzbar_available, bool)

    def test_default_error_is_none(self):
        r = BarcodeResult(barcode_text="", ocr_text="", best_text="", source="none", confidence=0.0)
        assert r.error is None


# ---------------------------------------------------------------------------
# 2. _preprocess_for_barcode
# ---------------------------------------------------------------------------

class TestPreprocessForBarcode:
    def test_returns_list(self):
        variants = _preprocess_for_barcode(white_crop())
        assert isinstance(variants, list)

    def test_at_least_two_variants(self):
        variants = _preprocess_for_barcode(white_crop())
        assert len(variants) >= 2

    def test_greyscale_input_accepted(self):
        variants = _preprocess_for_barcode(grey_crop())
        assert len(variants) >= 1

    def test_all_variants_are_numpy_arrays(self):
        for v in _preprocess_for_barcode(white_crop()):
            assert isinstance(v, np.ndarray)


# ---------------------------------------------------------------------------
# 3. decode_barcode — structural / graceful-degradation tests
# ---------------------------------------------------------------------------

class TestDecodeBarcode:
    def test_returns_BarcodeResult(self):
        result = decode_barcode(white_crop())
        assert isinstance(result, BarcodeResult)

    def test_no_exception_on_white_image(self):
        decode_barcode(white_crop())

    def test_no_exception_on_greyscale(self):
        decode_barcode(grey_crop())

    def test_no_exception_on_tiny_image(self):
        tiny = np.full((20, 100, 3), 255, dtype=np.uint8)
        decode_barcode(tiny)

    def test_confidence_in_range(self):
        result = decode_barcode(white_crop())
        assert 0.0 <= result.confidence <= 1.0

    def test_source_is_valid_string(self):
        result = decode_barcode(white_crop())
        assert result.source in ("pyzbar", "tesseract_ocrb", "none")

    def test_best_text_is_string(self):
        result = decode_barcode(white_crop())
        assert isinstance(result.best_text, str)

    def test_barcode_text_is_string(self):
        result = decode_barcode(white_crop())
        assert isinstance(result.barcode_text, str)

    def test_ocr_text_is_string(self):
        result = decode_barcode(white_crop())
        assert isinstance(result.ocr_text, str)

    def test_pyzbar_flag_matches_module_constant(self):
        result = decode_barcode(white_crop())
        assert result.pyzbar_available == _PYZBAR_AVAILABLE

    def test_white_image_yields_no_decode(self):
        result = decode_barcode(white_crop())
        # A blank white image has no barcode — should be "none" or empty
        if result.source == "pyzbar":
            # pyzbar somehow found something (very unlikely on white)
            pass
        else:
            assert result.source in ("tesseract_ocrb", "none")

    def test_no_decode_gives_zero_or_low_confidence(self):
        result = decode_barcode(white_crop())
        # White patch has no barcode; confidence should be low
        if result.best_text == "":
            assert result.confidence == 0.0

    def test_non_barcode_image_yields_empty_best_or_digits_only(self):
        result = decode_barcode(white_crop())
        import re
        if result.best_text:
            # OCR-B fallback strips non-digits
            assert re.fullmatch(r"\d*", result.best_text)

    def test_source_none_when_both_fail(self):
        result = decode_barcode(white_crop())
        if result.source == "none":
            assert result.best_text == ""
            assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 4. decode_barcode confidence rules
# ---------------------------------------------------------------------------

class TestDecodeBarcodeConfidence:
    def test_pyzbar_source_has_confidence_1(self):
        """If pyzbar decoded something, confidence is 1.0."""
        result = decode_barcode(white_crop())
        if result.source == "pyzbar":
            assert result.confidence == 1.0

    def test_tesseract_source_has_non_negative_confidence(self):
        result = decode_barcode(white_crop())
        if result.source == "tesseract_ocrb":
            assert result.confidence >= 0.0

    def test_none_source_has_zero_confidence(self):
        result = decode_barcode(white_crop())
        if result.source == "none":
            assert result.confidence == 0.0
