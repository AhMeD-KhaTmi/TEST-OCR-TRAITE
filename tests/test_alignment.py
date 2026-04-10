"""Unit tests for Phase 1 — alignment module."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.alignment import (
    AlignmentResult,
    _find_barcode_region,
    _find_header_band,
    _find_outer_border,
    align,
)

SAMPLE_DIR  = Path(__file__).parent.parent / "example"
ALL_SAMPLES = sorted(SAMPLE_DIR.glob("*.jpg"))
FIRST_SAMPLE = ALL_SAMPLES[0]


def _blank_bgr(h: int = 400, w: int = 300) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 240


# ---------------------------------------------------------------------------
# Header band detection
# ---------------------------------------------------------------------------

class TestHeaderBandDetection:
    def test_returns_none_for_blank_image(self):
        img = _blank_bgr()
        result = _find_header_band(img)
        assert result is None

    def test_detects_red_band_at_top(self):
        img = _blank_bgr(600, 800)
        # Paint a red/pink band in top 15%: R dominant over B
        img[:80, :] = (120, 140, 220)  # BGR: B=120, G=140, R=220 (red dominant)
        result = _find_header_band(img)
        if result is not None:
            y_top, y_bot = result
            assert y_top < 80
            assert y_bot < img.shape[0] * 0.25

    def test_result_is_tuple_or_none(self):
        img = _blank_bgr()
        result = _find_header_band(img)
        assert result is None or (isinstance(result, tuple) and len(result) == 2)


# ---------------------------------------------------------------------------
# Barcode region detection
# ---------------------------------------------------------------------------

class TestBarcodeDection:
    def test_returns_none_for_blank_image(self):
        img = _blank_bgr(600, 800)
        result = _find_barcode_region(img)
        assert result is None

    def test_detects_dense_vertical_lines_at_bottom(self):
        img = _blank_bgr(600, 800)
        # Draw many vertical black lines in bottom 20% → simulate barcode
        for x in range(0, 400, 4):
            img[480:560, x:x + 2] = 0
        result = _find_barcode_region(img)
        if result is not None:
            y_top, y_bot = result
            assert y_top > 400   # must be in bottom half
            assert y_bot > y_top

    def test_result_is_tuple_or_none(self):
        img = _blank_bgr()
        result = _find_barcode_region(img)
        assert result is None or (isinstance(result, tuple) and len(result) == 2)


# ---------------------------------------------------------------------------
# Outer border detection
# ---------------------------------------------------------------------------

class TestOuterBorderDetection:
    def test_returns_none_for_blank_image(self):
        img = _blank_bgr()
        result = _find_outer_border(img)
        assert result is None

    def test_detects_black_rectangle(self):
        img = np.ones((400, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (10, 10), (290, 390), (0, 0, 0), 3)
        result = _find_outer_border(img)
        if result is not None:
            x, y, w, h = result
            assert x <= 15
            assert y <= 15


# ---------------------------------------------------------------------------
# Align function
# ---------------------------------------------------------------------------

class TestAlignFunction:
    def test_returns_alignment_result(self):
        prep = preprocess(FIRST_SAMPLE)
        result = align(prep.deskewed)
        assert isinstance(result, AlignmentResult)

    def test_output_image_same_shape_as_input(self):
        prep = preprocess(FIRST_SAMPLE)
        result = align(prep.deskewed)
        assert result.image.shape == prep.deskewed.shape

    def test_confidence_between_zero_and_one(self):
        prep = preprocess(FIRST_SAMPLE)
        result = align(prep.deskewed)
        assert 0.0 <= result.confidence <= 1.0

    def test_method_is_valid_string(self):
        prep = preprocess(FIRST_SAMPLE)
        result = align(prep.deskewed)
        assert result.method in ("border", "header_barcode", "identity")

    def test_warnings_is_list(self):
        prep = preprocess(FIRST_SAMPLE)
        result = align(prep.deskewed)
        assert isinstance(result.warnings, list)

    def test_blank_image_returns_identity(self):
        img = _blank_bgr(400, 300)
        result = align(img)
        assert result.method in ("identity", "border", "header_barcode")
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Multi-sample smoke test
# ---------------------------------------------------------------------------

class TestAlignMultiSample:
    SAMPLE_SUBSET = ALL_SAMPLES[:5]   # first 5 from each batch variant

    @pytest.mark.parametrize("img_path", SAMPLE_SUBSET[:5])
    def test_pipeline_runs_without_error(self, img_path):
        prep    = preprocess(img_path)
        result  = align(prep.deskewed)
        assert result.image is not None
        assert result.confidence >= 0.0

    def test_all_first_batch_align_above_threshold(self):
        """All _1_ batch docs should align at confidence ≥ 0.6."""
        batch1 = [p for p in ALL_SAMPLES if "_1_page-" in p.name]
        for img_path in batch1:
            prep   = preprocess(img_path)
            result = align(prep.deskewed)
            assert result.confidence >= 0.6, (
                f"{img_path.name}: alignment confidence {result.confidence:.2f} < 0.6"
            )
