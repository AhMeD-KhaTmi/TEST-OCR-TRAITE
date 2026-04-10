"""Tests for stamp_preprocessor.py — Phase 4."""
import numpy as np
import pytest

from src.ocr_pipeline.stamp_preprocessor import (
    CropVariants,
    generate_crop_variants,
    select_best_text,
    _make_high_contrast,
    _make_stamp_suppressed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colour_image(w=120, h=80) -> np.ndarray:
    """Return a simple BGR test image (white background)."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Add some fake black text pixels
    img[20:40, 10:80] = (50, 50, 50)
    return img


def _stamp_blue_image(w=120, h=80) -> np.ndarray:
    """Return image with a blue stamp blob (HSV ~hue 110°, sat 180, val 160)."""
    import cv2
    img = _colour_image(w, h)
    # Draw a filled blue circle simulating stamp ink
    cv2.circle(img, (60, 40), 25, (160, 90, 20), thickness=-1)
    return img


# ---------------------------------------------------------------------------
# CropVariants
# ---------------------------------------------------------------------------

class TestCropVariants:
    def test_all_variants_original_only(self):
        img = _colour_image()
        cv = CropVariants(roi_id="R05", original=img)
        assert len(cv.all_variants) == 1

    def test_all_variants_with_high_contrast(self):
        img = _colour_image()
        hc = _make_high_contrast(img)
        cv = CropVariants(roi_id="R05", original=img, high_contrast=hc)
        assert len(cv.all_variants) == 2

    def test_all_variants_all_three(self):
        img = _colour_image()
        hc = _make_high_contrast(img)
        ss = _make_stamp_suppressed(img)
        cv = CropVariants(roi_id="R05", original=img, high_contrast=hc, stamp_suppressed=ss)
        assert len(cv.all_variants) == 3

    def test_original_is_first(self):
        img = _colour_image()
        hc = _make_high_contrast(img)
        cv = CropVariants(roi_id="R05", original=img, high_contrast=hc)
        assert cv.all_variants[0] is img


# ---------------------------------------------------------------------------
# _make_high_contrast
# ---------------------------------------------------------------------------

class TestMakeHighContrast:
    def test_returns_ndarray(self):
        img = _colour_image()
        result = _make_high_contrast(img)
        assert isinstance(result, np.ndarray)

    def test_output_is_single_channel(self):
        img = _colour_image()
        result = _make_high_contrast(img)
        assert result.ndim == 2

    def test_output_same_height_and_width(self):
        img = _colour_image(120, 80)
        result = _make_high_contrast(img)
        assert result.shape == (80, 120)

    def test_output_is_binary(self):
        img = _colour_image()
        result = _make_high_contrast(img)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_handles_greyscale_input(self):
        import cv2
        img = _colour_image()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = _make_high_contrast(grey)
        assert result.ndim == 2

    def test_input_not_mutated(self):
        img = _colour_image()
        original = img.copy()
        _make_high_contrast(img)
        assert np.array_equal(img, original)


# ---------------------------------------------------------------------------
# _make_stamp_suppressed
# ---------------------------------------------------------------------------

class TestMakeStampSuppressed:
    def test_returns_ndarray(self):
        img = _stamp_blue_image()
        result = _make_stamp_suppressed(img)
        assert isinstance(result, np.ndarray)

    def test_output_same_shape(self):
        img = _stamp_blue_image(120, 80)
        result = _make_stamp_suppressed(img)
        assert result.shape == img.shape

    def test_stamp_pixels_become_white(self):
        import cv2
        img = _stamp_blue_image()
        result = _make_stamp_suppressed(img)
        # Check centre of the blue circle — should be white after suppression
        centre_pixel = result[40, 60]
        # At least 2 channels should be near 255
        assert np.sum(centre_pixel >= 200) >= 2

    def test_dark_pixels_preserved(self):
        """Black text (near-zero saturation) should not be affected."""
        img = _colour_image()
        # Black text pixel at row 30, col 30
        img[30, 30] = (10, 10, 10)
        result = _make_stamp_suppressed(img)
        # Should still be dark (not turned white)
        assert np.mean(result[30, 30]) < 100

    def test_input_not_mutated(self):
        img = _stamp_blue_image()
        original = img.copy()
        _make_stamp_suppressed(img)
        assert np.array_equal(img, original)

    def test_handles_non_bgr_gracefully(self):
        grey = np.full((80, 120), 200, dtype=np.uint8)
        result = _make_stamp_suppressed(grey)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# generate_crop_variants
# ---------------------------------------------------------------------------

class TestGenerateCropVariants:
    def test_stamp_affected_generates_all_three(self):
        img = _colour_image()
        cv = generate_crop_variants(img, "R05", stamp_affected=True)
        assert cv.original is not None
        assert cv.high_contrast is not None
        assert cv.stamp_suppressed is not None

    def test_not_stamp_affected_returns_original_only(self):
        img = _colour_image()
        cv = generate_crop_variants(img, "R05", stamp_affected=False)
        assert cv.high_contrast is None
        assert cv.stamp_suppressed is None

    def test_none_crop_returns_empty_variants(self):
        cv = generate_crop_variants(None, "R05", stamp_affected=True)
        assert cv.high_contrast is None

    def test_roi_id_stored(self):
        img = _colour_image()
        cv = generate_crop_variants(img, "R14", stamp_affected=True)
        assert cv.roi_id == "R14"

    def test_original_is_same_object(self):
        img = _colour_image()
        cv = generate_crop_variants(img, "R05", stamp_affected=True)
        assert cv.original is img


# ---------------------------------------------------------------------------
# select_best_text
# ---------------------------------------------------------------------------

class TestSelectBestText:
    def test_original_wins_when_no_questions(self):
        text, source = select_best_text("3000,000", ["30?0,000", "3000?000"])
        assert text == "3000,000"
        assert source == "original"

    def test_variant_chosen_when_fewer_questions(self):
        text, source = select_best_text("3?00,000", ["3000,000"])
        assert text == "3000,000"
        assert source == "variant_1"

    def test_digit_field_prefers_more_digits(self):
        # Both have 0 '?' — prefer more digits
        text, source = select_best_text("3000", ["300000"], is_digit_field=True)
        assert text == "300000"

    def test_empty_original_uses_variant(self):
        text, source = select_best_text("", ["3000,000"])
        assert text == "3000,000"

    def test_no_variants_returns_original(self):
        text, source = select_best_text("hello", [])
        assert text == "hello"
        assert source == "original"

    def test_all_empty_returns_original(self):
        text, source = select_best_text("", ["", ""])
        assert text == ""
        assert source == "original"

    def test_tie_keeps_original(self):
        # Both have same question count — original wins (it's listed first)
        text, source = select_best_text("3?00", ["3?00"])
        assert source == "original"
