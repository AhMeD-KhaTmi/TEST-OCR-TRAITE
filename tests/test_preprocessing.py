"""Unit tests for Phase 1 — preprocessing module."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.preprocessing import (
    PreprocessResult,
    _detect_skew_angle,
    _estimate_dpi_scale,
    _resize_to_target,
    adaptive_binarize,
    correct_perspective,
    deskew,
    extract_blue_channel,
    gentle_denoise,
    preprocess,
    suppress_red_channel,
)

SAMPLE_DIR = Path(__file__).parent.parent / "example"
FIRST_SAMPLE = sorted(SAMPLE_DIR.glob("*.jpg"))[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_color_img(h: int = 300, w: int = 200, bgr=(200, 200, 200)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def _grey_img(h: int = 300, w: int = 200, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# DPI scale estimation
# ---------------------------------------------------------------------------

class TestDpiScale:
    def test_a4_sized_image_returns_near_one(self):
        img = np.zeros((3508, 2480, 3), dtype=np.uint8)  # exact A4 @ 300 DPI
        scale = _estimate_dpi_scale(img)
        assert abs(scale - 1.0) < 0.02

    def test_half_size_image_returns_near_two(self):
        img = np.zeros((1754, 1240, 3), dtype=np.uint8)  # half A4
        scale = _estimate_dpi_scale(img)
        assert 1.8 < scale < 2.2

    def test_tiny_image_does_not_crash(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        scale = _estimate_dpi_scale(img)
        assert scale > 0


class TestResize:
    def test_no_op_when_within_two_percent(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _resize_to_target(img, 1.01)
        assert out.shape == img.shape

    def test_upscales_correctly(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _resize_to_target(img, 2.0)
        assert out.shape == (200, 200, 3)

    def test_downscales_correctly(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = _resize_to_target(img, 0.5)
        assert out.shape == (50, 50, 3)


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------

class TestDeskew:
    def test_no_rotation_on_nearly_straight_image(self):
        # Horizontal black line on white background — zero skew expected
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.line(img, (20, 100), (380, 100), (0, 0, 0), 2)
        out, angle = deskew(img)
        assert abs(angle) < 1.0
        assert out.shape == img.shape

    def test_returns_ndarray_and_float(self):
        img = _solid_color_img()
        out, angle = deskew(img)
        assert isinstance(out, np.ndarray)
        assert isinstance(angle, float)

    def test_skew_angle_detection_near_horizontal(self):
        # Draw a slightly tilted line (~5°) → skew angle should be detected
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.line(img, (10, 200), (590, 235), (0, 0, 0), 2)  # slight tilt
        angle = _detect_skew_angle(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        assert isinstance(angle, float)
        assert -20 <= angle <= 20


# ---------------------------------------------------------------------------
# Color channel helpers
# ---------------------------------------------------------------------------

class TestChannels:
    def test_extract_blue_channel_shape(self):
        img = _solid_color_img(bgr=(200, 100, 50))
        blue = extract_blue_channel(img)
        assert blue.shape == (300, 200)
        assert blue.dtype == np.uint8

    def test_extract_blue_channel_passes_grayscale_unchanged(self):
        grey = _grey_img()
        out = extract_blue_channel(grey)
        assert out.shape == grey.shape

    def test_suppress_red_zeros_red_channel(self):
        img = _solid_color_img(bgr=(100, 150, 200))  # B=100, G=150, R=200
        out = suppress_red_channel(img)
        assert np.all(out[:, :, 2] == 0)   # red zeroed
        assert np.all(out[:, :, 0] == 100)  # blue preserved
        assert np.all(out[:, :, 1] == 150)  # green preserved


# ---------------------------------------------------------------------------
# Binarization & denoise
# ---------------------------------------------------------------------------

class TestBinarization:
    def test_output_is_binary(self):
        grey = _grey_img()
        binary = adaptive_binarize(grey)
        unique_vals = set(np.unique(binary))
        assert unique_vals.issubset({0, 255})

    def test_output_shape_matches_input(self):
        grey = _grey_img(100, 150)
        binary = adaptive_binarize(grey)
        assert binary.shape == grey.shape

    def test_gentle_denoise_returns_same_shape(self):
        img = _solid_color_img(200, 300)
        out = gentle_denoise(img)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

class TestPerspective:
    def test_returns_tuple_of_ndarray_and_bool(self):
        img = _solid_color_img(300, 200)
        out, applied = correct_perspective(img)
        assert isinstance(out, np.ndarray)
        assert isinstance(applied, bool)

    def test_plain_colour_image_not_corrected(self):
        # No document edges → corners not found → applied=False
        img = _solid_color_img(200, 300, bgr=(230, 230, 230))
        _, applied = correct_perspective(img)
        assert applied is False


# ---------------------------------------------------------------------------
# Full pipeline on real image
# ---------------------------------------------------------------------------

class TestPreprocessPipeline:
    def test_returns_preprocess_result(self):
        result = preprocess(FIRST_SAMPLE)
        assert isinstance(result, PreprocessResult)

    def test_all_images_are_ndarray(self):
        result = preprocess(FIRST_SAMPLE)
        assert isinstance(result.original, np.ndarray)
        assert isinstance(result.normalised, np.ndarray)
        assert isinstance(result.deskewed, np.ndarray)
        assert isinstance(result.binarized, np.ndarray)
        assert isinstance(result.blue_channel, np.ndarray)

    def test_binarized_is_single_channel(self):
        result = preprocess(FIRST_SAMPLE)
        assert result.binarized.ndim == 2

    def test_blue_channel_is_single_channel(self):
        result = preprocess(FIRST_SAMPLE)
        assert result.blue_channel.ndim == 2

    def test_scale_factor_positive(self):
        result = preprocess(FIRST_SAMPLE)
        assert result.scale_factor > 0

    def test_skew_angle_reasonable(self):
        result = preprocess(FIRST_SAMPLE)
        assert -45 <= result.skew_angle_deg <= 45

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            preprocess("nonexistent_file.jpg")

    def test_warnings_is_list(self):
        result = preprocess(FIRST_SAMPLE)
        assert isinstance(result.warnings, list)
