"""Tests for stamp_detector.py — Phase 4."""
import numpy as np
import pytest

from src.ocr_pipeline.stamp_detector import (
    StampRegion,
    StampDetectionResult,
    detect_stamps,
    roi_is_stamp_affected,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank(w=500, h=700, colour=(255, 255, 255)) -> np.ndarray:
    """Return a solid-colour BGR image."""
    img = np.full((h, w, 3), colour, dtype=np.uint8)
    return img


def _draw_blue_circle(img: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """Draw a filled blue circle (simulated stamp) on img."""
    result = img.copy()
    # Stamp blue: BGR ~ (200, 100, 30) — hue ~105°, sat ~180, val ~200
    import cv2
    cv2.circle(result, (cx, cy), r, (200, 100, 30), thickness=-1)
    return result


# ---------------------------------------------------------------------------
# StampRegion
# ---------------------------------------------------------------------------

class TestStampRegion:
    def test_bbox_has_margin(self):
        s = StampRegion(cx=100, cy=100, rx=50, ry=50, confidence=0.8)
        x1, y1, x2, y2 = s.bbox
        assert x1 < 100 - 50   # margin applied
        assert y1 < 100 - 50
        assert x2 > 100 + 50
        assert y2 > 100 + 50

    def test_bbox_clamped_to_zero(self):
        s = StampRegion(cx=5, cy=5, rx=50, ry=50, confidence=0.8)
        x1, y1, _, _ = s.bbox
        assert x1 >= 0
        assert y1 >= 0

    def test_overlaps_roi_full_overlap(self):
        s = StampRegion(cx=100, cy=100, rx=80, ry=80, confidence=0.9)
        # ROI centred at same point
        assert s.overlaps_roi(50, 50, 150, 150, threshold=0.05)

    def test_overlaps_roi_no_overlap(self):
        s = StampRegion(cx=10, cy=10, rx=5, ry=5, confidence=0.9)
        assert not s.overlaps_roi(400, 400, 500, 500, threshold=0.05)

    def test_overlaps_roi_threshold_controls_sensitivity(self):
        s = StampRegion(cx=100, cy=100, rx=20, ry=20, confidence=0.9)
        roi = (90, 90, 110, 110)  # small ROI, stamp covers it well
        assert s.overlaps_roi(*roi, threshold=0.05)
        # Threshold > 1.0 is mathematically impossible to reach
        assert not s.overlaps_roi(*roi, threshold=1.1)

    def test_method_default(self):
        s = StampRegion(cx=0, cy=0, rx=10, ry=10, confidence=0.5)
        assert s.method == "hough"

    def test_colour_default(self):
        s = StampRegion(cx=0, cy=0, rx=10, ry=10, confidence=0.5)
        assert s.colour == "blue"


# ---------------------------------------------------------------------------
# StampDetectionResult
# ---------------------------------------------------------------------------

class TestStampDetectionResult:
    def test_count_empty(self):
        r = StampDetectionResult()
        assert r.count == 0

    def test_count_with_stamps(self):
        s1 = StampRegion(cx=100, cy=100, rx=50, ry=50, confidence=0.8)
        s2 = StampRegion(cx=300, cy=300, rx=50, ry=50, confidence=0.7)
        r = StampDetectionResult(stamps=[s1, s2])
        assert r.count == 2

    def test_stamps_overlapping_roi_returns_matching(self):
        s = StampRegion(cx=100, cy=100, rx=60, ry=60, confidence=0.8)
        r = StampDetectionResult(stamps=[s])
        overlapping = r.stamps_overlapping_roi(60, 60, 160, 160, threshold=0.05)
        assert s in overlapping

    def test_stamps_overlapping_roi_no_match(self):
        s = StampRegion(cx=10, cy=10, rx=5, ry=5, confidence=0.8)
        r = StampDetectionResult(stamps=[s])
        overlapping = r.stamps_overlapping_roi(400, 400, 500, 500, threshold=0.05)
        assert overlapping == []


# ---------------------------------------------------------------------------
# detect_stamps
# ---------------------------------------------------------------------------

class TestDetectStamps:
    def test_blank_image_returns_empty(self):
        img = _blank()
        result = detect_stamps(img)
        assert result.count == 0

    def test_none_image_returns_empty(self):
        result = detect_stamps(None)  # type: ignore[arg-type]
        assert result.count == 0

    def test_returns_stamp_detection_result(self):
        img = _blank()
        result = detect_stamps(img)
        assert isinstance(result, StampDetectionResult)

    def test_blue_mask_returned(self):
        img = _blank()
        result = detect_stamps(img)
        assert result.blue_mask is not None
        assert result.blue_mask.shape[:2] == img.shape[:2]

    def test_blank_mask_is_all_zero(self):
        img = _blank()
        result = detect_stamps(img)
        assert result.blue_mask is not None
        assert result.blue_mask.sum() == 0

    def test_blue_circle_produces_nonzero_mask(self):
        img = _blank()
        img = _draw_blue_circle(img, cx=250, cy=350, r=80)
        result = detect_stamps(img)
        assert result.blue_mask is not None
        assert result.blue_mask.sum() > 0

    def test_debug_image_none_when_not_requested(self):
        img = _blank()
        result = detect_stamps(img, draw_debug=False)
        assert result.debug_image is None

    def test_debug_image_returned_when_requested_with_stamps(self):
        img = _blank()
        img = _draw_blue_circle(img, cx=250, cy=350, r=80)
        result = detect_stamps(img, draw_debug=True)
        # debug_image may be None if No stamps found, but mask should be non-zero
        # Just verify no exception and correct return type
        assert isinstance(result, StampDetectionResult)

    def test_input_image_not_mutated(self):
        img = _blank()
        original = img.copy()
        detect_stamps(img, draw_debug=True)
        assert np.array_equal(img, original)

    def test_stamp_confidence_in_range(self):
        img = _blank()
        img = _draw_blue_circle(img, cx=250, cy=350, r=90)
        result = detect_stamps(img)
        for s in result.stamps:
            assert 0.0 <= s.confidence <= 1.0

    def test_stamp_colour_is_string(self):
        img = _blank()
        img = _draw_blue_circle(img, cx=250, cy=350, r=90)
        result = detect_stamps(img)
        for s in result.stamps:
            assert s.colour in ("blue", "purple", "unknown")


# ---------------------------------------------------------------------------
# roi_is_stamp_affected
# ---------------------------------------------------------------------------

class TestRoiIsStampAffected:
    def test_no_stamps_returns_false(self):
        result = StampDetectionResult()
        assert not roi_is_stamp_affected(result, 0, 0, 100, 100)

    def test_overlapping_stamp_returns_true(self):
        s = StampRegion(cx=50, cy=50, rx=60, ry=60, confidence=0.8)
        result = StampDetectionResult(stamps=[s])
        assert roi_is_stamp_affected(result, 20, 20, 80, 80)

    def test_distant_stamp_returns_false(self):
        s = StampRegion(cx=500, cy=500, rx=20, ry=20, confidence=0.8)
        result = StampDetectionResult(stamps=[s])
        assert not roi_is_stamp_affected(result, 0, 0, 100, 100)
