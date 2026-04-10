"""Unit tests for Phase 1 — ROI extractor module."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.alignment import align
from ocr_pipeline.roi_extractor import (
    ROICrop,
    ROIExtractor,
    draw_roi_overlay,
    save_crops,
)

SAMPLE_DIR   = Path(__file__).parent.parent / "example"
ALL_SAMPLES  = sorted(SAMPLE_DIR.glob("*.jpg"))
FIRST_SAMPLE = ALL_SAMPLES[0]

# Expected ROI IDs from the config
EXPECTED_ROI_IDS = {
    "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08",
    "R09", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17",
}

TIER1_ROI_IDS = {"R02", "R03", "R05", "R06", "R09", "R10", "R12", "R13", "R14", "R15"}


@pytest.fixture(scope="module")
def extractor():
    return ROIExtractor()


@pytest.fixture(scope="module")
def aligned_first():
    prep = preprocess(FIRST_SAMPLE)
    aln  = align(prep.deskewed)
    return aln.image


# ---------------------------------------------------------------------------
# ROIExtractor construction
# ---------------------------------------------------------------------------

class TestROIExtractorInit:
    def test_loads_config_without_error(self, extractor):
        assert extractor is not None

    def test_has_17_rois(self, extractor):
        assert len(extractor.rois) == 17

    def test_reference_dimensions_positive(self, extractor):
        assert extractor.ref_w > 0
        assert extractor.ref_h > 0

    def test_stamp_affected_rois_is_set(self, extractor):
        assert isinstance(extractor.stamp_rois, set)
        assert len(extractor.stamp_rois) > 0


# ---------------------------------------------------------------------------
# extract_all
# ---------------------------------------------------------------------------

class TestExtractAll:
    def test_returns_all_17_rois(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        assert set(crops.keys()) == EXPECTED_ROI_IDS

    def test_each_crop_is_roi_crop_instance(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        for roi_id, crop in crops.items():
            assert isinstance(crop, ROICrop), f"{roi_id} is not ROICrop"

    def test_colour_crops_are_3_channel(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        for roi_id, crop in crops.items():
            assert crop.colour.ndim == 3, f"{roi_id} colour crop not 3-channel"
            assert crop.colour.shape[2] == 3

    def test_binarized_crops_are_single_channel(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        for roi_id, crop in crops.items():
            assert crop.binarized is not None, f"{roi_id} binarized is None"
            assert crop.binarized.ndim == 2, f"{roi_id} binarized not 2D"

    def test_blue_channel_crops_are_single_channel(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        for roi_id, crop in crops.items():
            assert crop.blue_channel is not None, f"{roi_id} blue_channel is None"
            assert crop.blue_channel.ndim == 2

    def test_all_crops_have_positive_dimensions(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        for roi_id, crop in crops.items():
            h, w = crop.colour.shape[:2]
            assert h > 0 and w > 0, f"{roi_id} has zero-size crop"

    def test_roi_ids_stored_correctly(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first)
        for roi_id, crop in crops.items():
            assert crop.roi_id == roi_id


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------

class TestTierFilter:
    def test_extract_tier1_returns_only_tier1(self, extractor, aligned_first):
        crops = extractor.extract_tier1(aligned_first)
        for roi_id in crops:
            tier = extractor.rois[roi_id]["tier"]
            assert tier == 1, f"{roi_id} has tier {tier}, expected 1"

    def test_tier1_contains_expected_rois(self, extractor, aligned_first):
        crops = extractor.extract_tier1(aligned_first)
        assert TIER1_ROI_IDS.issubset(set(crops.keys()))

    def test_tier_filter_set3_returns_city_rois(self, extractor, aligned_first):
        crops = extractor.extract_all(aligned_first, tier_filter={3})
        assert "R04" in crops  # city_upper
        assert "R11" in crops  # city_lower
        # Should NOT contain tier 1 ROIs
        assert "R05" not in crops  # rib_upper is tier 1


# ---------------------------------------------------------------------------
# Stamp-affected ROIs get extra padding
# ---------------------------------------------------------------------------

class TestStampPadding:
    def test_stamp_rois_use_larger_padding(self, extractor):
        """Stamp-affected ROIs (R05, R14, R15, R16) must have padding >= 0.15."""
        for roi_id in extractor.stamp_rois:
            padding = extractor.rois[roi_id].get("padding", 0.05)
            assert padding >= 0.15, (
                f"{roi_id} in stamp_affected_rois but padding={padding} < 0.15"
            )


# ---------------------------------------------------------------------------
# OCR engine assignments
# ---------------------------------------------------------------------------

class TestOcrEngineAssignments:
    def test_rib_rois_use_tesseract_first(self, extractor):
        for roi_id in ("R05", "R14"):
            engine = extractor.rois[roi_id].get("ocr_engine", "")
            assert "tesseract" in engine, f"{roi_id} should be tesseract-first"

    def test_amount_digit_rois_use_tesseract_first(self, extractor):
        for roi_id in ("R06", "R10"):
            engine = extractor.rois[roi_id].get("ocr_engine", "")
            assert "tesseract" in engine, f"{roi_id} should be tesseract-first"

    def test_handwriting_rois_use_qwen(self, extractor):
        for roi_id in ("R09", "R15"):
            engine = extractor.rois[roi_id].get("ocr_engine", "")
            assert "qwen" in engine, f"{roi_id} should be qwen"


# ---------------------------------------------------------------------------
# draw_roi_overlay
# ---------------------------------------------------------------------------

class TestDrawRoiOverlay:
    def test_returns_ndarray_same_shape(self, extractor, aligned_first):
        crops   = extractor.extract_all(aligned_first)
        overlay = draw_roi_overlay(aligned_first, crops, extractor)
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == aligned_first.shape

    def test_does_not_mutate_original(self, extractor, aligned_first):
        original_copy = aligned_first.copy()
        crops = extractor.extract_all(aligned_first)
        draw_roi_overlay(aligned_first, crops, extractor)
        assert np.array_equal(aligned_first, original_copy)


# ---------------------------------------------------------------------------
# save_crops
# ---------------------------------------------------------------------------

class TestSaveCrops:
    def test_saves_files_to_disk(self, extractor, aligned_first, tmp_path):
        crops = extractor.extract_all(aligned_first)
        save_crops(crops, tmp_path)
        saved = list(tmp_path.glob("*.jpg"))
        # Colour + binary per ROI → at least 17 * 2 = 34 files
        assert len(saved) >= 17

    def test_output_dir_created_if_missing(self, extractor, aligned_first, tmp_path):
        crops   = extractor.extract_all(aligned_first)
        new_dir = tmp_path / "deep" / "nested" / "dir"
        save_crops(crops, new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# Multi-sample smoke test
# ---------------------------------------------------------------------------

class TestExtractMultiSample:
    @pytest.mark.parametrize("img_path", ALL_SAMPLES[:10])
    def test_extract_tier1_no_error(self, extractor, img_path):
        prep  = preprocess(img_path)
        aln   = align(prep.deskewed)
        crops = extractor.extract_tier1(aln.image)
        assert len(crops) > 0

    def test_no_zero_size_crops_in_first_batch(self, extractor):
        """Tier-1 crops for the _1_ batch must all have positive dimensions."""
        batch1 = [p for p in ALL_SAMPLES if "_1_page-" in p.name]
        for img_path in batch1:
            prep  = preprocess(img_path)
            aln   = align(prep.deskewed)
            crops = extractor.extract_tier1(aln.image)
            for roi_id, crop in crops.items():
                h, w = crop.colour.shape[:2]
                assert h > 0 and w > 0, (
                    f"{img_path.name} / {roi_id}: zero-size crop {h}x{w}"
                )
