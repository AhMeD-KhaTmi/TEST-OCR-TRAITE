"""
Phase 4 — Stamp-aware multi-crop preprocessing.

For ROIs identified as stamp-affected, generates up to three crop variants:
  1. Original colour crop  (always the primary — never discarded)
  2. High-contrast binarised crop (adaptive threshold, aggressive)
  3. Gentle colour-range crop (suppress ONLY the specific stamp-blue shade,
     NOT a broad blue-channel wipe — handwriting is also blue)

These variants are fed to the OCR engine in priority order.
The best result is selected by the merge rules in ocr_engine.py.

⚠️  Critical constraints (plan section 3.7):
    - The original colour crop is ALWAYS kept as primary.
    - Colour-range filtering targets only the narrow HSV band of stamp ink,
      not the entire blue channel.
    - If a variant produces a result that conflicts with the original, the
      original wins unless the variant has measurably more digits.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

import cv2
import numpy as np

from .stamp_detector import StampRegion, StampDetectionResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CropVariants:
    """Alternative crop versions for one stamp-affected ROI."""
    roi_id: str
    original: np.ndarray                        # BGR — always present
    high_contrast: Optional[np.ndarray] = None  # binarised, aggressive threshold
    stamp_suppressed: Optional[np.ndarray] = None  # stamp ink masked to white

    @property
    def all_variants(self) -> list[np.ndarray]:
        """Return all non-None variants in priority order."""
        variants = [self.original]
        if self.high_contrast is not None:
            variants.append(self.high_contrast)
        if self.stamp_suppressed is not None:
            variants.append(self.stamp_suppressed)
        return variants


# ---------------------------------------------------------------------------
# HSV range for stamp ink suppression
# Deliberately narrower than the detection range to avoid killing handwriting
# ---------------------------------------------------------------------------

# Stamp business-blue ink: saturated, medium-dark
_STAMP_BLUE_LOW  = np.array([105, 100, 40],  dtype=np.uint8)
_STAMP_BLUE_HIGH = np.array([128, 255, 180], dtype=np.uint8)

# Stamp purple ink
_STAMP_PURPLE_LOW  = np.array([133, 60, 40],  dtype=np.uint8)
_STAMP_PURPLE_HIGH = np.array([158, 255, 200], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Individual crop variant generators
# ---------------------------------------------------------------------------

def _make_high_contrast(crop: np.ndarray) -> np.ndarray:
    """Produce an aggressively binarised crop.

    Converts to greyscale → adaptive Gaussian threshold (block 25, C 8).
    Useful when the stamp makes adaptive binarisation unstable because it
    introduces a large dark-ink blob.
    """
    if crop.ndim == 3:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        grey = crop.copy()

    # Gentle denoise first (stamp ink creates uneven illumination)
    grey = cv2.medianBlur(grey, 3)

    binary = cv2.adaptiveThreshold(
        grey,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=8,
    )
    # Return as single-channel (works with Tesseract directly)
    return binary


def _make_stamp_suppressed(crop: np.ndarray) -> np.ndarray:
    """Replace stamp-ink pixels with white without touching handwriting.

    Strategy:
      1. Build a narrow HSV mask targeting ONLY stamp-blue/purple ink.
      2. Dilate the mask slightly so nearby transition pixels are also cleaned.
      3. Replace masked pixels with 255 (white background).

    This leaves black handwriting (which is low-saturation) untouched.
    Blue/black handwriting (medium saturation) is at the margin — the narrow
    saturation lower bound (100) is specifically chosen to preserve most
    handwriting which has saturation < 80 in practice.
    """
    if crop.ndim != 3 or crop.shape[2] != 3:
        return crop.copy()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    blue_mask   = cv2.inRange(hsv, _STAMP_BLUE_LOW,   _STAMP_BLUE_HIGH)
    purple_mask = cv2.inRange(hsv, _STAMP_PURPLE_LOW,  _STAMP_PURPLE_HIGH)
    combined    = cv2.bitwise_or(blue_mask, purple_mask)

    # Dilate to clean up edges (3 × 3 ellipse, 1 iteration)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.dilate(combined, kernel, iterations=1)

    result = crop.copy()
    result[combined > 0] = (255, 255, 255)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_crop_variants(
    crop_colour: np.ndarray,
    roi_id: str,
    stamp_affected: bool = True,
) -> CropVariants:
    """Generate alternative crop variants for a stamp-affected ROI.

    Args:
        crop_colour:    Original colour (BGR) ROI crop.
        roi_id:         Identifier string (e.g. "R05") — for logging only.
        stamp_affected: If False, only the original crop is returned
                        (no extra processing).

    Returns:
        CropVariants with up to 3 crop images.
    """
    if not stamp_affected or crop_colour is None or crop_colour.size == 0:
        return CropVariants(roi_id=roi_id, original=crop_colour)

    high_contrast    = _make_high_contrast(crop_colour)
    stamp_suppressed = _make_stamp_suppressed(crop_colour)

    return CropVariants(
        roi_id=roi_id,
        original=crop_colour,
        high_contrast=high_contrast,
        stamp_suppressed=stamp_suppressed,
    )


def select_best_text(
    original_text: str,
    variant_texts: list[str],
    is_digit_field: bool = True,
) -> tuple[str, str]:
    """Choose the best text from original + variant OCR results.

    Heuristic selection rules (applied in order):
    1. If original has no '?' characters, it wins.
    2. Among all results (original + variants), prefer the one with
       the fewest '?' characters.
    3. Tie-break: for digit fields, prefer the result with the most digits.

    Returns:
        (best_text, source)  where source is "original" or "variant_N"
    """
    candidates = [(original_text, "original")] + [
        (t, f"variant_{i}") for i, t in enumerate(variant_texts, 1)
    ]

    if all(t == "" for t, _ in candidates):
        return original_text, "original"

    def score(text: str) -> tuple[int, int]:
        q_count = text.count("?")
        digit_count = sum(c.isdigit() for c in text) if is_digit_field else 0
        return (q_count, -digit_count)  # lower q, more digits = better

    best_text, best_source = min(candidates, key=lambda c: score(c[0]))
    return best_text, best_source


def generate_variants_for_affected_rois(
    crops: dict,    # dict[str, ROICrop] from roi_extractor
    stamp_result: StampDetectionResult,
    roi_config: dict,       # ROI config dict: roi_id -> {x1, y1, x2, y2, ...}
    ref_w: int,
    ref_h: int,
    img_w: int,
    img_h: int,
    overlap_threshold: float = 0.10,
) -> dict[str, CropVariants]:
    """Generate crop variants for every ROI that overlaps a detected stamp.

    Args:
        crops:               Raw crops dict from ROIExtractor.extract_all().
        stamp_result:        Output from detect_stamps().
        roi_config:          ROI config entries keyed by roi_id.
        ref_w, ref_h:        Reference dimensions the ROI coords are normalised against.
        img_w, img_h:        Actual image dimensions.
        overlap_threshold:   Fraction of ROI area that must be covered to flag.

    Returns:
        dict mapping roi_id -> CropVariants for every stamp-affected ROI.
        Clean ROIs are NOT included (callers can still use the original crop).
    """
    variants: dict[str, CropVariants] = {}

    for roi_id, crop in crops.items():
        if roi_id not in roi_config:
            continue
        cfg = roi_config[roi_id]
        # Convert relative coords to pixel coords
        x1 = int(cfg["x1"] * img_w)
        y1 = int(cfg["y1"] * img_h)
        x2 = int(cfg["x2"] * img_w)
        y2 = int(cfg["y2"] * img_h)

        affected = bool(stamp_result.stamps_overlapping_roi(
            x1, y1, x2, y2, threshold=overlap_threshold
        ))
        if affected:
            colour_img = crop.colour if hasattr(crop, "colour") else crop
            variants[roi_id] = generate_crop_variants(
                colour_img, roi_id, stamp_affected=True
            )

    return variants
