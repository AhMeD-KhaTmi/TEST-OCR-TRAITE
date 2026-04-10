"""
Phase 4 — Stamp Detection.

Detects blue circular/oval stamps on Tunisian Lettre de Change scans using:
  1. Colour-range segmentation (HSV blue/purple band)
  2. Hough circle detection on the binary mask
  3. Contour-based ellipse fitting for non-circular stamps

The output is a list of StampRegion instances that describe where stamps are
located on the document image.  These regions are used by two consumers:

  • stamp_preprocessor.py — to generate alternative crops for stamp-affected ROIs
  • anomaly_explainer.py  — to annotate the full-document prompt with stamp
                            region information for the Pass 3 LLM call

⚠️  Critical design constraint (from plan section 3.7):
    Stamp detection is for LOCALISATION ONLY.  The output of this module is
    NEVER used to modify or delete portions of the primary image.  Even
    morphological-filtered crops are generated in a SEPARATE processing branch
    and compared against the original crop — they never replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StampRegion:
    """A detected stamp position in image coordinates."""
    cx: int                    # centre x (pixels)
    cy: int                    # centre y (pixels)
    rx: int                    # half-width  (radius x for circle/ellipse)
    ry: int                    # half-height (radius y; == rx for circles)
    confidence: float          # 0.0–1.0  (detection confidence)
    method: str = "hough"      # "hough" | "contour"
    colour: str = "blue"       # "blue" | "purple" | "unknown"

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box (x1, y1, x2, y2) with a 10 px margin."""
        margin = 10
        return (
            max(0, self.cx - self.rx - margin),
            max(0, self.cy - self.ry - margin),
            self.cx + self.rx + margin,
            self.cy + self.ry + margin,
        )

    def overlaps_roi(
        self,
        roi_x1: int, roi_y1: int, roi_x2: int, roi_y2: int,
        threshold: float = 0.10,
    ) -> bool:
        """Return True if this stamp overlaps the ROI by more than threshold fraction.

        Fraction is computed as intersection area / ROI area.
        """
        sx1, sy1, sx2, sy2 = self.bbox
        ix1 = max(sx1, roi_x1)
        iy1 = max(sy1, roi_y1)
        ix2 = min(sx2, roi_x2)
        iy2 = min(sy2, roi_y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        intersection = (ix2 - ix1) * (iy2 - iy1)
        roi_area = max(1, (roi_x2 - roi_x1) * (roi_y2 - roi_y1))
        return (intersection / roi_area) >= threshold


@dataclass
class StampDetectionResult:
    """Full stamp detection output for one document image."""
    stamps: list[StampRegion] = dc_field(default_factory=list)
    blue_mask: Optional[np.ndarray] = None   # uint8 binary mask (same WxH as input)
    debug_image: Optional[np.ndarray] = None  # colour copy with annotations

    @property
    def count(self) -> int:
        return len(self.stamps)

    def stamps_overlapping_roi(
        self,
        roi_x1: int, roi_y1: int, roi_x2: int, roi_y2: int,
        threshold: float = 0.10,
    ) -> list[StampRegion]:
        return [
            s for s in self.stamps
            if s.overlaps_roi(roi_x1, roi_y1, roi_x2, roi_y2, threshold)
        ]


# ---------------------------------------------------------------------------
# Colour range definitions (HSV)
# ---------------------------------------------------------------------------

# Blue business stamps: hue 100–130, moderate-high saturation, moderate value
_BLUE_LOW_1  = np.array([100, 80, 50],  dtype=np.uint8)
_BLUE_HIGH_1 = np.array([130, 255, 220], dtype=np.uint8)

# Purple/fiscal stamps: hue 130–160
_PURPLE_LOW  = np.array([130, 60, 60],  dtype=np.uint8)
_PURPLE_HIGH = np.array([160, 255, 230], dtype=np.uint8)

# Minimum pixel area for a blob to be considered a stamp (avoids noise)
_MIN_STAMP_AREA_PX = 2_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_colour_mask(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build separate blue and purple masks from a BGR image.

    Returns (blue_mask, purple_mask) as uint8 binary arrays.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue   = cv2.inRange(hsv, _BLUE_LOW_1, _BLUE_HIGH_1)
    purple = cv2.inRange(hsv, _PURPLE_LOW,  _PURPLE_HIGH)
    # Morphological closing to fill small gaps inside stamp ink rings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blue   = cv2.morphologyEx(blue,   cv2.MORPH_CLOSE, kernel, iterations=2)
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, kernel, iterations=2)
    return blue, purple


def _detect_by_hough(
    mask: np.ndarray,
    colour: str,
) -> list[StampRegion]:
    """Run Hough circle detection on a binary mask."""
    regions: list[StampRegion] = []
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=60,
        param1=50,
        param2=25,
        minRadius=30,
        maxRadius=200,
    )
    if circles is None:
        return regions
    for cx, cy, r in np.round(circles[0]).astype(int):
        # Verify the circle is mostly within a dense mask region
        roi = mask[max(0, cy - r): cy + r, max(0, cx - r): cx + r]
        density = np.count_nonzero(roi) / max(1, roi.size)
        if density < 0.08:
            continue  # mostly empty — likely a false positive
        conf = min(1.0, 0.5 + density)
        regions.append(StampRegion(
            cx=int(cx), cy=int(cy),
            rx=int(r), ry=int(r),
            confidence=round(conf, 3),
            method="hough",
            colour=colour,
        ))
    return regions


def _detect_by_contour(
    mask: np.ndarray,
    colour: str,
) -> list[StampRegion]:
    """Fit ellipses to large contours in the binary mask."""
    regions: list[StampRegion] = []
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_STAMP_AREA_PX:
            continue
        if len(cnt) < 5:
            # Need at least 5 points for fitEllipse
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            rx, ry = w // 2, h // 2
            conf = 0.50
        else:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, mb), _ = ellipse
            cx, cy = int(round(cx)), int(round(cy))
            rx, ry = int(round(ma / 2)), int(round(mb / 2))
            # Quality: ratio of ellipse area to contour bounding-rect area
            ellipse_area = np.pi * rx * ry
            conf = min(1.0, float(area) / max(1, ellipse_area) * 0.8)
        regions.append(StampRegion(
            cx=cx, cy=cy,
            rx=max(rx, ry),  # use max radius as conservative bound
            ry=max(rx, ry),
            confidence=round(conf, 3),
            method="contour",
            colour=colour,
        ))
    return regions


def _deduplicate(regions: list[StampRegion], iou_threshold: float = 0.5) -> list[StampRegion]:
    """Remove duplicate stamp detections by IoU-based merging."""
    if not regions:
        return []
    # Sort by confidence descending
    regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
    kept: list[StampRegion] = []
    for candidate in regions:
        duplicate = False
        cx1, cy1, rx1, ry1 = candidate.cx, candidate.cy, candidate.rx, candidate.ry
        for existing in kept:
            cx2, cy2, rx2, ry2 = existing.cx, existing.cy, existing.rx, existing.ry
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            overlap_threshold = (rx1 + rx2) * 0.6
            if dist < overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_stamps(
    image: np.ndarray,
    draw_debug: bool = False,
) -> StampDetectionResult:
    """Detect stamps in a document image.

    Args:
        image:      BGR document image (typically the aligned full-page image).
        draw_debug: If True, annotate a copy of the image with detected
                    stamp boundaries for visual inspection.

    Returns:
        StampDetectionResult with all detected stamp regions.

    ⚠️  The returned masks and debug images reference NEW arrays.  The input
        ``image`` is never modified.
    """
    if image is None or image.size == 0:
        return StampDetectionResult()

    blue_mask, purple_mask = _build_colour_mask(image)

    stamps: list[StampRegion] = []

    # --- Hough detection ---
    stamps += _detect_by_hough(blue_mask,   colour="blue")
    stamps += _detect_by_hough(purple_mask, colour="purple")

    # --- Contour-ellipse fallback for non-circular stamps ---
    stamps += _detect_by_contour(blue_mask,   colour="blue")
    stamps += _detect_by_contour(purple_mask, colour="purple")

    stamps = _deduplicate(stamps)

    # Combined mask for downstream consumers
    combined_mask = cv2.bitwise_or(blue_mask, purple_mask)

    debug_image: Optional[np.ndarray] = None
    if draw_debug and stamps:
        debug_image = image.copy()
        for s in stamps:
            colour_bgr = (255, 120, 0) if s.colour == "blue" else (200, 0, 200)
            cv2.ellipse(
                debug_image,
                (s.cx, s.cy), (s.rx, s.ry), 0, 0, 360,
                colour_bgr, thickness=3,
            )
            label = f"{s.colour[:1].upper()} {s.confidence:.2f}"
            cv2.putText(
                debug_image, label,
                (s.cx - s.rx, s.cy - s.ry - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 2,
            )

    return StampDetectionResult(
        stamps=stamps,
        blue_mask=combined_mask,
        debug_image=debug_image,
    )


def roi_is_stamp_affected(
    stamp_result: StampDetectionResult,
    roi_x1: int, roi_y1: int, roi_x2: int, roi_y2: int,
    threshold: float = 0.10,
) -> bool:
    """Quick helper: returns True if any detected stamp overlaps the ROI."""
    return bool(stamp_result.stamps_overlapping_roi(
        roi_x1, roi_y1, roi_x2, roi_y2, threshold
    ))
