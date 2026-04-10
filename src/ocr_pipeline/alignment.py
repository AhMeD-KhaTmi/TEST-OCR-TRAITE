"""
Phase 1 — Template alignment.

Detects anchor points on the document (header band, barcode, outer border)
and computes an affine/homography transform so all downstream ROI coordinates
work on a canonical coordinate system.

Fallback hierarchy (per plan section 3.2):
  1. Outer border rectangle (contour analysis)
  2. Header band top edge + barcode bottom edge
  3. Percentage-based pass-through (no transform)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    image: np.ndarray          # aligned colour image (may be same as input)
    transform: Optional[np.ndarray]  # 3x3 homography, or None if identity
    method: str                # 'border' | 'header_barcode' | 'identity'
    confidence: float          # 0.0-1.0
    anchor_debug: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anchor detection helpers
# ---------------------------------------------------------------------------

def _find_outer_border(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Find the outer rectangle of the form using contour analysis.

    Returns (x, y, w, h) of the best candidate, or None.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)

    # Close small gaps so the border is a single contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = img.shape[:2]
    img_area = w * h

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area_ratio = cv2.contourArea(cnt) / img_area
        if area_ratio < 0.30:
            break  # all remaining are smaller

        x, y, cw, ch = cv2.boundingRect(cnt)
        # Accept if it covers most of the document
        if cw > w * 0.70 and ch > h * 0.70:
            return (x, y, cw, ch)

    return None


def _find_header_band(img: np.ndarray) -> Optional[tuple[int, int]]:
    """Detect the top and bottom y-coordinate of the header band.

    The header band has a pinkish/salmon hue (R slightly dominant, saturation
    above background white but below printed text).

    Returns (y_top, y_bottom) or None.
    """
    # Convert to float for channel math
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)

    # Pixels where red dominates over blue by at least 15 counts and
    # the image is neither too dark (<80) nor pure white (>240 mean)
    mean_val = (b + g + r) / 3.0
    red_dominant = (r - b > 10) & (r - g > 5) & (mean_val > 80) & (mean_val < 245)

    row_red_frac = red_dominant.astype(np.float32).mean(axis=1)

    # Header rows: at least 10% of pixels in the row show red dominance
    header_candidates = np.where(row_red_frac > 0.10)[0]

    # The header is a contiguous band at the top ~15% of the document
    h = img.shape[0]
    top_region = header_candidates[header_candidates < h * 0.20]

    if len(top_region) < 3:
        return None

    return int(top_region[0]), int(top_region[-1])


def _find_barcode_region(img: np.ndarray) -> Optional[tuple[int, int]]:
    """Detect the barcode strip near the bottom of the document.

    Returns (y_top, y_bottom) of the barcode band, or None.
    """
    h, w = img.shape[:2]
    # Only look in bottom 20%
    search_top = int(h * 0.80)
    bottom_region = img[search_top:, :]

    grey = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)

    # Barcode: columns with many alternating dark/light transitions per row
    # Measure vertical variance per column in the binary image
    col_variance = binary.astype(np.float32).var(axis=0)
    high_var_cols = np.sum(col_variance > 3000)

    if high_var_cols < w * 0.10:
        return None

    # Find the rows with many dark pixels (barcode lines span full height of strip)
    row_dark = (binary > 0).sum(axis=1)
    barcode_rows = np.where(row_dark > w * 0.05)[0]

    if len(barcode_rows) < 5:
        return None

    y_top = search_top + int(barcode_rows[0])
    y_bot = search_top + int(barcode_rows[-1])
    return (y_top, y_bot)


# ---------------------------------------------------------------------------
# Alignment strategies
# ---------------------------------------------------------------------------

def _align_via_border(img: np.ndarray) -> AlignmentResult:
    """Strategy 1: warp using the detected outer border rectangle."""
    bbox = _find_outer_border(img)
    if bbox is None:
        return AlignmentResult(
            image=img, transform=None, method='identity',
            confidence=0.3,
            warnings=['Outer border not detected — using identity.']
        )

    x, y, bw, bh = bbox
    h, w = img.shape[:2]

    # Source corners (detected border)
    src = np.float32([
        [x,      y],
        [x + bw, y],
        [x + bw, y + bh],
        [x,      y + bh],
    ])
    # Destination: full image canvas
    dst = np.float32([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    aligned = cv2.warpPerspective(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    coverage = (bw * bh) / (w * h)
    confidence = min(1.0, coverage * 1.2)

    return AlignmentResult(
        image=aligned, transform=M, method='border',
        confidence=confidence,
        anchor_debug={'border_bbox': bbox},
    )


def _align_via_header_barcode(img: np.ndarray) -> AlignmentResult:
    """Strategy 2: use header top edge and barcode bottom edge as anchors."""
    h, w = img.shape[:2]
    warnings: list[str] = []

    header = _find_header_band(img)
    barcode = _find_barcode_region(img)

    if header is None and barcode is None:
        return AlignmentResult(
            image=img, transform=None, method='identity',
            confidence=0.2,
            warnings=['No anchors found — identity transform.']
        )

    # Light normalisation: crop to header-top .. barcode-bottom
    y_top = header[0] if header else 0
    y_bot = barcode[1] if barcode else h - 1

    if y_bot <= y_top:
        warnings.append('Anchor ordering error — identity transform.')
        return AlignmentResult(image=img, transform=None, method='identity',
                               confidence=0.2, warnings=warnings)

    cropped = img[y_top:y_bot + 1, :]
    # Resize back to original canvas size
    aligned = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    # Build approximate transform for debugging (scale only)
    scale_y = h / (y_bot - y_top + 1)
    M = np.float32([
        [1,      0, 0],
        [0, scale_y, -y_top * scale_y],
        [0,      0, 1],
    ])

    confidence = 0.5 + 0.2 * (header is not None) + 0.2 * (barcode is not None)

    return AlignmentResult(
        image=aligned, transform=M, method='header_barcode',
        confidence=confidence,
        anchor_debug={'header': header, 'barcode': barcode},
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Main alignment entry point
# ---------------------------------------------------------------------------

def align(img: np.ndarray) -> AlignmentResult:
    """Attempt template alignment using the best available strategy.

    Falls back through the hierarchy in section 3.2 of the plan.
    """
    result = _align_via_border(img)

    if result.confidence < 0.7:
        result2 = _align_via_header_barcode(img)
        if result2.confidence > result.confidence:
            result = result2

    if result.confidence < 0.7:
        result.warnings.append(
            f"Alignment confidence {result.confidence:.2f} < 0.7 — "
            "ROI extraction may be inaccurate. Flag for manual review."
        )

    return result
