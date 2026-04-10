"""
Phase 1 — Image preprocessing pipeline.

Steps applied in order:
1. Load and normalise resolution to 300 DPI equivalent
2. Perspective correction (photographed docs)
3. Deskew via Hough line transform
4. Color-channel separation helpers
5. Adaptive (Sauvola-style) binarization
6. Gentle noise reduction
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    original: np.ndarray          # untouched colour image
    normalised: np.ndarray        # colour image at target DPI scale
    deskewed: np.ndarray          # colour image after rotation correction
    binarized: np.ndarray         # B/W image (uint8 0/255)
    blue_channel: np.ndarray      # blue channel isolated (uint8 greyscale)
    skew_angle_deg: float = 0.0
    perspective_corrected: bool = False
    scale_factor: float = 1.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DPI = 300          # All coordinates are calibrated at this DPI
A4_W_PX = 2480            # A4 @ 300 DPI
A4_H_PX = 3508


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_dpi_scale(img: np.ndarray) -> float:
    """Return scale factor so that after resizing image is ~A4 @ 300 DPI.

    We assume the document fills most of the frame.
    """
    h, w = img.shape[:2]
    # Use the longer dimension to estimate
    scale_h = A4_H_PX / max(h, 1)
    scale_w = A4_W_PX / max(w, 1)
    # Choose the scale that brings both dims closest to A4 target
    return (scale_h + scale_w) / 2


def _resize_to_target(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 0.02:     # don't resize if already within 2%
        return img
    new_w = int(round(img.shape[1] * scale))
    new_h = int(round(img.shape[0] * scale))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


# ---------------------------------------------------------------------------
# 1. Perspective correction
# ---------------------------------------------------------------------------

def _find_document_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """Detect the 4 corners of the document rectangle.

    Returns (4, 2) float32 array in TL, TR, BR, BL order, or None.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # Must cover at least 30% of the image area
            h, w = img.shape[:2]
            if cv2.contourArea(approx) > 0.30 * h * w:
                pts = approx.reshape(4, 2).astype(np.float32)
                return _order_corners(pts)
    return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def correct_perspective(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Apply perspective correction if 4 document corners are found."""
    corners = _find_document_corners(img)
    if corners is None:
        return img, False

    tl, tr, br, bl = corners
    width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl)),
    )
    height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl)),
    )
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, True


# ---------------------------------------------------------------------------
# 2. Deskew
# ---------------------------------------------------------------------------

def _detect_skew_angle(grey: np.ndarray) -> float:
    """Estimate document skew using Hough lines on a binarized edge image.

    Returns angle in degrees; positive = counter-clockwise tilt.
    """
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold=100)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert to degrees relative to horizontal
        angle = math.degrees(theta) - 90
        # Only keep near-horizontal lines (within ±20°)
        if -20 <= angle <= 20:
            angles.append(angle)

    if not angles:
        return 0.0

    # Use the median to avoid outlier influence
    return float(np.median(angles))


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate image to correct skew. Returns (corrected_img, angle_deg)."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    angle = _detect_skew_angle(grey)

    if abs(angle) < 0.3:   # skip trivial rotation
        return img, angle

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


# ---------------------------------------------------------------------------
# 3. Color channel separation
# ---------------------------------------------------------------------------

def extract_blue_channel(img: np.ndarray) -> np.ndarray:
    """Return the blue channel as a grayscale uint8 image.

    Handwriting and business stamps in Tunisian lettres de change are blue ink.
    Isolating the blue channel suppresses the red/pink pre-printed header band
    and black printed text, making handwriting stand out.
    """
    if img.ndim == 2:
        return img  # already grayscale
    # OpenCV stores as BGR
    return img[:, :, 0]  # index 0 = Blue channel


def suppress_red_channel(img: np.ndarray) -> np.ndarray:
    """Zero out the red channel to suppress the pre-printed form lines/header."""
    out = img.copy()
    out[:, :, 2] = 0  # index 2 = Red channel in BGR
    return out


# ---------------------------------------------------------------------------
# 4. Adaptive binarization (Sauvola approximation via OpenCV)
# ---------------------------------------------------------------------------

def adaptive_binarize(grey: np.ndarray, block_size: int = 31, C: int = 15) -> np.ndarray:
    """Sauvola-style adaptive binarization using OpenCV adaptiveThreshold.

    block_size: neighbourhood window (must be odd, 15-51 typical for 300 DPI)
    C: constant subtracted from the mean (higher = more text preserved)
    """
    # Mild blur to reduce salt-and-pepper noise before threshold
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C,
    )
    return binary


# ---------------------------------------------------------------------------
# 5. Noise reduction (gentle — preserve thin handwritten strokes)
# ---------------------------------------------------------------------------

def gentle_denoise(img: np.ndarray) -> np.ndarray:
    """Very mild Gaussian blur (σ=0.5) before binarization.

    Aggressive denoising destroys thin handwritten strokes.
    """
    return cv2.GaussianBlur(img, (3, 3), sigmaX=0.5, sigmaY=0.5)


# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess(image_path: str | Path) -> PreprocessResult:
    """Full preprocessing pipeline for a lettre de change scan.

    Returns a PreprocessResult with all intermediate images preserved.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    warnings: list[str] = []

    # --- Step 1: Normalise resolution ---
    scale = _estimate_dpi_scale(img)
    normalised = _resize_to_target(img, scale)

    # --- Step 2: Perspective correction (optional) ---
    corrected, persp_applied = correct_perspective(normalised)
    if not persp_applied:
        warnings.append("Perspective correction skipped — document corners not detected.")

    # --- Step 3: Deskew ---
    deskewed, skew_angle = deskew(corrected)
    if abs(skew_angle) > 5:
        warnings.append(f"Large skew detected: {skew_angle:.1f}°. Check alignment quality.")

    # --- Step 4: Blue channel (for handwriting isolation) ---
    blue_ch = extract_blue_channel(deskewed)

    # --- Step 5: Binarize (full greyscale, not just blue channel) ---
    grey = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    denoised_grey = gentle_denoise(grey)
    binarized = adaptive_binarize(denoised_grey)

    return PreprocessResult(
        original=img,
        normalised=normalised,
        deskewed=deskewed,
        binarized=binarized,
        blue_channel=blue_ch,
        skew_angle_deg=skew_angle,
        perspective_corrected=persp_applied,
        scale_factor=scale,
        warnings=warnings,
    )
