"""
Phase 2 — Barcode decoder.

Implements two strategies for reading the barcode / CMC-7 line at R17:
  1. pyzbar (primary): decode 1D/2D barcodes from the crop image
  2. Tesseract OCR-B (fallback): OCR the numeric text printed below/in the barcode

Both results are returned so the caller can pick the best one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
    _PYZBAR_AVAILABLE = True
except (ImportError, OSError):
    _PYZBAR_AVAILABLE = False

# Import Tesseract fallback from the same package
from .tesseract_ocr import run_tesseract, TESSERACT_AVAILABLE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BarcodeResult:
    barcode_text: str               # raw barcode data (pyzbar decode)
    ocr_text: str                   # OCR-B fallback text
    best_text: str                  # resolved best value
    source: str                     # "pyzbar" | "tesseract_ocrb" | "none"
    confidence: float               # 0.0–1.0
    pyzbar_available: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# pyzbar strategy
# ---------------------------------------------------------------------------

def _preprocess_for_barcode(crop: np.ndarray) -> list[np.ndarray]:
    """Return a list of crop variants optimised for barcode scanning."""
    variants = [crop]

    # Greyscale
    if crop.ndim == 3:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        grey = crop

    # High-contrast binary
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)

    # Inverted binary (some decoders prefer dark-on-white)
    variants.append(cv2.bitwise_not(binary))

    # Scaled up ×2 (improves decode for small/blurry areas)
    upscaled = cv2.resize(grey, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)

    return variants


def _try_pyzbar(crop: np.ndarray) -> Optional[str]:
    """Try pyzbar on multiple preprocessing variants. Return decoded text or None."""
    if not _PYZBAR_AVAILABLE:
        return None

    for variant in _preprocess_for_barcode(crop):
        try:
            codes = pyzbar.decode(variant)
            for code in codes:
                text = code.data.decode("utf-8", errors="replace").strip()
                if text:
                    return text
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Tesseract OCR-B fallback
# ---------------------------------------------------------------------------

_OCRB_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789 "


def _try_tesseract_ocrb(crop: np.ndarray) -> tuple[str, float]:
    """OCR the barcode area with Tesseract using digit-only whitelist.

    Returns (text, confidence).
    """
    if not TESSERACT_AVAILABLE:
        return "", 0.0

    # Preprocess: greyscale + threshold for clean OCR-B digits
    if crop.ndim == 3:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        grey = crop

    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = run_tesseract(binary, tesseract_config=_OCRB_CONFIG, lang="fra")
    # Strip any non-digit characters
    import re
    digits_only = re.sub(r"[^0-9]", "", result.text)
    return digits_only, result.confidence


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

def decode_barcode(crop: np.ndarray) -> BarcodeResult:
    """Decode the barcode / CMC-7 number from the R17 crop.

    Strategy:
      1. Try pyzbar on multiple image variants
      2. Fallback to Tesseract OCR-B digit extraction
      3. Pick the best result based on digit count and format

    Returns a BarcodeResult describing what was found.
    """
    errors: list[str] = []

    # --- Strategy 1: pyzbar ---
    barcode_text = ""
    if _PYZBAR_AVAILABLE:
        try:
            barcode_text = _try_pyzbar(crop) or ""
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pyzbar: {exc}")
    else:
        errors.append("pyzbar not available")

    # --- Strategy 2: Tesseract OCR-B ---
    ocr_text, ocr_conf = _try_tesseract_ocrb(crop)

    # --- Resolution ---
    if barcode_text:
        best = barcode_text
        source = "pyzbar"
        confidence = 1.0
    elif ocr_text:
        best = ocr_text
        source = "tesseract_ocrb"
        confidence = ocr_conf
    else:
        best = ""
        source = "none"
        confidence = 0.0

    return BarcodeResult(
        barcode_text=barcode_text,
        ocr_text=ocr_text,
        best_text=best,
        source=source,
        confidence=confidence,
        pyzbar_available=_PYZBAR_AVAILABLE,
        error="; ".join(errors) if errors else None,
    )
