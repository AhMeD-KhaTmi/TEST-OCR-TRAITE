"""
Phase 2 — Pass 1: Tesseract OCR engine.

Wraps pytesseract with:
- Automatic Tesseract binary discovery on Windows
- Field-specific PSM modes and character whitelists (digit, date, text)
- Digit confusion correction (O→0, l→1, S→5, I→1)
- Confidence extraction from HOCR / image_to_data
- Graceful degradation when Tesseract is unavailable
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tesseract binary discovery
# ---------------------------------------------------------------------------

_TESSERACT_SEARCH_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\THOURAYA\AppData\Local\Tesseract-OCR\tesseract.exe",
]


def _find_tesseract() -> Optional[str]:
    """Return path to tesseract.exe or None if not found."""
    # 1. Check PATH first
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    # 2. Known install locations (Windows)
    for p in _TESSERACT_SEARCH_PATHS:
        if Path(p).exists():
            return p
    return None


def _configure_tesseract() -> bool:
    """Set pytesseract.tesseract_cmd if found. Return True if available."""
    if not _PYTESSERACT_AVAILABLE:
        return False
    tess_path = _find_tesseract()
    if tess_path is None:
        return False
    pytesseract.pytesseract.tesseract_cmd = tess_path
    return True


TESSERACT_AVAILABLE: bool = _configure_tesseract()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TesseractResult:
    text: str                 # cleaned extracted text
    raw_text: str             # raw pytesseract output before cleaning
    confidence: float         # 0.0–1.0 (mean word confidence from data)
    engine_available: bool    # False if Tesseract not installed
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Digit confusion correction
# ---------------------------------------------------------------------------

_DIGIT_CONFUSION = str.maketrans({
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "|": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
})

_DATE_CONFUSION = str.maketrans({
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
})


def _apply_digit_confusion(text: str) -> str:
    """Fix common OCR digit/letter confusion for numeric fields."""
    return text.translate(_DIGIT_CONFUSION)


def _apply_date_confusion(text: str) -> str:
    """Fix digit confusion for date fields (keep / - . separators)."""
    return text.translate(_DATE_CONFUSION)


def _normalize_whitespace(text: str) -> str:
    """Strip and collapse multiple spaces/newlines to single space."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Core OCR functions
# ---------------------------------------------------------------------------

def _image_to_data_confidence(crop: np.ndarray, config: str) -> tuple[str, float]:
    """Run pytesseract.image_to_data and return (text, mean_word_confidence)."""
    data = pytesseract.image_to_data(
        crop,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words, confs = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        if word and conf != -1:
            words.append(word)
            confs.append(int(conf))

    text = " ".join(words)
    confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return text, confidence


def run_tesseract(
    crop: np.ndarray,
    tesseract_config: str = "--psm 7",
    lang: str = "fra",
) -> TesseractResult:
    """Run Tesseract on a single crop image.

    Args:
        crop: NumPy BGR or grayscale image (ROI crop)
        tesseract_config: Tesseract CLI flags (PSM mode, char whitelist, etc.)
        lang: Tesseract language. 'fra' for French; 'fra+eng' for mixed.

    Returns:
        TesseractResult with text, confidence, and diagnostics.
    """
    if not TESSERACT_AVAILABLE:
        return TesseractResult(
            text="",
            raw_text="",
            confidence=0.0,
            engine_available=False,
            error="Tesseract not installed or not found.",
        )

    # Ensure grayscale for OCR (colour adds no value and slows it down)
    if crop.ndim == 3:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        grey = crop

    config_str = f"--oem 3 {tesseract_config} -l {lang}"

    try:
        raw_text, confidence = _image_to_data_confidence(grey, config_str)
    except Exception as exc:  # noqa: BLE001
        return TesseractResult(
            text="",
            raw_text="",
            confidence=0.0,
            engine_available=True,
            error=str(exc),
        )

    return TesseractResult(
        text=_normalize_whitespace(raw_text),
        raw_text=raw_text,
        confidence=confidence,
        engine_available=True,
    )


# ---------------------------------------------------------------------------
# Specialised entry points (called by ocr_engine.py)
# ---------------------------------------------------------------------------

def ocr_digits(crop: np.ndarray, config: str = "--psm 7 -c tessedit_char_whitelist=0123456789") -> TesseractResult:
    """OCR for pure digit fields (RIB, payment order, barcode text)."""
    result = run_tesseract(crop, tesseract_config=config, lang="fra")
    # Apply digit confusion correction
    result.text = _apply_digit_confusion(result.text)
    result.text = re.sub(r"[^0-9]", "", result.text)  # strip non-digits
    return result


def ocr_amounts(crop: np.ndarray) -> TesseractResult:
    """OCR for amount fields: digits + ,. separators + # hash delimiters."""
    config = "--psm 7 -c tessedit_char_whitelist=0123456789,."
    result = run_tesseract(crop, tesseract_config=config, lang="fra")
    result.text = _apply_digit_confusion(result.text)
    # Clean: keep only digits, commas, dots
    result.text = re.sub(r"[^0-9,.]", "", result.text)
    return result


def ocr_date(crop: np.ndarray) -> TesseractResult:
    """OCR for date fields: digits + / - . separators."""
    config = "--psm 7 -c tessedit_char_whitelist=0123456789/-.  "
    result = run_tesseract(crop, tesseract_config=config, lang="fra")
    result.text = _apply_date_confusion(result.text)
    return result


def ocr_text(crop: np.ndarray) -> TesseractResult:
    """OCR for general text fields (names, city) — full French character set."""
    config = "--psm 6"
    result = run_tesseract(crop, tesseract_config=config, lang="fra")
    return result


def ocr_from_roi(crop: np.ndarray, tesseract_config: str) -> TesseractResult:
    """Generic dispatcher: runs Tesseract with the exact config from roi_config.json."""
    result = run_tesseract(crop, tesseract_config=tesseract_config, lang="fra")

    # Auto-correct based on whitelist
    if "tessedit_char_whitelist=0123456789" in tesseract_config:
        if "," in tesseract_config or "." in tesseract_config:
            result.text = _apply_digit_confusion(result.text)
            result.text = re.sub(r"[^0-9,.]", "", result.text)
        else:
            result.text = _apply_digit_confusion(result.text)
            result.text = re.sub(r"[^0-9]", "", result.text)

    return result
