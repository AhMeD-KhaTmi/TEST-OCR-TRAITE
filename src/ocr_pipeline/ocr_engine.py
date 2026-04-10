"""
Phase 2 — OCR engine orchestrator.

Routes each ROI to the correct OCR strategy:
  - tesseract        → Pass 1 only (fast path for clean printed fields)
  - tesseract_then_qwen → Pass 1; escalate to Pass 2 if confidence < threshold
  - qwen             → Pass 2 directly (handwriting, semantic fields)
  - barcode_decoder  → pyzbar + OCR-B fallback

Implements the Pass 1 + Pass 2 merge algorithm from plan section 3.4:
  1. If Tess and Qwen agree → high confidence, use value
  2. If they disagree → use Qwen but mark medium confidence
  3. If Qwen contains '?' chars → merge: Tess digits where readable, keep '?'
  4. If Qwen changes > 30% of Tess digits → reject Qwen, keep Tess + flag
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from .roi_extractor import ROICrop
from .tesseract_ocr import ocr_from_roi, TesseractResult, TESSERACT_AVAILABLE
from .qwen_ocr import run_qwen, QwenResult, qwen_diverges_too_much
from .barcode_decoder import decode_barcode, BarcodeResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum Tesseract confidence to bypass Qwen for tesseract_then_qwen fields
TESS_CONFIDENCE_GATE: float = 0.80

# Digit-divergence threshold
DIGIT_DIVERGENCE_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OCRFieldResult:
    roi_id: str
    field_name: str
    text: str                          # final resolved text
    confidence: float                  # 0.0–1.0

    source: str = "none"               # "tesseract" | "qwen" | "merged" | "barcode" | "none"
    tess_result: Optional[TesseractResult] = None
    qwen_result: Optional[QwenResult] = None
    barcode_result: Optional[BarcodeResult] = None

    tess_text: str = ""                # raw Tesseract output
    qwen_text: str = ""                # raw Qwen output

    readable: bool = True
    qwen_digits_changed: int = 0       # for audit log (qwen_corrections in JSON)
    tess_digits_total: int = 0

    flags: list[str] = dc_field(default_factory=list)
    error: Optional[str] = None


@dataclass
class OCRBatch:
    """All OCR results for one document."""
    doc_id: str
    fields: dict[str, OCRFieldResult] = dc_field(default_factory=dict)

    @property
    def qwen_call_count(self) -> int:
        return sum(
            1 for r in self.fields.values()
            if r.qwen_result is not None and r.qwen_result.engine_available
        )


# ---------------------------------------------------------------------------
# Text comparison helpers
# ---------------------------------------------------------------------------

def _texts_agree(a: str, b: str) -> bool:
    """Return True if two OCR texts agree after normalisation."""
    a_norm = re.sub(r"\s+", "", a).upper()
    b_norm = re.sub(r"\s+", "", b).upper()
    return a_norm == b_norm and len(a_norm) > 0


def _merge_question_marks(tess: str, qwen: str) -> str:
    """Where Qwen returns '?', substitute the corresponding Tesseract digit.

    Both strings are treated as sequences of characters (not digit-only).
    Where Qwen has '?' and Tesseract has a digit, Tesseract wins.
    """
    tess_ch = list(tess)
    qwen_ch = list(qwen)
    merged = []
    t_idx = 0
    for q_ch in qwen_ch:
        if q_ch == "?" and t_idx < len(tess_ch) and tess_ch[t_idx].isdigit():
            merged.append(tess_ch[t_idx])
        else:
            merged.append(q_ch)
        t_idx += 1
    return "".join(merged)


def _count_question_marks(text: str) -> int:
    return text.count("?")


# ---------------------------------------------------------------------------
# Merge logic (plan §3.4)
# ---------------------------------------------------------------------------

def _merge_tess_qwen(
    tess: TesseractResult,
    qwen: QwenResult,
    roi_id: str,
) -> tuple[str, float, str, list[str]]:
    """
    Merge Tesseract + Qwen results per the plan's 4-rule algorithm.

    Returns: (final_text, confidence, source, flag_list)
    """
    flags: list[str] = []
    tess_text = tess.text
    qwen_text  = qwen.text if qwen.engine_available else ""

    # Rule 0: Qwen not available → use Tesseract only
    if not qwen.engine_available:
        flags.append("qwen_unavailable")
        return tess_text, tess.confidence, "tesseract", flags

    # Rule 3: Qwen has '?' → merge immediately (before divergence check)
    # '?' means Qwen explicitly marked characters as unreadable — this is
    # intentional, not hallucination, so the divergence rule must not fire.
    if "?" in qwen_text:
        merged = _merge_question_marks(tess_text, qwen_text) if tess_text else qwen_text
        rem_q = _count_question_marks(merged)
        conf = 0.70 - (rem_q * 0.05)
        if rem_q > 0:
            flags.append(f"partial_unreadable_{rem_q}_chars")
        return merged, max(conf, 0.20), "merged", flags

    # Rule 4: digit-divergence rejection (only applies when no '?' present)
    if qwen_diverges_too_much(tess_text, qwen_text, DIGIT_DIVERGENCE_THRESHOLD):
        changed = len(re.findall(r"\d", tess_text))  # approximate
        flags.append(f"qwen_rejected_divergence")
        return tess_text, tess.confidence * 0.85, "tesseract", flags

    # Rule 1: agreement → high confidence
    if _texts_agree(tess_text, qwen_text):
        conf = 0.4 * 1.0 + 0.3 * 1.0 + 0.2 * tess.confidence + 0.1 * tess.confidence
        return tess_text, min(conf, 1.0), "tesseract", flags

    # Rule 2: disagreement → use Qwen, medium confidence
    flags.append("tess_qwen_disagree")
    conf = 0.4 * 0.3 + 0.3 * 0.5 + 0.2 * tess.confidence + 0.1 * tess.confidence
    return qwen_text, min(conf, 0.75), "qwen", flags


# ---------------------------------------------------------------------------
# Per-ROI dispatchers
# ---------------------------------------------------------------------------

def _run_tesseract_only(crop: ROICrop) -> OCRFieldResult:
    tess = ocr_from_roi(crop.binarized if crop.binarized is not None else crop.colour,
                        crop.tesseract_config)
    return OCRFieldResult(
        roi_id=crop.roi_id,
        field_name=crop.name,
        text=tess.text,
        confidence=tess.confidence,
        source="tesseract",
        tess_result=tess,
        tess_text=tess.text,
        error=tess.error,
        flags=[] if tess.engine_available else ["tesseract_unavailable"],
    )


def _run_qwen_only(crop: ROICrop) -> OCRFieldResult:
    qwen = run_qwen(crop.colour, crop.roi_id)
    return OCRFieldResult(
        roi_id=crop.roi_id,
        field_name=crop.name,
        text=qwen.text,
        confidence=0.70 if qwen.engine_available else 0.0,
        source="qwen" if qwen.engine_available else "none",
        qwen_result=qwen,
        qwen_text=qwen.text,
        readable=qwen.readable,
        error=qwen.error,
        flags=[] if qwen.engine_available else ["qwen_unavailable"],
    )


def _run_tesseract_then_qwen(crop: ROICrop) -> OCRFieldResult:
    """Pass 1 → escalate to Pass 2 only if Tess confidence < gate."""
    tess = ocr_from_roi(
        crop.binarized if crop.binarized is not None else crop.colour,
        crop.tesseract_config,
    )

    # Fast path: Tesseract alone is good enough
    if tess.confidence >= TESS_CONFIDENCE_GATE and not tess.error:
        return OCRFieldResult(
            roi_id=crop.roi_id,
            field_name=crop.name,
            text=tess.text,
            confidence=tess.confidence,
            source="tesseract",
            tess_result=tess,
            tess_text=tess.text,
            error=tess.error,
        )

    # Escalate to Qwen
    qwen = run_qwen(crop.colour, crop.roi_id)
    final_text, confidence, source, flags = _merge_tess_qwen(tess, qwen, crop.roi_id)

    # Count digit changes for audit
    tess_d = re.findall(r"\d", tess.text)
    qwen_d = re.findall(r"\d", qwen.text)
    changed = 0
    for t, q in zip(tess_d, qwen_d):
        if t != q:
            changed += 1
    changed += abs(len(tess_d) - len(qwen_d))

    return OCRFieldResult(
        roi_id=crop.roi_id,
        field_name=crop.name,
        text=final_text,
        confidence=confidence,
        source=source,
        tess_result=tess,
        qwen_result=qwen,
        tess_text=tess.text,
        qwen_text=qwen.text,
        readable=qwen.readable if qwen.engine_available else True,
        qwen_digits_changed=changed,
        tess_digits_total=len(tess_d),
        flags=flags,
        error=qwen.error if not qwen.engine_available else tess.error,
    )


def _run_barcode(crop: ROICrop) -> OCRFieldResult:
    result = decode_barcode(crop.colour)
    return OCRFieldResult(
        roi_id=crop.roi_id,
        field_name=crop.name,
        text=result.best_text,
        confidence=result.confidence,
        source=f"barcode_{result.source}",
        barcode_result=result,
        error=result.error,
        flags=[] if result.best_text else ["barcode_unreadable"],
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_ocr(
    crops: dict[str, ROICrop],
    skip_qwen: bool = False,
) -> dict[str, OCRFieldResult]:
    """Run OCR on all ROI crops, routing each to the correct engine.

    Args:
        crops:      dict from ROIExtractor.extract_all()
        skip_qwen:  if True, force Tesseract-only for all fields (test/debug mode)

    Returns:
        dict mapping roi_id → OCRFieldResult
    """
    results: dict[str, OCRFieldResult] = {}

    for roi_id, crop in crops.items():
        engine = crop.ocr_engine  # set by roi_config.json

        if engine == "barcode_decoder":
            results[roi_id] = _run_barcode(crop)

        elif engine == "qwen" and not skip_qwen:
            results[roi_id] = _run_qwen_only(crop)

        elif engine == "tesseract_then_qwen" and not skip_qwen:
            results[roi_id] = _run_tesseract_then_qwen(crop)

        else:
            # "tesseract" engines + skip_qwen fallbacks
            results[roi_id] = _run_tesseract_only(crop)

    return results


def run_ocr_batch(
    crops_list: list[dict[str, ROICrop]],
    doc_ids: list[str],
    skip_qwen: bool = False,
) -> list[OCRBatch]:
    """Process multiple documents sequentially.

    Args:
        crops_list: list of dicts (one per document) from ROIExtractor
        doc_ids:    parallel list of document identifiers
        skip_qwen:  force Tesseract-only

    Returns:
        list of OCRBatch objects
    """
    batches = []
    for doc_id, crops in zip(doc_ids, crops_list):
        field_results = run_ocr(crops, skip_qwen=skip_qwen)
        batch = OCRBatch(doc_id=doc_id, fields=field_results)
        batches.append(batch)
    return batches
