"""
Phase 5 — Production pipeline: single public entry-point.

This module is the top-level API for the Tunisian Lettre de Change OCR system.
It wraps the full five-phase pipeline into one function call and provides the
structured JSON output schema defined in the project blueprint.

Usage
-----
from ocr_pipeline.pipeline import process_document, ProcessingConfig

result = process_document("path/to/scan.jpg")
print(result.to_json())

# With Qwen OCR + Pass 3 anomaly explanation:
cfg = ProcessingConfig(skip_qwen=False, skip_pass3=False)
result = process_document("path/to/scan.jpg", config=cfg)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np

from .preprocessing import preprocess
from .alignment import align
from .roi_extractor import ROIExtractor
from .ocr_engine import run_ocr, OCRBatch
from .document_result import build_document_result, DocumentResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProcessingConfig:
    """Runtime configuration for the OCR pipeline.

    All boolean flags default to the conservative/safe values suitable for
    production without a live LLM server.
    """
    # OCR pass settings
    skip_qwen: bool = True
    """Skip Qwen3 VL 8B OCR pass (Pass 2).  True = Tesseract-only mode."""

    skip_pass3: bool = True
    """Skip Pass 3 anomaly explanation LLM call.  True = no LLM call."""

    # Stamp detection
    run_stamp_detection: bool = True
    """Run Phase 4 stamp detection on the aligned document image."""

    # Output
    save_crops: bool = False
    """Save each ROI crop to disk in the output directory."""

    output_dir: Optional[Path] = None
    """Directory for saving crops and overlays.  None = do not save."""

    # Performance
    tesseract_timeout: int = 30
    """Per-field Tesseract timeout in seconds."""


# ---------------------------------------------------------------------------
# Pipeline result wrapper
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full result of a single document processing run."""
    doc_result: DocumentResult
    processing_time_s: float
    aligned_image: Optional[np.ndarray] = None
    crops: Optional[dict] = None
    warnings: list[str] = dc_field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialise the DocumentResult to a JSON string."""
        from .document_result import document_result_to_json
        return document_result_to_json(self.doc_result, indent=indent)

    def to_dict(self) -> dict:
        """Convert the DocumentResult to a plain dict."""
        from .document_result import document_result_to_dict
        return document_result_to_dict(self.doc_result)

    @property
    def document_id(self) -> str:
        return self.doc_result.document_id

    @property
    def needs_human_review(self) -> bool:
        return self.doc_result.needs_human_review

    @property
    def document_confidence(self) -> float:
        return self.doc_result.document_confidence


# ---------------------------------------------------------------------------
# Module-level extractor (initialised once per process)
# ---------------------------------------------------------------------------

_extractor: Optional[ROIExtractor] = None


def _get_extractor() -> ROIExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ROIExtractor()
    return _extractor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(
    image_path: str | Path,
    doc_id: Optional[str] = None,
    config: Optional[ProcessingConfig] = None,
) -> PipelineResult:
    """Process a single Lettre de Change scan through the full OCR pipeline.

    Stages executed (in order):
      1. Preprocessing (DPI normalisation, deskew, binarisation)
      2. Alignment (header band + barcode anchor detection)
      3. ROI extraction (17 field crops)
      4. OCR engine (Tesseract + optional Qwen routing)
      5. Field parsing (RIB, date, amount, name parsers)
      6. Cross-field validation (9 rules, confidence scoring)
      7. Stamp detection (Phase 4 — always runs if config.run_stamp_detection)
      8. Pass 3 anomaly explanation (Phase 4 — optional LLM call)

    Args:
        image_path:  Path to the input JPEG/PNG scan.
        doc_id:      Document identifier.  If None, derived from the filename.
        config:      Processing configuration.  Uses safe defaults if None.

    Returns:
        PipelineResult containing the DocumentResult, timing, and aligned image.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError:        If the image cannot be loaded.
    """
    cfg   = config or ProcessingConfig()
    path  = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if doc_id is None:
        doc_id = path.stem

    start = time.perf_counter()
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Phase 1: Preprocessing
    # ------------------------------------------------------------------
    prep = preprocess(path)
    warnings.extend(prep.warnings)

    # ------------------------------------------------------------------
    # Phase 1: Alignment
    # ------------------------------------------------------------------
    aln = align(prep.deskewed)
    if aln.confidence < 0.7:
        warnings.append(
            f"Low alignment confidence ({aln.confidence:.2f}) — "
            f"ROI positions may be inaccurate."
        )

    # ------------------------------------------------------------------
    # Phase 1: ROI extraction
    # ------------------------------------------------------------------
    extractor = _get_extractor()
    crops     = extractor.extract_all(aln.image)

    # ------------------------------------------------------------------
    # Phase 2: OCR engine
    # ------------------------------------------------------------------
    ocr_results = run_ocr(crops, skip_qwen=cfg.skip_qwen)
    batch       = OCRBatch(doc_id=doc_id, fields=ocr_results)

    # ------------------------------------------------------------------
    # Phases 3 & 4: Parse + Validate + Stamp detection + Pass 3
    # ------------------------------------------------------------------
    doc_result = build_document_result(
        doc_id,
        batch,
        document_image=aln.image if cfg.run_stamp_detection else None,
        skip_pass3=cfg.skip_pass3,
    )

    elapsed = time.perf_counter() - start

    # ------------------------------------------------------------------
    # Optional: save crops to disk
    # ------------------------------------------------------------------
    if cfg.save_crops and cfg.output_dir:
        _save_crops(crops, cfg.output_dir / doc_id, extractor)

    return PipelineResult(
        doc_result=doc_result,
        processing_time_s=round(elapsed, 3),
        aligned_image=aln.image,
        crops=crops,
        warnings=warnings,
    )


def process_batch(
    image_paths: list[str | Path],
    config: Optional[ProcessingConfig] = None,
    on_progress: Optional[callable] = None,
) -> list[PipelineResult]:
    """Process a list of document scans sequentially.

    Args:
        image_paths:  List of paths to process.
        config:       Shared configuration for all documents.
        on_progress:  Optional callback called after each document with
                      (index, total, doc_id, result).

    Returns:
        List of PipelineResult, one per input path.  Results for documents
        that raised exceptions have ``doc_result.needs_human_review=True``
        and a warning describing the error.
    """
    cfg     = config or ProcessingConfig()
    results = []

    for i, path in enumerate(image_paths):
        path = Path(path)
        try:
            result = process_document(path, config=cfg)
        except Exception as exc:
            # Build a minimal error result so the batch never stops
            error_result = _make_error_result(path.stem, str(exc))
            result = PipelineResult(
                doc_result=error_result,
                processing_time_s=0.0,
                warnings=[f"Processing failed: {exc}"],
            )
        results.append(result)
        if on_progress:
            on_progress(i, len(image_paths), path.stem, result)

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_crops(crops: dict, out_dir: Path, extractor: ROIExtractor) -> None:
    """Save all ROI crops to PNG files in out_dir."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    for roi_id, crop in crops.items():
        cv2.imwrite(str(out_dir / f"{roi_id}_colour.png"), crop.colour)
        cv2.imwrite(str(out_dir / f"{roi_id}_binarised.png"), crop.binarised)


def _make_error_result(doc_id: str, error_msg: str) -> DocumentResult:
    """Return a minimal DocumentResult indicating an unrecoverable error."""
    from datetime import datetime
    from .document_result import (
        DocumentResult, QwenCorrections, FlaggedField, StampInfo
    )
    from .validator import ErrorType
    return DocumentResult(
        document_id=doc_id,
        payment_order_number=None,
        extraction_timestamp=datetime.utcnow().isoformat() + "Z",
        echeance=None,
        creation_date=None,
        creation_city=None,
        rib=None,
        amount=None,
        tireur=None,
        beneficiary=None,
        tire=None,
        domiciliation=None,
        validation={},
        document_confidence=0.0,
        needs_human_review=True,
        qwen_corrections=QwenCorrections(),
        flagged_fields=[
            FlaggedField(
                field="pipeline",
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Pipeline exception: {error_msg}",
                is_hard_failure=True,
            )
        ],
        stamp_info=StampInfo(),
        anomaly_explanation="",
    )
