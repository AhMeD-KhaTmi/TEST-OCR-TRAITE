"""
Phase 5 — Audit logger: structured logging and audit trail.

Every call to process_document() can be recorded to an audit log so that:
  - Each extraction can be traced back to its inputs
  - All intermediate values (raw OCR, parsed values, confidence) are preserved
  - Human review corrections can be stored alongside original extraction
  - System-level metrics (throughput, error rate, confidence distribution) can
    be computed from the log

Log format: one JSON-Lines (JSONL) record per document, appended to a
persistent log file. Each record is self-contained and human-readable.

Usage
-----
from ocr_pipeline.audit_logger import AuditLogger

logger = AuditLogger("output/audit.jsonl")
logger.log(pipeline_result, image_path="scans/doc001.jpg")

# Later: read back all records
for record in logger.iter_records():
    print(record["doc_id"], record["document_confidence"])
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Optional

from .pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _serialise(obj):
    """JSON default handler: converts Decimal, Path, ndarray, etc."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape}>"
    except ImportError:
        pass
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _build_record(
    result: PipelineResult,
    image_path: Optional[str | Path] = None,
    operator_id: Optional[str] = None,
) -> dict:
    """Build one JSONL audit record from a PipelineResult."""
    dr = result.doc_result
    record = {
        "schema_version": "1.0",
        "logged_at": datetime.utcnow().isoformat() + "Z",
        "doc_id": dr.document_id,
        "image_path": str(image_path) if image_path else None,
        "operator_id": operator_id,
        "extraction_timestamp": dr.extraction_timestamp,

        # Pipeline performance
        "processing_time_s": result.processing_time_s,
        "warnings": result.warnings,

        # Outcome summary
        "document_confidence": dr.document_confidence,
        "needs_human_review": dr.needs_human_review,

        # Stamp info
        "stamp_count": dr.stamp_info.stamp_count,
        "stamp_affected_rois": dr.stamp_info.affected_rois,

        # Qwen correction stats
        "qwen_source": dr.qwen_corrections.source,
        "qwen_rib_digits_changed": dr.qwen_corrections.rib_digits_changed,
        "qwen_amount_digits_changed": dr.qwen_corrections.amount_digits_changed,
        "qwen_date_digits_changed": dr.qwen_corrections.date_digits_changed,

        # Key extracted values
        "payment_order_number": dr.payment_order_number,

        "rib": {
            "full": dr.rib.full if dr.rib else None,
            "key_valid": dr.rib.key_valid if dr.rib else None,
            "bank_name": dr.rib.bank_name if dr.rib else None,
            "confidence": dr.rib.confidence if dr.rib else None,
            "source": dr.rib.source if dr.rib else None,
        },

        "amount": {
            "value_numeric": str(dr.amount.value_numeric) if dr.amount else None,
            "value_text": dr.amount.value_text if dr.amount else None,
            "numeric_text_match": dr.amount.numeric_text_match if dr.amount else None,
            "confidence": dr.amount.confidence if dr.amount else None,
        },

        "echeance": {
            "value": dr.echeance.value if dr.echeance else None,
            "confidence": dr.echeance.confidence if dr.echeance else None,
        },

        "creation_date": {
            "value": dr.creation_date.value if dr.creation_date else None,
            "confidence": dr.creation_date.confidence if dr.creation_date else None,
        },

        # Validation results
        "validation": dr.validation,

        # Flags
        "flagged_fields": [
            {
                "field": f.field,
                "error_type": f.error_type,
                "message": f.message,
                "is_hard_failure": f.is_hard_failure,
            }
            for f in dr.flagged_fields
        ],
        "hard_failure_count": sum(1 for f in dr.flagged_fields if f.is_hard_failure),
        "soft_warning_count": sum(1 for f in dr.flagged_fields if not f.is_hard_failure),

        # Anomaly explanation (Pass 3)
        "anomaly_explanation": dr.anomaly_explanation or None,
    }
    return record


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Thread-safe append-only JSONL audit logger.

    Each ``log()`` call appends one JSON-Lines record to the log file.
    Records are flushed immediately.  Multiple processes must NOT write to
    the same log file concurrently (use dated shards instead).

    Args:
        log_path:    Path to the JSONL log file (created if absent).
        auto_rotate: If True, daily rotation is applied: a date suffix is
                     appended before the extension (e.g. audit_2026-04-11.jsonl).
    """

    def __init__(
        self,
        log_path: str | Path,
        auto_rotate: bool = False,
    ) -> None:
        self._base_path = Path(log_path)
        self._auto_rotate = auto_rotate
        self._lock = threading.Lock()
        self._base_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _log_path(self) -> Path:
        if not self._auto_rotate:
            return self._base_path
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self._base_path.with_stem(f"{self._base_path.stem}_{today}")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def log(
        self,
        result: PipelineResult,
        image_path: Optional[str | Path] = None,
        operator_id: Optional[str] = None,
    ) -> None:
        """Append one audit record to the log file.

        Args:
            result:       PipelineResult from process_document().
            image_path:   Original scan path (for traceability).
            operator_id:  Optional identifier for the operator or system
                          that triggered this extraction.
        """
        record = _build_record(result, image_path=image_path, operator_id=operator_id)
        line = json.dumps(record, ensure_ascii=False, default=_serialise)
        path = self._log_path()
        with self._lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    def iter_records(self, date: Optional[str] = None) -> Iterator[dict]:
        """Iterate over all records in the log file (most recent log by default).

        Args:
            date: If not None, read a rotated shard for this date
                  (YYYY-MM-DD format).  Only relevant when auto_rotate=True.

        Yields:
            Parsed record dicts.
        """
        if date and self._auto_rotate:
            path = self._base_path.with_stem(f"{self._base_path.stem}_{date}")
        else:
            path = self._log_path()

        if not path.exists():
            return

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def get_summary_stats(self) -> dict:
        """Compute aggregate statistics over all records in the current log.

        Returns:
            dict with keys: total_docs, error_rate, avg_confidence,
            review_rate, stamp_rate, avg_processing_time_s.
        """
        total             = 0
        needs_review      = 0
        hard_failures     = 0
        conf_sum          = 0.0
        stamp_docs        = 0
        time_sum          = 0.0

        for rec in self.iter_records():
            total += 1
            if rec.get("needs_human_review"):
                needs_review += 1
            hard_failures += rec.get("hard_failure_count", 0)
            conf_sum      += rec.get("document_confidence", 0.0)
            if rec.get("stamp_count", 0) > 0:
                stamp_docs += 1
            time_sum      += rec.get("processing_time_s", 0.0)

        if total == 0:
            return {"total_docs": 0}

        return {
            "total_docs": total,
            "review_rate": round(needs_review / total, 4),
            "hard_failure_rate": round(hard_failures / total, 4),
            "avg_confidence": round(conf_sum / total, 4),
            "stamp_detection_rate": round(stamp_docs / total, 4),
            "avg_processing_time_s": round(time_sum / total, 3),
        }
