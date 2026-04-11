"""Tests for audit_logger.py — Phase 5."""
import json
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.audit_logger import AuditLogger, _build_record, _serialise
from decimal import Decimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pipeline_result(
    doc_id="doc001",
    confidence=0.85,
    needs_review=False,
    processing_time=1.23,
):
    from ocr_pipeline.document_result import (
        DocumentResult, QwenCorrections, StampInfo
    )
    from ocr_pipeline.pipeline import PipelineResult
    dr = DocumentResult(
        document_id=doc_id,
        payment_order_number="001",
        extraction_timestamp="2026-04-11T00:00:00Z",
        echeance=None, creation_date=None, creation_city=None,
        rib=None, amount=None, tireur=None, beneficiary=None,
        tire=None, domiciliation=None,
        validation={"rib_key_valid": True},
        document_confidence=confidence,
        needs_human_review=needs_review,
        qwen_corrections=QwenCorrections(source="tesseract_only"),
        flagged_fields=[],
        stamp_info=StampInfo(stamp_count=2),
        anomaly_explanation="Stamp covers upper RIB.",
    )
    return PipelineResult(
        doc_result=dr,
        processing_time_s=processing_time,
        warnings=["low alignment confidence"],
    )


# ---------------------------------------------------------------------------
# _serialise
# ---------------------------------------------------------------------------

class TestSerialise:
    def test_decimal_becomes_string(self):
        assert _serialise(Decimal("3000.000")) == "3000.000"

    def test_path_becomes_string(self):
        p = Path("/some/path.json")
        assert _serialise(p) == str(p)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            _serialise(object())


# ---------------------------------------------------------------------------
# _build_record
# ---------------------------------------------------------------------------

class TestBuildRecord:
    def test_returns_dict(self):
        r = _mock_pipeline_result()
        rec = _build_record(r)
        assert isinstance(rec, dict)

    def test_doc_id_set(self):
        r = _mock_pipeline_result(doc_id="doc42")
        rec = _build_record(r)
        assert rec["doc_id"] == "doc42"

    def test_schema_version_set(self):
        r = _mock_pipeline_result()
        rec = _build_record(r)
        assert rec["schema_version"] == "1.0"

    def test_confidence_set(self):
        r = _mock_pipeline_result(confidence=0.77)
        rec = _build_record(r)
        assert rec["document_confidence"] == 0.77

    def test_stamp_count_set(self):
        r = _mock_pipeline_result()
        rec = _build_record(r)
        assert rec["stamp_count"] == 2

    def test_anomaly_explanation_set(self):
        r = _mock_pipeline_result()
        rec = _build_record(r)
        assert rec["anomaly_explanation"] == "Stamp covers upper RIB."

    def test_warnings_set(self):
        r = _mock_pipeline_result()
        rec = _build_record(r)
        assert "low alignment confidence" in rec["warnings"]

    def test_image_path_stored(self):
        r = _mock_pipeline_result()
        rec = _build_record(r, image_path="/scans/doc001.jpg")
        assert rec["image_path"] == "/scans/doc001.jpg"

    def test_operator_id_stored(self):
        r = _mock_pipeline_result()
        rec = _build_record(r, operator_id="user_01")
        assert rec["operator_id"] == "user_01"

    def test_processing_time_set(self):
        r = _mock_pipeline_result(processing_time=2.5)
        rec = _build_record(r)
        assert rec["processing_time_s"] == 2.5


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def test_log_creates_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        result = _mock_pipeline_result()
        logger.log(result)
        assert log_path.exists()

    def test_log_appends_multiple_records(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        for i in range(3):
            logger.log(_mock_pipeline_result(doc_id=f"doc{i:03d}"))
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        for i in range(3):
            logger.log(_mock_pipeline_result(doc_id=f"doc{i:03d}"))
        for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
            rec = json.loads(line)
            assert "doc_id" in rec

    def test_iter_records_yields_all(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        for i in range(5):
            logger.log(_mock_pipeline_result(doc_id=f"doc{i:03d}"))
        records = list(logger.iter_records())
        assert len(records) == 5

    def test_iter_records_empty_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        records = list(logger.iter_records())
        assert records == []

    def test_get_summary_stats_empty(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        stats = logger.get_summary_stats()
        assert stats["total_docs"] == 0

    def test_get_summary_stats_values(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log(_mock_pipeline_result(confidence=0.90, needs_review=False))
        logger.log(_mock_pipeline_result(confidence=0.30, needs_review=True))
        stats = logger.get_summary_stats()
        assert stats["total_docs"] == 2
        assert stats["review_rate"] == 0.5
        assert abs(stats["avg_confidence"] - 0.6) < 0.01

    def test_thread_safe_concurrent_writes(self, tmp_path):
        """Multiple threads logging simultaneously should not corrupt the file."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        threads = []
        for i in range(20):
            t = threading.Thread(
                target=logger.log,
                args=(_mock_pipeline_result(doc_id=f"doc{i:03d}"),),
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records = list(logger.iter_records())
        assert len(records) == 20

    def test_auto_rotate_uses_date_suffix(self, tmp_path):
        from datetime import datetime
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path, auto_rotate=True)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        rotated = logger._log_path()
        assert today in rotated.name

    def test_log_writes_valid_utf8(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)
        logger.log(_mock_pipeline_result(), image_path="/scans/résumé.jpg")
        content = log_path.read_bytes().decode("utf-8")
        assert "résumé" in content
