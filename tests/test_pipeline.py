"""Tests for pipeline.py — Phase 5."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.pipeline import (
    ProcessingConfig,
    PipelineResult,
    process_batch,
    _make_error_result,
)
from ocr_pipeline.document_result import DocumentResult


# ---------------------------------------------------------------------------
# ProcessingConfig
# ---------------------------------------------------------------------------

class TestProcessingConfig:
    def test_defaults_are_safe(self):
        cfg = ProcessingConfig()
        assert cfg.skip_qwen is True
        assert cfg.skip_pass3 is True
        assert cfg.run_stamp_detection is True
        assert cfg.save_crops is False

    def test_custom_values(self):
        cfg = ProcessingConfig(skip_qwen=False, skip_pass3=False)
        assert cfg.skip_qwen is False
        assert cfg.skip_pass3 is False

    def test_output_dir_default_none(self):
        cfg = ProcessingConfig()
        assert cfg.output_dir is None


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

def _mock_pipeline_result(needs_review=False, confidence=0.85):
    """Build a minimal PipelineResult for testing."""
    from ocr_pipeline.document_result import (
        DocumentResult, QwenCorrections, StampInfo
    )
    dr = DocumentResult(
        document_id="test_doc",
        payment_order_number="123",
        extraction_timestamp="2026-04-11T00:00:00Z",
        echeance=None, creation_date=None, creation_city=None,
        rib=None, amount=None, tireur=None, beneficiary=None,
        tire=None, domiciliation=None,
        validation={},
        document_confidence=confidence,
        needs_human_review=needs_review,
        qwen_corrections=QwenCorrections(),
        flagged_fields=[],
        stamp_info=StampInfo(),
        anomaly_explanation="",
    )
    return PipelineResult(doc_result=dr, processing_time_s=1.5)


class TestPipelineResult:
    def test_document_id_property(self):
        r = _mock_pipeline_result()
        assert r.document_id == "test_doc"

    def test_needs_human_review_true(self):
        r = _mock_pipeline_result(needs_review=True)
        assert r.needs_human_review is True

    def test_needs_human_review_false(self):
        r = _mock_pipeline_result(needs_review=False, confidence=0.9)
        assert r.needs_human_review is False

    def test_document_confidence_property(self):
        r = _mock_pipeline_result(confidence=0.77)
        assert r.document_confidence == 0.77

    def test_to_json_returns_string(self):
        r = _mock_pipeline_result()
        js = r.to_json()
        assert isinstance(js, str)
        assert "test_doc" in js

    def test_to_dict_returns_dict(self):
        r = _mock_pipeline_result()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["document_id"] == "test_doc"

    def test_warnings_default_empty(self):
        r = _mock_pipeline_result()
        assert r.warnings == []


# ---------------------------------------------------------------------------
# _make_error_result
# ---------------------------------------------------------------------------

class TestMakeErrorResult:
    def test_returns_document_result(self):
        result = _make_error_result("err_doc", "something broke")
        assert isinstance(result, DocumentResult)

    def test_doc_id_set(self):
        result = _make_error_result("err_doc", "oops")
        assert result.document_id == "err_doc"

    def test_needs_human_review_true(self):
        result = _make_error_result("x", "fail")
        assert result.needs_human_review is True

    def test_confidence_zero(self):
        result = _make_error_result("x", "fail")
        assert result.document_confidence == 0.0

    def test_flagged_fields_have_error(self):
        result = _make_error_result("x", "critical failure")
        assert len(result.flagged_fields) == 1
        assert result.flagged_fields[0].is_hard_failure is True
        assert "critical failure" in result.flagged_fields[0].message

    def test_all_fields_none(self):
        result = _make_error_result("x", "fail")
        assert result.rib is None
        assert result.amount is None
        assert result.echeance is None


# ---------------------------------------------------------------------------
# process_document — file not found
# ---------------------------------------------------------------------------

class TestProcessDocumentErrors:
    def test_file_not_found(self):
        from ocr_pipeline.pipeline import process_document
        with pytest.raises(FileNotFoundError):
            process_document("/nonexistent/path/scan.jpg")


# ---------------------------------------------------------------------------
# process_batch — error handling
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_returns_one_result_per_input(self):
        """process_batch returns list same length as input, even on failures."""
        paths = ["/nonexistent/a.jpg", "/nonexistent/b.jpg"]
        results = process_batch(paths)
        assert len(results) == 2

    def test_failure_does_not_stop_batch(self):
        paths = ["/nonexistent/a.jpg", "/nonexistent/b.jpg", "/nonexistent/c.jpg"]
        results = process_batch(paths)
        assert len(results) == 3

    def test_failed_docs_have_review_flag(self):
        paths = ["/nonexistent/x.jpg"]
        results = process_batch(paths)
        assert results[0].needs_human_review is True

    def test_on_progress_called(self):
        calls = []
        def cb(idx, total, doc_id, result):
            calls.append((idx, total))
        process_batch(["/nonexistent/x.jpg"], on_progress=cb)
        assert len(calls) == 1
        assert calls[0] == (0, 1)
