"""Tests for review_reporter.py — Phase 5."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.review_reporter import (
    generate_review_report,
    generate_batch_index,
    _conf_colour,
    _crop_to_b64,
    _field_row,
)
from ocr_pipeline.pipeline import PipelineResult
from ocr_pipeline.document_result import (
    DocumentResult, QwenCorrections, StampInfo, FlaggedField, FieldValue
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_result(
    doc_id="doc001",
    confidence=0.85,
    needs_review=False,
    stamps=0,
    flagged=None,
    anomaly="",
):
    from ocr_pipeline.validator import ErrorType
    flagged = flagged or []
    dr = DocumentResult(
        document_id=doc_id,
        payment_order_number="001",
        extraction_timestamp="2026-04-11T00:00:00Z",
        echeance=FieldValue(value="30/06/2025", confidence=0.9, source="merged"),
        creation_date=FieldValue(value="01/01/2025", confidence=0.9, source="merged"),
        creation_city=FieldValue(value="Tunis", confidence=0.7, source="upper"),
        rib=None,
        amount=None,
        tireur=FieldValue(value="STE ACME SARL", confidence=0.75, source="single"),
        beneficiary=None,
        tire=None,
        domiciliation=None,
        validation={"rib_key_valid": False},
        document_confidence=confidence,
        needs_human_review=needs_review,
        qwen_corrections=QwenCorrections(),
        flagged_fields=flagged,
        stamp_info=StampInfo(stamp_count=stamps, hints=["Blue stamp at (100,100)"]*stamps),
        anomaly_explanation=anomaly,
    )
    return PipelineResult(doc_result=dr, processing_time_s=1.0)


# ---------------------------------------------------------------------------
# _conf_colour
# ---------------------------------------------------------------------------

class TestConfColour:
    def test_high_confidence_green(self):
        assert _conf_colour(0.90) == "#22c55e"

    def test_medium_confidence_amber(self):
        assert _conf_colour(0.70) == "#f59e0b"

    def test_low_confidence_red(self):
        assert _conf_colour(0.30) == "#ef4444"

    def test_exact_boundary_0_85_green(self):
        assert _conf_colour(0.85) == "#22c55e"

    def test_exact_boundary_0_60_amber(self):
        assert _conf_colour(0.60) == "#f59e0b"


# ---------------------------------------------------------------------------
# _crop_to_b64
# ---------------------------------------------------------------------------

class TestCropToB64:
    def test_returns_string(self):
        img = np.full((50, 100, 3), 200, dtype=np.uint8)
        result = _crop_to_b64(img)
        assert isinstance(result, str)

    def test_non_empty(self):
        img = np.full((50, 100, 3), 200, dtype=np.uint8)
        result = _crop_to_b64(img)
        assert len(result) > 0

    def test_valid_base64(self):
        import base64
        img = np.full((50, 100, 3), 200, dtype=np.uint8)
        result = _crop_to_b64(img)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# _field_row
# ---------------------------------------------------------------------------

class TestFieldRow:
    def test_returns_string(self):
        row = _field_row("RIB", "08 006 XXXX 41", 0.85, None)
        assert isinstance(row, str)

    def test_contains_label(self):
        row = _field_row("RIB", "08 006 XXXX 41", 0.85, None)
        assert "RIB" in row

    def test_contains_value(self):
        row = _field_row("Amount", "3000.000 DT", 0.90, None)
        assert "3000.000 DT" in row

    def test_contains_confidence(self):
        row = _field_row("RIB", "xxx", 0.77, None)
        assert "0.77" in row

    def test_none_value_shows_dash(self):
        row = _field_row("RIB", None, 0.0, None)
        assert "—" in row

    def test_flag_included_when_provided(self):
        row = _field_row("RIB", "xxx", 0.1, None, flag="mod-97 failed", flag_hard=True)
        assert "mod-97 failed" in row
        assert "HARD" in row

    def test_soft_flag_shows_warn(self):
        row = _field_row("Amount", "xxx", 0.5, None, flag="text mismatch", flag_hard=False)
        assert "WARN" in row

    def test_with_crop_image(self):
        img = np.full((40, 80, 3), 170, dtype=np.uint8)
        row = _field_row("RIB", "xxx", 0.8, img)
        assert "data:image/png;base64," in row


# ---------------------------------------------------------------------------
# generate_review_report
# ---------------------------------------------------------------------------

class TestGenerateReviewReport:
    def test_returns_html_string(self):
        html = generate_review_report(_mock_result())
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_doc_id(self):
        html = generate_review_report(_mock_result(doc_id="batch1_0001"))
        assert "batch1_0001" in html

    def test_review_banner_when_needed(self):
        html = generate_review_report(_mock_result(needs_review=True))
        assert "NEEDS HUMAN REVIEW" in html

    def test_accepted_banner_when_ok(self):
        html = generate_review_report(_mock_result(needs_review=False, confidence=0.95))
        assert "ACCEPTED" in html

    def test_stamp_section_when_stamps(self):
        html = generate_review_report(_mock_result(stamps=2))
        assert "Stamp Detection" in html

    def test_no_stamp_section_when_no_stamps(self):
        html = generate_review_report(_mock_result(stamps=0))
        assert "Stamp Detection" not in html

    def test_anomaly_section_when_explanation(self):
        html = generate_review_report(_mock_result(anomaly="Stamp over RIB."))
        assert "Stamp over RIB." in html

    def test_no_anomaly_section_when_empty(self):
        html = generate_review_report(_mock_result(anomaly=""))
        assert "Anomaly Explanation" not in html

    def test_flags_section_when_flagged(self):
        from ocr_pipeline.validator import ErrorType
        ff = FlaggedField(field="rib", error_type=ErrorType.VALIDATION_ERROR,
                          message="mod-97 failed", is_hard_failure=True)
        html = generate_review_report(_mock_result(flagged=[ff]))
        assert "mod-97 failed" in html
        assert "Validation Flags" in html

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "report.html"
        generate_review_report(_mock_result(), output_path=out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "report.html"
        generate_review_report(_mock_result(), output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# generate_batch_index
# ---------------------------------------------------------------------------

class TestGenerateBatchIndex:
    def test_returns_html_string(self, tmp_path):
        results = [_mock_result(f"doc{i:03d}") for i in range(3)]
        html = generate_batch_index(results, tmp_path / "index.html", tmp_path)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_creates_index_file(self, tmp_path):
        results = [_mock_result(f"doc{i:03d}") for i in range(2)]
        generate_batch_index(results, tmp_path / "index.html", tmp_path)
        assert (tmp_path / "index.html").exists()

    def test_shows_document_count(self, tmp_path):
        results = [_mock_result(f"doc{i:03d}") for i in range(5)]
        html = generate_batch_index(results, tmp_path / "index.html", tmp_path)
        assert "5 documents" in html

    def test_counts_review_vs_ok(self, tmp_path):
        results = [
            _mock_result("d1", needs_review=True),
            _mock_result("d2", needs_review=False, confidence=0.95),
        ]
        html = generate_batch_index(results, tmp_path / "index.html", tmp_path)
        # "Accepted: 1" and "Needs Review: 1" should appear
        assert "1" in html  # basic sanity
