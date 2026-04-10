"""
Tests for Phase 2 — ocr_engine.py

Coverage:
  - _texts_agree
  - _merge_question_marks
  - _merge_tess_qwen (all 4 rules from plan §3.4)
  - OCRFieldResult dataclass structure
  - OCRBatch.qwen_call_count
  - run_ocr with skip_qwen=True on synthetic crops
  - run_ocr routes barcode correctly
  - run_ocr routes tesseract_then_qwen (skipped in skip_qwen mode)
  - run_ocr_batch
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.tesseract_ocr import TesseractResult
from ocr_pipeline.qwen_ocr import QwenResult
from ocr_pipeline.roi_extractor import ROICrop
from ocr_pipeline.ocr_engine import (
    _texts_agree,
    _merge_question_marks,
    _merge_tess_qwen,
    OCRFieldResult,
    OCRBatch,
    run_ocr,
    run_ocr_batch,
    TESS_CONFIDENCE_GATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def white_crop(roi_id: str = "R01", engine: str = "tesseract") -> ROICrop:
    img = np.full((60, 200, 3), 255, dtype=np.uint8)
    grey = np.full((60, 200), 255, dtype=np.uint8)
    return ROICrop(
        roi_id=roi_id,
        name=f"field_{roi_id}",
        label=roi_id,
        colour=img,
        binarized=grey,
        blue_channel=grey,
        ocr_engine=engine,
        tesseract_config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )


def make_tess(text: str, conf: float = 0.90, available: bool = True) -> TesseractResult:
    return TesseractResult(
        text=text,
        raw_text=text,
        confidence=conf,
        engine_available=available,
    )


def make_qwen(text: str, available: bool = True, readable: bool = True) -> QwenResult:
    return QwenResult(
        text=text,
        readable=readable,
        engine_available=available,
    )


# ---------------------------------------------------------------------------
# 1. _texts_agree
# ---------------------------------------------------------------------------

class TestTextsAgree:
    def test_identical_strings_agree(self):
        assert _texts_agree("12345", "12345")

    def test_case_insensitive(self):
        assert _texts_agree("hello", "HELLO")

    def test_spaces_ignored(self):
        assert _texts_agree("08 006 123", "08006123")

    def test_different_strings_disagree(self):
        assert not _texts_agree("12345", "12346")

    def test_both_empty_disagree(self):
        # empty strings should not be treated as agreement
        assert not _texts_agree("", "")

    def test_one_empty_disagrees(self):
        assert not _texts_agree("123", "")


# ---------------------------------------------------------------------------
# 2. _merge_question_marks
# ---------------------------------------------------------------------------

class TestMergeQuestionMarks:
    def test_no_questions_unchanged(self):
        assert _merge_question_marks("12345", "12345") == "12345"

    def test_question_filled_by_tess_digit(self):
        result = _merge_question_marks("12345", "12?45")
        assert result == "12345"

    def test_question_where_tess_is_non_digit_stays(self):
        # Tess has a letter at position 2 → can't substitute
        result = _merge_question_marks("12A45", "12?45")
        assert result[2] == "?"

    def test_multiple_questions_filled(self):
        result = _merge_question_marks("08006", "0?00?")
        assert result == "08006"

    def test_qwen_longer_than_tess(self):
        result = _merge_question_marks("123", "12?56")
        # position 2: tess has "3" → fills "?"
        assert result[2] == "3"

    def test_empty_tess(self):
        result = _merge_question_marks("", "???")
        assert result == "???"


# ---------------------------------------------------------------------------
# 3. _merge_tess_qwen — Rule 0: Qwen unavailable
# ---------------------------------------------------------------------------

class TestMergeTessQwenRule0:
    def test_qwen_unavailable_uses_tess(self):
        tess = make_tess("12345", conf=0.75)
        qwen = make_qwen("", available=False)
        text, conf, source, flags = _merge_tess_qwen(tess, qwen, "R05")
        assert text == "12345"
        assert source == "tesseract"
        assert "qwen_unavailable" in flags

    def test_qwen_unavailable_confidence_from_tess(self):
        tess = make_tess("999", conf=0.82)
        qwen = make_qwen("", available=False)
        _, conf, _, _ = _merge_tess_qwen(tess, qwen, "R05")
        assert conf == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# 4. _merge_tess_qwen — Rule 4: Digit divergence rejection
# ---------------------------------------------------------------------------

class TestMergeTessQwenRule4:
    def test_divergence_above_30_percent_rejected(self):
        # 5 tess digits, qwen changes 3 of them → 60% divergence
        tess = make_tess("11111", conf=0.80)
        qwen = make_qwen("11444")  # positions 2,3,4 differ
        text, conf, source, flags = _merge_tess_qwen(tess, qwen, "R05")
        assert source == "tesseract"
        assert "qwen_rejected_divergence" in flags

    def test_divergence_below_30_percent_not_rejected(self):
        # 10 tess digits, qwen changes 2 → 20% → OK
        tess = make_tess("1234567890", conf=0.80)
        qwen = make_qwen("1234507890")  # 1 digit differs
        _, _, source, flags = _merge_tess_qwen(tess, qwen, "R05")
        assert "qwen_rejected_divergence" not in flags

    def test_no_tess_digits_no_rejection(self):
        tess = make_tess("BIAT Sousse", conf=0.60)
        qwen = make_qwen("BIAT Sfax")
        _, _, source, flags = _merge_tess_qwen(tess, qwen, "R16")
        assert "qwen_rejected_divergence" not in flags


# ---------------------------------------------------------------------------
# 5. _merge_tess_qwen — Rule 3: Question marks
# ---------------------------------------------------------------------------

class TestMergeTessQwenRule3:
    def test_question_mark_triggers_merge(self):
        tess = make_tess("08006", conf=0.85)
        qwen = make_qwen("0?006")
        text, _, source, _ = _merge_tess_qwen(tess, qwen, "R05")
        assert source == "merged"
        assert "?" not in text  # filled by Tess digit

    def test_unresolvable_question_lowers_confidence(self):
        tess = make_tess("", conf=0.0)  # tess has nothing to offer
        qwen = make_qwen("0???6")
        _, conf, _, flags = _merge_tess_qwen(tess, qwen, "R05")
        # Remaining ?s should lower confidence and produce a flag
        any_partial_flag = any("partial_unreadable" in f for f in flags)
        assert any_partial_flag
        assert conf < 0.70


# ---------------------------------------------------------------------------
# 6. _merge_tess_qwen — Rule 1: Agreement
# ---------------------------------------------------------------------------

class TestMergeTessQwenRule1:
    def test_agreement_high_confidence(self):
        tess = make_tess("15/06/2025", conf=0.88)
        qwen = make_qwen("15/06/2025")
        text, conf, source, flags = _merge_tess_qwen(tess, qwen, "R02")
        assert text == "15/06/2025"
        assert conf > 0.85
        assert flags == []

    def test_agreement_case_insensitive(self):
        tess = make_tess("biat", conf=0.80)
        qwen = make_qwen("BIAT")
        text, conf, source, _ = _merge_tess_qwen(tess, qwen, "R16")
        # Should agree (case-insensitive)
        assert source == "tesseract"


# ---------------------------------------------------------------------------
# 7. _merge_tess_qwen — Rule 2: Disagreement
# ---------------------------------------------------------------------------

class TestMergeTessQwenRule2:
    def test_disagreement_uses_qwen(self):
        tess = make_tess("15062025", conf=0.55)
        qwen = make_qwen("15/06/2025")
        text, conf, source, flags = _merge_tess_qwen(tess, qwen, "R02")
        assert text == "15/06/2025"
        assert source == "qwen"
        assert "tess_qwen_disagree" in flags

    def test_disagreement_confidence_capped_at_75(self):
        tess = make_tess("123", conf=0.50)
        qwen = make_qwen("456")
        _, conf, _, _ = _merge_tess_qwen(tess, qwen, "R06")
        assert conf <= 0.75


# ---------------------------------------------------------------------------
# 8. OCRFieldResult dataclass
# ---------------------------------------------------------------------------

class TestOCRFieldResult:
    def test_required_fields_exist(self):
        r = OCRFieldResult(roi_id="R01", field_name="payment_order", text="123", confidence=0.9)
        assert r.roi_id == "R01"
        assert r.field_name == "payment_order"
        assert r.text == "123"
        assert r.confidence == 0.9

    def test_default_values(self):
        r = OCRFieldResult(roi_id="R01", field_name="x", text="", confidence=0.0)
        assert r.source == "none"
        assert r.tess_result is None
        assert r.qwen_result is None
        assert r.readable is True
        assert r.flags == []
        assert r.error is None


# ---------------------------------------------------------------------------
# 9. OCRBatch
# ---------------------------------------------------------------------------

class TestOCRBatch:
    def test_qwen_call_count_zero_when_no_qwen(self):
        batch = OCRBatch(doc_id="test_doc")
        batch.fields["R01"] = OCRFieldResult(
            roi_id="R01", field_name="x", text="", confidence=0.0, source="tesseract"
        )
        assert batch.qwen_call_count == 0

    def test_qwen_call_count_counts_available_qwen(self):
        batch = OCRBatch(doc_id="test_doc")
        r = OCRFieldResult(
            roi_id="R05", field_name="rib", text="", confidence=0.7, source="qwen",
            qwen_result=make_qwen("12345", available=True),
        )
        batch.fields["R05"] = r
        assert batch.qwen_call_count == 1

    def test_qwen_call_count_ignores_unavailable(self):
        batch = OCRBatch(doc_id="test_doc")
        r = OCRFieldResult(
            roi_id="R05", field_name="rib", text="", confidence=0.0, source="none",
            qwen_result=make_qwen("", available=False),
        )
        batch.fields["R05"] = r
        assert batch.qwen_call_count == 0


# ---------------------------------------------------------------------------
# 10. run_ocr (skip_qwen=True) on synthetic ROI crops
# ---------------------------------------------------------------------------

class TestRunOcrSkipQwen:
    def test_all_17_rois_processed(self):
        crops = {f"R{i:02d}": white_crop(f"R{i:02d}", engine="tesseract") for i in range(1, 18)}
        results = run_ocr(crops, skip_qwen=True)
        assert len(results) == 17

    def test_result_type_is_OCRFieldResult(self):
        crops = {"R01": white_crop("R01")}
        results = run_ocr(crops, skip_qwen=True)
        assert isinstance(results["R01"], OCRFieldResult)

    def test_no_qwen_calls_when_skip_qwen(self):
        crops = {"R05": white_crop("R05", engine="tesseract_then_qwen")}
        results = run_ocr(crops, skip_qwen=True)
        assert results["R05"].qwen_result is None

    def test_tesseract_engine_uses_tesseract_source(self):
        crops = {"R01": white_crop("R01", engine="tesseract")}
        results = run_ocr(crops, skip_qwen=True)
        assert results["R01"].source == "tesseract"

    def test_confidence_in_range(self):
        crops = {"R06": white_crop("R06", engine="tesseract")}
        results = run_ocr(crops, skip_qwen=True)
        assert 0.0 <= results["R06"].confidence <= 1.0

    def test_barcode_engine_routes_to_barcode(self):
        crops = {"R17": white_crop("R17", engine="barcode_decoder")}
        results = run_ocr(crops, skip_qwen=True)
        assert results["R17"].source.startswith("barcode_")

    def test_all_results_have_text_attribute(self):
        crops = {f"R{i:02d}": white_crop(f"R{i:02d}") for i in range(1, 6)}
        for r in run_ocr(crops, skip_qwen=True).values():
            assert isinstance(r.text, str)

    def test_qwen_engine_skipped_falls_to_tesseract(self):
        crops = {"R09": white_crop("R09", engine="qwen")}
        results = run_ocr(crops, skip_qwen=True)
        # skip_qwen=True forces Tesseract path
        assert results["R09"].source in ("tesseract", "none")


# ---------------------------------------------------------------------------
# 11. run_ocr_batch
# ---------------------------------------------------------------------------

class TestRunOcrBatch:
    def test_returns_list_of_batches(self):
        crops = {"R01": white_crop("R01")}
        batches = run_ocr_batch([crops, crops], ["doc1", "doc2"], skip_qwen=True)
        assert len(batches) == 2
        assert batches[0].doc_id == "doc1"
        assert batches[1].doc_id == "doc2"

    def test_each_batch_has_fields(self):
        crops = {"R01": white_crop("R01"), "R02": white_crop("R02")}
        batches = run_ocr_batch([crops], ["doc1"], skip_qwen=True)
        assert len(batches[0].fields) == 2

    def test_empty_input(self):
        batches = run_ocr_batch([], [], skip_qwen=True)
        assert batches == []
