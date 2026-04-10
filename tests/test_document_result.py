"""Tests for document_result.py — Phase 3.

Tests the full document assembly pipeline that combines
OCR engine output with parsers and validator into the final JSON schema.
"""
from decimal import Decimal
import pytest

from src.ocr_pipeline.ocr_engine import OCRBatch, OCRFieldResult
from src.ocr_pipeline.document_result import (
    build_document_result,
    document_result_to_dict,
    document_result_to_json,
    DocumentResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_rib_str(bank="08", branch="006", account="0110510000870") -> str:
    n = int(bank + branch + account)
    key = 97 - ((n * 100) % 97)
    if key == 97:
        key = 0
    return f"{bank} {branch} {account} {key:02d}"


def _field(roi_id: str, name: str, text: str, confidence: float = 0.9) -> OCRFieldResult:
    return OCRFieldResult(
        roi_id=roi_id,
        field_name=name,
        text=text,
        confidence=confidence,
        source="tesseract",
        tess_text=text,
    )


def _make_batch(overrides: dict | None = None) -> OCRBatch:
    """Return a minimal OCRBatch with sensible default values for all 17 ROIs."""
    rib = _make_valid_rib_str()
    defaults = {
        "R01": _field("R01", "payment_order", "01118862916"),
        "R02": _field("R02", "echeance_upper", "15/06/2025"),
        "R03": _field("R03", "creation_upper", "01/01/2025"),
        "R04": _field("R04", "city_upper", "TUNIS"),
        "R05": _field("R05", "rib_upper", rib),
        "R06": _field("R06", "amount_upper", "3000,000"),
        "R07": _field("R07", "tireur", "STE DELTA SARL"),
        "R08": _field("R08", "beneficiary", "AHMED BEN ALI"),
        "R09": _field("R09", "amount_words", "trois mille dinars"),
        "R10": _field("R10", "amount_lower", "3000,000"),
        "R11": _field("R11", "city_lower", "TUNIS"),
        "R12": _field("R12", "creation_lower", "01/01/2025"),
        "R13": _field("R13", "echeance_lower", "15/06/2025"),
        "R14": _field("R14", "rib_lower", rib),
        "R15": _field("R15", "drawee", "BIAT SA"),
        "R16": _field("R16", "domiciliation", "BIAT"),
        "R17": _field("R17", "barcode", "01118862916", confidence=0.0),
    }
    if overrides:
        defaults.update(overrides)
    return OCRBatch(doc_id="test_doc", fields=defaults)


# ---------------------------------------------------------------------------
# build_document_result — basic structure
# ---------------------------------------------------------------------------

class TestBuildDocumentResult:
    def test_returns_document_result(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert isinstance(result, DocumentResult)

    def test_document_id_set(self):
        batch = _make_batch()
        result = build_document_result("my_doc_001", batch)
        assert result.document_id == "my_doc_001"

    def test_extraction_timestamp_set(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.extraction_timestamp.endswith("Z")

    def test_payment_order_number(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.payment_order_number == "01118862916"

    def test_rib_parsed(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.rib is not None
        assert result.rib.bank_code == "08"
        assert result.rib.key_valid

    def test_echeance_parsed(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.echeance is not None
        assert result.echeance.value == "15/06/2025"

    def test_creation_date_parsed(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.creation_date is not None
        assert result.creation_date.value == "01/01/2025"

    def test_amount_parsed(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.amount is not None
        assert result.amount.value_numeric == "3000.000"

    def test_tireur_parsed(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.tireur is not None
        assert "DELTA" in result.tireur.value

    def test_domiciliation_set(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.domiciliation is not None
        assert result.domiciliation.value == "BIAT"

    def test_validation_dict_present(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert "rib_key_valid" in result.validation
        assert "upper_lower_consistency" in result.validation

    def test_upper_lower_rib_consistent(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.validation["upper_lower_consistency"]["rib"]

    def test_upper_lower_amount_consistent(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.validation["upper_lower_consistency"]["amount"]

    def test_document_confidence_set(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert 0.0 <= result.document_confidence <= 1.0

    def test_needs_human_review_set(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert isinstance(result.needs_human_review, bool)

    def test_qwen_corrections_set(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert result.qwen_corrections is not None

    def test_flagged_fields_list(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        assert isinstance(result.flagged_fields, list)


# ---------------------------------------------------------------------------
# build_document_result — with invalid fields
# ---------------------------------------------------------------------------

class TestBuildWithInvalidFields:
    def test_invalid_rib_flagged(self):
        batch = _make_batch(overrides={
            "R05": _field("R05", "rib_upper", "invalid_rib"),
            "R14": _field("R14", "rib_lower", "invalid_rib"),
        })
        result = build_document_result("test_doc", batch)
        assert result.needs_human_review

    def test_mismatched_amounts_flagged(self):
        batch = _make_batch(overrides={
            "R06": _field("R06", "amount_upper", "3000,000"),
            "R10": _field("R10", "amount_lower", "4000,000"),
        })
        result = build_document_result("test_doc", batch)
        hard_fields = [f for f in result.flagged_fields if f.is_hard_failure]
        assert len(hard_fields) > 0

    def test_missing_rib_result_is_none(self):
        batch = OCRBatch(doc_id="test_doc", fields={})
        result = build_document_result("test_doc", batch)
        assert result.rib is None

    def test_empty_batch_still_returns_document(self):
        batch = OCRBatch(doc_id="test_doc", fields={})
        result = build_document_result("test_doc", batch)
        assert result.document_id == "test_doc"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_is_dict(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        d = document_result_to_dict(result)
        assert isinstance(d, dict)

    def test_to_json_is_str(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        j = document_result_to_json(result)
        assert isinstance(j, str)
        assert "document_id" in j

    def test_decimal_serialised_as_string(self):
        batch = _make_batch()
        result = build_document_result("test_doc", batch)
        j = document_result_to_json(result)
        # "3000.000" should appear as a JSON string, not cause TypeError
        assert "3000" in j
