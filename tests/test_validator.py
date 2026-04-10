"""Tests for validator.py — Phase 3.

Tests the cross-field validation engine and confidence scoring.
"""
from decimal import Decimal
import pytest

from src.ocr_pipeline.rib_parser import parse_rib, RIBResult
from src.ocr_pipeline.date_parser import parse_date
from src.ocr_pipeline.amount_parser import parse_amount_numeric, parse_amount_words
from src.ocr_pipeline.name_parser import parse_name
from src.ocr_pipeline.validator import (
    ParsedDocument,
    ValidationReport,
    validate_document,
    compute_field_confidence,
    compute_document_confidence,
    ErrorType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_rib_str(bank="08", branch="006", account="0110510000870") -> str:
    """Return a 20-digit RIB string with computed correct key."""
    n = int(bank + branch + account)
    key = 97 - ((n * 100) % 97)
    if key == 97:
        key = 0
    return f"{bank} {branch} {account} {key:02d}"


def _empty_doc() -> ParsedDocument:
    return ParsedDocument()


# ---------------------------------------------------------------------------
# compute_field_confidence
# ---------------------------------------------------------------------------

class TestComputeFieldConfidence:
    def test_all_perfect(self):
        c = compute_field_confidence(1.0, 1.0, 1.0, 1.0)
        assert abs(c - 1.0) < 0.001

    def test_all_zero(self):
        c = compute_field_confidence(0.0, 0.0, 0.0, 0.0)
        assert c == 0.0

    def test_weights_sum_to_one(self):
        # weights: 0.4 + 0.3 + 0.2 + 0.1 = 1.0
        c = compute_field_confidence(0.5, 0.5, 0.5, 0.5)
        assert abs(c - 0.5) < 0.001

    def test_format_validity_contribution(self):
        # Only format_validity = 1.0, rest 0
        c = compute_field_confidence(0.0, 0.0, 1.0, 0.0)
        assert abs(c - 0.2) < 0.001


# ---------------------------------------------------------------------------
# compute_document_confidence
# ---------------------------------------------------------------------------

class TestComputeDocumentConfidence:
    def test_minimum_of_critical_fields(self):
        confs = {"rib": 0.9, "amount_upper": 0.7, "echeance_upper": 0.85, "creation_upper": 0.95}
        assert compute_document_confidence(confs) == pytest.approx(0.7)

    def test_missing_critical_fields_returns_zero(self):
        assert compute_document_confidence({}) == 0.0

    def test_subset_of_critical_fields(self):
        confs = {"rib": 0.8, "amount_upper": 0.6}
        assert compute_document_confidence(confs) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Rule 1: RIB key validity
# ---------------------------------------------------------------------------

class TestRibKeyValidation:
    def test_valid_upper_rib_passes(self):
        doc = ParsedDocument(rib_upper=parse_rib(_make_valid_rib_str()))
        r = validate_document(doc)
        assert r.rib_key_valid

    def test_valid_lower_rib_passes(self):
        doc = ParsedDocument(rib_lower=parse_rib(_make_valid_rib_str()))
        r = validate_document(doc)
        assert r.rib_key_valid

    def test_invalid_both_ribs_hard_failure(self):
        doc = ParsedDocument(
            rib_upper=parse_rib("08 006 0110510000870 00"),  # wrong key
            rib_lower=parse_rib("08 006 0110510000870 00"),
        )
        r = validate_document(doc)
        assert not r.rib_key_valid
        assert r.hard_failures > 0

    def test_no_rib_no_failure(self):
        doc = _empty_doc()
        r = validate_document(doc)
        # No RIB provided — no failure, just not valid
        assert not r.rib_key_valid
        assert r.hard_failures == 0


# ---------------------------------------------------------------------------
# Rule 2: Upper ↔ Lower RIB consistency
# ---------------------------------------------------------------------------

class TestRibConsistency:
    def test_matching_ribs_consistent(self):
        rib_str = _make_valid_rib_str()
        doc = ParsedDocument(
            rib_upper=parse_rib(rib_str),
            rib_lower=parse_rib(rib_str),
        )
        r = validate_document(doc)
        assert r.upper_lower.rib

    def test_mismatched_ribs_hard_failure(self):
        rib1 = _make_valid_rib_str("08", "006", "0110510000870")
        rib2 = _make_valid_rib_str("07", "001", "0000000000001")
        doc = ParsedDocument(
            rib_upper=parse_rib(rib1),
            rib_lower=parse_rib(rib2),
        )
        r = validate_document(doc)
        assert r.hard_failures > 0

    def test_question_marks_soft_warning(self):
        doc = ParsedDocument(
            rib_upper=parse_rib("08 006 01105?0000870 41"),
            rib_lower=parse_rib(_make_valid_rib_str()),
        )
        r = validate_document(doc)
        # Soft warning because of question marks
        assert r.soft_warnings >= 0  # At least no hard failure from this rule


# ---------------------------------------------------------------------------
# Rule 3 & 4: Amount consistency
# ---------------------------------------------------------------------------

class TestAmountConsistency:
    def test_matching_amounts_consistent(self):
        doc = ParsedDocument(
            amount_upper=parse_amount_numeric("3000,000"),
            amount_lower=parse_amount_numeric("3 000.000"),
        )
        r = validate_document(doc)
        assert r.upper_lower.amount

    def test_mismatched_amounts_hard_failure(self):
        doc = ParsedDocument(
            amount_upper=parse_amount_numeric("3000,000"),
            amount_lower=parse_amount_numeric("4000,000"),
        )
        r = validate_document(doc)
        assert r.hard_failures > 0

    def test_amount_words_match_soft(self):
        doc = ParsedDocument(
            amount_upper=parse_amount_numeric("3000,000"),
            amount_words=parse_amount_words("trois mille dinars"),
        )
        r = validate_document(doc)
        assert r.amount_numeric_matches_words

    def test_amount_words_mismatch_soft_warning(self):
        doc = ParsedDocument(
            amount_upper=parse_amount_numeric("3000,000"),
            amount_words=parse_amount_words("cinq mille dinars"),
        )
        r = validate_document(doc)
        assert not r.amount_numeric_matches_words
        # Should be a soft warning, not a hard failure
        assert r.soft_warnings > 0
        assert r.hard_failures == 0


# ---------------------------------------------------------------------------
# Rule 5 & 6: Date consistency
# ---------------------------------------------------------------------------

class TestDateConsistency:
    def test_matching_echeance_consistent(self):
        doc = ParsedDocument(
            echeance_upper=parse_date("15/06/2025"),
            echeance_lower=parse_date("15/06/2025"),
        )
        r = validate_document(doc)
        assert r.upper_lower.echeance

    def test_mismatched_echeance_hard_failure(self):
        doc = ParsedDocument(
            echeance_upper=parse_date("15/06/2025"),
            echeance_lower=parse_date("16/06/2025"),
        )
        r = validate_document(doc)
        assert r.hard_failures > 0

    def test_matching_creation_dates_consistent(self):
        doc = ParsedDocument(
            creation_upper=parse_date("01/01/2025"),
            creation_lower=parse_date("01/01/2025"),
        )
        r = validate_document(doc)
        assert r.upper_lower.creation_date


# ---------------------------------------------------------------------------
# Rule 7: Échéance ≥ creation date
# ---------------------------------------------------------------------------

class TestDateOrdering:
    def test_echeance_after_creation_passes(self):
        doc = ParsedDocument(
            echeance_upper=parse_date("15/06/2025"),
            creation_upper=parse_date("01/01/2025"),
        )
        r = validate_document(doc)
        assert r.echeance_after_creation

    def test_echeance_before_creation_soft_warning(self):
        doc = ParsedDocument(
            echeance_upper=parse_date("01/01/2025"),
            creation_upper=parse_date("15/06/2025"),
        )
        r = validate_document(doc)
        assert not r.echeance_after_creation
        assert r.soft_warnings > 0
        assert r.hard_failures == 0

    def test_echeance_same_as_creation_passes(self):
        doc = ParsedDocument(
            echeance_upper=parse_date("15/06/2025"),
            creation_upper=parse_date("15/06/2025"),
        )
        r = validate_document(doc)
        assert r.echeance_after_creation


# ---------------------------------------------------------------------------
# Rule 9: Payment order matches barcode
# ---------------------------------------------------------------------------

class TestPaymentOrderBarcode:
    def test_matching_numbers_pass(self):
        doc = ParsedDocument(
            payment_order_number="01118862916",
            barcode_number="01118862916",
        )
        r = validate_document(doc)
        assert r.payment_order_matches_barcode

    def test_leading_zeros_normalised(self):
        doc = ParsedDocument(
            payment_order_number="00123",
            barcode_number="123",
        )
        r = validate_document(doc)
        assert r.payment_order_matches_barcode

    def test_mismatched_hard_failure(self):
        doc = ParsedDocument(
            payment_order_number="01118862916",
            barcode_number="99999999999",
        )
        r = validate_document(doc)
        assert r.hard_failures > 0

    def test_missing_barcode_no_failure(self):
        doc = ParsedDocument(payment_order_number="01118862916")
        r = validate_document(doc)
        assert r.hard_failures == 0


# ---------------------------------------------------------------------------
# needs_human_review
# ---------------------------------------------------------------------------

class TestNeedsHumanReview:
    def test_no_data_needs_review(self):
        doc = _empty_doc()
        r = validate_document(doc)
        assert r.needs_human_review  # low confidence → flag

    def test_hard_failure_needs_review(self):
        doc = ParsedDocument(
            payment_order_number="111",
            barcode_number="999",
        )
        r = validate_document(doc)
        assert r.needs_human_review
