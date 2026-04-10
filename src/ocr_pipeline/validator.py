"""
Phase 3 — Cross-field validation & confidence scoring.

Implements all 10 validation rules from section 3.6 of the plan, plus the
confidence scoring formula from section 3.10.

Public API
----------
validate_document(fields: dict[str, OCRFieldResult], parsed: ParsedDocument)
    -> ValidationReport

compute_field_confidence(ocr_agreement, redundancy_match, format_validity, engine_confidence)
    -> float

compute_document_confidence(field_confidences: dict[str, float]) -> float
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from decimal import Decimal
from typing import Optional

from .rib_parser import RIBResult, bank_name_for_code
from .date_parser import DateResult, dates_equal, date_a_on_or_after_b
from .amount_parser import AmountResult, AmountWordsResult, amounts_equal
from .name_parser import NameResult


# ---------------------------------------------------------------------------
# Error types (section 3.8)
# ---------------------------------------------------------------------------

class ErrorType:
    OCR_ERROR       = "OCR_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MISSING_FIELD   = "MISSING_FIELD"
    INCONSISTENCY   = "INCONSISTENCY"
    STAMP_OCCLUSION = "STAMP_OCCLUSION"
    FORMAT_ERROR    = "FORMAT_ERROR"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldFlag:
    field: str
    error_type: str
    message: str
    is_hard_failure: bool = False  # True → document rejected for human review


@dataclass
class UpperLowerConsistency:
    rib: bool = False
    amount: bool = False
    echeance: bool = False
    creation_date: bool = False


@dataclass
class ValidationReport:
    # Rule results
    rib_key_valid: bool = False
    rib_bank_code_matches_domiciliation: bool = False
    amount_numeric_matches_words: bool = False
    echeance_after_creation: bool = False
    payment_order_matches_barcode: bool = False
    upper_lower: UpperLowerConsistency = dc_field(default_factory=UpperLowerConsistency)

    # Flags
    flags: list[FieldFlag] = dc_field(default_factory=list)
    hard_failures: int = 0
    soft_warnings: int = 0

    # Confidence
    field_confidences: dict[str, float] = dc_field(default_factory=dict)
    document_confidence: float = 0.0
    needs_human_review: bool = True


@dataclass
class ParsedDocument:
    """Container for all parsed field results, passed into validate_document()."""
    # RIB
    rib_upper: Optional[RIBResult] = None
    rib_lower: Optional[RIBResult] = None
    rib_best: Optional[RIBResult] = None      # chosen after redundancy logic

    # Dates
    echeance_upper: Optional[DateResult] = None
    echeance_lower: Optional[DateResult] = None
    creation_upper: Optional[DateResult] = None
    creation_lower: Optional[DateResult] = None

    # Amounts
    amount_upper: Optional[AmountResult] = None
    amount_lower: Optional[AmountResult] = None
    amount_words: Optional[AmountWordsResult] = None

    # Names
    tireur: Optional[NameResult] = None
    beneficiary: Optional[NameResult] = None
    drawee: Optional[NameResult] = None

    # Single-occurrence fields
    domiciliation_text: str = ""   # raw text from R16
    payment_order_number: str = "" # raw text from R01
    barcode_number: str = ""       # decoded from R17


# ---------------------------------------------------------------------------
# Confidence scoring (section 3.10)
# ---------------------------------------------------------------------------

def compute_field_confidence(
    ocr_agreement: float,
    redundancy_match: float,
    format_validity: float,
    engine_confidence: float,
) -> float:
    """Weighted confidence score per the plan formula:

    confidence = 0.4 * ocr_agreement
               + 0.3 * redundancy_match
               + 0.2 * format_validity
               + 0.1 * engine_confidence
    """
    return (
        0.4 * ocr_agreement
        + 0.3 * redundancy_match
        + 0.2 * format_validity
        + 0.1 * engine_confidence
    )


def compute_document_confidence(field_confidences: dict[str, float]) -> float:
    """Document confidence = minimum of all critical field confidences."""
    critical = ["rib", "amount_upper", "echeance_upper", "creation_upper"]
    values = [
        field_confidences[f]
        for f in critical
        if f in field_confidences
    ]
    if not values:
        return 0.0
    return min(values)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flag_hard(report: ValidationReport, field: str, error_type: str, msg: str) -> None:
    report.flags.append(FieldFlag(field=field, error_type=error_type, message=msg, is_hard_failure=True))
    report.hard_failures += 1


def _flag_soft(report: ValidationReport, field: str, error_type: str, msg: str) -> None:
    report.flags.append(FieldFlag(field=field, error_type=error_type, message=msg, is_hard_failure=False))
    report.soft_warnings += 1


def _bank_name_in_domiciliation(bank_code: str, domiciliation: str) -> bool:
    """Check if the bank name for the given code appears in the domiciliation text."""
    name = bank_name_for_code(bank_code)
    if name is None:
        return False  # Unknown code — cannot verify
    return name.lower() in domiciliation.lower()


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_document(
    parsed: ParsedDocument,
    ocr_engine_confidences: Optional[dict[str, float]] = None,
) -> ValidationReport:
    """Run all cross-field validation rules and compute confidence scores.

    Args:
        parsed:                  ParsedDocument with all parsers' outputs.
        ocr_engine_confidences:  Raw OCR engine confidence values per field
                                 (used in confidence scoring). Dict of field_id → float.
                                 Defaults to 0.5 for all fields if not supplied.

    Returns:
        ValidationReport with flags, hard/soft counts, and confidence scores.
    """
    report = ValidationReport()
    confs = ocr_engine_confidences or {}

    # ------------------------------------------------------------------
    # Rule 1: RIB key validity
    # ------------------------------------------------------------------
    # Choose the best RIB (prefer the one that passes key check)
    rib_upper_valid = parsed.rib_upper is not None and parsed.rib_upper.key_valid
    rib_lower_valid = parsed.rib_lower is not None and parsed.rib_lower.key_valid

    if rib_upper_valid or rib_lower_valid:
        report.rib_key_valid = True
    else:
        # Both invalid or missing
        if parsed.rib_upper is not None or parsed.rib_lower is not None:
            _flag_hard(
                report, "rib", ErrorType.VALIDATION_ERROR,
                "RIB check key (mod 97) failed on both upper and lower instances"
            )

    # ------------------------------------------------------------------
    # Rule 2: Upper ↔ Lower RIB consistency
    # ------------------------------------------------------------------
    if parsed.rib_upper and parsed.rib_lower:
        both_have_digits = (
            len(parsed.rib_upper.digits) == 20
            and len(parsed.rib_lower.digits) == 20
            and "?" not in parsed.rib_upper.digits
            and "?" not in parsed.rib_lower.digits
        )
        if both_have_digits:
            if parsed.rib_upper.digits == parsed.rib_lower.digits:
                report.upper_lower.rib = True
            else:
                _flag_hard(
                    report, "rib", ErrorType.INCONSISTENCY,
                    f"Upper RIB '{parsed.rib_upper.digits}' ≠ Lower RIB '{parsed.rib_lower.digits}'"
                )
        else:
            # At least one has question marks — soft warning
            report.upper_lower.rib = rib_upper_valid or rib_lower_valid
            if parsed.rib_upper.has_question_marks or parsed.rib_lower.has_question_marks:
                _flag_soft(
                    report, "rib", ErrorType.STAMP_OCCLUSION,
                    "RIB contains unreadable positions — upper/lower comparison skipped"
                )

    # ------------------------------------------------------------------
    # Rule 3: Amount upper ↔ lower consistency
    # ------------------------------------------------------------------
    if parsed.amount_upper and parsed.amount_lower:
        if parsed.amount_upper.valid and parsed.amount_lower.valid:
            if amounts_equal(parsed.amount_upper, parsed.amount_lower):
                report.upper_lower.amount = True
            else:
                _flag_hard(
                    report, "amount", ErrorType.INCONSISTENCY,
                    f"Upper amount '{parsed.amount_upper.normalised}' ≠ "
                    f"Lower amount '{parsed.amount_lower.normalised}'"
                )
        else:
            if not parsed.amount_upper.valid:
                _flag_soft(report, "amount_upper", ErrorType.FORMAT_ERROR,
                           f"Upper amount parse failed: {parsed.amount_upper.error}")
            if not parsed.amount_lower.valid:
                _flag_soft(report, "amount_lower", ErrorType.FORMAT_ERROR,
                           f"Lower amount parse failed: {parsed.amount_lower.error}")

    # ------------------------------------------------------------------
    # Rule 4: Amount numeric vs. amount in words (SOFT)
    # ------------------------------------------------------------------
    best_amount = (
        parsed.amount_upper if (parsed.amount_upper and parsed.amount_upper.valid)
        else parsed.amount_lower
    )
    if best_amount and best_amount.valid and parsed.amount_words and parsed.amount_words.valid:
        tolerance = Decimal("0.001")
        if abs(best_amount.value - parsed.amount_words.value) <= tolerance:
            report.amount_numeric_matches_words = True
        else:
            _flag_soft(
                report, "amount", ErrorType.INCONSISTENCY,
                f"Numeric amount '{best_amount.normalised}' ≠ "
                f"words amount '{parsed.amount_words.value}' (soft warning)"
            )

    # ------------------------------------------------------------------
    # Rule 5: Échéance upper ↔ lower consistency
    # ------------------------------------------------------------------
    if parsed.echeance_upper and parsed.echeance_lower:
        if dates_equal(parsed.echeance_upper, parsed.echeance_lower):
            report.upper_lower.echeance = True
        elif parsed.echeance_upper.valid and parsed.echeance_lower.valid:
            _flag_hard(
                report, "echeance", ErrorType.INCONSISTENCY,
                f"Upper échéance '{parsed.echeance_upper.normalised}' ≠ "
                f"Lower échéance '{parsed.echeance_lower.normalised}'"
            )

    # ------------------------------------------------------------------
    # Rule 6: Creation date upper ↔ lower consistency
    # ------------------------------------------------------------------
    if parsed.creation_upper and parsed.creation_lower:
        if dates_equal(parsed.creation_upper, parsed.creation_lower):
            report.upper_lower.creation_date = True
        elif parsed.creation_upper.valid and parsed.creation_lower.valid:
            _flag_hard(
                report, "creation_date", ErrorType.INCONSISTENCY,
                f"Upper creation date '{parsed.creation_upper.normalised}' ≠ "
                f"Lower creation date '{parsed.creation_lower.normalised}'"
            )

    # ------------------------------------------------------------------
    # Rule 7: Échéance ≥ creation date
    # ------------------------------------------------------------------
    best_echeance  = parsed.echeance_upper  or parsed.echeance_lower
    best_creation  = parsed.creation_upper  or parsed.creation_lower
    if best_echeance and best_creation and best_echeance.valid and best_creation.valid:
        if date_a_on_or_after_b(best_echeance, best_creation):
            report.echeance_after_creation = True
        else:
            _flag_soft(
                report, "echeance", ErrorType.VALIDATION_ERROR,
                f"Échéance '{best_echeance.normalised}' is before "
                f"creation date '{best_creation.normalised}'"
            )

    # ------------------------------------------------------------------
    # Rule 8: RIB bank code matches domiciliation (SOFT)
    # ------------------------------------------------------------------
    best_rib = parsed.rib_best or parsed.rib_upper or parsed.rib_lower
    if (
        best_rib
        and best_rib.bank_code
        and not best_rib.has_question_marks
        and parsed.domiciliation_text
    ):
        if _bank_name_in_domiciliation(best_rib.bank_code, parsed.domiciliation_text):
            report.rib_bank_code_matches_domiciliation = True
        else:
            _flag_soft(
                report, "rib", ErrorType.VALIDATION_ERROR,
                f"Bank code '{best_rib.bank_code}' "
                f"({bank_name_for_code(best_rib.bank_code)}) "
                f"not found in domiciliation '{parsed.domiciliation_text}'"
            )

    # ------------------------------------------------------------------
    # Rule 9: Payment order number matches barcode
    # ------------------------------------------------------------------
    if parsed.payment_order_number and parsed.barcode_number:
        # Normalise: strip spaces, leading zeros optional
        po = re.sub(r"\s", "", parsed.payment_order_number)
        bc = re.sub(r"\s", "", parsed.barcode_number)
        if po == bc or po.lstrip("0") == bc.lstrip("0"):
            report.payment_order_matches_barcode = True
        else:
            _flag_hard(
                report, "payment_order", ErrorType.INCONSISTENCY,
                f"Payment order '{po}' ≠ barcode '{bc}'"
            )

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------
    def _field_conf(
        field_id: str,
        ocr_agree: float,
        redundancy: float,
        fmt_valid: float,
    ) -> float:
        eng_conf = confs.get(field_id, 0.5)
        return compute_field_confidence(ocr_agree, redundancy, fmt_valid, eng_conf)

    # RIB confidence
    rib_fmt   = 1.0 if (best_rib and best_rib.valid) else 0.0
    rib_redun = 1.0 if report.upper_lower.rib else (0.5 if (parsed.rib_upper or parsed.rib_lower) else 0.0)
    rib_agree = 1.0 if report.rib_key_valid else 0.3
    report.field_confidences["rib"] = _field_conf("R05", rib_agree, rib_redun, rib_fmt)

    # Amount confidence
    amt_fmt   = 1.0 if (best_amount and best_amount.valid) else 0.0
    amt_redun = 1.0 if report.upper_lower.amount else 0.5
    amt_agree = 1.0 if (best_amount and best_amount.valid) else 0.3
    report.field_confidences["amount_upper"] = _field_conf("R06", amt_agree, amt_redun, amt_fmt)

    # Échéance confidence
    echo_fmt   = 1.0 if (best_echeance and best_echeance.valid) else 0.0
    echo_redun = 1.0 if report.upper_lower.echeance else 0.5
    echo_agree = 1.0 if (best_echeance and best_echeance.valid) else 0.3
    report.field_confidences["echeance_upper"] = _field_conf("R02", echo_agree, echo_redun, echo_fmt)

    # Creation date confidence
    crea_fmt   = 1.0 if (best_creation and best_creation.valid) else 0.0
    crea_redun = 1.0 if report.upper_lower.creation_date else 0.5
    crea_agree = 1.0 if (best_creation and best_creation.valid) else 0.3
    report.field_confidences["creation_upper"] = _field_conf("R03", crea_agree, crea_redun, crea_fmt)

    report.document_confidence = compute_document_confidence(report.field_confidences)

    # Human review flag: any hard failure OR document confidence < 0.85
    report.needs_human_review = (
        report.hard_failures > 0
        or report.document_confidence < 0.85
    )

    return report


# ---------------------------------------------------------------------------
# Import needed inside validate_document
# ---------------------------------------------------------------------------
import re  # noqa: E402  (already imported at module level in helpers)
