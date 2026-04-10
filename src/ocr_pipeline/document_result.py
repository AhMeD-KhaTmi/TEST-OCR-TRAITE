"""
Phase 3 — Document result: full JSON output schema.

Assembles the final structured output for a single Lettre de Change document
by combining the OCR engine output (Phase 2) with the parsed and validated
results (Phase 3 parsers + validator).

Public API
----------
build_document_result(doc_id, ocr_batch_result, *, skip_qwen=False) -> DocumentResult
document_result_to_dict(result: DocumentResult) -> dict
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any

from .ocr_engine import OCRBatch, OCRFieldResult
from .rib_parser import parse_rib, RIBResult
from .date_parser import parse_date, DateResult
from .amount_parser import parse_amount_numeric, parse_amount_words, AmountResult, AmountWordsResult
from .name_parser import parse_name, NameResult
from .validator import (
    ParsedDocument,
    ValidationReport,
    validate_document,
)


# ---------------------------------------------------------------------------
# Output schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FieldValue:
    value: Optional[str]
    confidence: float
    source: str                   # "upper" | "lower" | "merged" | "single"
    raw_ocr: dict[str, str] = dc_field(default_factory=dict)
    needs_review: bool = False


@dataclass
class RIBValue:
    bank_code: str
    branch_code: str
    account_number: str
    key: str
    full: str
    key_valid: bool
    bank_name: Optional[str]
    confidence: float
    source: str
    raw_ocr: dict[str, str] = dc_field(default_factory=dict)
    needs_review: bool = False


@dataclass
class AmountValue:
    value_numeric: Optional[str]   # Decimal as string e.g. "3000.000"
    value_text: Optional[str]      # raw amount-in-words OCR
    currency: str = "DT"
    numeric_text_match: bool = False
    confidence: float = 0.0
    raw_ocr: dict[str, str] = dc_field(default_factory=dict)
    needs_review: bool = False


@dataclass
class FlaggedField:
    field: str
    error_type: str
    message: str
    is_hard_failure: bool = False


@dataclass
class QwenCorrections:
    rib_digits_changed: int = 0
    amount_digits_changed: int = 0
    date_digits_changed: int = 0
    source: str = "tesseract_only"


@dataclass
class DocumentResult:
    document_id: str
    payment_order_number: Optional[str]
    extraction_timestamp: str

    echeance: Optional[FieldValue]
    creation_date: Optional[FieldValue]
    creation_city: Optional[FieldValue]

    rib: Optional[RIBValue]
    amount: Optional[AmountValue]

    tireur: Optional[FieldValue]
    beneficiary: Optional[FieldValue]
    tire: Optional[FieldValue]
    domiciliation: Optional[FieldValue]

    validation: dict[str, Any]
    document_confidence: float
    needs_human_review: bool
    qwen_corrections: QwenCorrections
    flagged_fields: list[FlaggedField] = dc_field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(batch: OCRBatch, roi_id: str) -> Optional[OCRFieldResult]:
    for f in batch.fields:
        if f.roi_id == roi_id:
            return f
    return None


def _text(field: Optional[OCRFieldResult]) -> str:
    return field.final_text if field else ""


def _conf(field: Optional[OCRFieldResult]) -> float:
    return field.confidence if field else 0.0


def _merge_redundant_date(upper: Optional[DateResult], lower: Optional[DateResult]) -> tuple[Optional[str], str, float]:
    """Choose the best date value; return (normalised_str, source, confidence)."""
    if upper and upper.valid and lower and lower.valid:
        if upper.value == lower.value:
            return upper.normalised, "merged", 1.0
        # Prefer upper (closer to the top of the document)
        return upper.normalised, "upper", 0.6
    if upper and upper.valid:
        return upper.normalised, "upper", 0.7
    if lower and lower.valid:
        return lower.normalised, "lower", 0.7
    return None, "upper", 0.0


def _best_rib(upper: Optional[RIBResult], lower: Optional[RIBResult]) -> tuple[Optional[RIBResult], str]:
    """Choose the best RIB result: prefer key-valid, then non-question-mark."""
    if upper and upper.key_valid:
        if lower and lower.key_valid:
            return upper, "merged"  # both valid and equal (validated by caller)
        return upper, "upper"
    if lower and lower.key_valid:
        return lower, "lower"
    # Neither key-valid — prefer the one without question marks
    if upper and not upper.has_question_marks:
        return upper, "upper"
    if lower and not lower.has_question_marks:
        return lower, "lower"
    return upper, "upper"


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def build_document_result(
    doc_id: str,
    batch: OCRBatch,
) -> DocumentResult:
    """Build the final document result from a completed OCR batch.

    Runs all Phase 3 parsers and the validator internally.
    """
    ts = datetime.utcnow().isoformat() + "Z"

    # ----------------------------------------------------------------
    # Extract raw OCR texts per ROI
    # ----------------------------------------------------------------
    r01 = _get_field(batch, "R01")
    r02 = _get_field(batch, "R02")
    r03 = _get_field(batch, "R03")
    r04 = _get_field(batch, "R04")
    r05 = _get_field(batch, "R05")
    r06 = _get_field(batch, "R06")
    r07 = _get_field(batch, "R07")
    r08 = _get_field(batch, "R08")
    r09 = _get_field(batch, "R09")
    r10 = _get_field(batch, "R10")
    r11 = _get_field(batch, "R11")
    r12 = _get_field(batch, "R12")
    r13 = _get_field(batch, "R13")
    r14 = _get_field(batch, "R14")
    r15 = _get_field(batch, "R15")
    r16 = _get_field(batch, "R16")
    r17 = _get_field(batch, "R17")

    # ----------------------------------------------------------------
    # Parse all fields
    # ----------------------------------------------------------------
    rib_upper  = parse_rib(_text(r05))  if r05 else None
    rib_lower  = parse_rib(_text(r14))  if r14 else None
    echo_upper = parse_date(_text(r02)) if r02 else None
    echo_lower = parse_date(_text(r13)) if r13 else None
    crea_upper = parse_date(_text(r03)) if r03 else None
    crea_lower = parse_date(_text(r12)) if r12 else None
    amt_upper  = parse_amount_numeric(_text(r06)) if r06 else None
    amt_lower  = parse_amount_numeric(_text(r10)) if r10 else None
    amt_words  = parse_amount_words(_text(r09))   if r09 else None
    tireur_res = parse_name(_text(r07)) if r07 else None
    bene_res   = parse_name(_text(r08)) if r08 else None
    tire_res   = parse_name(_text(r15)) if r15 else None

    # ----------------------------------------------------------------
    # Validate
    # ----------------------------------------------------------------
    best_rib, rib_source = _best_rib(rib_upper, rib_lower)

    parsed = ParsedDocument(
        rib_upper=rib_upper,
        rib_lower=rib_lower,
        rib_best=best_rib,
        echeance_upper=echo_upper,
        echeance_lower=echo_lower,
        creation_upper=crea_upper,
        creation_lower=crea_lower,
        amount_upper=amt_upper,
        amount_lower=amt_lower,
        amount_words=amt_words,
        tireur=tireur_res,
        beneficiary=bene_res,
        drawee=tire_res,
        domiciliation_text=_text(r16),
        payment_order_number=_text(r01),
        barcode_number=_text(r17),
    )

    ocr_confs = {f.roi_id: f.confidence for f in batch.fields}
    report: ValidationReport = validate_document(parsed, ocr_confs)

    # ----------------------------------------------------------------
    # Assemble RIB value
    # ----------------------------------------------------------------
    rib_value: Optional[RIBValue] = None
    if best_rib and best_rib.digits:
        rib_value = RIBValue(
            bank_code=best_rib.bank_code,
            branch_code=best_rib.branch_code,
            account_number=best_rib.account_number,
            key=best_rib.key,
            full=best_rib.digits,
            key_valid=best_rib.key_valid,
            bank_name=best_rib.bank_name,
            confidence=report.field_confidences.get("rib", 0.0),
            source=rib_source,
            raw_ocr={"upper": _text(r05), "lower": _text(r14)},
            needs_review=not best_rib.valid,
        )

    # ----------------------------------------------------------------
    # Assemble amount value
    # ----------------------------------------------------------------
    best_amt = amt_upper if (amt_upper and amt_upper.valid) else amt_lower
    amount_value: Optional[AmountValue] = None
    if best_amt:
        amount_value = AmountValue(
            value_numeric=best_amt.normalised if best_amt.valid else None,
            value_text=_text(r09),
            currency="DT",
            numeric_text_match=report.amount_numeric_matches_words,
            confidence=report.field_confidences.get("amount_upper", 0.0),
            raw_ocr={
                "numeric_upper": _text(r06),
                "numeric_lower": _text(r10),
                "text": _text(r09),
            },
            needs_review=not best_amt.valid,
        )

    # ----------------------------------------------------------------
    # Assemble date values
    # ----------------------------------------------------------------
    echo_val, echo_src, echo_conf = _merge_redundant_date(echo_upper, echo_lower)
    crea_val, crea_src, crea_conf = _merge_redundant_date(crea_upper, crea_lower)

    echeance_fv = FieldValue(
        value=echo_val,
        confidence=report.field_confidences.get("echeance_upper", echo_conf),
        source=echo_src,
        raw_ocr={"upper": _text(r02), "lower": _text(r13)},
        needs_review=echo_val is None,
    )
    creation_fv = FieldValue(
        value=crea_val,
        confidence=report.field_confidences.get("creation_upper", crea_conf),
        source=crea_src,
        raw_ocr={"upper": _text(r03), "lower": _text(r12)},
        needs_review=crea_val is None,
    )

    # ----------------------------------------------------------------
    # City (no redundancy logic needed — just validate non-empty)
    # ----------------------------------------------------------------
    city_upper = _text(r04)
    city_lower = _text(r11)
    city_val   = city_upper or city_lower or None
    city_src   = "upper" if city_upper else ("lower" if city_lower else "single")
    city_fv = FieldValue(
        value=city_val,
        confidence=_conf(r04) if city_upper else _conf(r11),
        source=city_src,
        raw_ocr={"upper": city_upper, "lower": city_lower},
        needs_review=city_val is None,
    )

    # ----------------------------------------------------------------
    # Name fields
    # ----------------------------------------------------------------
    def _name_fv(res: Optional[NameResult], roi: Optional[OCRFieldResult]) -> Optional[FieldValue]:
        if res is None:
            return None
        return FieldValue(
            value=res.value if res.valid else None,
            confidence=_conf(roi),
            source="single",
            raw_ocr={"raw": res.raw},
            needs_review=not res.valid or res.has_stamp_occlusion,
        )

    # ----------------------------------------------------------------
    # Validation summary dict
    # ----------------------------------------------------------------
    validation_dict = {
        "rib_key_valid": report.rib_key_valid,
        "rib_bank_code_matches_domiciliation": report.rib_bank_code_matches_domiciliation,
        "amount_numeric_matches_text": report.amount_numeric_matches_words,
        "echeance_after_creation": report.echeance_after_creation,
        "payment_order_matches_barcode": report.payment_order_matches_barcode,
        "upper_lower_consistency": {
            "rib":           report.upper_lower.rib,
            "amount":        report.upper_lower.amount,
            "echeance":      report.upper_lower.echeance,
            "creation_date": report.upper_lower.creation_date,
        },
    }

    # ----------------------------------------------------------------
    # Qwen corrections summary
    # ----------------------------------------------------------------
    qwen_corr = QwenCorrections(source="tesseract_only")
    for f in batch.fields:
        if f.rule_applied in ("rule2_disagree_use_qwen",):
            qwen_corr.source = "qwen_fallback"
            break
        if f.rule_applied == "rule0_qwen_unavailable":
            qwen_corr.source = "tesseract_only"

    # ----------------------------------------------------------------
    # Flagged fields
    # ----------------------------------------------------------------
    flagged = [
        FlaggedField(
            field=fl.field,
            error_type=fl.error_type,
            message=fl.message,
            is_hard_failure=fl.is_hard_failure,
        )
        for fl in report.flags
    ]

    return DocumentResult(
        document_id=doc_id,
        payment_order_number=_text(r01) or None,
        extraction_timestamp=ts,
        echeance=echeance_fv,
        creation_date=creation_fv,
        creation_city=city_fv,
        rib=rib_value,
        amount=amount_value,
        tireur=_name_fv(tireur_res, r07),
        beneficiary=_name_fv(bene_res, r08),
        tire=_name_fv(tire_res, r15),
        domiciliation=FieldValue(
            value=_text(r16) or None,
            confidence=_conf(r16),
            source="single",
            raw_ocr={"raw": _text(r16)},
            needs_review=not bool(_text(r16)),
        ),
        validation=validation_dict,
        document_confidence=report.document_confidence,
        needs_human_review=report.needs_human_review,
        qwen_corrections=qwen_corr,
        flagged_fields=flagged,
    )


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _decimal_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def document_result_to_dict(result: DocumentResult) -> dict:
    """Convert a DocumentResult to a plain dict suitable for JSON serialisation."""
    d = asdict(result)
    return d


def document_result_to_json(result: DocumentResult, indent: int = 2) -> str:
    """Serialise a DocumentResult to a JSON string."""
    return json.dumps(document_result_to_dict(result), indent=indent, default=_decimal_default)
