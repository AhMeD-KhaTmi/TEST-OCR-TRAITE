# Tunisian Lettre de Change — OCR Pipeline

A production-grade OCR extraction pipeline for Tunisian bills of exchange (*lettres de change*). Extracts all financial fields from scanned documents using ROI-based field isolation, multi-pass OCR (Tesseract + Qwen3 VL 8B), cross-field validation exploiting built-in document redundancy, and structured JSON output with per-field confidence scoring.

---

## Project Structure

```
OCR TRAITE/
├── src/
│   └── ocr_pipeline/
│       ├── preprocessing.py     # DPI normalisation, deskew, binarisation
│       ├── alignment.py         # Anchor detection, affine transform
│       └── roi_extractor.py     # Crop 17 field regions from aligned image
├── config/
│   ├── roi_config.json          # Calibrated ROI coordinates (all 17 fields)
│   └── bank_codes.json          # Tunisian bank code → bank name lookup
├── tests/
│   ├── test_preprocessing.py
│   ├── test_alignment.py
│   ├── test_roi_extractor.py
│   └── test_verify_helpers.py
├── example/                     # 100 sample scans (5 batches × 20 pages)
├── output/                      # Generated — gitignored
├── verify_phase1.py             # End-to-end Phase 1 verification script
├── Tunisian-Lettre-de-Change-OCR-Plan.md   # Full technical blueprint
└── PHASE1_SUMMARY.md            # Phase 1 completion report
```

---

## Setup

**Requirements**: Python 3.10+, Windows/Linux/macOS

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install opencv-python numpy pytest
```

> Tesseract and Qwen3 VL 8B are required for Phase 2 (OCR). Phase 1 (preprocessing + alignment + ROI extraction) runs on OpenCV and NumPy only.

---

## Quick Start

### Run the Phase 1 verification (preprocessing → alignment → ROI extraction)

```bash
venv\Scripts\python.exe verify_phase1.py
```

Output is saved to `output/verify/`:
- `output/verify/<doc_id>/overlay.jpg` — full document with all 17 ROI boxes drawn
- `output/verify/<doc_id>/crops/` — individual colour and binarised crops per field
- `output/verify/summary.txt` — alignment method, confidence, and skew per document

### Run tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

84 tests, all passing.

---

## Pipeline Overview

```
Image Input
    │
    ▼
[1. Preprocessing]   deskew, denoise, binarise, colour channel separation
    │
    ▼
[2. Alignment]       detect header band + barcode anchors → affine transform
    │
    ▼
[3. ROI Extraction]  crop 17 field regions (colour + binarised + blue-channel)
    │
    ▼
[4. Multi-Pass OCR]  Pass 1: Tesseract on clean crops
                     Pass 2: Qwen3 VL 8B on original crops
                     Pass 3: Qwen3 VL 8B full-document anomaly explanation
    │
    ▼
[5. Field Parsing]   RIB, date, amount, name parsers
    │
    ▼
[6. Cross-Validation] redundancy checks, RIB mod-97 key, format rules
    │
    ▼
[7. Confidence Scoring] per-field and document-level
    │
    ▼
[8. JSON Output]     structured result + flagged fields for human review
```

---

## Fields Extracted (17 ROIs)

| ROI | Field | Tier | OCR Engine |
|-----|-------|------|-----------|
| R01 | Payment order number | 2 | Tesseract (digits) |
| R02 | Échéance — upper | 1 | Tesseract |
| R03 | Date de création — upper | 1 | Tesseract |
| R04 | City — upper | 3 | Qwen |
| R05 | RIB — upper | 1 | Tesseract → Qwen fallback |
| R06 | Montant chiffres — upper | 1 | Tesseract → Qwen fallback |
| R07 | Tireur | 2 | Qwen |
| R08 | Bénéficiaire | 2 | Qwen |
| R09 | Montant en lettres | 1 | Qwen |
| R10 | Montant chiffres — lower | 1 | Tesseract → Qwen fallback |
| R11 | Lieu de création — lower | 3 | Qwen |
| R12 | Date de création — lower | 1 | Tesseract |
| R13 | Échéance — lower | 1 | Tesseract |
| R14 | RIB — lower | 1 | Tesseract → Qwen fallback |
| R15 | Nom et adresse du Tiré | 1 | Qwen |
| R16 | Domiciliation | 2 | Qwen |
| R17 | Barcode | 2 | Barcode decoder |

**Tier 1** = MVP fields (RIB, amounts, dates, drawee) — implement first.  
**Tier 2/3** = add after Tier 1 is stable.

---

## Key Design Decisions

- **ROI-based extraction** over full-page OCR — all 100 samples confirm a fixed government-standardised template layout.
- **Tesseract-first for digits** — RIBs and amounts use Tesseract with a digit whitelist as the primary engine; Qwen3 VL 8B is a fallback only. Qwen is unreliable for exact digit extraction.
- **Redundancy as the primary validation weapon** — RIB, amount, dates, and échéance all appear twice (upper and lower halves). Mismatches are detected automatically.
- **Reject over guess** — fields that fail validation are flagged for human review rather than returned with a guessed value.
- **Anti-hallucination by design** — Qwen receives a strict system prompt forbidding guessing, normalisation, or digit completion. A digit-divergence rule (>30% of Tesseract digits changed → reject Qwen output) catches silent hallucinations.

See [`Tunisian-Lettre-de-Change-OCR-Plan.md`](Tunisian-Lettre-de-Change-OCR-Plan.md) for the full technical blueprint.

---

## Implementation Status

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1 — Foundation** | Preprocessing, alignment, ROI extraction, verification | ✅ Complete |
| **Phase 2 — Core OCR** | Tesseract + Qwen integration, barcode decoding, result merging | Pending |
| **Phase 3 — Parsing & Validation** | Field parsers, cross-validation engine, confidence scoring | Pending |
| **Phase 4 — Stamp Handling** | Stamp detection, multi-crop strategy, anomaly explanation pass | Pending |
| **Phase 5 — Production** | JSON output, human review interface, logging, performance | Pending |

---

## Output JSON Schema (target)

```json
{
  "document_id": "string",
  "rib": {
    "full": "20-digit string",
    "key_valid": true,
    "confidence": 0.97,
    "source": "lower"
  },
  "amount": {
    "value_numeric": 3000.000,
    "value_text": "trois mille dinars",
    "numeric_text_match": true,
    "confidence": 0.95
  },
  "echeance": { "value": "30/06/2025", "confidence": 0.99 },
  "validation": {
    "rib_key_valid": true,
    "upper_lower_consistency": { "rib": true, "amount": true }
  },
  "document_confidence": 0.95,
  "needs_human_review": false,
  "flagged_fields": []
}
```

---

## Known Limitations (current)

1. **Confidence formula is heuristic** — weights (0.4/0.3/0.2/0.1) are reasonable defaults. Must be calibrated with Platt scaling on 200+ labelled documents before production use.
2. **Amount text parser is fragile** — French number words + OCR noise = unreliable parse. Numeric amount is always authoritative; text amount is soft validation only.
3. **Stamp handling relies on LLM prompt compliance** — the strongest safety nets are cross-field redundancy and RIB mod-97 validation, not LLM stamp awareness.
