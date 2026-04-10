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
│       ├── roi_extractor.py     # Crop 17 field regions from aligned image
│       ├── tesseract_ocr.py     # Tesseract wrapper: PSM modes, digit whitelists, confusion fixes
│       ├── qwen_ocr.py          # Qwen3 VL 8B client: field prompts, anti-hallucination, dual-mode
│       ├── barcode_decoder.py   # pyzbar primary + Tesseract OCR-B fallback
│       ├── ocr_engine.py        # OCR orchestrator: routes ROIs, merges Tess+Qwen results
│       ├── rib_parser.py        # Tunisian RIB parser: mod-97 key, bank code lookup
│       ├── date_parser.py       # Date parser: DD/MM/YYYY, OCR corrections, 2-digit year
│       ├── amount_parser.py     # Numeric amount + French number-word parser
│       ├── name_parser.py       # Name/address: normalisation, company detection, stamp markers
│       ├── validator.py         # Cross-field validation: 9 rules, confidence scoring
│       └── document_result.py   # Full JSON schema assembler: DocumentResult
├── config/
│   ├── roi_config.json          # Calibrated ROI coordinates (all 17 fields)
│   └── bank_codes.json          # Tunisian bank code → bank name lookup (21 banks)
├── tests/
│   ├── test_preprocessing.py    # 30 tests
│   ├── test_alignment.py        # 20 tests
│   ├── test_roi_extractor.py    # 34 tests
│   ├── test_verify_helpers.py   # 6 tests
│   ├── test_tesseract_ocr.py    # 30 tests
│   ├── test_ocr_engine.py       # 39 tests
│   ├── test_barcode_decoder.py  # 23 tests
│   ├── test_qwen_ocr.py         # 42 tests
│   ├── test_rib_parser.py       # 13 tests
│   ├── test_amount_parser.py    # 41 tests
│   ├── test_date_parser.py      # 38 tests
│   ├── test_name_parser.py      # 22 tests
│   ├── test_validator.py        # 38 tests
│   └── test_document_result.py  # 28 tests
├── example/                     # 100 sample scans (5 batches × 20 pages)
├── output/                      # Generated — gitignored
├── .env.example                 # Inference config template (copy to .env)
├── verify_phase1.py             # Phase 1 end-to-end verification
├── verify_phase2.py             # Phase 2 end-to-end verification
├── verify_phase3.py             # Phase 3 end-to-end verification (full pipeline)
├── Tunisian-Lettre-de-Change-OCR-Plan.md   # Full technical blueprint
├── PHASE1_SUMMARY.md            # Phase 1 completion report
├── PHASE2_SUMMARY.md            # Phase 2 completion report
└── PHASE3_SUMMARY.md            # Phase 3 completion report
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
pip install opencv-python numpy pytest pytesseract pyzbar pillow python-dotenv requests
```

**Tesseract OCR** must be installed separately:
- Windows: download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr tesseract-ocr-fra`

**Qwen3 VL 8B** (optional — required only for `--with-qwen` runs):
- Copy `.env.example` to `.env` and fill in your inference config
- **API mode**: set `INFERENCE_MODE=api` and `OPENROUTER_API_KEY` in `.env`
- **Local mode**: set `INFERENCE_MODE=local` and run `ollama serve` with `qwen3-vl-8b` loaded

---

## Quick Start

### Run Phase 1 — preprocessing → alignment → ROI extraction

```bash
venv\Scripts\python.exe verify_phase1.py
```

Output saved to `output/verify_phase1/`:
- `<doc_id>/overlay.jpg` — full document with all 17 ROI boxes drawn
- `<doc_id>/crops/` — individual colour and binarised crops per field
- `summary.txt` — alignment method, confidence, and skew per document

### Run Phase 2 — OCR engine (Tesseract + Qwen routing)

```bash
# Tesseract-only baseline (no API key required)
venv\Scripts\python.exe verify_phase2.py

# With live Qwen inference (uses API tokens or local server)
venv\Scripts\python.exe verify_phase2.py --with-qwen
```

Output saved to `output/verify_phase2/`:
- `<doc_id>/ocr_results.json` — raw OCR output for all 17 fields
- `<doc_id>/overlay.jpg` — ROI overlay image
- `summary.txt` — per-document OCR summary

### Run Phase 3 — full pipeline (parsing + validation + JSON output)

```bash
# Tesseract-only baseline
venv\Scripts\python.exe verify_phase3.py

# With live Qwen inference
venv\Scripts\python.exe verify_phase3.py --with-qwen
```

Output saved to `output/verify_phase3/`:
- `<doc_id>/document_result.json` — fully validated structured JSON per document
- `<doc_id>/overlay.jpg` — ROI overlay image
- `summary.txt` — per-document validation summary (RIB key, amounts, dates, confidence)

### Run all tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

**385 tests, all passing.**

---

## Pipeline Overview

```
Image Input
    |
    v
[1. Preprocessing]    deskew, denoise, binarise, colour channel separation    (Phase 1)
    |
    v
[2. Alignment]        detect header band + barcode anchors -> affine transform  (Phase 1)
    |
    v
[3. ROI Extraction]   crop 17 field regions (colour + binarised + blue-channel) (Phase 1)
    |
    v
[4. Multi-Pass OCR]   Pass 1: Tesseract on binarised crops                      (Phase 2)
                      Pass 2: Qwen3 VL 8B on colour crops (escalation only)
                      Pass 3: Qwen3 VL 8B full-document anomaly explanation     (Phase 4)
    |
    v
[5. Field Parsing]    RIB, date, amount (numeric + words), name parsers         (Phase 3)
    |
    v
[6. Cross-Validation] 9 rules: RIB mod-97, upper/lower consistency,             (Phase 3)
                      date ordering, bank code vs domiciliation, barcode vs R01
    |
    v
[7. Confidence]       per-field (0.4xOCR + 0.3xredundancy + 0.2xformat + 0.1xengine)
    |
    v
[8. JSON Output]      DocumentResult with all fields, validation, flags         (Phase 3)
```

---

## Fields Extracted (17 ROIs)

| ROI | Field | Tier | OCR Engine |
|-----|-------|------|------------|
| R01 | Payment order number | 2 | Tesseract (digits) |
| R02 | Echeance — upper | 1 | Tesseract -> Qwen fallback |
| R03 | Date de creation — upper | 1 | Tesseract -> Qwen fallback |
| R04 | City — upper | 3 | Qwen |
| R05 | RIB — upper | 1 | Tesseract -> Qwen fallback |
| R06 | Montant chiffres — upper | 1 | Tesseract -> Qwen fallback |
| R07 | Tireur | 2 | Qwen |
| R08 | Beneficiaire | 2 | Qwen |
| R09 | Montant en lettres | 1 | Qwen |
| R10 | Montant chiffres — lower | 1 | Tesseract -> Qwen fallback |
| R11 | Lieu de creation — lower | 3 | Qwen |
| R12 | Date de creation — lower | 1 | Tesseract -> Qwen fallback |
| R13 | Echeance — lower | 1 | Tesseract -> Qwen fallback |
| R14 | RIB — lower | 1 | Tesseract -> Qwen fallback |
| R15 | Nom et adresse du Tire | 1 | Qwen |
| R16 | Domiciliation | 2 | Qwen |
| R17 | Barcode | 2 | Barcode decoder (pyzbar + OCR-B fallback) |

**Tier 1** = MVP fields (RIB, amounts, dates, drawee).  
**Tier 2** = secondary fields (tireur, beneficiary, domiciliation, barcode).  
**Tier 3** = city — low priority, simple once pipeline is stable.

---

## Key Design Decisions

- **ROI-based extraction** over full-page OCR — all 100 samples confirm a fixed government-standardised template layout.
- **Tesseract-first for digits** — RIBs and amounts use Tesseract with a digit whitelist as the primary engine; Qwen3 VL 8B is a fallback only. Qwen is unreliable for exact digit extraction.
- **Redundancy as the primary validation weapon** — RIB, amount, dates, and echeance all appear twice (upper and lower halves). Mismatches are detected automatically.
- **Reject over guess** — fields that fail validation are flagged for human review rather than returned with a guessed value.
- **Anti-hallucination by design** — Qwen receives a strict system prompt forbidding guessing, normalisation, or digit completion. A digit-divergence rule (>30% of Tesseract digits changed -> reject Qwen output) catches silent hallucinations.
- **Dual-mode inference** — switch between OpenRouter API and a local server (Ollama / vLLM / LMStudio) via `.env` with no code changes.

See [`Tunisian-Lettre-de-Change-OCR-Plan.md`](Tunisian-Lettre-de-Change-OCR-Plan.md) for the full technical blueprint.

---

## Implementation Status

| Phase | Scope | Status | Tests |
|-------|-------|--------|-------|
| **Phase 1 — Foundation** | Preprocessing, alignment, ROI extraction | Complete | 90 passing |
| **Phase 2 — Core OCR** | Tesseract + Qwen integration, barcode decoding, result merging | Complete | 141 passing |
| **Phase 3 — Parsing & Validation** | Field parsers, cross-validation engine, confidence scoring, JSON output | Complete | 154 passing |
| **Phase 4 — Stamp Handling** | Stamp detection, multi-crop strategy, anomaly explanation pass | Pending | — |
| **Phase 5 — Production** | Human review interface, logging, performance optimisation | Pending | — |

**Total: 385 tests, 0 failures.**

---

## Output JSON Schema

The full `DocumentResult` schema produced by `build_document_result()`:

```json
{
  "document_id": "1_0001",
  "payment_order_number": "01118862916",
  "extraction_timestamp": "2026-04-10T22:00:00Z",

  "rib": {
    "bank_code": "08",
    "branch_code": "006",
    "account_number": "0110510000870",
    "key": "41",
    "full": "08006011051000087041",
    "key_valid": true,
    "bank_name": "BIAT",
    "confidence": 0.85,
    "source": "lower",
    "raw_ocr": { "upper": "...", "lower": "..." },
    "needs_review": false
  },

  "amount": {
    "value_numeric": "3000.000",
    "value_text": "trois mille dinars",
    "currency": "DT",
    "numeric_text_match": true,
    "confidence": 0.90,
    "raw_ocr": { "numeric_upper": "...", "numeric_lower": "...", "text": "..." }
  },

  "echeance":      { "value": "30/06/2025", "confidence": 0.92, "source": "merged" },
  "creation_date": { "value": "01/01/2025", "confidence": 0.92, "source": "merged" },
  "creation_city": { "value": "TUNIS",      "confidence": 0.70, "source": "upper" },

  "tireur":        { "value": "STE DELTA SARL", "confidence": 0.75, "source": "single" },
  "beneficiary":   { "value": "AHMED BEN ALI",  "confidence": 0.70, "source": "single" },
  "tire":          { "value": "BIAT SA",         "confidence": 0.80, "source": "single" },
  "domiciliation": { "value": "BIAT AGENCE TUNIS", "confidence": 0.72, "source": "single" },

  "validation": {
    "rib_key_valid": true,
    "rib_bank_code_matches_domiciliation": true,
    "amount_numeric_matches_text": true,
    "echeance_after_creation": true,
    "payment_order_matches_barcode": false,
    "upper_lower_consistency": {
      "rib": true,
      "amount": true,
      "echeance": true,
      "creation_date": true
    }
  },

  "document_confidence": 0.88,
  "needs_human_review": false,
  "qwen_corrections": {
    "rib_digits_changed": 0,
    "amount_digits_changed": 0,
    "date_digits_changed": 0,
    "source": "tesseract_only"
  },
  "flagged_fields": []
}
```

---

## Known Limitations

1. **Confidence formula is heuristic** — weights (0.4/0.3/0.2/0.1) are reasonable defaults. Must be calibrated with Platt scaling on 200+ labelled documents before production use.
2. **Amount text parser is fragile** — French number words + OCR noise = unreliable parse. Numeric amount is always authoritative; text amount is soft validation only.
3. **Stamp handling relies on LLM prompt compliance** — the strongest safety nets are cross-field redundancy and RIB mod-97 validation, not LLM stamp awareness. A dedicated stamp segmentation model is planned for Phase 4.
4. **CMC-7 barcode (R17) unreadable** — pyzbar does not support CMC-7 MICR font. ~39% of documents have an unreadable barcode flag. A dedicated CMC-7 decoder is tracked for Phase 4.
