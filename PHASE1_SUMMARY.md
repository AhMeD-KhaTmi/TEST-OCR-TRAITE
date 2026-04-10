# Phase 1 — Foundation: Summary Report

**Date completed**: 2026-04-10  
**Documents processed**: 100 (5 batches × 20 pages)  
**Pipeline status**: All 100 documents pass without errors

---

## What Was Built

Phase 1 delivers the complete foundation — preprocessing, alignment, and ROI extraction — on which all subsequent phases (OCR, parsing, validation) depend.

### Files delivered

| File | Description |
|------|-------------|
| `src/ocr_pipeline/preprocessing.py` | Full preprocessing pipeline: DPI normalisation, perspective correction, deskew, colour channel separation, adaptive binarisation |
| `src/ocr_pipeline/alignment.py` | Template alignment with 3-level fallback hierarchy: outer border → header+barcode → identity passthrough |
| `src/ocr_pipeline/roi_extractor.py` | ROI extraction from aligned image; returns colour, binarised, and blue-channel crops; tier filtering; draw/save helpers |
| `config/roi_config.json` | Calibrated ROI coordinates for all 17 fields (relative 0.0–1.0 fractions, with per-ROI padding) |
| `config/bank_codes.json` | Tunisian bank code lookup: 21 codes covering all active Tunisian commercial banks |
| `verify_phase1.py` | End-to-end verification script: runs full pipeline on every sample and saves overlay images + crop files + summary |
| `tests/test_preprocessing.py` | 30 unit tests for the preprocessing module |
| `tests/test_alignment.py` | 20 unit tests for the alignment module |
| `tests/test_roi_extractor.py` | 34 unit tests for ROI extraction |
| `tests/test_verify_helpers.py` | 6 unit tests for the verify script's `_make_doc_id` helper |

---

## Verification Results (100 documents)

| Metric | Result |
|--------|--------|
| Documents processed | **100 / 100** |
| Errors (unhandled exceptions) | **0** |
| Alignment method used | `header_barcode` — **100%** of documents |
| Alignment confidence | **0.90** — all documents (above the 0.7 gate) |
| ROI crops per document | **17** (all tiers) |
| Max skew detected | **2°** (2 documents); all others ≤ 1° |
| Documents with scale 1.74× | **100%** (all scanned at same resolution) |

---

## Problems Found and Fixed

### 1. verify_phase1.py — doc ID collision (bug)

**Problem**: The original verify script used:

```python
page = img_path.stem.split("page-")[-1]  # e.g. "0010"
```

For 4 of the 5 batches (`_1_page-`, `_2_page-`, `_3_page-`, `_4_page-`), this always produced `"0001"` through `"0020"`. Each batch overwrote the previous batch's output in `output/verify/<page>/`. The fifth batch (`_pages-to-jpg-`) happened to produce a different suffix (`s-to-jpg-XXXX`) by accident.

**Impact**: The previous run (before this fix) processed all 100 images but only kept the last batch's visual output. The summary.txt had only 20 entries (the last batch), making it appear Phase 1 had only been verified on 20 documents.

**Fix**: Replaced with a `_make_doc_id()` function that extracts both the batch number and page number to produce collision-free IDs:

- `_1_page-0001` → `1_0001`  
- `_4_page-0020` → `4_0020`  
- `_pages-to-jpg-0001` → `ptj_0001`

**Also fixed**: a secondary off-by-one in the `ptj` path — `split("_")[-1]` returned `"pages-to-jpg-0001"` rather than `"0001"` because that batch uses hyphens inside the last underscore-separated segment. Fixed with `.split("-")[-1]`.

---

### 2. pytest not installed in venv

**Problem**: Running `python -m pytest tests/` failed immediately — pytest was not in the venv.

**Fix**: `pip install pytest` added to the venv. No code change required.

---

### 3. Perspective correction always skips (expected, not a bug)

**Observation**: All 100 documents produce the warning:
> "Perspective correction skipped — document corners not detected."

**Root cause**: These are flatbed scans where the document fills the entire frame edge-to-edge. The `_find_document_corners()` function looks for a large quadrilateral contour covering ≥ 30% of the image, but because the form boundary is the image boundary, there is no interior contour to detect.

**Impact**: None. Flatbed scans do not have perspective distortion. The warning is informational.

**Action taken**: None required. This is the expected behaviour for this scan type. The warning is considered benign and documented here for future reference.

---

### 4. Outer border alignment strategy never activates (expected)

**Observation**: `align()` uses `header_barcode` for all 100 documents and never `border`.

**Root cause**: `_find_outer_border()` requires a large contour that covers ≥ 70% of the document's width and height. Because these are flatbed scans (same as above), no clear bordered form crops inward enough to register as a separate contour.

**Impact**: None. The `header_barcode` strategy produces a 0.90 confidence for all documents — above the 0.7 gate — so the fallback hierarchy is working as designed. The `border` strategy remains available for photographed documents (non-flatbed).

---

### 5. Scale factor constant at 1.74 × for all documents

**Observation**: Every document has `scale_factor=1.74`, meaning the raw scans are approximately 1426 × 1006 px (estimated input size for a 1.74× upscale to A4 at 300 DPI).

**Impact**: None. All documents are from one scan batch at a fixed resolution. The DPI normalisation step scales them correctly. If documents from a different scanner (e.g., higher-resolution at 600 DPI, scale ≈ 0.87) are added later, the normalisation handles them without code changes.

---

## Stale Artifacts

The `output/verify/` directory contains 20 legacy subdirectories (`0001/` through `0020/`) from the initial run before the doc ID collision bug was discovered. These contain the `pages-to-jpg` batch's crops (the last batch to run in sorted order). They are harmless but misleading — they can be safely deleted once Phase 2 begins.

---

## Test Coverage (84 tests, all passing)

| Test file | Tests | Coverage focus |
|-----------|-------|----------------|
| `test_preprocessing.py` | 30 | DPI scaling, deskew, colour channels, binarisation, full pipeline on real image |
| `test_alignment.py` | 20 | Anchor detection, align strategy, confidence gate, multi-sample smoke test |
| `test_roi_extractor.py` | 34 | Config loading, 17 ROIs present, crop dimensions, tier filtering, stamp padding, OCR engine assignments, save/draw helpers |
| `test_verify_helpers.py` | 6 | `_make_doc_id` correctness and uniqueness across all 100 filenames |

Run with: `venv/Scripts/python.exe -m pytest tests/ -v`

---

## What Phase 2 Receives

- **Input contract**: `preprocess(path)` → `align(result.deskewed)` → `extractor.extract_all(result.image)` returns a `dict[str, ROICrop]` with `colour`, `binarised`, and `blue_channel` crops plus OCR engine hints for each of 17 ROIs.
- **Tier 1 ROIs ready** (10 MVP fields): R02/R03/R12/R13 (dates), R05/R14 (RIBs), R06/R10 (amounts), R09 (amount text), R15 (drawee).
- **Stamp-affected ROIs padded** at 15-20% (R05, R07, R14, R15, R16) per plan section 3.2.
- **OCR engine routing**: `tesseract_then_qwen` for digit fields, `qwen` for handwriting, `barcode_decoder` for R17.
