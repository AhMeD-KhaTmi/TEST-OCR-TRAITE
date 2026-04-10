# Tunisian Lettre de Change OCR System

## TL;DR
Build a production-grade OCR extraction pipeline for Tunisian bills of exchange using ROI-based field isolation, multi-pass OCR (Tesseract + Qwen3 VL 8B local vision LLM), strong cross-field validation exploiting the document's built-in redundancy (fields appear twice), and structured JSON output with confidence scoring. No code — this is a technical blueprint.

---

## 1. Executive Summary

**System**: An automated document intelligence pipeline that ingests scanned images of Tunisian "lettres de change" (bills of exchange), extracts all financial fields with exact fidelity, validates them against business rules, and outputs structured JSON.

**Key Challenges** (observed from 20 real samples):
- **Stamp occlusion**: Blue circular business stamps (acceptation, aval, tireur) frequently overlap critical fields — especially RIB numbers, drawee info, and bank domiciliation. Stamps contain their own text (phone numbers, MF codes) that creates false field readings.
- **Handwriting variability**: Documents range from clean machine-printed (e.g., page-0010, page-0018) to fully handwritten with poor legibility (e.g., page-0003 — messy cursive amounts, dates, RIBs). No single OCR engine handles both equally well.
- **Amount format inconsistency**: Observed formats include 3000,000, #5.000.000#, 2,893,192, 500,000, 25 000,000 DT, # 3 000.000#. Hash delimiters, spaces, dots, and commas are used interchangeably.
- **Hallucination risk**: Vision LLMs tend to "guess" plausible-looking RIB digits or invent amount text when stamps occlude the actual data. This is unacceptable for financial documents.
- **Field duplication as opportunity**: The template deliberately prints RIB, amount, and échéance twice (upper half and lower half). This built-in redundancy is the system's strongest validation weapon.

**High-Level Approach**: ROI-based crop -> multi-pass OCR (classical + LLM) -> field-specific parsers -> cross-field validation exploiting redundancy -> confidence-scored JSON output. Reject low-confidence extractions rather than hallucinate.

---

## 2. System Architecture

**End-to-End Pipeline**:

```text
Image Input
    │
    ▼
[1. Preprocessing] -> deskew, denoise, binarize, normalize
    │
    ▼
[2. Template Alignment] -> detect anchor points, compute affine transform
    │
    ▼
[3. ROI Extraction] -> crop 15+ field regions from aligned image
    │
    ▼
[4. Multi-Pass OCR] -> Pass 1: Tesseract on clean crops
                     -> Pass 2: Vision LLM on original crops
                     -> Pass 3: Vision LLM on full document (anomaly explanation)
    │
    ▼
[5. Field Parsing] -> type-specific parsers (RIB, date, amount, name)
    │
    ▼
[6. Cross-Validation] -> redundancy checks, format rules, Luhn-like key verification
    │
    ▼
[7. Confidence Scoring] -> per-field and document-level confidence
    │
    ▼
[8. Output] -> Structured JSON + flagged fields for human review
```

**Data flow**: Each component passes a DocumentResult object forward, accumulating raw OCR results, parsed values, validation flags, and confidence scores. Original crops are preserved for audit.

---

## 3. Detailed Pipeline Design

### 3.1 Image Preprocessing

**Techniques**:
- **Deskew**: Detect document skew via Hough line transform on the printed border lines and the header band. Correct rotation to ±0.5°. The header "Lettre de Change" band and the bottom barcode line are reliable angle references.
- **Perspective correction**: If the document was photographed (not scanned), apply 4-point perspective transform using the document's rectangular border corners detected via contour analysis.
- **Adaptive binarization**: Use Sauvola or Niblack binarization (not global Otsu) because the document has:
  - Red/pink pre-printed header band
  - Blue handwritten ink
  - Blue stamp ink
  - Black printed text
  - Background varies from white to yellowish
- **Noise reduction**: Gentle Gaussian blur (σ=0.5) before binarization only. Aggressive denoising destroys handwritten thin strokes.
- **Color channel separation**: Extract blue channel specifically for handwritten content (handwriting and stamps are blue; pre-printed form is red/black). This naturally suppresses the red header band and isolates handwritten text.
- **Resolution normalization**: Resample to 300 DPI equivalent. Documents scanned at lower resolution lose digit legibility in RIB fields.

**Why needed**: The samples show varied scan quality — some are clean flatbed scans, others show shadows, slight rotations, and background noise. Preprocessing must normalize without destroying handwritten detail.

### 3.2 ROI Extraction Strategy

**Why fixed ROIs are optimal**: All 20 samples confirm the lettre de change is a government-standardized template. The form layout, box positions, and label positions are identical across all samples. ROI coordinates can be calibrated once and applied universally (after alignment).

**Anchor-based alignment before extraction** (with fallback hierarchy):

**Primary anchors**:
1. Detect the header band ("Lettre de Change" / "Bill of Exchange") — it spans the full width with a distinctive red/stippled pattern. Use template matching or color segmentation.
2. Detect the barcode region at the bottom — it's always present and provides a reliable bottom anchor.
3. Detect the outer border rectangle of the form.
4. Compute affine transform mapping detected anchors to reference coordinates.

**Fallback anchors** (if primary fails — e.g., cropped pages, damaged borders):
5. Text label detection: locate printed labels ("échéance", "RIB", "Montant", "Nom et adresse du Tiré") via OCR and use their positions as alignment points.
6. Percentage-based ROIs: if fewer than 2 anchors are detected, fall back to relative layout (ROI coordinates as percentage of page width/height). This tolerates moderate cropping.
7. **Alignment confidence gate**: if anchor detection confidence < 0.7, flag the document for manual ROI verification rather than proceed with potentially misaligned crops.

**ROI definitions** (15 field regions to extract):

| ROI ID | Field | Location Description |
|--------|-------|----------------------|
| R01 | Ordre de paiement (L-C N°) | Top-right, boxed field |
| R02 | Échéance (due date) — upper | Top-center, boxed |
| R03 | Creation date — upper | Top-center, after "Le" |
| R04 | City (lieu) — upper | Top-center, after "A" |
| R05 | RIB — upper | Top-center, 4-part boxed field |
| R06 | Montant (amount digits) — upper | Right column, boxed |
| R07 | Tireur (drawer) | Left side, below signature du tiré |
| R08 | Beneficiary ("payer à l'ordre de") | Center, below tireur |
| R09 | Montant en lettres (amount text) | Center, wide field |
| R10 | Montant (amount digits) — lower | Right column, second occurrence |
| R11 | Lieu de création | Lower-left, boxed |
| R12 | Date de création — lower | Lower-center, boxed |
| R13 | Échéance — lower | Lower-center, boxed |
| R14 | RIB — lower | Lower area, 4-part boxed field |
| R15 | Nom et adresse du Tiré (drawee) | Lower-center |
| R16 | Domiciliation (bank) | Lower-right, boxed |
| R17 | Barcode number | Bottom strip, OCR-B font |

**Calibration process**: Manually annotate 5 reference documents with bounding boxes for all ROIs. Compute median coordinates relative to anchor points. Add 5-10% padding to each ROI to handle minor alignment errors. Store as a JSON configuration file.

**Adaptive padding**: For fields known to be stamp-affected (R05 upper RIB, R14 lower RIB, R15 drawee, R16 domiciliation), use wider crops (15-20% padding) to give the LLM surrounding context for reconstruction.

### 3.3 OCR Strategy

**Vision LLM: Qwen3 VL 8B (local deployment)**

Selected model for all LLM passes. Key characteristics:
- **Strengths**: Handwritten text recognition, context understanding, French/Arabic bilingual support, zero API cost (runs locally)
- **Weaknesses**: Unreliable exact digit extraction, tendency to normalize/clean output, overconfident on uncertain fields, no built-in uncertainty indication
- **Golden rule**: Qwen = semantic OCR (handwriting, names, text amounts, contextual fields). Tesseract = deterministic OCR (digits, dates, structured fields). **Qwen outputs are NEVER trusted without validation.**

**Per-field extraction logic**:

| Field Type | Primary Engine | Reason |
|-----------|----------------|--------|
| Payment order number (R01) | Tesseract (digits mode) | Always printed, clean, boxed |
| Dates (R02, R03, R12, R13) | Tesseract + LLM fallback | Printed = Tesseract; handwritten = LLM |
| RIB (R05, R14) | Tesseract (digit whitelist) → LLM fallback | **Revised**: Digits-first approach. Step 1: Tesseract with `--psm 7 -c tessedit_char_whitelist=0123456789` on binarized+high-contrast crop. Step 2: If Tesseract confidence < 0.8 or digit count ≠ 20, THEN use LLM to fill ONLY missing/uncertain digits (return `?` for unreadable). LLM = fallback, NOT primary — LLMs are unreliable digit extractors under occlusion. |
| Amount digits (R06, R10) | Tesseract (digit whitelist) → Qwen fallback | **Revised**: Same digits-first approach as RIB. Tesseract with digit whitelist + `#.,` characters on binarized crop. Qwen fills gaps only. Handwritten amounts still need Qwen but result must pass format validation. |
| Amount text (R09) | Qwen (primary) | Handwritten cursive French — Qwen's strength. Soft validation only (see 3.5). |
| Names (R07, R08, R15) | Qwen (primary) | Mix of printed company names and handwritten names — semantic OCR. |
| Domiciliation (R16) | Qwen (primary) | Often stamp-occluded; needs contextual reading — Qwen's strength. |
| Barcode (R17) | Barcode decoder (pyzbar) + Tesseract OCR-B | Machine-readable, should be decoded not OCR'd |

**Prompting strategy for LLM** (anti-hallucination design):
- **Field-specific system prompts**: Each ROI gets a tailored prompt. Example for RIB:
  - *"Extract the bank identification number from this image. The RIB format is: 2-digit bank code, 3-digit branch code, 13-digit account number, 2-digit key. Return ONLY the digits you can see. If any digits are obscured by a stamp, return '?' for each unreadable digit. Do NOT guess."*
- **Constrained output format**: Require the LLM to return structured JSON with explicit readable: true/false flags per sub-field.
- **Negative examples in prompts**:
  - *"The blue circular stamp text (phone numbers, MF codes) is NOT part of the field. Ignore stamp text entirely."*
- **Temperature = 0**: Always. No creative generation.
- **Multiple samples**: For critical fields (RIB, amounts), query the LLM **at most 2 times** and compare results against rule-based validation. If results disagree AND validation fails, flag as uncertain. (**Revised**: no 3-way majority voting — too expensive/slow. 2 passes + validation rules is sufficient.)

### 3.4 Multi-Pass OCR Design

**Pass 1 — Classical OCR (Tesseract)**:
- Input: Binarized, cleaned ROI crops
- Target fields: Payment order number, printed dates, printed amounts, barcode
- Post-processing: Strip whitespace, normalize digit confusion (O->0, l->1, S->5)
- Role: Establishes a high-confidence baseline for clean printed fields. Fast, no API cost.

**Pass 2 — Qwen3 VL 8B on individual ROI crops**:
- Input: Original (non-binarized) color ROI crops at full resolution
- Target fields: All fields, but especially handwritten content, stamp-occluded areas
- Prompts: Field-type-specific (see 3.3 above)
- **Qwen-specific anti-hallucination system prompt** (CRITICAL — prepend to all field prompts):
  ```
  You are an OCR system. Rules:
  - Copy EXACTLY what is visible in the image
  - DO NOT guess missing or unclear characters
  - If a character is unreadable → output "?"
  - DO NOT complete partial numbers
  - DO NOT normalize formatting (keep spaces, commas, dots as-is)
  - DO NOT clean up or reformat the text
  Return only the raw extracted text, nothing else.
  ```
- **For RIB fields specifically**: Force space-separated group output (`"08 006 0110510000870 41"` not `"08006011051000087041"`) — easier to validate and debug at the group level
- Role: Handles handwriting and contextual reconstruction where Tesseract fails. Each crop is processed independently — no cross-field context leakage (prevents hallucination from one field bleeding into another).

**Pass 3 — Vision LLM on full document** (anomaly explanation, NOT validation authority):
- Input: Full document image (optionally with annotations showing ROI boundaries)
- Prompt:
  - *"This is a Tunisian lettre de change. The following fields were extracted: [Pass 1+2 merged results]. For any fields where upper and lower values disagree, or where validation rules failed, describe what you observe in the document that might explain the discrepancy. Do not invent new values. Do not override the extracted values."*
- Role: **Anomaly explanation only** — helps human reviewers understand WHY a mismatch occurred (e.g., "stamp covers upper RIB digits 5-8"). Does NOT have authority to validate or override Pass 1+2 results. All validation remains rule-based (section 3.6).
- **Important**: LLM vision is not a deterministic verifier. It can miss inconsistencies and give false confidence. Keep validation in the rule-based layer.

**Result merging logic**:
1. If Pass 1 and Pass 2 agree -> high confidence, use the value.
2. If Pass 1 and Pass 2 disagree -> use Pass 2 (Qwen) but mark as medium confidence. Apply rule-based validation (section 3.6) as the authority.
3. If Pass 2 returns ? characters -> field partially readable. Use Pass 1 digits for readable positions, keep ? for unreadable.
4. **Digit-divergence rejection rule** (numeric fields only): If Qwen modifies >30% of digits compared to Tesseract output, reject the Qwen result entirely and keep Tesseract + flag for human review. Rationale: Qwen replacing many digits at once likely indicates hallucination, not correction.
5. Pass 3 output is **informational only** — attached to flagged fields as `anomaly_explanation` for human reviewers. It does NOT influence field selection or confidence scoring.

### 3.5 Data Parsing Layer

**RIB Parsing**:
- Expected structure: BB AAA CCCCCCCCCCCCC KK (bank 2, branch 3, account 13, key 2 = 20 digits total)
- Parser: Strip all spaces, hashes, dots. Validate total length = 20 digits.
- Bank code validation: Lookup against known Tunisian bank codes (08=BIAT, 07=Amen Bank, 04=Attijari, 12=WIB, 02=STB, 10=BT, 03=UIB, 14=BH, etc.)
- Branch code validation: Cross-reference with domiciliation field (if bank="BIAT" but bank code≠08, flag error)
- Key verification: Compute RIB check digit using modulo 97 algorithm (standard Tunisian RIB key calculation: key = 97 - ((bank*100000000000000000 + branch*100000000000000 + account) * 100 mod 97)). If computed key ≠ extracted key, flag.

**Date Parsing**:
- Normalize separators: accept /, -, ., spaces
- Parse as DD/MM/YYYY (Tunisian standard)
- Validate: day 1-31, month 1-12, year 2020-2030 (reasonable range)
- Business rule: échéance must be ≥ date de création
- Business rule: échéance typically ≤ 12 months after creation date (flag but don't reject if violated)

**Amount Parsing**:
- Strip hash delimiters #, "DT" suffix, spaces
- Normalize: dots and commas. Tunisian convention: comma = decimal separator, dot = thousands separator. But observed inconsistency — some use dots as decimal.
- Heuristic: If amount has exactly 3 digits after last separator and there's no other separator after it, treat as millimes (e.g., 3000,000 = 3000.000 DT = 3000 DT). Tunisian dinars have 3 decimal places (millimes).
- Cross-validate: numeric amount vs. amount in words. Parse French number text ("trois mille dinars" = 3000, "vingt-cinq mille dinars" = 25000, "deux mille huit cent quatre-vingt-treize dinars cent quatre-vingt-douze millimes" = 2893.192).

**Amount-in-Words Parser** (soft validation only — NOT authoritative):
- **Important**: The numeric amount (R06/R10) is the PRIMARY source of truth. The text amount (R09) serves as **soft validation only**. Reasons:
  - French number text parsing is non-trivial NLP, especially with OCR noise, spelling errors, missing accents, and abbreviations
  - A mismatch between numeric and text does NOT necessarily mean the numeric amount is wrong — the text parse may be unreliable
- Build a French number word parser handling: units, teens, tens, hundreds, "mille", "million"
- Handle Tunisian conventions: "dinars" (whole) + "millimes" (fractional, 3 digits)
- Handle informal writing: "cinq cent(s)", "dinar(s)", abbreviated forms
- **On mismatch**: Log a soft warning. Trust the numeric value. Flag for human review only if BOTH numeric instances (upper/lower) also disagree with each other.

**Name Parsing**:
- Minimal transformation: trim whitespace, normalize multiple spaces to single
- Do NOT attempt to correct spelling — extract exactly as written
- Distinguish between company names (uppercase, often with "STE", "SARL", etc.) and personal names

### 3.6 Validation Layer

**Cross-field validation rules** (exploiting built-in document redundancy):

| Rule | Fields Compared | Action |
|------|-----------------|--------|
| RIB match | R05 (upper RIB) vs R14 (lower RIB) | Must be identical. If differ, take the one with higher confidence. If both uncertain, flag. |
| Amount match | R06 (upper amount) vs R10 (lower amount) | Must be identical after normalization. |
| Amount vs words | R06/R10 vs R09 (amount in letters) | Numeric must match parsed word amount. |
| Échéance match | R02 (upper) vs R13 (lower) | Must be identical. |
| Date match | R03 (upper creation) vs R12 (lower creation) | Must be identical. |
| Date ordering | Échéance vs Date de création | Échéance ≥ Creation date |
| RIB vs Bank | R05/R14 bank code vs R16 domiciliation | Bank code must correspond to bank name |
| Payment order vs barcode | R01 vs R17 | Payment order number must match barcode decode |
| RIB key check | R05/R14 | Computed key must match extracted key (mod 97) |
| City consistency | R04 (upper) vs R11 (lower lieu de création) | Should match |

**Error detection strategy**:
- **Hard failures**: RIB key mismatch, payment order ≠ barcode, amount upper ≠ amount lower when both high-confidence -> document rejected for human review.
- **Soft warnings**: Bank code doesn't match bank name (could be legitimate branch), date ordering violation, amount-in-words doesn't match digits (handwriting misread likely).
- **Rescue via redundancy**: If upper RIB has stamp occlusion (low confidence) but lower RIB is clean -> use lower RIB. Vice versa. This is the system's primary error recovery mechanism.

### 3.7 Noise & Stamp Handling

**Observed stamp patterns** (from samples):
- **Blue circular business stamps**: Contain company name, phone numbers ("Mob: 25 333 137"), MF codes ("I.U: 0377116/A"), addresses. These overlap the Signature du tiré area, Acceptation area, and often bleed into RIB and drawee fields.
- **Blue oval company stamps**: Contain company name and address. Overlap tireur and signature areas.
- **Government fiscal stamps**: Small purple/orange stamps in bottom-right corner. Rarely occlude data fields.

**Mitigation techniques**:
1. **⚠️ CRITICAL: Preserve original image.** Blue channel suppression is DANGEROUS because handwriting is ALSO blue. Aggressive stamp removal risks deleting actual field data. Always keep the unmodified image as the primary input.
2. **Multi-crop strategy** (primary technique): For stamp-affected ROIs, generate 2-3 crop variants:
   - Original color crop (always primary)
   - High-contrast binarized crop
   - Optionally: gentle color-range filtering targeting only the stamp's specific blue shade (NOT broad blue channel removal)
   Run OCR on all variants and take consensus.
3. **LLM-based stamp awareness** (primary technique): In Pass 2 prompts, explicitly instruct:
   - *"This image contains blue circular stamps overlaying the text. The stamps contain company information (phone numbers, tax IDs). IGNORE all stamp text. Extract only the field text printed in the form boxes."*
4. **Recovery via redundancy**: The strongest mitigation. When a stamp covers the upper RIB (very common — signature/stamp area is top-left, directly over upper RIB), the lower RIB is often clean. The system should automatically prefer the clean instance.
5. **Hough circle detection**: Use ONLY for identifying stamp boundaries to inform the LLM prompt (e.g., "stamp detected in region X-Y"). Do NOT use for image modification/deletion.
6. **Morphological filtering**: Use sparingly and ONLY on a separate processing branch — never on the primary image. Compare results against the unmodified crop to catch data loss.

**Unrecoverable cases**: When stamps cover BOTH instances of a field (rare but possible for small fields like drawee name that appear only once), the system must flag the field as "unreadable" rather than guess. Output confidence: 0.0 and needs_review: true.

### 3.8 Error Classification

Every flagged field must include an `error_type` for debugging and analytics:

| Error Type | Description | Example |
|------------|-------------|---------|
| `OCR_ERROR` | OCR engine returned garbled/low-confidence text | Tesseract reads "8I4T" instead of "BIAT" |
| `VALIDATION_ERROR` | Field extracted but fails a validation rule | RIB key check fails (mod 97 mismatch) |
| `MISSING_FIELD` | ROI crop contains no extractable content | Empty RIB boxes, blank amount field |
| `INCONSISTENCY` | Upper and lower instances of same field disagree | Upper RIB = 08001...42, Lower RIB = 08001...47 |
| `STAMP_OCCLUSION` | Stamp detected over the field region | Blue stamp covering RIB digits 5-12 |
| `FORMAT_ERROR` | Extracted value doesn't match expected format | Date extracted as "32/13/2025" |

This taxonomy enables:
- Targeted pipeline improvements (e.g., if 80% of errors are `STAMP_OCCLUSION`, invest in better stamp handling)
- Meaningful dashboards for monitoring
- Efficient human review (reviewer knows what to look for)

### 3.9 Deterministic Post-Processing Layer

Apply strict normalization BEFORE validation — no guessing, only deterministic transforms:

**Universal transforms**:
- Strip leading/trailing whitespace
- Collapse multiple spaces to single space
- Remove null bytes and control characters

**RIB post-processing**:
- Remove all spaces, dots, hashes, dashes
- Strip any alphabetic characters (stamp bleed-through)
- Validate: result must be exactly 20 digits or flag as `FORMAT_ERROR`

**Date post-processing**:
- Unify separators: replace `-`, `.`, spaces with `/`
- Validate DD/MM/YYYY bounds

**Amount post-processing**:
- Strip `#` delimiters, "DT" suffix, spaces
- Normalize to 3 decimal places (Tunisian millimes)
- Unify separator convention

**Name post-processing**:
- Uppercase all characters
- Collapse multiple spaces
- Trim — but do NOT correct spelling

### 3.10 Output Structuring

**JSON schema**:

```json
{
  "document_id": "string — derived from filename or payment order",
  "payment_order_number": "string — e.g., 01118862916",
  "extraction_timestamp": "ISO 8601",

  "echeance": {
    "value": "DD/MM/YYYY",
    "confidence": 0.0-1.0,
    "source": "upper|lower|merged",
    "raw_ocr": {"upper": "...", "lower": "..."}
  },
  "creation_date": {
    "value": "DD/MM/YYYY",
    "confidence": 0.0-1.0,
    "source": "upper|lower|merged",
    "raw_ocr": {"upper": "...", "lower": "..."}
  },
  "creation_city": {
    "value": "string",
    "confidence": 0.0-1.0
  },
  "rib": {
    "bank_code": "string — 2 digits",
    "branch_code": "string — 3 digits",
    "account_number": "string — 13 digits",
    "key": "string — 2 digits",
    "full": "string — 20 digits concatenated",
    "key_valid": true/false,
    "confidence": 0.0-1.0,
    "source": "upper|lower|merged",
    "raw_ocr": {"upper": "...", "lower": "..."}
  },
  "amount": {
    "value_numeric": "float — e.g., 3000.000",
    "value_text": "string — e.g., trois mille dinars",
    "currency": "DT",
    "numeric_text_match": true/false,
    "confidence": 0.0-1.0,
    "raw_ocr": {"numeric_upper": "...", "numeric_lower": "...", "text": "..."}
  },
  "tireur": {
    "name": "string",
    "address": "string|null",
    "confidence": 0.0-1.0
  },
  "beneficiary": {
    "name": "string",
    "confidence": 0.0-1.0
  },
  "tire": {
    "name": "string",
    "address": "string|null",
    "confidence": 0.0-1.0
  },
  "domiciliation": {
    "bank_name": "string",
    "branch": "string|null",
    "confidence": 0.0-1.0
  },

  "validation": {
    "rib_key_valid": true/false,
    "rib_bank_code_matches_domiciliation": true/false,
    "amount_numeric_matches_text": true/false,
    "echeance_after_creation": true/false,
    "payment_order_matches_barcode": true/false,
    "upper_lower_consistency": {
      "rib": true/false,
      "amount": true/false,
      "echeance": true/false,
      "creation_date": true/false
    }
  },

  "document_confidence": 0.0-1.0,
  "needs_human_review": true/false,
  "qwen_corrections": {
    "rib_digits_changed": 0,
    "amount_digits_changed": 0,
    "date_digits_changed": 0,
    "source": "tesseract_only|qwen_fallback|qwen_primary"
  },
  "flagged_fields": [
    {
      "field": "field_name",
      "error_type": "OCR_ERROR|VALIDATION_ERROR|MISSING_FIELD|INCONSISTENCY|STAMP_OCCLUSION|FORMAT_ERROR",
      "anomaly_explanation": "string — from Pass 3 LLM, describes what was observed (optional)"
    }
  ]
}
```

**Confidence scoring formula** (concrete computation):
- Per-field confidence = weighted sum:
  ```
  confidence = 0.4 * ocr_agreement     # Pass 1 vs Pass 2 agree (1.0) or disagree (0.3)
             + 0.3 * redundancy_match   # upper vs lower match (1.0), partial (0.5), conflict (0.0)
             + 0.2 * format_validity    # passes format rules (1.0) or not (0.0)
             + 0.1 * engine_confidence  # raw OCR engine confidence score (0.0-1.0)
  ```
- Document-level confidence = minimum of all critical field confidences (RIB, amount, dates)
- needs_human_review = true if any critical field confidence < 0.85 (initial threshold — must be recalibrated per-field using a labeled validation set before production; see section 9 known limitation 3.1) or any hard validation failure

---

## 4. Risk Analysis

| Failure Case | Why It Happens | Frequency (est.) | Mitigation |
|---|---|---|---|
| **Stamp covers all RIB digits** | Business stamp placed directly over RIB boxes in both upper and lower sections | Low (5%) — usually only one instance is covered | Use lower/upper fallback. If both occluded, flag. Apply stamp suppression preprocessing. |
| **Handwritten amount illegible** | Poor penmanship, especially for large numbers with many words | Medium (15%) | Cross-validate numeric vs text amount. Use upper+lower numeric redundancy. If words are illegible but digits are clear, trust digits. |
| **Date misread (digit confusion)** | 1↔7, 0↔6, 2↔7 in handwriting | Medium (10%) | Cross-validate upper vs lower dates. Validate date logic (valid day/month, échéance > creation). |
| **Wrong field extracted (crop misalignment)** | Template alignment failed due to missing/damaged borders | Low (3%) | Multiple anchor detection (header, barcode, border). Sanity-check extracted values (RIB should be digits, dates should match DD/MM/YYYY). |
| **LLM hallucinates plausible RIB** | LLM "completes" partially visible digits with plausible but wrong values | High risk if not mitigated | Force ? output for unreadable digits. Cross-validate with key check (mod 97). Compare upper vs lower. Use max 2 LLM passes + validation rules (no 3-way voting). |
| **Stamp text misread as field text** | OCR reads "Mob: 25 333 137" from stamp as part of RIB | Medium (10%) | Stamp detection + suppression preprocessing. Prompt engineering to ignore stamp text. Validate RIB length = 20 digits. |
| **Amount format misparse** | Ambiguous decimal/thousands separators (3000,000 vs 3.000,000) | Medium (10%) | Tunisian dinar uses 3 millimes -> always 3 fractional digits. Cross-validate with text amount. |
| **Barcode unreadable** | Partial barcode, ink smear, fold damage | Low (5%) | Fall back to OCR-B text below barcode. If both fail, use payment order number from R01 field. |
| **Fully handwritten document** | All fields handwritten including RIB — lower OCR accuracy across the board | ~20% of samples | Qwen-primary strategy for these documents. Lower confidence thresholds. More aggressive human review flagging. |
| **Qwen digit hallucination** | Qwen silently replaces uncertain digits with plausible but wrong values (e.g., `08 006 01105??000870 41` → `08 006 0110510000870 41`) | High risk for numeric fields | Digit-divergence rejection: if Qwen changes >30% of Tesseract digits, reject Qwen output. RIB mod 97 catches remaining errors. Tesseract is always primary for digits. |
| **Qwen format normalization** | Qwen strips delimiters, collapses spaces, or removes decimal separators (e.g., `3000,000` → `3000`) | Medium (15%) for amount fields | Anti-hallucination prompt explicitly forbids normalization. Post-processing enforces 3-decimal-place format. Cross-validate upper vs lower amounts. |
| **Qwen overconfidence** | Qwen returns clean, confident-looking output even when underlying image is ambiguous — no built-in uncertainty signal | Constant risk | Never trust Qwen without validation. Confidence scoring weights OCR agreement and redundancy match higher than engine output. Track `qwen_corrections` in JSON for audit. |

---

## 5. Optimization Strategy

**Accuracy Improvements**:
- **Field-specific image preprocessing**: Don't use one-size-fits-all. Apply blue-channel extraction for handwritten fields, standard binarization for printed fields, and stamp suppression only for known stamp-affected regions.
- **Ensemble OCR**: For critical fields (RIB, amounts), run 2-3 different OCR passes and vote. The cost of over-reading is much lower than the cost of a wrong extraction.
- **Feedback loop**: Log all human corrections. After 100+ corrections, analyze error patterns. Adjust ROI coordinates, prompt wording, or confidence thresholds based on real failure modes.
- **Adaptive confidence thresholds**: Start conservative (flag anything < 0.9). After measuring false positive rate, gradually lower thresholds for fields that prove reliable.

**Performance Improvements**:
- **Parallel ROI processing**: After alignment, all 17 ROI crops are independent — process them in parallel.
- **Conditional Qwen calls**: Skip Pass 2 (Qwen) for fields where Pass 1 (Tesseract) achieves high confidence AND passes validation. This avoids unnecessary inference for clean printed documents. **Revised strategy**: 1 pass (Tesseract only) for clean fields; multi-pass only for complex/handwritten/stamp-occluded fields. This can reduce processing time by 50-70% on clean documents.
- **Skip Pass 3 (anomaly explanation) entirely** when all fields from Passes 1+2 are high-confidence and all validations pass.
- **Batch processing**: Process multiple documents in parallel at the pipeline level.

**Cost Optimization** (API usage):
- Qwen3 VL 8B runs locally — no API cost. Primary optimization target is **inference time**, not cost.
- Tesseract is instant — maximize its use for printed fields.
- Qwen inference is slower — use conditional routing (only call Qwen when Tesseract confidence is low or field is handwritten).
- **Estimated Qwen inference calls per document**:
  - Best case (clean printed): 0 Qwen calls (Tesseract handles everything)
  - Average case: 5-8 Qwen calls (handwritten fields + stamp-occluded fields)
  - Worst case (fully handwritten + stamps): 17 ROI calls + 1 full-document call = 18 calls
- **Single model**: Qwen3 VL 8B for all LLM passes. No model tiering needed (runs locally, no per-call cost).
- **Crop size optimization**: Send tightly cropped ROIs, not full-resolution images. Reduces inference time and memory usage.

---

## 6. Implementation Roadmap

**Field priority strategy** — all 17 ROIs are defined but implementation is phased:
- **Tier 1 (MVP — implement first)**: RIB (R05/R14), amount digits (R06/R10), amount text (R09), dates (R02/R03/R12/R13), drawee (R15) — these cover ~80% of business value
- **Tier 2 (add after Tier 1 is stable)**: domiciliation (R16), tireur (R07), beneficiary (R08), barcode (R17), payment order (R01)
- **Tier 3 (add last)**: city (R04/R11) — low business priority, simple once pipeline works

This avoids premature complexity: validate the pipeline end-to-end on 7 critical fields before expanding to all 17.

### Phase 1: Foundation (Week 1-2)
1. **Image preprocessing pipeline**: Deskew, denoise, normalize, color channel separation
2. **Template alignment**: Anchor detection (header band, barcode, borders), affine transform
3. **ROI definition**: Manually annotate 5 reference images, calibrate ROI coordinates, store as JSON config
4. **ROI extraction**: Crop all 17 regions per aligned document
5. **Verification**: Visually inspect all 20 sample documents' ROI crops to confirm alignment accuracy

### Phase 2: Core OCR (Week 3-4)
1. **Tesseract integration**: Configure for French + digit modes. Run on all ROIs.
2. **LLM integration**: Build prompt templates for each field type. Implement Pass 2.
3. **Barcode decoding**: Integrate pyzbar for barcode reading.
4. **Result merging logic**: Implement Pass 1 + Pass 2 merge algorithm.
5. **Verification**: Compare OCR output against manually transcribed ground truth for all 20 samples.

### Phase 3: Parsing & Validation (Week 5-6)
1. **Field parsers**: RIB, date, amount (numeric + text), name parsers
2. **French number word parser**: Handle full range of amount-in-words expressions
3. **Cross-field validation engine**: Implement all 10 validation rules from section 3.6
4. **Confidence scoring**: Implement per-field and document-level scoring
5. **Verification**: Run full pipeline on 20 samples, measure field-level accuracy

### Phase 4: Stamp Handling & Hardening (Week 7-8)
1. **Stamp detection**: Hough circle detection, color segmentation
2. **Stamp suppression preprocessing**: Implement multi-crop strategy
3. **Pass 3 (anomaly explanation pass)**: Implement full-document LLM pass for explaining discrepancies — NOT a validation authority
4. **Edge case handling**: Fully handwritten documents, extreme occlusion, missing fields
5. **Verification**: Re-run pipeline, measure improvement on stamp-affected fields

### Phase 5: Production Readiness (Week 9-10)
1. **JSON output generation**: Implement full schema with confidence scores
2. **Human review interface**: Flag low-confidence fields, display original crops alongside extracted values
3. **Logging & audit trail**: Store all intermediate results (crops, raw OCR, parsed values)
4. **Performance optimization**: Conditional LLM routing, parallel processing
5. **Documentation**: API docs, configuration guide, deployment instructions

---

## 7. Testing & Evaluation

**Metrics to track**:
- **Field-level accuracy**: Exact match rate per field (strict equality after normalization)
- **Character-level accuracy**: CER (Character Error Rate) per field — important for RIB and amounts where a single digit error is critical
- **Document-level accuracy**: Percentage of documents where ALL critical fields are 100% correct
- **Rejection rate**: Percentage of documents flagged for human review
- **False acceptance rate**: Documents that passed validation but contain errors (most dangerous — measure via periodic human audit)
- **Processing time**: Seconds per document end-to-end
- **Inference time per document**: Qwen calls per document (average and P95)

**How to measure accuracy**:
1. **Build ground truth dataset**: Manually transcribe all 20 sample documents into the target JSON schema. Double-annotate (two independent annotators) to establish gold standard. Resolve disagreements.
2. **Automated evaluation script**: Compare pipeline output JSON against ground truth JSON. Compute per-field exact match and CER.
3. **Stratified analysis**: Break down accuracy by:
   - Printed vs handwritten (based on manual labeling)
   - Stamp-affected vs clean
   - Field type (RIB, date, amount, name)

**Dataset requirements**:
- **Minimum 20 annotated documents** for initial development (available now)
- **Target 100+ documents** for robust evaluation across all edge cases
- **Distribution should include**: fully printed (~40%), partially handwritten (~40%), fully handwritten (~20%); clean (~60%), stamp-affected (~40%)
- **Negative test cases**: Deliberately corrupted images (extreme blur, heavy occlusion, torn documents) to verify the system correctly rejects rather than hallucinates

**Regression testing**: After any pipeline change, re-run full evaluation suite. No field-level accuracy regression allowed on the test set.

---

## 8. Future Improvements

**Advanced Enhancements**:
- **Fine-tuned handwriting model**: Train a custom CRNN or TrOCR model on Tunisian handwriting for amounts and names. Requires ~500+ annotated field crops. Reduces LLM dependency for handwritten fields.
- **Template auto-detection**: If the system needs to handle other Tunisian financial documents (chèques, billets à ordre), implement a document classifier at the pipeline entry to route to the correct ROI configuration.
- **Active learning loop**: When human reviewers correct a field, automatically add the corrected crop+label to a training dataset. Periodically retrain field-specific models.
- **Barcode-driven validation**: The CMC-7 line at the bottom encodes structured data. Implement full CMC-7 parsing to provide additional cross-validation against OCR results.

**Scaling Strategy**:
- **Queue-based architecture**: Wrap the pipeline in a worker that reads from a message queue (Redis/RabbitMQ). Scale workers horizontally.
- **GPU acceleration**: If deploying custom OCR models, ensure the preprocessing and model inference can run on GPU.
- **Cache LLM prompts**: For identical prompt templates, leverage Qwen's KV-cache to reduce latency on repeated field-type inferences within a batch.
- **Target throughput**: 10+ documents/minute with parallel processing and conditional LLM routing.

**Model Improvements**:
- **Qwen3 VL 8B is already self-hosted** — no data privacy concern from external API calls. All bank details stay local. If accuracy proves insufficient, consider upgrading to a larger Qwen variant (e.g., 72B with quantization) or fine-tuning the 8B model on domain-specific crops.
- **Specialized digit recognizer**: Train a lightweight CNN specifically for printed and handwritten digits in the Tunisian document context. Use for RIB and amount fields as a fast first pass before LLM.
- **Confidence calibration** (PRODUCTION BLOCKER — see section 9 known limitation 3.1): Collect a large set of predictions with ground truth. Calibrate confidence scores so that "0.95 confidence" actually means 95% of the time the value is correct (Platt scaling or isotonic regression).

---

## 9. Review Findings & Revisions (April 2026)

### What is excellent (keep as-is)
- ROI-based extraction — correct for fixed template
- Multi-pass OCR (Tesseract + LLM) — industry-grade hybrid pipeline
- Redundancy exploitation (upper/lower fields) — strongest design element
- Validation layer (RIB mod 97) — bank-grade logic
- Reject-over-guess philosophy — mandatory for financial systems

### Critical fixes applied
See inline revisions in sections 3.2, 3.3, 3.4, 3.5, 3.7, and new sections 3.8/3.9/3.10.

### Consistency fixes applied (Round 3)
- Removed stale "majority vote" reference in Risk table (section 4) — aligned with 2-pass + validation strategy
- Renamed Pass 3 everywhere: "context-aware verification" / "validation pass" → "anomaly explanation pass"
- Marked 0.85 confidence threshold as "initial" — must become dynamic per-field before production
- Added field priority tiers (Tier 1/2/3) to roadmap section 6 — controls over-engineering by phasing implementation

### Simplification notes
- Start with priority fields (RIB, amounts, dates, drawee) — 80% value with 30% effort
- Use conditional OCR passes: 1 pass for clean fields, multi-pass only for complex ones
- Cap LLM calls at 2 per field max + validation rules (no 3-way majority voting)

### Qwen3 VL 8B integration (Round 4)
- Selected Qwen3 VL 8B as the vision LLM — local deployment, zero API cost, good French/Arabic/handwriting support
- **Golden rule enforced**: Qwen = semantic OCR (handwriting, names, text), Tesseract = deterministic OCR (digits, dates, structured). Qwen never authoritative without validation.
- Added Qwen-specific anti-hallucination system prompt to Pass 2 (forbids guessing, normalizing, completing)
- Added digit-divergence rejection rule: Qwen changes >30% of Tesseract digits → reject Qwen, flag for review
- Added `qwen_corrections` tracking in JSON output for audit
- Replaced all generic LLM/API references with Qwen-specific language
- Added 3 Qwen-specific risks to Risk table: digit hallucination, format normalization, overconfidence
- Amount digits (R06/R10) promoted to Tesseract-first (same as RIB), Qwen fallback only
- Cost optimization section reframed: target = inference time, not API cost (Qwen is local)

### Known limitations (acceptable now, must fix for production)

**3.1 Confidence formula is heuristic (not calibrated)**
- Current weights (0.4/0.3/0.2/0.1) are reasonable defaults but arbitrary
- **Acceptable for MVP**: the formula produces usable relative rankings and the initial reject threshold (< 0.85) is conservative enough to catch most errors. This threshold should become dynamic (per-field, calibration-set-driven) before production.
- **Required for production**: collect 200+ predictions with ground truth, then calibrate via Platt scaling or isotonic regression so that "0.95 confidence" genuinely means 95% correct
- **Production gate**: confidence calibration curve must show monotonic reliability (higher score = higher accuracy) on a held-out test set before going live

**3.2 Amount text parsing remains fragile in edge cases**
- French number text + OCR noise + missing accents/words = unreliable parser
- **Acceptable for MVP**: numeric amount is authoritative, text is soft validation only — parser failures produce warnings, not rejections
- **Safety net**: upper/lower numeric redundancy catches real amount errors regardless of text parser quality
- **Required for production**: log all text-vs-numeric mismatches. Analyze patterns. Improve parser iteratively based on real failures, not hypothetical edge cases

**3.3 Stamp handling depends on LLM prompt compliance**
- LLM instruction to "ignore stamp text" is not 100% reliable — stamps with digits near RIB fields can still bleed into extractions
- **Acceptable for MVP**: redundancy (upper/lower) and validation (mod 97, format checks) catch most stamp-induced errors regardless of LLM behavior
- **Safety net hierarchy**: redundancy > validation rules > LLM stamp awareness (in order of reliability)
- **Required for production**: measure stamp-induced error rate across 100+ documents. If > 5% of stamp-affected fields have undetected errors, invest in a dedicated stamp segmentation model (U-Net or similar trained on stamp vs. text regions)

---

## Relevant files
- `example` — 20 sample documents for development and initial testing
- ROI configuration (to be created) — JSON file defining all 17 ROI coordinates relative to anchors
- Tunisian bank code lookup table (to be created) — mapping bank codes to names for validation

## Decisions
- ROI-based approach over full-page OCR — justified by fixed template layout across all samples
- Multi-pass OCR (Tesseract + LLM) over single-engine — required due to printed/handwritten mix
- Reject-over-guess philosophy — financial documents demand precision, not recall
- Redundancy-first validation — the document template's deliberate field duplication is the strongest accuracy lever
