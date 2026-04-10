"""
Phase 3 verification script.

Runs the full Phase 3 pipeline on every sample document:
  - Phase 1: preprocess -> align -> ROI extract
  - Phase 2: OCR engine on all ROI crops (Tesseract + Qwen routing)
  - Phase 3: Parse all fields, cross-field validate, assemble DocumentResult

By default Qwen calls are SKIPPED (Tesseract-only baseline).
Pass --with-qwen to enable live Qwen/OpenRouter calls (costs API tokens).

Output:
  output/verify_phase3/<doc_id>/document_result.json   — full validated JSON
  output/verify_phase3/<doc_id>/overlay.jpg             — ROI overlay image
  output/verify_phase3/summary.txt                      — per-doc summary

Run from the project root (venv activated):
    venv/Scripts/python.exe verify_phase3.py
    venv/Scripts/python.exe verify_phase3.py --with-qwen
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2

# On Windows, stdout defaults to cp1252 which cannot encode many Unicode chars.
# Reconfigure stdout to UTF-8 so summary lines with special chars don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.alignment import align
from ocr_pipeline.roi_extractor import ROIExtractor, draw_roi_overlay
from ocr_pipeline.ocr_engine import run_ocr, OCRBatch
from ocr_pipeline.document_result import (
    build_document_result,
    document_result_to_json,
)

SAMPLES_DIR = Path(__file__).parent / "example"
OUTPUT_DIR  = Path(__file__).parent / "output" / "verify_phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extractor = ROIExtractor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_id(img_path: Path) -> str:
    stem = img_path.stem
    if "_page-" in stem:
        batch = stem.split("_page-")[0].rsplit("_", 1)[-1]
        page  = stem.split("_page-")[1]
        return f"{batch}_{page}"
    else:
        number = stem.split("_")[-1].split("-")[-1]
        return f"ptj_{number}"


def _format_flags(flags: list) -> str:
    if not flags:
        return "none"
    parts = []
    for f in flags:
        prefix = "HARD" if f.is_hard_failure else "soft"
        parts.append(f"[{prefix}:{f.field}:{f.error_type}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(with_qwen: bool) -> None:
    skip_qwen = not with_qwen
    mode_label = "Tesseract+Qwen (API)" if with_qwen else "Tesseract-only"
    print(f"Phase 3 verification — mode: {mode_label}")
    print(f"Processing images in: {SAMPLES_DIR}\n")

    summary_lines: list[str] = [
        f"Phase 3 verification — mode: {mode_label}",
        "=" * 70,
    ]

    total     = 0
    errors    = 0
    needs_review_count = 0
    hard_fail_count    = 0

    for img_path in sorted(SAMPLES_DIR.glob("*.jpg")):
        doc_id = _make_doc_id(img_path)
        print(f"[{doc_id}] {img_path.name}...")

        try:
            # --- Phase 1 (foundation) ---
            prep  = preprocess(img_path)
            aln   = align(prep.deskewed)
            crops = extractor.extract_all(aln.image)

            # --- Phase 2: OCR ---
            ocr_results = run_ocr(crops, skip_qwen=skip_qwen)
            batch = OCRBatch(doc_id=doc_id, fields=ocr_results)

            # --- Phase 3: Parse + Validate ---
            doc_result = build_document_result(doc_id, batch)

            # --- Save overlay ---
            doc_dir = OUTPUT_DIR / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            overlay = draw_roi_overlay(aln.image, crops, extractor)
            cv2.imwrite(str(doc_dir / "overlay.jpg"), overlay)

            # --- Save document result JSON ---
            json_str = document_result_to_json(doc_result)
            (doc_dir / "document_result.json").write_text(json_str, encoding="utf-8")

            # --- Per-doc summary line ---
            hard_fails = [f for f in doc_result.flagged_fields if f.is_hard_failure]
            soft_warns = [f for f in doc_result.flagged_fields if not f.is_hard_failure]
            rib_ok  = "Y" if doc_result.validation.get("rib_key_valid") else "N"
            amt_ok  = "Y" if doc_result.validation.get("upper_lower_consistency", {}).get("amount") else "N"
            echo_ok = "Y" if doc_result.validation.get("upper_lower_consistency", {}).get("echeance") else "N"
            review  = "REVIEW" if doc_result.needs_human_review else "OK"

            line = (
                f"[{doc_id}] {review}  "
                f"conf={doc_result.document_confidence:.2f}  "
                f"rib={rib_ok} amt={amt_ok} echo={echo_ok}  "
                f"hard={len(hard_fails)} soft={len(soft_warns)}"
            )
            if hard_fails:
                line += f"\n        flags: {_format_flags(hard_fails)}"

            summary_lines.append(line)
            print(f"  -> {line}")

            if doc_result.needs_human_review:
                needs_review_count += 1
            if hard_fails:
                hard_fail_count += 1
            total += 1

        except Exception as exc:
            err = f"[{doc_id}] ERROR: {exc}"
            summary_lines.append(err)
            print(f"  -> {err}")
            errors += 1
            total += 1

    # --- Totals ---
    summary_lines += [
        "=" * 70,
        f"Total documents     : {total}",
        f"Successful          : {total - errors}",
        f"Errors (exceptions) : {errors}",
        f"Needs human review  : {needs_review_count}",
        f"Hard failures       : {hard_fail_count}",
    ]

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nDone. Results in    : {OUTPUT_DIR}")
    print(f"Summary             : {summary_path}")
    print(f"Errors              : {errors} / {total}")
    print(f"Needs human review  : {needs_review_count} / {total - errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 verification")
    parser.add_argument(
        "--with-qwen",
        action="store_true",
        help="Enable live Qwen/OpenRouter API calls (uses tokens).",
    )
    args = parser.parse_args()
    main(with_qwen=args.with_qwen)
