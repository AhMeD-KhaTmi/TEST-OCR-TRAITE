"""
Phase 2 verification script.

Runs the full Phase-2 OCR pipeline on every sample document:
  - Phase 1: preprocess -> align -> ROI extract (same as verify_phase1.py)
  - Phase 2: OCR engine on all ROI crops (Tesseract + Qwen routing)

By default Qwen calls are SKIPPED (--skip-qwen flag or INFERENCE_MODE not set).
Pass --with-qwen to enable live Qwen/OpenRouter calls (costs API tokens).

Output:
  output/verify_phase2/<doc_id>/ocr_results.json   — per-field OCR output
  output/verify_phase2/<doc_id>/overlay.jpg         — ROI overlay (same as P1)
  output/verify_phase2/summary.txt                  — per-doc summary

Run from the project root (venv activated):
    venv/Scripts/python.exe verify_phase2.py
    venv/Scripts/python.exe verify_phase2.py --with-qwen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.alignment import align
from ocr_pipeline.roi_extractor import ROIExtractor, draw_roi_overlay
from ocr_pipeline.ocr_engine import run_ocr, OCRFieldResult

SAMPLES_DIR = Path(__file__).parent / "example"
OUTPUT_DIR  = Path(__file__).parent / "output" / "verify_phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extractor = ROIExtractor()


# ---------------------------------------------------------------------------
# Helpers (shared with verify_phase1.py)
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


def _result_to_dict(r: OCRFieldResult) -> dict:
    """Serialise OCRFieldResult to a JSON-safe dict."""
    return {
        "roi_id":            r.roi_id,
        "field_name":        r.field_name,
        "text":              r.text,
        "confidence":        round(r.confidence, 4),
        "source":            r.source,
        "readable":          r.readable,
        "tess_text":         r.tess_text,
        "qwen_text":         r.qwen_text,
        "qwen_digits_changed": r.qwen_digits_changed,
        "tess_digits_total": r.tess_digits_total,
        "flags":             r.flags,
        "error":             r.error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(with_qwen: bool) -> None:
    skip_qwen = not with_qwen
    mode_label = "Tesseract+Qwen (API)" if with_qwen else "Tesseract-only"
    print(f"Phase 2 verification — mode: {mode_label}")
    print(f"Processing images in: {SAMPLES_DIR}\n")

    summary_lines: list[str] = [
        f"Phase 2 verification — mode: {mode_label}",
        "=" * 60,
    ]

    total = 0
    errors = 0
    qwen_calls_total = 0

    for img_path in sorted(SAMPLES_DIR.glob("*.jpg")):
        doc_id = _make_doc_id(img_path)
        print(f"[{doc_id}] {img_path.name}...")

        try:
            # --- Phase 1 (foundation) ---
            prep   = preprocess(img_path)
            aln    = align(prep.deskewed)
            crops  = extractor.extract_all(aln.image)

            # --- Phase 2: OCR ---
            ocr_results = run_ocr(crops, skip_qwen=skip_qwen)

            # Count Qwen calls
            qwen_calls = sum(
                1 for r in ocr_results.values()
                if r.qwen_result is not None and r.qwen_result.engine_available
            )
            qwen_calls_total += qwen_calls

            # --- Save overlay ---
            doc_dir = OUTPUT_DIR / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            overlay = draw_roi_overlay(aln.image, crops, extractor)
            cv2.imwrite(str(doc_dir / "overlay.jpg"), overlay)

            # --- Save OCR JSON ---
            ocr_dict = {roi_id: _result_to_dict(r) for roi_id, r in ocr_results.items()}
            (doc_dir / "ocr_results.json").write_text(
                json.dumps(ocr_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # --- Per-doc summary ---
            flagged = [
                f"{r.roi_id}:{','.join(r.flags)}"
                for r in ocr_results.values() if r.flags
            ]
            avg_conf = sum(r.confidence for r in ocr_results.values()) / len(ocr_results)
            line = (
                f"[{doc_id}] OK  fields={len(ocr_results)}  "
                f"avg_conf={avg_conf:.2f}  qwen_calls={qwen_calls}"
            )
            if flagged:
                line += f"\n        flags: {'; '.join(flagged)}"
            summary_lines.append(line)
            print(f"  -> {line}")
            total += 1

        except Exception as exc:
            err = f"[{doc_id}] ERROR: {exc}"
            summary_lines.append(err)
            print(f"  -> {err}")
            errors += 1
            total += 1

    # --- Overall totals ---
    summary_lines += [
        "=" * 60,
        f"Total documents : {total}",
        f"Successful      : {total - errors}",
        f"Errors          : {errors}",
        f"Total Qwen calls: {qwen_calls_total}",
    ]

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nDone. Results in: {OUTPUT_DIR}")
    print(f"Summary        : {summary_path}")
    print(f"Errors         : {errors} / {total}")
    print(f"Total Qwen calls: {qwen_calls_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 verification")
    parser.add_argument(
        "--with-qwen",
        action="store_true",
        help="Enable live Qwen/OpenRouter API calls (uses tokens).",
    )
    args = parser.parse_args()
    main(with_qwen=args.with_qwen)
