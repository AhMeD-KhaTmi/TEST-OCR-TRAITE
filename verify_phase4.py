"""
Phase 4 verification script.

Runs the full Phase 4 pipeline on every sample document:
  Phase 1: preprocess -> align -> ROI extract
  Phase 2: OCR engine (Tesseract + optional Qwen routing)
  Phase 3: Parse + cross-field validate + JSON assembly
  Phase 4:
    - Stamp detection on the aligned full-page image
    - Multi-crop variants for stamp-affected ROIs
    - Optional Pass 3 anomaly explanation LLM call (--with-pass3)

Output per document:
  output/verify_phase4/<doc_id>/document_result.json   — full validated JSON
  output/verify_phase4/<doc_id>/stamp_debug.jpg        — overlay with stamp boundaries
  output/verify_phase4/<doc_id>/overlay.jpg            — ROI overlay
  output/verify_phase4/<doc_id>/variants/              — alternative crops for flagged ROIs
  output/verify_phase4/summary.txt                     — aggregate summary

Usage:
  # Tesseract-only, no LLM anomaly explanation (default)
  venv\\Scripts\\python.exe verify_phase4.py

  # With live Qwen inference (Pass 2 + Pass 3 anomaly explanation)
  venv\\Scripts\\python.exe verify_phase4.py --with-qwen --with-pass3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.alignment import align
from ocr_pipeline.roi_extractor import ROIExtractor, draw_roi_overlay
from ocr_pipeline.ocr_engine import run_ocr, OCRBatch
from ocr_pipeline.stamp_detector import detect_stamps, roi_is_stamp_affected
from ocr_pipeline.stamp_preprocessor import generate_crop_variants
from ocr_pipeline.document_result import (
    build_document_result,
    document_result_to_json,
)

SAMPLES_DIR = Path(__file__).parent / "example"
OUTPUT_DIR  = Path(__file__).parent / "output" / "verify_phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extractor   = ROIExtractor()
ROI_CONFIG  = extractor.rois  # dict[str, dict] with x,y,w,h (relative 0-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_id(img_path: Path) -> str:
    stem = img_path.stem
    if "_page-" in stem:
        batch = stem.split("_page-")[0].rsplit("_", 1)[-1]
        page  = stem.split("_page-")[1]
        return f"{batch}_{page}"
    number = stem.split("_")[-1].split("-")[-1]
    return f"ptj_{number}"


def _format_stamp_summary(stamp_count: int, affected: list[str]) -> str:
    if stamp_count == 0:
        return "stamps=0"
    rois = ",".join(affected) if affected else "none"
    return f"stamps={stamp_count} affected_rois=[{rois}]"


def _affected_rois(
    stamp_result,
    img_w: int,
    img_h: int,
    threshold: float = 0.10,
) -> list[str]:
    """Return list of ROI IDs whose boxes are covered by a detected stamp."""
    affected = []
    for roi_id, cfg in ROI_CONFIG.items():
        # Config uses x,y,w,h (top-left origin, relative 0-1)
        x1 = int(cfg["x"] * img_w)
        y1 = int(cfg["y"] * img_h)
        x2 = int((cfg["x"] + cfg["w"]) * img_w)
        y2 = int((cfg["y"] + cfg["h"]) * img_h)
        if roi_is_stamp_affected(stamp_result, x1, y1, x2, y2, threshold):
            affected.append(roi_id)
    return affected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(with_qwen: bool, with_pass3: bool) -> None:
    skip_qwen  = not with_qwen
    skip_pass3 = not with_pass3
    mode_parts = []
    if with_qwen:
        mode_parts.append("Qwen OCR")
    if with_pass3:
        mode_parts.append("Pass3 explanation")
    mode_label = "+".join(mode_parts) if mode_parts else "Tesseract-only"
    print(f"Phase 4 verification -- mode: {mode_label}")
    print(f"Processing images in: {SAMPLES_DIR}\n")

    summary_lines = [
        f"Phase 4 verification -- mode: {mode_label}",
        "=" * 70,
    ]

    total         = 0
    errors        = 0
    review_count  = 0
    stamp_docs    = 0
    total_stamps  = 0

    for img_path in sorted(SAMPLES_DIR.glob("*.jpg")):
        doc_id = _make_doc_id(img_path)
        print(f"[{doc_id}] {img_path.name}...")

        try:
            # --- Phase 1 ---
            prep  = preprocess(img_path)
            aln   = align(prep.deskewed)
            crops = extractor.extract_all(aln.image)
            img_h, img_w = aln.image.shape[:2]

            # --- Phase 4: Stamp detection (before OCR) ---
            stamp_result = detect_stamps(aln.image, draw_debug=True)
            affected     = _affected_rois(stamp_result, img_w, img_h)
            if stamp_result.count > 0:
                stamp_docs   += 1
                total_stamps += stamp_result.count

            # --- Phase 4: Multi-crop variants for stamp-affected ROIs ---
            variant_map = {}
            for roi_id in affected:
                if roi_id in crops:
                    crop_colour = crops[roi_id].colour
                    variants = generate_crop_variants(
                        crop_colour, roi_id, stamp_affected=True
                    )
                    variant_map[roi_id] = variants

            # --- Phase 2: OCR ---
            ocr_results = run_ocr(crops, skip_qwen=skip_qwen)
            batch = OCRBatch(doc_id=doc_id, fields=ocr_results)

            # --- Phase 3 + 4: Parse, validate, stamp info, Pass 3 ---
            doc_result = build_document_result(
                doc_id,
                batch,
                document_image=aln.image,
                skip_pass3=skip_pass3,
            )

            # --- Save outputs ---
            doc_dir = OUTPUT_DIR / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # ROI overlay
            overlay = draw_roi_overlay(aln.image, crops, extractor)
            cv2.imwrite(str(doc_dir / "overlay.jpg"), overlay)

            # Stamp debug overlay
            if stamp_result.debug_image is not None:
                cv2.imwrite(str(doc_dir / "stamp_debug.jpg"), stamp_result.debug_image)
            else:
                # Save raw aligned image when no stamps found
                cv2.imwrite(str(doc_dir / "stamp_debug.jpg"), aln.image)

            # Multi-crop variants
            if variant_map:
                variant_dir = doc_dir / "variants"
                variant_dir.mkdir(exist_ok=True)
                for roi_id, variants in variant_map.items():
                    cv2.imwrite(
                        str(variant_dir / f"{roi_id}_original.jpg"),
                        variants.original,
                    )
                    if variants.high_contrast is not None:
                        cv2.imwrite(
                            str(variant_dir / f"{roi_id}_high_contrast.jpg"),
                            variants.high_contrast,
                        )
                    if variants.stamp_suppressed is not None:
                        cv2.imwrite(
                            str(variant_dir / f"{roi_id}_stamp_suppressed.jpg"),
                            variants.stamp_suppressed,
                        )

            # Document result JSON
            json_str = document_result_to_json(doc_result)
            (doc_dir / "document_result.json").write_text(json_str, encoding="utf-8")

            # --- Per-doc summary ---
            hard_fails = [f for f in doc_result.flagged_fields if f.is_hard_failure]
            review     = "REVIEW" if doc_result.needs_human_review else "OK"
            stamp_str  = _format_stamp_summary(
                doc_result.stamp_info.stamp_count,
                doc_result.stamp_info.affected_rois,
            )
            anom_str   = f" [explanation: {len(doc_result.anomaly_explanation)} chars]" \
                         if doc_result.anomaly_explanation else ""

            line = (
                f"[{doc_id}] {review}  conf={doc_result.document_confidence:.2f}  "
                f"{stamp_str}  hard={len(hard_fails)}{anom_str}"
            )
            summary_lines.append(line)
            print(f"  -> {line}")

            if doc_result.needs_human_review:
                review_count += 1
            total += 1

        except Exception as exc:
            err = f"[{doc_id}] ERROR: {exc}"
            summary_lines.append(err)
            print(f"  -> {err}")
            errors += 1
            total  += 1

    summary_lines += [
        "=" * 70,
        f"Total documents     : {total}",
        f"Successful          : {total - errors}",
        f"Errors (exceptions) : {errors}",
        f"Needs human review  : {review_count}",
        f"Docs with stamps    : {stamp_docs}",
        f"Total stamps found  : {total_stamps}",
    ]

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nDone. Results in   : {OUTPUT_DIR}")
    print(f"Summary            : {summary_path}")
    print(f"Errors             : {errors} / {total}")
    print(f"Docs with stamps   : {stamp_docs} / {total - errors}")
    print(f"Total stamps found : {total_stamps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 verification")
    parser.add_argument("--with-qwen",  action="store_true",
                        help="Enable live Qwen OCR (Pass 2).")
    parser.add_argument("--with-pass3", action="store_true",
                        help="Enable Pass 3 LLM anomaly explanation (requires --with-qwen).")
    args = parser.parse_args()
    main(with_qwen=args.with_qwen, with_pass3=args.with_pass3)
