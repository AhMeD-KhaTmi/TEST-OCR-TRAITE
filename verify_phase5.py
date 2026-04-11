"""
Phase 5 verification script — Production pipeline end-to-end.

Runs the full production pipeline on all 100 sample documents using the new
high-level process_document() API and generates:
  - Per-document HTML review reports
  - Batch HTML index page
  - JSONL audit log
  - JSON output per document
  - Summary text report

Usage:
  # Tesseract-only baseline (default, no LLM required)
  venv\\Scripts\\python.exe verify_phase5.py

  # With full Qwen inference + Pass 3 anomaly explanation
  venv\\Scripts\\python.exe verify_phase5.py --with-qwen --with-pass3

  # Limit to first N documents (for quick tests)
  venv\\Scripts\\python.exe verify_phase5.py --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr_pipeline.pipeline import (
    ProcessingConfig, process_document, PipelineResult
)
from ocr_pipeline.audit_logger import AuditLogger
from ocr_pipeline.review_reporter import generate_review_report, generate_batch_index
from ocr_pipeline.document_result import document_result_to_json

SAMPLES_DIR = Path(__file__).parent / "example"
OUTPUT_DIR  = Path(__file__).parent / "output" / "verify_phase5"


def _make_doc_id(img_path: Path) -> str:
    stem = img_path.stem
    if "_page-" in stem:
        batch = stem.split("_page-")[0].rsplit("_", 1)[-1]
        page  = stem.split("_page-")[1]
        return f"{batch}_{page}"
    number = stem.split("_")[-1].split("-")[-1]
    return f"ptj_{number}"


def main(with_qwen: bool, with_pass3: bool, limit: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir  = OUTPUT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    audit_path   = OUTPUT_DIR / "audit.jsonl"

    mode_parts = []
    if with_qwen:
        mode_parts.append("Qwen OCR")
    if with_pass3:
        mode_parts.append("Pass3")
    mode_label = "+".join(mode_parts) if mode_parts else "Tesseract-only"

    print(f"Phase 5 verification -- mode: {mode_label}")
    print(f"Processing images in : {SAMPLES_DIR}")
    print(f"Output directory     : {OUTPUT_DIR}\n")

    cfg = ProcessingConfig(
        skip_qwen=not with_qwen,
        skip_pass3=not with_pass3,
        run_stamp_detection=True,
        save_crops=False,
    )

    logger    = AuditLogger(audit_path)
    all_paths = sorted(SAMPLES_DIR.glob("*.jpg"))
    if limit:
        all_paths = all_paths[:limit]

    results: list[PipelineResult] = []
    errors = 0

    for img_path in all_paths:
        doc_id = _make_doc_id(img_path)
        print(f"[{doc_id}] {img_path.name}...")
        try:
            result = process_document(img_path, doc_id=doc_id, config=cfg)

            # Save JSON output
            json_str = result.to_json()
            (OUTPUT_DIR / f"{doc_id}.json").write_text(json_str, encoding="utf-8")

            # Save HTML review report (always — helps human reviewers even for OK docs)
            report_path = reports_dir / f"{doc_id}.html"
            generate_review_report(result, output_path=report_path)

            # Log to audit trail
            logger.log(result, image_path=img_path)

            results.append(result)
            dr = result.doc_result
            review  = "REVIEW" if dr.needs_human_review else "OK"
            hard    = sum(1 for f in dr.flagged_fields if f.is_hard_failure)
            stamps  = dr.stamp_info.stamp_count
            time_s  = result.processing_time_s
            print(
                f"  -> [{doc_id}] {review}  conf={dr.document_confidence:.2f}"
                f"  stamps={stamps}  hard={hard}  time={time_s:.2f}s"
            )
        except Exception as exc:
            print(f"  -> [{doc_id}] ERROR: {exc}")
            errors += 1

    # ------------------------------------------------------------------
    # Batch index + aggregate stats
    # ------------------------------------------------------------------
    generate_batch_index(results, OUTPUT_DIR / "index.html", reports_dir)

    stats = logger.get_summary_stats()
    review_count = sum(1 for r in results if r.needs_human_review)

    summary_lines = [
        f"Phase 5 verification -- mode: {mode_label}",
        "=" * 70,
        f"Total              : {len(all_paths)}",
        f"Successful         : {len(results)}",
        f"Errors             : {errors}",
        f"Needs human review : {review_count}",
        f"Avg confidence     : {stats.get('avg_confidence', 0):.4f}",
        f"Avg processing     : {stats.get('avg_processing_time_s', 0):.3f}s",
        f"Stamp detection    : {stats.get('stamp_detection_rate', 0)*100:.1f}% of docs",
        "",
        f"Audit log          : {audit_path}",
        f"Batch index        : {OUTPUT_DIR}/index.html",
        f"Individual reports : {reports_dir}/",
    ]
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Errors              : {errors} / {len(all_paths)}")
    print(f"  Needs human review  : {review_count} / {len(results)}")
    print(f"  Avg confidence      : {stats.get('avg_confidence', 0):.4f}")
    print(f"  Avg processing time : {stats.get('avg_processing_time_s', 0):.3f}s")
    print(f"  Batch index         : {OUTPUT_DIR / 'index.html'}")
    print(f"  Audit log           : {audit_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 production pipeline verification")
    parser.add_argument("--with-qwen",  action="store_true", help="Enable Qwen OCR pass.")
    parser.add_argument("--with-pass3", action="store_true", help="Enable Pass 3 anomaly explanation.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N documents.")
    args = parser.parse_args()
    main(with_qwen=args.with_qwen, with_pass3=args.with_pass3, limit=args.limit)
