"""
Phase 1 verification script.

Runs the full Phase-1 pipeline (preprocess -> align -> ROI extract) on every
sample document and saves:
  - output/verify/<page>/overlay.jpg   — full doc with all ROI boxes drawn
  - output/verify/<page>/crops/        — individual ROI crops (colour + binary)
  - output/verify/summary.txt          — alignment confidence and warnings

Run from the project root:
    venv/Scripts/python.exe verify_phase1.py
"""

from pathlib import Path
import sys, json

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.alignment import align
from ocr_pipeline.roi_extractor import ROIExtractor, save_crops, draw_roi_overlay

SAMPLES_DIR = Path(__file__).parent / "example"
OUTPUT_DIR  = Path(__file__).parent / "output" / "verify"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extractor = ROIExtractor()

summary_lines: list[str] = []


def _make_doc_id(img_path: Path) -> str:
    """Build a unique doc ID that includes batch info to avoid collisions.

    Handles two filename patterns present in the sample set:
      _N_page-XXXX  ->  N_XXXX   (e.g. "1_0001", "2_0020")
      _pages-to-jpg-XXXX  ->  ptj_XXXX
    """
    stem = img_path.stem
    if "_page-" in stem:
        batch = stem.split("_page-")[0].rsplit("_", 1)[-1]  # digit before "_page-"
        page  = stem.split("_page-")[1]
        return f"{batch}_{page}"
    else:
        # pages-to-jpg-XXXX pattern: trailing token after last "_" is "pages-to-jpg-XXXX"
        # → keep only the 4-digit suffix after the last hyphen
        number = stem.split("_")[-1].split("-")[-1]
        return f"ptj_{number}"


for img_path in sorted(SAMPLES_DIR.glob("*.jpg")):
    doc_id = _make_doc_id(img_path)
    print(f"[{doc_id}] Processing {img_path.name}...")

    try:
        # --- Phase 1 pipeline ---
        prep = preprocess(img_path)
        aln  = align(prep.deskewed)
        crops = extractor.extract_all(aln.image)  # all tiers

        # --- Save visual output ---
        doc_dir   = OUTPUT_DIR / doc_id
        crops_dir = doc_dir / "crops"
        doc_dir.mkdir(parents=True, exist_ok=True)

        overlay = draw_roi_overlay(aln.image, crops, extractor)

        import cv2
        cv2.imwrite(str(doc_dir / "overlay.jpg"), overlay)
        save_crops(crops, crops_dir, prefix=f"{doc_id}_")

        # --- Summary line ---
        warns = prep.warnings + aln.warnings
        line = (
            f"[{doc_id}] align={aln.method}({aln.confidence:.2f}) "
            f"skew={prep.skew_angle_deg:.1f}° scale={prep.scale_factor:.2f} "
            f"crops={len(crops)}"
        )
        if warns:
            line += f"\n        WARNINGS: {'; '.join(warns)}"
        summary_lines.append(line)
        print(f"  -> {line}")

    except Exception as exc:
        err = f"[{doc_id}] ERROR: {exc}"
        summary_lines.append(err)
        print(f"  -> {err}")

# Write summary
(OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
print(f"\nDone. Visual output in: {OUTPUT_DIR}")
print(f"Summary: {OUTPUT_DIR / 'summary.txt'}")
