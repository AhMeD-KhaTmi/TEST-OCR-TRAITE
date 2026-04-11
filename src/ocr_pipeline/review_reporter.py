"""
Phase 5 — Human review report generator.

Produces a standalone HTML report for each document that needs human review,
showing:
  - Side-by-side: original ROI crop image vs extracted value
  - Confidence score (colour-coded: green ≥ 0.85, yellow ≥ 0.6, red < 0.6)
  - Validation flags (HARD failures vs soft warnings)
  - Stamp detection summary
  - Anomaly explanation (Pass 3, if available)
  - Full document JSON for reference

The HTML is self-contained (no external dependencies): all styles are inline,
crop images are embedded as base64 data URIs.

Usage
-----
from ocr_pipeline.review_reporter import generate_review_report

html = generate_review_report(pipeline_result)
Path("review/doc001.html").write_text(html, encoding="utf-8")

# Or save directly:
generate_review_report(pipeline_result, output_path="review/doc001.html")
"""

from __future__ import annotations

import base64
import json
from decimal import Decimal
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pipeline import PipelineResult
from .document_result import document_result_to_json


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _conf_colour(conf: float) -> str:
    if conf >= 0.85:
        return "#22c55e"   # green
    if conf >= 0.60:
        return "#f59e0b"   # amber
    return "#ef4444"       # red


def _crop_to_b64(crop: np.ndarray) -> str:
    """Encode a NumPy image as a base64 PNG data URI."""
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Field row rendering
# ---------------------------------------------------------------------------

def _field_row(
    label: str,
    value: Optional[str],
    confidence: Optional[float],
    crop: Optional[np.ndarray],
    flag: Optional[str] = None,
    flag_hard: bool = False,
) -> str:
    """Render one field row in the review table."""
    # Crop image
    img_html = ""
    if crop is not None and crop.size > 0:
        b64 = _crop_to_b64(crop)
        if b64:
            img_html = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:320px;max-height:80px;border:1px solid #334155;'
                f'border-radius:4px;background:#0f172a;" alt="{label} crop">'
            )

    # Confidence badge
    conf_html = ""
    if confidence is not None:
        col = _conf_colour(confidence)
        conf_html = (
            f'<span style="background:{col};color:#fff;padding:2px 8px;'
            f'border-radius:12px;font-size:0.75rem;font-weight:700;">'
            f'{confidence:.2f}</span>'
        )

    # Flag badge
    flag_html = ""
    if flag:
        severity_col = "#ef4444" if flag_hard else "#f59e0b"
        severity_lbl = "HARD" if flag_hard else "WARN"
        flag_html = (
            f'<span style="background:{severity_col};color:#fff;padding:2px 8px;'
            f'border-radius:12px;font-size:0.7rem;margin-left:6px;">'
            f'{severity_lbl}: {flag}</span>'
        )

    value_display = value if value else '<em style="color:#64748b;">—</em>'

    return f"""
    <tr>
      <td style="padding:8px 12px;font-weight:600;color:#94a3b8;white-space:nowrap;">{label}</td>
      <td style="padding:8px 12px;">{img_html}</td>
      <td style="padding:8px 12px;font-family:monospace;color:#e2e8f0;">{value_display}{flag_html}</td>
      <td style="padding:8px 12px;text-align:center;">{conf_html}</td>
    </tr>"""


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_CSS = """
  body { margin:0; font-family:'Segoe UI',Arial,sans-serif; background:#0f172a; color:#e2e8f0; }
  .container { max-width:1100px; margin:0 auto; padding:24px; }
  h1 { font-size:1.5rem; color:#f8fafc; margin-bottom:4px; }
  .subtitle { color:#64748b; font-size:0.9rem; margin-bottom:24px; }
  .card { background:#1e293b; border-radius:10px; padding:20px; margin-bottom:20px; }
  .card h2 { font-size:1rem; color:#94a3b8; margin:0 0 14px; text-transform:uppercase;
              letter-spacing:0.08em; border-bottom:1px solid #334155; padding-bottom:8px; }
  table { width:100%; border-collapse:collapse; }
  tr:nth-child(even) td { background:#1a2535; }
  .badge-ok  { color:#22c55e; font-weight:700; }
  .badge-err { color:#ef4444; font-weight:700; }
  .badge-warn{ color:#f59e0b; font-weight:700; }
  .json-box  { background:#0f172a; border:1px solid #334155; border-radius:8px;
               padding:14px; font-family:monospace; font-size:0.75rem;
               white-space:pre-wrap; overflow-x:auto; color:#94a3b8; }
  .stamp-hint { background:#1e3a5f; border-left:3px solid #3b82f6; padding:8px 12px;
                border-radius:0 6px 6px 0; margin:4px 0; font-size:0.85rem; }
  .anom-box  { background:#2d1f3f; border-left:3px solid #a855f7; padding:12px 14px;
               border-radius:0 8px 8px 0; color:#c4b5fd; font-style:italic; }
"""


def _build_html(result: PipelineResult) -> str:
    dr   = result.doc_result
    crops = result.crops or {}

    # ------------------------------------------------------------------
    # Build a map of field → flag message for quick lookup
    # ------------------------------------------------------------------
    flag_map: dict[str, tuple[str, bool]] = {}
    for ff in dr.flagged_fields:
        flag_map[ff.field] = (ff.message, ff.is_hard_failure)

    def get_crop(roi_id: str) -> Optional[np.ndarray]:
        c = crops.get(roi_id)
        if c is None:
            return None
        return c.colour if hasattr(c, "colour") else c

    def flag_for(field: str) -> tuple[Optional[str], bool]:
        if field in flag_map:
            msg, hard = flag_map[field]
            return msg, hard
        return None, False

    # ------------------------------------------------------------------
    # Review-needed banner
    # ------------------------------------------------------------------
    status_colour = "#ef4444" if dr.needs_human_review else "#22c55e"
    status_label  = "NEEDS HUMAN REVIEW" if dr.needs_human_review else "ACCEPTED"

    # ------------------------------------------------------------------
    # Overview card
    # ------------------------------------------------------------------
    overview_rows = [
        ("Document ID",   dr.document_id,                       None,  None),
        ("Payment Order", dr.payment_order_number or "—",       None,  None),
        ("Extracted at",  dr.extraction_timestamp,              None,  None),
        ("Processing",    f"{result.processing_time_s:.2f} s",  None,  None),
        ("Doc confidence",f"{dr.document_confidence:.3f}",      None,  None),
        ("Stamps detected", str(dr.stamp_info.stamp_count),     None,  None),
        ("Qwen source",   dr.qwen_corrections.source,           None,  None),
    ]
    overview_html = "".join(
        f'<tr><td style="padding:6px 12px;color:#94a3b8;">{r[0]}</td>'
        f'<td style="padding:6px 12px;font-family:monospace;">{r[1]}</td></tr>'
        for r in overview_rows
    )

    # ------------------------------------------------------------------
    # Fields card
    # ------------------------------------------------------------------
    rib_flag, rib_hard = flag_for("rib")
    amt_flag, amt_hard = flag_for("amount")
    ech_flag, ech_hard = flag_for("echeance")
    cdt_flag, cdt_hard = flag_for("creation_date")

    fields_html = ""
    # RIB
    fields_html += _field_row(
        "RIB",
        dr.rib.full if dr.rib else None,
        dr.rib.confidence if dr.rib else None,
        get_crop("R05"),
        rib_flag, rib_hard,
    )
    # Amount numeric
    fields_html += _field_row(
        "Amount (numeric)",
        str(dr.amount.value_numeric) + " DT" if dr.amount and dr.amount.value_numeric else None,
        dr.amount.confidence if dr.amount else None,
        get_crop("R06"),
        amt_flag, amt_hard,
    )
    # Amount in words
    fields_html += _field_row(
        "Amount (words)",
        dr.amount.value_text if dr.amount else None,
        None,
        get_crop("R09"),
    )
    # Echeance
    fields_html += _field_row(
        "Echeance",
        dr.echeance.value if dr.echeance else None,
        dr.echeance.confidence if dr.echeance else None,
        get_crop("R02"),
        ech_flag, ech_hard,
    )
    # Creation date
    fields_html += _field_row(
        "Creation date",
        dr.creation_date.value if dr.creation_date else None,
        dr.creation_date.confidence if dr.creation_date else None,
        get_crop("R03"),
        cdt_flag, cdt_hard,
    )
    # City
    fields_html += _field_row(
        "City",
        dr.creation_city.value if dr.creation_city else None,
        dr.creation_city.confidence if dr.creation_city else None,
        get_crop("R04"),
    )
    # Tireur
    fields_html += _field_row(
        "Tireur",
        dr.tireur.value if dr.tireur else None,
        dr.tireur.confidence if dr.tireur else None,
        get_crop("R07"),
    )
    # Beneficiary
    fields_html += _field_row(
        "Beneficiaire",
        dr.beneficiary.value if dr.beneficiary else None,
        dr.beneficiary.confidence if dr.beneficiary else None,
        get_crop("R08"),
    )
    # Tire
    fields_html += _field_row(
        "Tire (drawee)",
        dr.tire.value if dr.tire else None,
        dr.tire.confidence if dr.tire else None,
        get_crop("R15"),
    )
    # Domiciliation
    fields_html += _field_row(
        "Domiciliation",
        dr.domiciliation.value if dr.domiciliation else None,
        dr.domiciliation.confidence if dr.domiciliation else None,
        get_crop("R16"),
    )

    # ------------------------------------------------------------------
    # Flags card
    # ------------------------------------------------------------------
    if dr.flagged_fields:
        flags_rows = ""
        for ff in dr.flagged_fields:
            col   = "#ef4444" if ff.is_hard_failure else "#f59e0b"
            label = "HARD" if ff.is_hard_failure else "WARN"
            flags_rows += (
                f'<tr>'
                f'<td style="padding:6px 12px;"><span style="background:{col};color:#fff;'
                f'padding:2px 6px;border-radius:8px;font-size:0.7rem;">{label}</span></td>'
                f'<td style="padding:6px 12px;font-weight:600;">{ff.field}</td>'
                f'<td style="padding:6px 12px;color:#94a3b8;">{ff.message}</td>'
                f'</tr>'
            )
        flags_card = f"""
        <div class="card">
          <h2>Validation Flags ({len(dr.flagged_fields)})</h2>
          <table>{flags_rows}</table>
        </div>"""
    else:
        flags_card = ""

    # ------------------------------------------------------------------
    # Stamp hints card
    # ------------------------------------------------------------------
    if dr.stamp_info.hints:
        hints_html = "".join(
            f'<div class="stamp-hint">{h}</div>' for h in dr.stamp_info.hints
        )
        stamp_card = f"""
        <div class="card">
          <h2>Stamp Detection ({dr.stamp_info.stamp_count} detected)</h2>
          {hints_html}
        </div>"""
    else:
        stamp_card = ""

    # ------------------------------------------------------------------
    # Anomaly explanation card
    # ------------------------------------------------------------------
    if dr.anomaly_explanation:
        anom_card = f"""
        <div class="card">
          <h2>Pass 3 — Anomaly Explanation</h2>
          <div class="anom-box">{dr.anomaly_explanation}</div>
        </div>"""
    else:
        anom_card = ""

    # ------------------------------------------------------------------
    # JSON output card (truncated to 4000 chars)
    # ------------------------------------------------------------------
    full_json = document_result_to_json(dr, indent=2)
    json_preview = full_json[:4000] + ("\n… (truncated)" if len(full_json) > 4000 else "")

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Review — {dr.document_id}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>Lettre de Change — Review Report</h1>
  <div class="subtitle">Document: <strong>{dr.document_id}</strong> &nbsp;|&nbsp;
    <span style="color:{status_colour};font-weight:700;">{status_label}</span>
    &nbsp;|&nbsp; Confidence: <strong>{dr.document_confidence:.3f}</strong>
  </div>

  <div class="card">
    <h2>Overview</h2>
    <table>{overview_html}</table>
  </div>

  <div class="card">
    <h2>Extracted Fields</h2>
    <table>
      <thead>
        <tr>
          <th style="padding:8px 12px;text-align:left;color:#64748b;">Field</th>
          <th style="padding:8px 12px;text-align:left;color:#64748b;">ROI Crop</th>
          <th style="padding:8px 12px;text-align:left;color:#64748b;">Extracted Value</th>
          <th style="padding:8px 12px;text-align:center;color:#64748b;">Conf</th>
        </tr>
      </thead>
      <tbody>{fields_html}</tbody>
    </table>
  </div>

  {flags_card}
  {stamp_card}
  {anom_card}

  <div class="card">
    <h2>Full JSON Output</h2>
    <div class="json-box">{json_preview}</div>
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_review_report(
    result: PipelineResult,
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate a standalone HTML review report for one document.

    Args:
        result:       PipelineResult from process_document().
        output_path:  If given, write the HTML to this file as well.

    Returns:
        The full HTML string.
    """
    html = _build_html(result)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html


def generate_batch_index(
    results: list[PipelineResult],
    output_path: str | Path,
    report_dir: str | Path,
) -> str:
    """Generate an HTML index page listing all processed documents.

    Args:
        results:     List of PipelineResults.
        output_path: Path to write the index HTML.
        report_dir:  Directory where individual reports are saved (used to
                     build links).

    Returns:
        The full index HTML string.
    """
    report_dir = Path(report_dir)
    rows = ""
    for r in results:
        dr = r.doc_result
        status_col = "#ef4444" if dr.needs_human_review else "#22c55e"
        status_lbl = "REVIEW" if dr.needs_human_review else "OK"
        conf_col   = _conf_colour(dr.document_confidence)
        report_link = report_dir / f"{dr.document_id}.html"
        hard = sum(1 for f in dr.flagged_fields if f.is_hard_failure)
        rows += (
            f'<tr>'
            f'<td style="padding:6px 12px;">'
            f'<a href="{report_link}" style="color:#60a5fa;">{dr.document_id}</a></td>'
            f'<td style="padding:6px 12px;text-align:center;">'
            f'<span style="color:{status_col};font-weight:700;">{status_lbl}</span></td>'
            f'<td style="padding:6px 12px;text-align:center;">'
            f'<span style="color:{conf_col};">{dr.document_confidence:.3f}</span></td>'
            f'<td style="padding:6px 12px;text-align:center;">{hard}</td>'
            f'<td style="padding:6px 12px;text-align:center;">'
            f'{dr.stamp_info.stamp_count}</td>'
            f'<td style="padding:6px 12px;text-align:center;">'
            f'{r.processing_time_s:.1f}s</td>'
            f'</tr>'
        )

    review_count = sum(1 for r in results if r.doc_result.needs_human_review)
    ok_count     = len(results) - review_count

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Batch Review Index</title>
<style>
  body {{ margin:0; font-family:'Segoe UI',Arial,sans-serif;
         background:#0f172a; color:#e2e8f0; }}
  .container {{ max-width:900px; margin:0 auto; padding:24px; }}
  h1 {{ color:#f8fafc; }}
  .stats {{ display:flex; gap:20px; margin-bottom:24px; }}
  .stat {{ background:#1e293b; border-radius:8px; padding:14px 20px; flex:1; }}
  .stat .label {{ color:#64748b; font-size:0.8rem; margin-bottom:4px; }}
  .stat .value {{ font-size:1.4rem; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; background:#1e293b;
           border-radius:10px; overflow:hidden; }}
  th {{ padding:10px 12px; text-align:left; color:#64748b;
        background:#0f172a; font-size:0.8rem; text-transform:uppercase; }}
  tr:nth-child(even) td {{ background:#162032; }}
  a {{ text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<div class="container">
  <h1>Batch Review Index — {len(results)} documents</h1>
  <div class="stats">
    <div class="stat">
      <div class="label">Total</div>
      <div class="value" style="color:#e2e8f0;">{len(results)}</div>
    </div>
    <div class="stat">
      <div class="label">Accepted</div>
      <div class="value" style="color:#22c55e;">{ok_count}</div>
    </div>
    <div class="stat">
      <div class="label">Needs Review</div>
      <div class="value" style="color:#ef4444;">{review_count}</div>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Document ID</th><th>Status</th><th>Confidence</th>
        <th>Hard Flags</th><th>Stamps</th><th>Time</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return html
