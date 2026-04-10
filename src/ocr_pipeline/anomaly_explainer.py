"""
Phase 4 — Pass 3: Anomaly Explanation.

Implements the third OCR pass described in plan section 3.4 / 3.7:
a full-document vision LLM call whose sole purpose is to explain WHY
validated field mismatches or low-confidence flags occurred.

Role and constraints (from plan):
  - INFORMATIONAL ONLY — does NOT override Pass 1+2 extraction results.
  - Does NOT perform validation — all rule-based validation remains in
    validator.py.
  - Output is attached to flagged DocumentResult fields as
    ``anomaly_explanation`` strings for human reviewers.
  - If the LLM is unavailable, the explanation is an empty string and
    the pipeline continues normally (graceful degradation).

The prompt deliberately:
  - Tells the model the already-extracted values so it does not need to
    re-extract anything.
  - Asks ONLY about visual observations that explain discrepancies.
  - Forbids inventing new field values.
"""

from __future__ import annotations

import base64
import json
import os
import re
import textwrap
import urllib.request
import urllib.error
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# Re-use the same endpoint config as qwen_ocr.py
_INFERENCE_MODE: str      = os.getenv("INFERENCE_MODE", "local").lower()
_API_KEY: str             = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_ENDPOINT: str = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
_OPENROUTER_MODEL: str    = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-vl-8b-instruct")
_LOCAL_ENDPOINT: str      = os.getenv("LOCAL_ENDPOINT", "http://localhost:11434/v1/chat/completions")
_LOCAL_MODEL: str         = os.getenv("LOCAL_MODEL", "qwen3-vl-8b")
_LLM_TIMEOUT: int         = int(os.getenv("LLM_TIMEOUT", "90"))  # longer timeout for full-page

_ENDPOINT: str = _OPENROUTER_ENDPOINT if _INFERENCE_MODE == "api" else _LOCAL_ENDPOINT
_MODEL: str    = _OPENROUTER_MODEL    if _INFERENCE_MODE == "api" else _LOCAL_MODEL


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyExplanation:
    """Result of a Pass 3 anomaly explanation call."""
    explanation: str = ""           # human-readable LLM explanation
    stamp_regions_mentioned: list[str] = dc_field(default_factory=list)
    engine_available: bool = True
    error: Optional[str] = None

    @property
    def has_explanation(self) -> bool:
        return bool(self.explanation.strip())


# ---------------------------------------------------------------------------
# System prompt for Pass 3
# ---------------------------------------------------------------------------

_PASS3_SYSTEM = textwrap.dedent("""\
    You are a document analyst assistant. Your job is to OBSERVE and DESCRIBE
    what you see in a scanned Tunisian Lettre de Change (bill of exchange) image.

    Rules you MUST follow:
    - Do NOT extract new field values. The extraction has already been done.
    - Do NOT validate or override extracted values.
    - ONLY describe visual observations that explain WHY extracted values may be
      wrong, unreadable, or inconsistent.
    - Focus on: stamps covering text, ink bleed-through, poor scan quality,
      handwriting over printed fields, torn areas, fold lines.
    - Keep your response concise (3-6 sentences maximum).
    - If everything looks clean and you see no reason for the flags, say so briefly.
    - Do NOT invent or hallucinate observations you cannot see.
""")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_pass3_prompt(
    flagged_fields: list[str],
    extracted_summary: dict[str, str],
    stamp_region_hints: list[str],
) -> str:
    """Build the user prompt for the Pass 3 anomaly explanation call.

    Args:
        flagged_fields:     List of field names that failed validation
                            (e.g. ["rib", "amount"]).
        extracted_summary:  Dict of field_name -> extracted_value for key fields.
        stamp_region_hints: Strings describing detected stamp locations
                            (e.g. ["Blue stamp at top-left (x=120-380, y=50-290)"]).
    """
    lines = [
        "The following fields were extracted from this Lettre de Change:",
        "",
    ]
    for field, value in extracted_summary.items():
        lines.append(f"  {field}: {value!r}")

    if flagged_fields:
        lines += [
            "",
            f"The following fields are flagged (failed validation or low confidence):",
            "  " + ", ".join(flagged_fields),
        ]

    if stamp_region_hints:
        lines += [
            "",
            "Automatic stamp detection found the following stamp region(s):",
        ]
        for hint in stamp_region_hints:
            lines.append(f"  - {hint}")

    lines += [
        "",
        "Look at the document image carefully. For each flagged field, describe",
        "only what you visually observe that might explain the problem",
        "(e.g. a stamp is covering the upper RIB digits, or the amount box has",
        "ink smudging). Do NOT provide corrected values.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image_b64(image: np.ndarray) -> str:
    """Encode a BGR image as base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Failed to encode document image to JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_for_llm(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Downscale the full-page image so it fits within max_side pixels.

    The full A4 image at 300 DPI is ~2480 × 3508 px — too large for most
    LLM API limits.  We downscale to ≤1280 px on the longest side while
    preserving aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------

def _call_llm(prompt_user: str, image_b64: str) -> str:
    """Make a single LLM call with the full-page image.

    Returns the raw response string.
    Raises urllib.error.URLError / OSError on network problems.
    """
    payload = {
        "model": _MODEL,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": _PASS3_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt_user},
                ],
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _INFERENCE_MODE == "api" and _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
        headers["HTTP-Referer"]   = "https://github.com/tunisian-lcr-ocr"
        headers["X-Title"]        = "Tunisian Lettre de Change OCR — Pass 3"

    req = urllib.request.Request(
        _ENDPOINT, data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as resp:
        return resp.read().decode("utf-8")


def _parse_response(raw: str) -> str:
    """Extract the text explanation from the API envelope."""
    try:
        envelope = json.loads(raw)
        content = envelope["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return raw.strip()
    # Strip markdown fences if present
    content = re.sub(r"```(?:\w+)?\s*", "", content).strip().rstrip("```").strip()
    return content


# ---------------------------------------------------------------------------
# Stamp hint builder
# ---------------------------------------------------------------------------

def build_stamp_hints(stamps: list) -> list[str]:
    """Convert StampRegion objects to human-readable location strings.

    Args:
        stamps: list of StampRegion from stamp_detector.StampDetectionResult.

    Returns:
        List of strings like "Blue stamp centred at (250, 180), radius ~85 px"
    """
    hints = []
    for s in stamps:
        hints.append(
            f"{s.colour.capitalize()} stamp centred at ({s.cx}, {s.cy}), "
            f"radius ~{s.rx} px (detected via {s.method}, "
            f"confidence {s.confidence:.2f})"
        )
    return hints


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_anomalies(
    document_image: np.ndarray,
    flagged_fields: list[str],
    extracted_summary: dict[str, str],
    stamp_hints: Optional[list[str]] = None,
    skip: bool = False,
) -> AnomalyExplanation:
    """Run Pass 3: request a visual anomaly explanation from the LLM.

    Args:
        document_image:     Full aligned BGR document image.
        flagged_fields:     Field names that failed validation.
        extracted_summary:  Key extracted values for context.
        stamp_hints:        Pre-built stamp location strings (from build_stamp_hints).
        skip:               If True, return an empty explanation immediately
                            (used in Tesseract-only mode to skip API calls).

    Returns:
        AnomalyExplanation — empty but valid if LLM is unavailable or skip=True.
    """
    if skip or not flagged_fields:
        return AnomalyExplanation(explanation="", engine_available=not skip)

    prompt = _build_pass3_prompt(
        flagged_fields=flagged_fields,
        extracted_summary=extracted_summary,
        stamp_region_hints=stamp_hints or [],
    )

    try:
        resized = _resize_for_llm(document_image)
        image_b64 = _encode_image_b64(resized)
    except Exception as exc:
        return AnomalyExplanation(
            explanation="",
            engine_available=False,
            error=f"Image encoding failed: {exc}",
        )

    try:
        raw = _call_llm(prompt, image_b64)
    except (urllib.error.URLError, OSError) as exc:
        return AnomalyExplanation(
            explanation="",
            engine_available=False,
            error=f"LLM server unreachable: {exc}",
        )
    except Exception as exc:
        return AnomalyExplanation(
            explanation="",
            engine_available=False,
            error=f"LLM call failed: {exc}",
        )

    explanation = _parse_response(raw)
    return AnomalyExplanation(
        explanation=explanation,
        stamp_regions_mentioned=stamp_hints or [],
        engine_available=True,
    )
