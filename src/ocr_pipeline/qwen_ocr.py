"""
Phase 2 — Pass 2: Qwen3 VL 8B OCR engine.

Provides:
- Field-specific prompt templates with anti-hallucination system prompt
- Structured JSON output parsing (handles partial/malformed JSON gracefully)
- Digit-divergence rejection rule (>30% changed vs Tesseract → reject)
- Offline/stub mode when Qwen server is not running (returns placeholder result)

Qwen3 VL 8B is expected to be served locally via a compatible OpenAI-style API
(e.g., vLLM, Ollama, or LMStudio). Default endpoint: http://localhost:11434
"""

from __future__ import annotations

import base64
import json
import os
import re
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
    pass  # python-dotenv not installed; fall back to environment variables


# ---------------------------------------------------------------------------
# Configuration — loaded from .env
# ---------------------------------------------------------------------------

_INFERENCE_MODE: str       = os.getenv("INFERENCE_MODE", "local").lower()
_API_KEY: str              = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_ENDPOINT: str  = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
_OPENROUTER_MODEL: str     = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-vl-8b-instruct")
_LOCAL_ENDPOINT: str       = os.getenv("LOCAL_ENDPOINT", "http://localhost:11434/v1/chat/completions")
_LOCAL_MODEL: str          = os.getenv("LOCAL_MODEL", "qwen3-vl-8b")
_LOCAL_HF_MODEL: str       = os.getenv("HF_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
_LLM_TIMEOUT: int          = int(os.getenv("LLM_TIMEOUT", "60"))

# Computed active values for HTTP modes (api / local)
QWEN_ENDPOINT: str  = _OPENROUTER_ENDPOINT if _INFERENCE_MODE == "api" else _LOCAL_ENDPOINT
QWEN_MODEL: str     = _OPENROUTER_MODEL    if _INFERENCE_MODE == "api" else _LOCAL_MODEL
QWEN_TIMEOUT: int   = _LLM_TIMEOUT

# Lazy-loaded HF model components (used only when INFERENCE_MODE=hf)
_HF_MODEL_INSTANCE = None
_HF_PROCESSOR_INSTANCE = None

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QwenResult:
    text: str                          # primary extracted text
    readable: bool = True              # False if field is occluded/unreadable
    raw_response: str = ""             # raw model output
    engine_available: bool = True      # False when server is unreachable
    error: Optional[str] = None
    unreadable_positions: list[int] = dc_field(default_factory=list)


# ---------------------------------------------------------------------------
# Anti-hallucination system prompt (prepended to ALL field prompts)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an OCR system. Follow these rules absolutely:
- Copy EXACTLY what is visible in the image. Nothing more.
- DO NOT guess missing or unclear characters.
- If a character is unreadable → output "?" for that character.
- DO NOT complete partial numbers.
- DO NOT normalize formatting (keep spaces, commas, dots as-is).
- DO NOT clean up or reformat the text.
- The blue circular stamps contain company info (phone numbers, tax IDs). IGNORE all stamp text.
- Return only the raw extracted text as valid JSON, nothing else."""


# ---------------------------------------------------------------------------
# Field-specific prompt templates
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    # RIB fields
    "rib": (
        "Extract the bank identification number (RIB) from this image. "
        "The RIB format is: 2-digit bank code, space, 3-digit branch code, space, "
        "13-digit account number, space, 2-digit check key (total 20 digits). "
        "Return ONLY the digits you can see, grouped with spaces. "
        "If any digits are obscured by a stamp, return '?' for each unreadable digit. "
        "Do NOT guess. Do NOT invent digits. "
        "Return JSON: {\"text\": \"08 006 0110510000870 41\", \"readable\": true}"
    ),
    # Amount digit fields
    "amount_digits": (
        "Extract the monetary amount (in digits) from this image. "
        "Keep all punctuation exactly as written (commas, dots, # symbols). "
        "If digits are obscured, use '?' for each unreadable digit. "
        "Do NOT round or reformat. Do NOT remove # delimiters. "
        "Return JSON: {\"text\": \"3000,000\", \"readable\": true}"
    ),
    # Amount text fields
    "amount_text": (
        "Extract the amount written in words (montant en lettres) from this image. "
        "The text is handwritten French. Copy exactly what you read. "
        "If part of the text is unreadable, use '...' for that part. "
        "Return JSON: {\"text\": \"trois mille dinars\", \"readable\": true}"
    ),
    # Date fields
    "date": (
        "Extract the date from this image. Expected format is DD/MM/YYYY. "
        "Accept alternative separators (-, .). Copy exactly what you see. "
        "If digits are unclear, use '?' for each unclear digit. "
        "Return JSON: {\"text\": \"15/06/2025\", \"readable\": true}"
    ),
    # Name/address fields (tireur, beneficiary, drawee)
    "name": (
        "Extract the name and address text from this image. "
        "Copy exactly what is written, including any abbreviations. "
        "Ignore any blue circular stamp text. "
        "If part of the text is blocked by a stamp, use '[STAMP]' to indicate. "
        "Return JSON: {\"text\": \"STE ACME SARL\\nSousse\", \"readable\": true}"
    ),
    # Domiciliation (bank branch)
    "domiciliation": (
        "Extract the bank domiciliation information from this image. "
        "This typically shows a bank name and branch location. "
        "A blue stamp may partially cover this field — ignore stamp text. "
        "If the bank name is obscured, use '[STAMP]' for that part. "
        "Return JSON: {\"text\": \"BIAT Sousse Centre\", \"readable\": true}"
    ),
    # City fields
    "city": (
        "Extract the city name from this image. It is a single word or short phrase. "
        "Return JSON: {\"text\": \"Tunis\", \"readable\": true}"
    ),
    # Generic fallback
    "generic": (
        "Extract all visible text from this image exactly as it appears. "
        "If any part is unreadable, use '?' characters. "
        "Return JSON: {\"text\": \"...\", \"readable\": true}"
    ),
}

# ROI-to-prompt-type mapping
_ROI_PROMPT_MAP: dict[str, str] = {
    "R01": "generic",          # payment order — Tesseract primary
    "R02": "date",             # écheance upper
    "R03": "date",             # creation date upper
    "R04": "city",             # city upper
    "R05": "rib",              # RIB upper
    "R06": "amount_digits",    # amount upper
    "R07": "name",             # tireur
    "R08": "name",             # beneficiary
    "R09": "amount_text",      # amount in words
    "R10": "amount_digits",    # amount lower
    "R11": "city",             # city lower
    "R12": "date",             # creation date lower
    "R13": "date",             # écheance lower
    "R14": "rib",              # RIB lower
    "R15": "name",             # drawee
    "R16": "domiciliation",    # domiciliation
    "R17": "generic",          # barcode (fallback only)
}


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _encode_image_b64(crop: np.ndarray) -> str:
    """Encode a NumPy BGR image as a base64 JPEG string for API upload."""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Failed to encode crop image to JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# HuggingFace direct inference (INFERENCE_MODE=hf)
# ---------------------------------------------------------------------------

def _load_hf_model() -> None:
    """Lazy-load Qwen3-VL from HuggingFace cache with 4-bit quantization.

    Requires: torch (CUDA), transformers>=5.0, accelerate, bitsandbytes,
              qwen-vl-utils, Pillow.  The model is loaded once and reused.
    """
    global _HF_MODEL_INSTANCE, _HF_PROCESSOR_INSTANCE
    if _HF_MODEL_INSTANCE is not None:
        return

    import torch
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    _HF_PROCESSOR_INSTANCE = AutoProcessor.from_pretrained(
        _LOCAL_HF_MODEL, trust_remote_code=True
    )
    _HF_MODEL_INSTANCE = Qwen3VLForConditionalGeneration.from_pretrained(
        _LOCAL_HF_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _HF_MODEL_INSTANCE.eval()


def _run_hf_inference(user_prompt: str, image_b64: str) -> str:
    """Run inference directly through the cached HF model (no HTTP server).

    Returns the same API-envelope JSON string as ``_call_qwen()`` so that
    ``_parse_qwen_response()`` can handle the output without modification.
    """
    import io
    import torch
    import PIL.Image
    from qwen_vl_utils import process_vision_info

    _load_hf_model()

    # Decode base64 JPEG → PIL RGB image
    img_bytes = base64.b64decode(image_b64)
    pil_img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text_for_model = _HF_PROCESSOR_INSTANCE.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _HF_PROCESSOR_INSTANCE(
        text=[text_for_model],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_HF_MODEL_INSTANCE.device)

    with torch.no_grad():
        generated_ids = _HF_MODEL_INSTANCE.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    decoded = _HF_PROCESSOR_INSTANCE.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    content = decoded[0] if decoded else ""

    # Wrap in API-envelope format expected by _parse_qwen_response()
    return json.dumps({"choices": [{"message": {"content": content}}]})


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_qwen(prompt_user: str, image_b64: str) -> str:
    """Make a single call to the local Qwen3 VL endpoint.

    Returns the raw model response string.
    Raises urllib.error.URLError / ConnectionRefusedError on network errors.
    """
    payload = {
        "model": QWEN_MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
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
        headers["X-Title"]        = "Tunisian Lettre de Change OCR"

    req = urllib.request.Request(
        QWEN_ENDPOINT,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    return body


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_qwen_response(raw: str) -> tuple[str, bool]:
    """Parse the API response envelope and extract model text.

    Returns (content_string, readable).
    """
    try:
        envelope = json.loads(raw)
        content = envelope["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return raw, True  # fallback: treat the whole response as content

    # Try to parse the content as JSON {"text": ..., "readable": ...}
    content = content.strip()
    # Strip markdown code fences if present
    content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("```").strip()

    try:
        parsed = json.loads(content)
        text = str(parsed.get("text", "")).strip()
        readable = bool(parsed.get("readable", True))
        return text, readable
    except (json.JSONDecodeError, TypeError):
        # Content is not JSON — return it as plain text
        return content, True


# ---------------------------------------------------------------------------
# Digit-divergence rejection rule
# ---------------------------------------------------------------------------

def _count_digit_changes(tess_text: str, qwen_text: str) -> tuple[int, int]:
    """Count digit changes between Tesseract and Qwen outputs.

    Returns (changed_digits, total_tesseract_digits).
    """
    tess_digits = re.findall(r"\d", tess_text)
    qwen_digits  = re.findall(r"\d", qwen_text)

    if not tess_digits:
        return 0, 0

    # Align digit sequences to compare position by position
    min_len = min(len(tess_digits), len(qwen_digits))
    changed = sum(1 for t, q in zip(tess_digits[:min_len], qwen_digits[:min_len]) if t != q)
    # Additions or deletions count as changes too
    changed += abs(len(tess_digits) - len(qwen_digits))
    return changed, len(tess_digits)


def qwen_diverges_too_much(
    tess_text: str,
    qwen_text: str,
    threshold: float = 0.30,
) -> bool:
    """Return True if Qwen changed > threshold fraction of Tesseract digits."""
    changed, total = _count_digit_changes(tess_text, qwen_text)
    if total == 0:
        return False
    return (changed / total) > threshold


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

def run_qwen(
    crop: np.ndarray,
    roi_id: str,
    prompt_override: Optional[str] = None,
) -> QwenResult:
    """Run Qwen3 VL 8B on a single ROI crop.

    Args:
        crop:            BGR image crop (NumPy array)
        roi_id:          ROI identifier string (e.g. "R05")
        prompt_override: If given, use this user prompt instead of the default.

    Returns:
        QwenResult with extracted text and metadata.
    """
    prompt_type = _ROI_PROMPT_MAP.get(roi_id, "generic")
    user_prompt = prompt_override if prompt_override else _PROMPTS[prompt_type]

    try:
        image_b64 = _encode_image_b64(crop)
    except Exception as exc:  # noqa: BLE001
        return QwenResult(
            text="",
            readable=False,
            engine_available=False,
            error=f"Image encoding failed: {exc}",
        )

    try:
        if _INFERENCE_MODE == "hf":
            raw = _run_hf_inference(user_prompt, image_b64)
        else:
            raw = _call_qwen(user_prompt, image_b64)
    except (urllib.error.URLError, OSError) as exc:
        return QwenResult(
            text="",
            readable=False,
            raw_response="",
            engine_available=False,
            error=f"Qwen server unreachable: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return QwenResult(
            text="",
            readable=False,
            raw_response="",
            engine_available=False,
            error=f"Qwen call failed: {exc}",
        )

    text, readable = _parse_qwen_response(raw)
    return QwenResult(
        text=text,
        readable=readable,
        raw_response=raw,
        engine_available=True,
    )


def get_prompt_for_roi(roi_id: str) -> str:
    """Return the user prompt that would be used for this ROI."""
    prompt_type = _ROI_PROMPT_MAP.get(roi_id, "generic")
    return _PROMPTS[prompt_type]
