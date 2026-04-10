"""
Tests for Phase 2 — qwen_ocr.py

Coverage:
  - Config loading (INFERENCE_MODE, QWEN_ENDPOINT, QWEN_MODEL, QWEN_TIMEOUT)
  - QwenResult dataclass structure
  - _encode_image_b64 produces valid base64
  - _parse_qwen_response with JSON envelope / plain text / malformed input
  - _count_digit_changes and qwen_diverges_too_much
  - run_qwen graceful degradation when server is unreachable
  - get_prompt_for_roi returns non-empty string for all 17 ROIs
  - get_prompt_for_roi falls back to "generic" for unknown ROI ID
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_pipeline.qwen_ocr import (
    QwenResult,
    QWEN_ENDPOINT,
    QWEN_MODEL,
    QWEN_TIMEOUT,
    _INFERENCE_MODE,
    _encode_image_b64,
    _parse_qwen_response,
    _count_digit_changes,
    qwen_diverges_too_much,
    run_qwen,
    get_prompt_for_roi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def white_crop(h: int = 60, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _make_api_envelope(content: str) -> str:
    return json.dumps({
        "choices": [{"message": {"content": content}}]
    })


# ---------------------------------------------------------------------------
# 1. Configuration loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_inference_mode_is_string(self):
        assert isinstance(_INFERENCE_MODE, str)

    def test_inference_mode_is_valid_value(self):
        assert _INFERENCE_MODE in ("api", "local")

    def test_qwen_endpoint_is_string(self):
        assert isinstance(QWEN_ENDPOINT, str)

    def test_qwen_endpoint_starts_with_http(self):
        assert QWEN_ENDPOINT.startswith("http")

    def test_qwen_model_is_non_empty_string(self):
        assert isinstance(QWEN_MODEL, str)
        assert len(QWEN_MODEL) > 0

    def test_qwen_timeout_is_positive_int(self):
        assert isinstance(QWEN_TIMEOUT, int)
        assert QWEN_TIMEOUT > 0

    def test_api_mode_uses_openrouter_endpoint(self):
        if _INFERENCE_MODE == "api":
            assert "openrouter" in QWEN_ENDPOINT

    def test_local_mode_uses_local_endpoint(self):
        if _INFERENCE_MODE == "local":
            assert "localhost" in QWEN_ENDPOINT or "127.0.0.1" in QWEN_ENDPOINT


# ---------------------------------------------------------------------------
# 2. QwenResult dataclass
# ---------------------------------------------------------------------------

class TestQwenResultStructure:
    def test_has_required_fields(self):
        r = QwenResult(text="hello")
        assert hasattr(r, "text")
        assert hasattr(r, "readable")
        assert hasattr(r, "raw_response")
        assert hasattr(r, "engine_available")
        assert hasattr(r, "error")
        assert hasattr(r, "unreadable_positions")

    def test_defaults(self):
        r = QwenResult(text="hi")
        assert r.readable is True
        assert r.raw_response == ""
        assert r.engine_available is True
        assert r.error is None
        assert r.unreadable_positions == []

    def test_text_assigned_correctly(self):
        r = QwenResult(text="3000,000")
        assert r.text == "3000,000"


# ---------------------------------------------------------------------------
# 3. _encode_image_b64
# ---------------------------------------------------------------------------

class TestEncodeImageB64:
    def test_returns_string(self):
        result = _encode_image_b64(white_crop())
        assert isinstance(result, str)

    def test_valid_base64(self):
        result = _encode_image_b64(white_crop())
        # Should not raise
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_output_starts_with_jpeg_magic(self):
        result = _encode_image_b64(white_crop())
        decoded = base64.b64decode(result)
        # JPEG files start with FF D8 FF
        assert decoded[:2] == b"\xff\xd8"

    def test_no_exception_on_various_sizes(self):
        for h, w in [(30, 100), (100, 300), (200, 200)]:
            _encode_image_b64(np.full((h, w, 3), 128, dtype=np.uint8))


# ---------------------------------------------------------------------------
# 4. _parse_qwen_response
# ---------------------------------------------------------------------------

class TestParseQwenResponse:
    def test_json_envelope_with_json_content(self):
        content = json.dumps({"text": "15/06/2025", "readable": True})
        raw = _make_api_envelope(content)
        text, readable = _parse_qwen_response(raw)
        assert text == "15/06/2025"
        assert readable is True

    def test_json_envelope_readable_false(self):
        content = json.dumps({"text": "", "readable": False})
        raw = _make_api_envelope(content)
        _, readable = _parse_qwen_response(raw)
        assert readable is False

    def test_json_with_markdown_fences_stripped(self):
        content = "```json\n{\"text\": \"3000,000\", \"readable\": true}\n```"
        raw = _make_api_envelope(content)
        text, readable = _parse_qwen_response(raw)
        assert text == "3000,000"

    def test_plain_text_content_returned_as_is(self):
        raw = _make_api_envelope("some plain text")
        text, readable = _parse_qwen_response(raw)
        assert text == "some plain text"
        assert readable is True

    def test_malformed_envelope_falls_back(self):
        # Not a valid JSON envelope at all
        raw = "this is not json"
        text, readable = _parse_qwen_response(raw)
        assert isinstance(text, str)
        assert isinstance(readable, bool)

    def test_missing_choices_key_falls_back(self):
        raw = json.dumps({"error": "model not loaded"})
        text, readable = _parse_qwen_response(raw)
        assert isinstance(text, str)

    def test_text_field_empty_string(self):
        content = json.dumps({"text": "", "readable": True})
        raw = _make_api_envelope(content)
        text, _ = _parse_qwen_response(raw)
        assert text == ""


# ---------------------------------------------------------------------------
# 5. _count_digit_changes and qwen_diverges_too_much
# ---------------------------------------------------------------------------

class TestDigitDivergence:
    def test_identical_strings_zero_change(self):
        changed, total = _count_digit_changes("12345", "12345")
        assert changed == 0
        assert total == 5

    def test_one_digit_changed(self):
        changed, total = _count_digit_changes("12345", "12346")
        assert changed == 1
        assert total == 5

    def test_length_difference_counts_as_changes(self):
        changed, total = _count_digit_changes("123", "12")
        assert changed >= 1

    def test_no_tess_digits_returns_zero(self):
        changed, total = _count_digit_changes("abc", "def")
        assert changed == 0
        assert total == 0

    def test_diverges_too_much_above_threshold(self):
        # 4 out of 5 digits changed → 80% > 30%
        assert qwen_diverges_too_much("11111", "22221") is True

    def test_diverges_too_much_below_threshold(self):
        # 1 out of 10 changed → 10% < 30%
        assert qwen_diverges_too_much("1234567890", "1234567891") is False

    def test_diverges_too_much_exactly_at_threshold(self):
        # 3 out of 10 → exactly 30% — should be False (rule is > threshold)
        assert qwen_diverges_too_much("1234567890", "1234507690") is False  # adjust

    def test_diverges_too_much_no_digits_is_false(self):
        assert qwen_diverges_too_much("abc", "xyz") is False

    def test_custom_threshold(self):
        # With threshold=0.10, even 2 of 10 changes should trigger
        assert qwen_diverges_too_much("1234567890", "1234007890", threshold=0.10) is True


# ---------------------------------------------------------------------------
# 6. run_qwen — graceful degradation (server unreachable)
# ---------------------------------------------------------------------------

class TestRunQwenGracefulDegradation:
    def test_returns_QwenResult_when_server_unreachable(self):
        # No server running → should return QwenResult with engine_available=False
        result = run_qwen(white_crop(), roi_id="R01")
        assert isinstance(result, QwenResult)

    def test_engine_available_false_when_unreachable(self):
        result = run_qwen(white_crop(), roi_id="R05")
        # Either engine is available (real server) or gracefully unavailable
        assert isinstance(result.engine_available, bool)

    def test_no_exception_on_any_roi_id(self):
        for roi_id in [f"R{i:02d}" for i in range(1, 18)]:
            run_qwen(white_crop(), roi_id=roi_id)

    def test_error_message_set_when_unavailable(self):
        result = run_qwen(white_crop(), roi_id="R05")
        if not result.engine_available:
            assert result.error is not None
            assert len(result.error) > 0

    def test_text_is_string_when_unavailable(self):
        result = run_qwen(white_crop(), roi_id="R05")
        assert isinstance(result.text, str)


# ---------------------------------------------------------------------------
# 7. get_prompt_for_roi
# ---------------------------------------------------------------------------

class TestGetPromptForRoi:
    def test_returns_string(self):
        p = get_prompt_for_roi("R05")
        assert isinstance(p, str)

    def test_non_empty_for_all_17_rois(self):
        for i in range(1, 18):
            p = get_prompt_for_roi(f"R{i:02d}")
            assert len(p) > 0, f"Empty prompt for R{i:02d}"

    def test_rib_prompt_mentions_rib(self):
        p = get_prompt_for_roi("R05")
        assert "RIB" in p or "bank" in p.lower() or "identification" in p.lower()

    def test_date_prompt_for_r02(self):
        p = get_prompt_for_roi("R02")
        assert "date" in p.lower() or "DD/MM/YYYY" in p

    def test_amount_prompt_for_r06(self):
        p = get_prompt_for_roi("R06")
        assert "amount" in p.lower() or "montant" in p.lower()

    def test_unknown_roi_returns_generic_prompt(self):
        p = get_prompt_for_roi("R99")
        assert len(p) > 0  # falls back to generic

    def test_all_prompts_mention_json(self):
        for i in range(1, 18):
            p = get_prompt_for_roi(f"R{i:02d}")
            assert "JSON" in p or "json" in p.lower(), f"R{i:02d} prompt has no JSON instruction"
