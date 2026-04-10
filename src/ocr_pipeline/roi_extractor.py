"""
Phase 1 — ROI extraction.

Loads the JSON config, crops each of the 17 field regions from an aligned image,
and returns a dict of {roi_id: ROICrop}.  Adaptive padding is applied to
stamp-affected ROIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ROICrop:
    roi_id: str
    name: str
    label: str
    colour: np.ndarray       # original colour crop
    binarized: Optional[np.ndarray] = None  # pre-processed B/W crop
    blue_channel: Optional[np.ndarray] = None  # blue channel crop
    bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) in source image
    ocr_engine: str = "tesseract"
    tesseract_config: str = "--psm 7"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ROI Extractor class
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "roi_config.json"


class ROIExtractor:
    def __init__(self, config_path: str | Path = CONFIG_PATH):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._cfg = json.load(f)

        self.ref_w: int = self._cfg["reference_width"]
        self.ref_h: int = self._cfg["reference_height"]
        self.stamp_rois: set[str] = set(self._cfg.get("stamp_affected_rois", []))
        self.rois: dict = self._cfg["rois"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_bbox(self, roi_def: dict, img_h: int, img_w: int) -> tuple[int, int, int, int]:
        """Convert relative ROI definition to pixel bbox (x, y, w, h)."""
        padding: float = roi_def.get("padding", 0.05)

        x = roi_def["x"] * img_w
        y = roi_def["y"] * img_h
        w = roi_def["w"] * img_w
        h = roi_def["h"] * img_h

        # Apply padding
        pad_x = w * padding
        pad_y = h * padding

        x = max(0, int(x - pad_x))
        y = max(0, int(y - pad_y))
        w = min(img_w - x, int(w + 2 * pad_x))
        h = min(img_h - y, int(h + 2 * pad_y))

        return (x, y, w, h)

    def _make_binarized(self, colour_crop: np.ndarray) -> np.ndarray:
        """Adaptive binarization for digit-mode OCR."""
        grey = cv2.cvtColor(colour_crop, cv2.COLOR_BGR2GRAY)
        # Gentle noise reduction
        blurred = cv2.GaussianBlur(grey, (3, 3), 0.5)

        # For small crops, reduce block size proportionally
        h = blurred.shape[0]
        block_size = max(11, (h // 4) | 1)  # must be odd
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, 10
        )
        return binary

    def _extract_blue(self, colour_crop: np.ndarray) -> np.ndarray:
        """Return blue channel (isolates handwritten ink)."""
        return colour_crop[:, :, 0]  # BGR: index 0 = Blue

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract_all(
        self,
        aligned_img: np.ndarray,
        tier_filter: Optional[set[int]] = None,
    ) -> dict[str, ROICrop]:
        """Extract all ROI crops from an aligned image.

        Args:
            aligned_img: colour image after alignment
            tier_filter: if given, only extract ROIs of these priority tiers
                         e.g. {1} for Tier-1 (MVP) fields only

        Returns:
            dict mapping roi_id (e.g. "R01") to ROICrop
        """
        img_h, img_w = aligned_img.shape[:2]
        crops: dict[str, ROICrop] = {}

        for roi_id, roi_def in self.rois.items():
            tier = roi_def.get("tier", 1)
            if tier_filter is not None and tier not in tier_filter:
                continue

            bbox = self._compute_bbox(roi_def, img_h, img_w)
            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                crops[roi_id] = ROICrop(
                    roi_id=roi_id,
                    name=roi_def["name"],
                    label=roi_def["label"],
                    colour=np.zeros((10, 10, 3), dtype=np.uint8),
                    bbox_px=bbox,
                    ocr_engine=roi_def.get("ocr_engine", "tesseract"),
                    tesseract_config=roi_def.get("tesseract_config", "--psm 7"),
                    warnings=["Empty crop bbox — ROI coordinates may be miscalibrated."],
                )
                continue

            colour_crop = aligned_img[y:y + h, x:x + w].copy()
            binarized = self._make_binarized(colour_crop)
            blue_ch = self._extract_blue(colour_crop)

            crops[roi_id] = ROICrop(
                roi_id=roi_id,
                name=roi_def["name"],
                label=roi_def["label"],
                colour=colour_crop,
                binarized=binarized,
                blue_channel=blue_ch,
                bbox_px=bbox,
                ocr_engine=roi_def.get("ocr_engine", "tesseract"),
                tesseract_config=roi_def.get("tesseract_config", "--psm 7"),
            )

        return crops

    def extract_tier1(self, aligned_img: np.ndarray) -> dict[str, ROICrop]:
        """Shortcut: extract only Tier-1 (MVP) fields."""
        return self.extract_all(aligned_img, tier_filter={1})


# ---------------------------------------------------------------------------
# Standalone save helper (for verification / debugging)
# ---------------------------------------------------------------------------

def save_crops(
    crops: dict[str, ROICrop],
    output_dir: str | Path,
    prefix: str = "",
) -> None:
    """Save all ROI crops as image files for visual inspection."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for roi_id, crop in crops.items():
        base = f"{prefix}{roi_id}_{crop.name}"
        cv2.imwrite(str(out / f"{base}_colour.jpg"), crop.colour)
        if crop.binarized is not None:
            cv2.imwrite(str(out / f"{base}_binary.jpg"), crop.binarized)


def draw_roi_overlay(
    img: np.ndarray,
    crops: dict[str, ROICrop],
    extractor: ROIExtractor,
) -> np.ndarray:
    """Draw all ROI bounding boxes on the image with labels.

    Returns a copy with coloured overlays for visual verification.
    """
    overlay = img.copy()
    tier_colours = {
        1: (0, 200, 0),    # green
        2: (255, 140, 0),  # orange
        3: (200, 0, 200),  # purple
    }

    for roi_id, crop in crops.items():
        roi_def = extractor.rois.get(roi_id, {})
        tier = roi_def.get("tier", 1)
        colour = tier_colours.get(tier, (128, 128, 128))

        x, y, w, h = crop.bbox_px
        cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            overlay,
            f"{roi_id}:{crop.name[:10]}",
            (x + 2, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, colour, 1, cv2.LINE_AA,
        )

    return overlay
