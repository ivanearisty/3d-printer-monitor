#!/usr/bin/env python
"""
Image Analyzer for 3D Print Failure Detection

Sends images to the ML API and returns detection results.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single failure detection result."""
    confidence: float
    label: str
    box: tuple[float, float, float, float]  # x, y, width, height


@dataclass
class AnalysisResult:
    """Result of analyzing an image."""
    detections: list[Detection]
    max_confidence: float
    has_failure: bool

    @classmethod
    def empty(cls) -> "AnalysisResult":
        """Create an empty result (no detections)."""
        return cls(detections=[], max_confidence=0.0, has_failure=False)


class ImageAnalyzer:
    """Analyzes images for 3D print failures using the ML API."""

    def __init__(
        self,
        ml_api_url: str = "http://localhost:3333",
        timeout: int = 30,
    ):
        self.ml_api_url = ml_api_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        """Check if ML API is available."""
        try:
            response = requests.get(f"{self.ml_api_url}/hc/", timeout=5)
            return response.ok
        except Exception:
            return False

    def analyze(self, image: "Image.Image") -> AnalysisResult:
        """
        Analyze an image for print failures.

        Args:
            image: PIL Image to analyze

        Returns:
            AnalysisResult with detections and max confidence
        """
        try:
            # Convert PIL Image to base64 JPEG
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Send to ML API
            response = requests.post(
                f"{self.ml_api_url}/p/",
                json={"image": img_base64},
                timeout=self.timeout,
            )

            if not response.ok:
                logger.warning(f"ML API error: {response.status_code}")
                return AnalysisResult.empty()

            # Parse response
            data = response.json()
            raw_detections = data.get("detections", [])

            detections = [
                Detection(
                    confidence=d.get("confidence", 0),
                    label=d.get("label", "failure"),
                    box=tuple(d.get("box", [0, 0, 0, 0])),
                )
                for d in raw_detections
            ]

            max_confidence = max((d.confidence for d in detections), default=0.0)

            return AnalysisResult(
                detections=detections,
                max_confidence=max_confidence,
                has_failure=len(detections) > 0,
            )

        except requests.exceptions.ConnectionError:
            logger.warning("ML API not available")
            return AnalysisResult.empty()
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return AnalysisResult.empty()
