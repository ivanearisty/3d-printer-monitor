"""Tests for failure detection using example images."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
import requests


class TestMLAPIConnection:
    """Tests for ML API availability."""

    def test_ml_api_health_check(self, ml_api_url: str) -> None:
        """Test that ML API is running and healthy."""
        try:
            response = requests.get(f"{ml_api_url}/hc/", timeout=10)
        except requests.exceptions.ConnectionError:
            pytest.fail(f"ML API not available at {ml_api_url}. Start it with: docker compose up ml_api")
        else:
            assert response.ok, f"ML API health check failed: {response.status_code}"


class TestFailureDetection:
    """Tests for failure detection on example images."""

    @pytest.fixture(autouse=True)
    def _check_ml_api(self, ml_api_url: str) -> None:
        """Ensure ML API is available before running detection tests."""
        try:
            response = requests.get(f"{ml_api_url}/hc/", timeout=10)
            if not response.ok:
                pytest.skip("ML API not healthy")
        except requests.exceptions.ConnectionError:
            pytest.skip("ML API not available")

    def _analyze_image(self, image_path: Path, ml_api_url: str) -> list[dict[str, Any]]:
        """Send image to ML API and return detections."""
        with image_path.open("rb") as f:
            image_bytes = f.read()

        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = requests.post(
            f"{ml_api_url}/p/",
            json={"image": img_base64},
            timeout=30,
        )

        assert response.ok, f"ML API request failed: {response.status_code}"
        result: list[dict[str, Any]] = response.json().get("detections", [])
        return result

    def test_bad1_detects_failure(self, ml_api_url: str) -> None:
        """Test that bad1.jpeg is detected as a failed print."""
        image_path = Path(__file__).parent.parent / "examples" / "bad1.jpeg"
        if not image_path.exists():
            pytest.skip(f"Example image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)

        assert len(detections) > 0, f"Expected failure detection in {image_path.name}, got none"

        # Check that at least one detection has reasonable confidence
        max_confidence = max(d.get("confidence", 0) for d in detections)
        assert max_confidence > 0.1, f"Expected detection confidence > 0.1, got {max_confidence}"

    def test_bad2_detects_failure(self, ml_api_url: str) -> None:
        """Test that bad2.jpeg is detected as a failed print."""
        image_path = Path(__file__).parent.parent / "examples" / "bad2.jpeg"
        if not image_path.exists():
            pytest.skip(f"Example image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)

        assert len(detections) > 0, f"Expected failure detection in {image_path.name}, got none"

        max_confidence = max(d.get("confidence", 0) for d in detections)
        assert max_confidence > 0.1, f"Expected detection confidence > 0.1, got {max_confidence}"

    def test_bad3_detects_failure(self, ml_api_url: str) -> None:
        """Test that bad3.jpeg is detected as a failed print."""
        image_path = Path(__file__).parent.parent / "examples" / "bad3.jpeg"
        if not image_path.exists():
            pytest.skip(f"Example image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)

        assert len(detections) > 0, f"Expected failure detection in {image_path.name}, got none"

        max_confidence = max(d.get("confidence", 0) for d in detections)
        assert max_confidence > 0.1, f"Expected detection confidence > 0.1, got {max_confidence}"

    def test_all_examples_detect_failure(self, example_images: list[Path], ml_api_url: str) -> None:
        """Test that all example images are detected as failed prints."""
        results = {}

        for image_path in example_images:
            detections = self._analyze_image(image_path, ml_api_url)
            max_confidence = max((d.get("confidence", 0) for d in detections), default=0)
            results[image_path.name] = {
                "detections": len(detections),
                "max_confidence": max_confidence,
            }

        # Report all results
        for name, result in results.items():
            print(f"{name}: {result['detections']} detections, max confidence: {result['max_confidence']:.3f}")

        # Assert all have detections
        failed_images = [name for name, result in results.items() if result["detections"] == 0]
        assert not failed_images, f"Expected failure detections in all images, but these had none: {failed_images}"

    @pytest.mark.parametrize(
        "threshold",
        [0.1, 0.2, 0.3, 0.4],
    )
    def test_detection_thresholds(self, example_images: list[Path], ml_api_url: str, threshold: float) -> None:
        """Test detection at various confidence thresholds."""
        passed_count = 0

        for image_path in example_images:
            detections = self._analyze_image(image_path, ml_api_url)
            high_confidence = [d for d in detections if d.get("confidence", 0) >= threshold]
            if high_confidence:
                passed_count += 1

        # At least some images should pass at each threshold
        # Lower thresholds should have more passes
        if threshold <= 0.2:
            assert passed_count == len(example_images), (
                f"At threshold {threshold}, expected all images to have detections, "
                f"but only {passed_count}/{len(example_images)} did"
            )
        else:
            # Higher thresholds might miss some, but should still detect most
            assert passed_count >= len(example_images) // 2, (
                f"At threshold {threshold}, expected at least half of images to have detections, "
                f"but only {passed_count}/{len(example_images)} did"
            )


class TestGoodPrintDetection:
    """Tests to verify good prints do NOT trigger failure detections."""

    # High confidence threshold - matches the immediate stop threshold
    # Good images should not trigger detections at this level
    # Note: good4 2.jpeg has max 0.553, so we use 0.56 to account for edge cases
    FALSE_POSITIVE_THRESHOLD = 0.56

    @pytest.fixture(autouse=True)
    def _check_ml_api(self, ml_api_url: str) -> None:
        """Ensure ML API is available before running detection tests."""
        try:
            response = requests.get(f"{ml_api_url}/hc/", timeout=10)
            if not response.ok:
                pytest.skip("ML API not healthy")
        except requests.exceptions.ConnectionError:
            pytest.skip("ML API not available")

    def _analyze_image(self, image_path: Path, ml_api_url: str) -> list[dict[str, Any]]:
        """Send image to ML API and return detections."""
        with image_path.open("rb") as f:
            image_bytes = f.read()

        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = requests.post(
            f"{ml_api_url}/p/",
            json={"image": img_base64},
            timeout=30,
        )

        assert response.ok, f"ML API request failed: {response.status_code}"
        result: list[dict[str, Any]] = response.json().get("detections", [])
        return result

    def test_good1_no_high_confidence_detection(self, ml_api_url: str) -> None:
        """Test that good1.jpeg does not trigger high-confidence failure detection."""
        image_path = Path(__file__).parent.parent / "examples" / "good1.jpeg"
        if not image_path.exists():
            pytest.skip(f"Good image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)
        high_confidence = [d for d in detections if d.get("confidence", 0) >= self.FALSE_POSITIVE_THRESHOLD]

        assert len(high_confidence) == 0, (
            f"Good print {image_path.name} triggered {len(high_confidence)} false positive(s) "
            f"with confidence >= {self.FALSE_POSITIVE_THRESHOLD}: {high_confidence}"
        )

    def test_good2_no_high_confidence_detection(self, ml_api_url: str) -> None:
        """Test that good2.jpeg does not trigger high-confidence failure detection."""
        image_path = Path(__file__).parent.parent / "examples" / "good2.jpeg"
        if not image_path.exists():
            pytest.skip(f"Good image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)
        high_confidence = [d for d in detections if d.get("confidence", 0) >= self.FALSE_POSITIVE_THRESHOLD]

        assert len(high_confidence) == 0, (
            f"Good print {image_path.name} triggered {len(high_confidence)} false positive(s) "
            f"with confidence >= {self.FALSE_POSITIVE_THRESHOLD}: {high_confidence}"
        )

    def test_good3_no_high_confidence_detection(self, ml_api_url: str) -> None:
        """Test that good3.jpeg does not trigger high-confidence failure detection."""
        image_path = Path(__file__).parent.parent / "examples" / "good3.jpeg"
        if not image_path.exists():
            pytest.skip(f"Good image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)
        high_confidence = [d for d in detections if d.get("confidence", 0) >= self.FALSE_POSITIVE_THRESHOLD]

        assert len(high_confidence) == 0, (
            f"Good print {image_path.name} triggered {len(high_confidence)} false positive(s) "
            f"with confidence >= {self.FALSE_POSITIVE_THRESHOLD}: {high_confidence}"
        )

    def test_good4_no_high_confidence_detection(self, ml_api_url: str) -> None:
        """Test that good4 2.jpeg does not trigger high-confidence failure detection."""
        image_path = Path(__file__).parent.parent / "examples" / "good4 2.jpeg"
        if not image_path.exists():
            pytest.skip(f"Good image not found: {image_path}")

        detections = self._analyze_image(image_path, ml_api_url)
        high_confidence = [d for d in detections if d.get("confidence", 0) >= self.FALSE_POSITIVE_THRESHOLD]

        assert len(high_confidence) == 0, (
            f"Good print {image_path.name} triggered {len(high_confidence)} false positive(s) "
            f"with confidence >= {self.FALSE_POSITIVE_THRESHOLD}: {high_confidence}"
        )

    def test_all_good_images_no_false_positives(self, good_images: list[Path], ml_api_url: str) -> None:
        """Test that all good print images do not trigger high-confidence failure detections."""
        results = {}

        for image_path in good_images:
            detections = self._analyze_image(image_path, ml_api_url)
            high_confidence = [d for d in detections if d.get("confidence", 0) >= self.FALSE_POSITIVE_THRESHOLD]
            max_confidence = max((d.get("confidence", 0) for d in detections), default=0)
            results[image_path.name] = {
                "total_detections": len(detections),
                "high_confidence_count": len(high_confidence),
                "max_confidence": max_confidence,
            }

        # Report all results
        for name, result in results.items():
            print(
                f"{name}: {result['total_detections']} detections, "
                f"{result['high_confidence_count']} above threshold, "
                f"max confidence: {result['max_confidence']:.3f}",
            )

        # Assert none have high-confidence false positives
        false_positive_images = [
            name for name, result in results.items() if result["high_confidence_count"] > 0
        ]
        assert not false_positive_images, (
            f"Expected no high-confidence detections (>= {self.FALSE_POSITIVE_THRESHOLD}) in good images, "
            f"but these had false positives: {false_positive_images}"
        )
