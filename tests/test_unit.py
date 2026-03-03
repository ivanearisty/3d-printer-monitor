"""Unit tests for core logic (no hardware or services required)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stream_analyzer.bambu_controller import BambuController, PrinterState
from stream_analyzer.image_analyzer import AnalysisResult, Detection, ImageAnalyzer


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_empty_factory(self) -> None:
        """Test AnalysisResult.empty() returns correct defaults."""
        result = AnalysisResult.empty()
        assert result.detections == []
        assert result.max_confidence == 0.0
        assert result.has_failure is False


class TestImageAnalyzer:
    """Unit tests for ImageAnalyzer (mocked HTTP)."""

    def test_url_trailing_slash_stripped(self) -> None:
        """Test that trailing slashes are normalized to avoid double-slash URLs."""
        analyzer = ImageAnalyzer(ml_api_url="http://example.com/")
        assert analyzer.ml_api_url == "http://example.com"

    def test_health_check_connection_error(self) -> None:
        """Test health check returns False when API is unreachable."""
        import requests

        analyzer = ImageAnalyzer()
        with patch(
            "stream_analyzer.image_analyzer.requests.get",
            side_effect=requests.exceptions.ConnectionError,
        ):
            assert analyzer.health_check() is False

    def test_analyze_parses_detections(self) -> None:
        """Test analyze correctly parses ML API response into Detection objects."""
        from PIL import Image

        analyzer = ImageAnalyzer()
        img = Image.new("RGB", (100, 100))

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "detections": [
                {"label": "failure", "confidence": 0.75, "box": [10, 20, 30, 40]},
                {"label": "failure", "confidence": 0.50, "box": [50, 60, 70, 80]},
            ],
        }

        with patch("stream_analyzer.image_analyzer.requests.post", return_value=mock_resp):
            result = analyzer.analyze(img)

        assert result.has_failure is True
        assert len(result.detections) == 2
        assert result.max_confidence == 0.75
        assert result.detections[0].box == (10, 20, 30, 40)

    def test_analyze_empty_detections(self) -> None:
        """Test analyze returns no-failure result when API finds nothing."""
        from PIL import Image

        analyzer = ImageAnalyzer()
        img = Image.new("RGB", (100, 100))

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"detections": []}

        with patch("stream_analyzer.image_analyzer.requests.post", return_value=mock_resp):
            result = analyzer.analyze(img)

        assert result.has_failure is False
        assert result.max_confidence == 0.0

    def test_analyze_connection_error_returns_empty(self) -> None:
        """Test analyze degrades gracefully when API is unreachable."""
        import requests
        from PIL import Image

        analyzer = ImageAnalyzer()
        img = Image.new("RGB", (100, 100))
        with patch(
            "stream_analyzer.image_analyzer.requests.post",
            side_effect=requests.exceptions.ConnectionError,
        ):
            result = analyzer.analyze(img)

        assert result.has_failure is False
        assert result.detections == []

    def test_analyze_api_error_returns_empty(self) -> None:
        """Test analyze degrades gracefully on HTTP 500."""
        from PIL import Image

        analyzer = ImageAnalyzer()
        img = Image.new("RGB", (100, 100))

        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500

        with patch("stream_analyzer.image_analyzer.requests.post", return_value=mock_resp):
            result = analyzer.analyze(img)

        assert result.has_failure is False


class TestBambuControllerUnit:
    """Unit tests for BambuController (no real printer)."""

    def test_get_state_returns_unknown_when_disconnected(self) -> None:
        """Test get_state guard returns UNKNOWN when _printer is None."""
        ctrl = BambuController("192.168.1.1", "code", "serial")
        assert ctrl.get_state() == PrinterState.UNKNOWN

    def test_stop_print_returns_false_when_disconnected(self) -> None:
        """Test stop_print guard returns False when _printer is None."""
        ctrl = BambuController("192.168.1.1", "code", "serial")
        assert ctrl.stop_print() is False

    def test_disconnect_safe_when_not_connected(self) -> None:
        """Test disconnect does not raise when called before connect."""
        ctrl = BambuController("192.168.1.1", "code", "serial")
        ctrl.disconnect()  # should not raise


class TestFailureMonitor:
    """Unit tests for FailureMonitor two-tier detection logic."""

    @staticmethod
    def _make_monitor(
        immediate_threshold: float = 0.70,
        consecutive_threshold: float = 0.45,
        consecutive_failures_to_stop: int = 2,
    ) -> Any:
        """Create a FailureMonitor with mocked dependencies."""
        from main import FailureMonitor

        return FailureMonitor(
            controller=MagicMock(),
            camera=MagicMock(),
            analyzer=MagicMock(),
            immediate_threshold=immediate_threshold,
            consecutive_threshold=consecutive_threshold,
            consecutive_failures_to_stop=consecutive_failures_to_stop,
        )

    def test_tier1_immediate_stop_on_high_confidence(self) -> None:
        """Test that confidence >= immediate_threshold triggers immediate stop."""
        monitor = self._make_monitor()
        monitor.camera.connect.return_value = True
        monitor.camera.get_image.return_value = MagicMock()
        monitor.analyzer.analyze.return_value = AnalysisResult(
            detections=[Detection(confidence=0.85, label="failure", box=(0, 0, 0, 0))],
            max_confidence=0.85,
            has_failure=True,
        )

        assert monitor.check_for_failure() is True

    def test_below_all_thresholds_no_stop(self) -> None:
        """Test that confidence below consecutive_threshold does not stop or increment counter."""
        monitor = self._make_monitor()
        monitor.camera.connect.return_value = True
        monitor.camera.get_image.return_value = MagicMock()
        monitor.analyzer.analyze.return_value = AnalysisResult(
            detections=[Detection(confidence=0.30, label="failure", box=(0, 0, 0, 0))],
            max_confidence=0.30,
            has_failure=True,
        )

        assert monitor.check_for_failure() is False
        assert monitor.consecutive_failures == 0

    def test_tier2_consecutive_detections_trigger_stop(self) -> None:
        """Test that N consecutive detections above consecutive_threshold triggers stop."""
        monitor = self._make_monitor(consecutive_failures_to_stop=2)
        monitor.camera.connect.return_value = True
        monitor.camera.get_image.return_value = MagicMock()
        monitor.analyzer.analyze.return_value = AnalysisResult(
            detections=[Detection(confidence=0.50, label="failure", box=(0, 0, 0, 0))],
            max_confidence=0.50,
            has_failure=True,
        )

        # First detection — accumulates but doesn't stop
        assert monitor.check_for_failure() is False
        assert monitor.consecutive_failures == 1

        # Second consecutive — triggers stop
        assert monitor.check_for_failure() is True

    def test_consecutive_counter_resets_on_clean_frame(self) -> None:
        """Test that a below-threshold frame resets the consecutive failure counter."""
        monitor = self._make_monitor()
        monitor.camera.connect.return_value = True
        monitor.camera.get_image.return_value = MagicMock()

        # Accumulate one consecutive failure
        monitor.analyzer.analyze.return_value = AnalysisResult(
            detections=[Detection(confidence=0.50, label="failure", box=(0, 0, 0, 0))],
            max_confidence=0.50,
            has_failure=True,
        )
        monitor.check_for_failure()
        assert monitor.consecutive_failures == 1

        # Clean frame resets counter
        monitor.analyzer.analyze.return_value = AnalysisResult.empty()
        monitor.check_for_failure()
        assert monitor.consecutive_failures == 0

    def test_camera_connect_failure_returns_false(self) -> None:
        """Test that camera connection failure returns False without calling disconnect."""
        monitor = self._make_monitor()
        monitor.camera.connect.return_value = False

        assert monitor.check_for_failure() is False
        monitor.camera.disconnect.assert_not_called()

    def test_no_image_returns_false(self) -> None:
        """Test that failing to capture an image returns False and still disconnects."""
        monitor = self._make_monitor()
        monitor.camera.connect.return_value = True
        monitor.camera.get_image.return_value = None

        assert monitor.check_for_failure() is False
        monitor.camera.disconnect.assert_called_once()

    def test_camera_disconnected_even_on_exception(self) -> None:
        """Test finally block: camera.disconnect is called even when analyze raises."""
        monitor = self._make_monitor()
        monitor.camera.connect.return_value = True
        monitor.camera.get_image.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            monitor.check_for_failure()

        monitor.camera.disconnect.assert_called_once()
