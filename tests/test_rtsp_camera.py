"""Tests for RTSP camera connection and frame capture."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from stream_analyzer.rtsp_camera import RTSPCamera


class TestRTSPCamera:
    """Tests for connecting to the RTSP camera and grabbing frames."""

    @pytest.fixture(scope="session")
    def camera(self) -> Generator[RTSPCamera, None, None]:
        """Create and connect an RTSP camera for tests."""
        host = os.getenv("CAMERA_HOST")
        if not host:
            pytest.skip("CAMERA_HOST not configured")
        port = int(os.getenv("CAMERA_PORT", "554"))
        username = os.getenv("CAMERA_USERNAME", "")
        password = os.getenv("CAMERA_PASSWORD", "")
        stream_path = os.getenv("CAMERA_STREAM_PATH", "stream1")
        cam = RTSPCamera(host, port, username, password, stream_path)
        assert cam.connect(), "Camera should connect successfully"
        yield cam
        cam.disconnect()

    def test_camera_connected(self, camera: RTSPCamera) -> None:
        """Test that camera is connected and ready."""
        assert camera.is_connected(), "Camera should be connected"

    def test_camera_grab_frame(self, camera: RTSPCamera) -> None:
        """Test that we can grab a frame from the camera."""
        frame = camera.get_frame()
        assert frame is not None, "Should be able to grab a frame"
        assert frame.shape[0] > 0, "Frame height should be positive"
        assert frame.shape[1] > 0, "Frame width should be positive"

    def test_camera_grab_image(self, camera: RTSPCamera) -> None:
        """Test that we can get a PIL Image from the camera."""
        image = camera.get_image()
        assert image is not None, "Should be able to get a PIL Image"
        assert hasattr(image, "size"), "Image should have a size attribute"
