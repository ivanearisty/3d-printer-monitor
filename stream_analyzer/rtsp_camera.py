#!/usr/bin/env python
"""
RTSP Camera Module

Handles RTSP camera stream capture for 3D print monitoring.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class RTSPCamera:
    """Captures frames from an RTSP camera stream."""

    def __init__(
        self,
        host: str,
        port: int = 554,
        username: str = "",
        password: str = "",
        stream_path: str = "stream1",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.stream_path = stream_path

        # Build RTSP URL
        if username and password:
            self.rtsp_url = f"rtsp://{username}:{password}@{host}:{port}/{stream_path}"
        else:
            self.rtsp_url = f"rtsp://{host}:{port}/{stream_path}"

        self._cap: cv2.VideoCapture | None = None
        self._connected = False

        # Frame buffer for low-latency access
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=5)
        self._frame_lock = threading.Lock()
        self._grabber_thread: threading.Thread | None = None
        self._grabber_running = False

    def connect(self, timeout: int = 10) -> bool:
        """Connect to the RTSP stream."""
        logger.info(f"Connecting to RTSP camera at {self.host}:{self.port}...")

        # Force FFmpeg (used by OpenCV) to use TCP for RTSP to avoid UDP packet loss.
        # TCP guarantees packet delivery which prevents "bad cseq" and decoding errors
        # when the network is lossy (e.g., unstable WiFi). This may add a small latency.
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self._cap is None:
            logger.error("Failed to create VideoCapture")
            return False

        # Set buffer size to minimum for low latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.error("Failed to connect to RTSP stream")
            return False

        logger.info("✅ Connected to RTSP camera")

        # Start background frame grabber
        self._start_grabber()

        # Wait for first frame
        for _ in range(timeout * 10):
            with self._frame_lock:
                if self._frame_buffer:
                    self._connected = True
                    logger.info("📷 Camera ready")
                    return True
            time.sleep(0.1)

        logger.warning("Timeout waiting for first frame")
        return False

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        self._stop_grabber()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._connected = False
        with self._frame_lock:
            self._frame_buffer.clear()
        logger.info("📷 Camera disconnected")

    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._connected and self._cap is not None and self._cap.isOpened()

    def _start_grabber(self) -> None:
        """Start background frame grabber thread."""
        if self._grabber_thread is not None and self._grabber_thread.is_alive():
            return

        self._grabber_running = True
        self._grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self._grabber_thread.start()
        logger.debug("Started frame grabber thread")

    def _stop_grabber(self) -> None:
        """Stop background frame grabber thread."""
        self._grabber_running = False
        if self._grabber_thread is not None:
            self._grabber_thread.join(timeout=2.0)
            self._grabber_thread = None

    def _grabber_loop(self) -> None:
        """Continuously grab frames to keep buffer fresh."""
        while self._grabber_running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._frame_buffer.append(frame)
            else:
                time.sleep(0.01)

    def get_frame(self) -> np.ndarray | None:
        """Get the latest frame as numpy array (BGR)."""
        with self._frame_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1].copy()
        return None

    def get_image(self) -> Image.Image | None:
        """Get the latest frame as PIL Image (RGB)."""
        frame = self.get_frame()
        if frame is None:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def reconnect(self) -> bool:
        """Reconnect to the camera."""
        logger.info("Reconnecting to camera...")
        self.disconnect()
        time.sleep(1)
        return self.connect()
