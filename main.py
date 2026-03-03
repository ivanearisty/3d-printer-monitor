"""3D Print Failure Detection Monitor for Bambu Lab Printers.

Monitors a Bambu Lab printer using an external RTSP camera and
automatically stops prints when failures are detected.
"""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from stream_analyzer.bambu_controller import BambuController, PrinterState
from stream_analyzer.image_analyzer import ImageAnalyzer
from stream_analyzer.rtsp_camera import RTSPCamera

# Configure logging: console + rotating file handler
log_dir = os.getenv("LOG_DIR", "logs")
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = Path(log_dir) / "monitor.log"

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# Rotating file handler to avoid unbounded log growth
file_handler = RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


class FailureMonitor:
    """Monitor Bambu printer for print failures using AI detection."""

    # Two-tier detection thresholds
    IMMEDIATE_STOP_THRESHOLD = 0.70  # Immediate stop for obvious failures
    CONSECUTIVE_THRESHOLD = 0.45     # Lower threshold requiring consecutive detections

    def __init__(
        self,
        controller: BambuController,
        camera: RTSPCamera,
        analyzer: ImageAnalyzer,
        immediate_threshold: float = 0.70,
        consecutive_threshold: float = 0.45,
        check_interval_printing: float = 30.0,  # 30 seconds
        check_interval_idle: float = 60.0,
        consecutive_failures_to_stop: int = 2,
    ) -> None:
        """Initialize the failure monitor with printer components and thresholds."""
        self.controller: BambuController = controller
        self.camera: RTSPCamera = camera
        self.analyzer: ImageAnalyzer = analyzer
        self.immediate_threshold: float = immediate_threshold
        self.consecutive_threshold: float = consecutive_threshold
        self.check_interval_printing: float = check_interval_printing
        self.check_interval_idle: float = check_interval_idle
        self.consecutive_failures_to_stop: int = consecutive_failures_to_stop
        self.consecutive_failures = 0
        self.is_monitoring = False
        self.running = False

    def check_for_failure(self) -> bool:
        """Check current camera frame for failures.

        Returns True if print should be stopped.
        """
        # Connect briefly, grab a snapshot, then disconnect to minimize bandwidth.
        if not self.camera.connect():
            logger.warning("Camera not available")
            return False

        try:
            image = self.camera.get_image()
            if image is None:
                logger.warning("Failed to get camera frame")
                return False

            result = self.analyzer.analyze(image)
        finally:
            # Always disconnect to avoid keeping RTSP/FFmpeg running continuously.
            try:
                self.camera.disconnect()
            except OSError:
                logger.debug("Error while disconnecting camera", exc_info=True)

        # Log result
        if result.has_failure:
            logger.info("Detection: max_confidence=%s", f"{result.max_confidence:.3f}")
        else:
            logger.info("No failures detected")

        # Tier 1: Immediate stop for high confidence
        if result.max_confidence >= self.immediate_threshold:
            logger.warning(
                "HIGH CONFIDENCE FAILURE! %s >= %s - IMMEDIATE STOP",
                f"{result.max_confidence:.2f}",
                self.immediate_threshold,
            )
            return True

        # Tier 2: Consecutive detections at lower threshold
        if result.max_confidence >= self.consecutive_threshold:
            self.consecutive_failures += 1
            logger.warning(
                "Failure detected! Confidence: %s, Consecutive: %s/%s",
                f"{result.max_confidence:.2f}",
                self.consecutive_failures,
                self.consecutive_failures_to_stop,
            )
            if self.consecutive_failures >= self.consecutive_failures_to_stop:
                return True
        else:
            if self.consecutive_failures > 0:
                logger.info("No failure detected, resetting counter")
            self.consecutive_failures = 0

        return False

    def run(self) -> None:
        """Run the main monitoring loop."""
        self.running = True

        logger.info("=" * 60)
        logger.info("3D Print Failure Detection Monitor")
        logger.info("=" * 60)
        logger.info("Immediate Stop Threshold: %s", self.immediate_threshold)
        logger.info("Consecutive Threshold: %s", self.consecutive_threshold)
        logger.info("Check Interval (printing): %ss", self.check_interval_printing)
        logger.info("Check Interval (idle): %ss", self.check_interval_idle)
        logger.info("Consecutive Failures to Stop: %s", self.consecutive_failures_to_stop)
        logger.info("=" * 60)

        while self.running:
            try:
                state = self.controller.get_state()
                logger.info("Printer state: %s", state.value)

                if state == PrinterState.PRINTING:
                    # Start monitoring if not already
                    if not self.is_monitoring:
                        logger.info("Print started - beginning failure detection")
                        self.is_monitoring = True
                        self.consecutive_failures = 0

                    # Check for failures
                    should_stop = self.check_for_failure()
                    if should_stop:
                        self.controller.stop_print()
                        self.consecutive_failures = 0

                    time.sleep(self.check_interval_printing)

                elif state == PrinterState.PAUSED:
                    logger.info("Print paused")
                    time.sleep(self.check_interval_idle)

                else:
                    # Idle or unknown state
                    if self.is_monitoring:
                        logger.info("Print finished - stopping failure detection")
                        self.is_monitoring = False
                        self.consecutive_failures = 0

                    time.sleep(self.check_interval_idle)

            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Error in monitoring loop")
                time.sleep(self.check_interval_idle)

    def stop(self) -> None:
        """Stop the monitor."""
        self.running = False


def main() -> None:
    """Run the main entry point."""
    load_dotenv()

    # Required: Bambu printer credentials
    access_code = os.getenv("PRINTER_ACCESS_CODE")
    serial = os.getenv("PRINTER_SERIAL_NUMBER")
    ip_address = os.getenv("PRINTER_IP_ADDRESS")

    if not access_code:
        raise ValueError("PRINTER_ACCESS_CODE environment variable is required")
    if not serial:
        raise ValueError("PRINTER_SERIAL_NUMBER environment variable is required")
    if not ip_address:
        raise ValueError("PRINTER_IP_ADDRESS environment variable is required")

    # Required: RTSP camera settings
    camera_host = os.getenv("CAMERA_HOST")
    if not camera_host:
        raise ValueError("CAMERA_HOST environment variable is required")

    camera_username = os.getenv("CAMERA_USERNAME", "")
    camera_password = os.getenv("CAMERA_PASSWORD", "")
    camera_port = int(os.getenv("CAMERA_PORT", "554"))
    camera_stream_path = os.getenv("CAMERA_STREAM_PATH", "stream1")

    # Optional settings
    ml_api_url = os.getenv("ML_API_URL", "http://localhost:3333")
    check_interval_printing = float(os.getenv("CHECK_INTERVAL_PRINTING", "30"))
    check_interval_idle = float(os.getenv("CHECK_INTERVAL_IDLE", "60"))
    consecutive_to_stop = int(os.getenv("CONSECUTIVE_FAILURES_TO_STOP", "2"))

    # Create components
    controller = BambuController(
        ip_address=ip_address,
        access_code=access_code,
        serial_number=serial,
    )

    camera = RTSPCamera(
        host=camera_host,
        port=camera_port,
        username=camera_username,
        password=camera_password,
        stream_path=camera_stream_path,
    )

    analyzer = ImageAnalyzer(ml_api_url=ml_api_url)

    # Connect to printer
    if not controller.connect():
        raise RuntimeError("Failed to connect to printer")

    # Note: Camera connections are done on-demand by the monitor to minimize bandwidth.
    # We intentionally do not keep a persistent camera connection here.

    # Check ML API
    if not analyzer.health_check():
        logger.warning("ML API not available - will retry during monitoring")

    # Create and run monitor
    monitor = FailureMonitor(
        controller=controller,
        camera=camera,
        analyzer=analyzer,
        check_interval_printing=check_interval_printing,
        check_interval_idle=check_interval_idle,
        consecutive_failures_to_stop=consecutive_to_stop,
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        monitor.stop()
        camera.disconnect()
        controller.disconnect()


if __name__ == "__main__":
    main()
