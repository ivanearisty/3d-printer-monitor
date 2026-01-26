#!/usr/bin/env python
"""
Bambu Lab Printer Controller

Handles printer state monitoring and control via bambulabs_api.
Does NOT handle camera - use rtsp_camera.py for camera access.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class PrinterState(Enum):
    """Simplified printer states."""
    IDLE = "idle"
    PRINTING = "printing"
    PAUSED = "paused"
    ERROR = "error"
    UNKNOWN = "unknown"


class BambuController:
    """Controls and monitors a Bambu Lab printer (no camera)."""

    def __init__(
        self,
        ip_address: str,
        access_code: str,
        serial_number: str,
    ):
        self.ip_address = ip_address
        self.access_code = access_code
        self.serial_number = serial_number
        self._printer = None
        self._connected = False

    def connect(self, timeout: int = 30) -> bool:
        """Connect to the printer via MQTT."""
        from bambulabs_api import Printer

        logger.info(f"Connecting to Bambu printer at {self.ip_address}...")
        
        self._printer = Printer(
            access_code=self.access_code,
            serial=self.serial_number,
            ip_address=self.ip_address,
        )
        self._printer.connect()

        # Wait for MQTT connection
        for _ in range(timeout):
            if self._printer.mqtt_client_ready():
                self._connected = True
                logger.info("✅ Connected to Bambu printer")
                return True
            time.sleep(1)

        logger.error("Failed to connect to printer MQTT")
        return False

    def disconnect(self) -> None:
        """Disconnect from the printer."""
        if self._printer:
            self._printer.disconnect()
            self._connected = False
            logger.info("Disconnected from printer")

    def is_connected(self) -> bool:
        """Check if connected to printer."""
        return self._connected and self._printer is not None

    def get_state(self) -> PrinterState:
        """Get the current printer state."""
        if not self._printer:
            return PrinterState.UNKNOWN

        try:
            from bambulabs_api.states_info import GcodeState

            state = self._printer.get_state()

            if state in (GcodeState.RUNNING, GcodeState.PREPARE):
                return PrinterState.PRINTING
            elif state == GcodeState.PAUSE:
                return PrinterState.PAUSED
            elif state == GcodeState.IDLE:
                return PrinterState.IDLE
            else:
                return PrinterState.IDLE

        except Exception as e:
            logger.error(f"Error getting printer state: {e}")
            return PrinterState.UNKNOWN

    def is_printing(self) -> bool:
        """Check if printer is currently printing."""
        return self.get_state() == PrinterState.PRINTING

    def stop_print(self) -> bool:
        """Stop the current print job."""
        if not self._printer:
            return False

        logger.warning("🛑 STOPPING PRINT")
        try:
            result = self._printer.stop_print()
            if result:
                logger.info("Print stopped successfully")
            else:
                logger.error("Failed to stop print")
            return result
        except Exception as e:
            logger.error(f"Error stopping print: {e}")
            return False
