"""Tests for printer connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from stream_analyzer.bambu_controller import BambuController, PrinterState

if TYPE_CHECKING:
    import collections.abc

    from bambulabs_api import Printer


class TestPrinterConnection:
    """Tests for connecting to the Bambu printer."""

    def test_printer_connected(self, printer: Printer) -> None:
        """Test that printer is connected via MQTT."""
        assert printer.mqtt_client_connected(), "Printer should be connected"

    def test_printer_mqtt_ready(self, printer: Printer) -> None:
        """Test that printer MQTT client is ready."""
        assert printer.mqtt_client_ready(), "Printer MQTT should be ready"

    def test_get_printer_state(self, printer: Printer) -> None:
        """Test that we can get printer state."""
        state = printer.get_state()
        assert state is not None, "Should be able to get printer state"
        # State should have a name attribute (it's an enum)
        state_name = state.name if hasattr(state, "name") else str(state)
        assert state_name in (
            "IDLE",
            "RUNNING",
            "PREPARING",
            "PAUSED",
            "FINISHED",
            "FAILED",
            "UNKNOWN",
        ), f"Unexpected state: {state_name}"

    def test_get_wifi_signal(self, printer: Printer) -> None:
        """Test that we can get WiFi signal strength."""
        signal = printer.wifi_signal()
        assert signal is not None, "Should be able to get WiFi signal"


class TestBambuController:
    """Tests for connecting to the Bambu printer using BambuController."""

    @pytest.fixture(scope="session")
    def controller(self) -> collections.abc.Generator[BambuController, None, None]:
        """Create and connect a BambuController for tests."""
        import os

        access_code = os.getenv("PRINTER_ACCESS_CODE")
        serial = os.getenv("PRINTER_SERIAL_NUMBER")
        ip_address = os.getenv("PRINTER_IP_ADDRESS")
        # Type narrowing
        assert access_code is not None
        assert serial is not None
        assert ip_address is not None
        ctrl = BambuController(ip_address, access_code, serial)
        assert ctrl.connect(), "Controller should connect successfully"
        yield ctrl
        ctrl.disconnect()

    def test_printer_connected(self, controller: BambuController) -> None:
        """Test that printer is connected via MQTT."""
        assert controller.is_connected(), "Printer should be connected"

    def test_get_printer_state(self, controller: BambuController) -> None:
        """Test that we can get printer state."""
        state = controller.get_state()
        assert state in (
            PrinterState.IDLE,
            PrinterState.PRINTING,
            PrinterState.PAUSED,
            PrinterState.ERROR,
            PrinterState.UNKNOWN,
        ), f"Unexpected state: {state}"
