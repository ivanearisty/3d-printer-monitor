"""Pytest fixtures for 3D printer failure detection tests."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import load_dotenv

if TYPE_CHECKING:
    from bambulabs_api import Printer

# Load environment variables
load_dotenv()

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="session")
def printer() -> Generator["Printer", None, None]:
    """Create and connect to printer for the test session."""
    from bambulabs_api import Printer

    access_code = os.getenv("PRINTER_ACCESS_CODE")
    serial = os.getenv("PRINTER_SERIAL_NUMBER")
    ip_address = os.getenv("PRINTER_IP_ADDRESS")

    if not all([access_code, serial, ip_address]):
        pytest.skip("Printer credentials not configured in environment")

    # Assert for type narrowing after the skip check
    assert access_code is not None
    assert serial is not None
    assert ip_address is not None

    printer = Printer(access_code=access_code, serial=serial, ip_address=ip_address)
    printer.connect()

    # Wait for MQTT connection
    for _ in range(30):
        if printer.mqtt_client_ready():
            break
        time.sleep(1)
    else:
        pytest.fail("Failed to connect to printer MQTT")

    yield printer

    printer.disconnect()


@pytest.fixture(scope="session")
def ml_api_url() -> str:
    """Get ML API URL from environment."""
    return os.getenv("ML_API_URL", "http://localhost:3333")


@pytest.fixture
def example_images() -> list[Path]:
    """Get list of bad/failure example images (bad*.jpeg)."""
    if not EXAMPLES_DIR.exists():
        pytest.skip("Examples directory not found")

    # Only include failure examples (bad1.jpeg, bad2.jpeg, etc.)
    images = sorted(EXAMPLES_DIR.glob("bad*.jpeg"))

    if not images:
        pytest.skip("No failure example images found")

    return images


@pytest.fixture
def good_images() -> list[Path]:
    """Get list of good print images (good*.jpeg) that should NOT trigger detections."""
    if not EXAMPLES_DIR.exists():
        pytest.skip("Examples directory not found")

    # Only include good examples (good1.jpeg, good2.jpeg, etc.)
    images = sorted(EXAMPLES_DIR.glob("good*.jpeg"))

    if not images:
        pytest.skip("No good example images found")

    return images
