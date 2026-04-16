# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Tests for package logging and in-memory log retention."""

import json
from pathlib import Path

import numpy as np
import pytest

from pywellsfm import (
    clear_stored_logs,
    configure_logging,
    export_stored_logs,
    export_stored_logs_to_json_file,
    export_stored_logs_to_text_file,
    get_logger,
    get_stored_log_messages,
    get_stored_logs,
)
from pywellsfm.model.Well import Well
from pywellsfm.simulator.AccommodationSimulator import AccommodationSimulator


def test_logging_retention_collects_info_warning_error() -> None:
    """Stored logs capture expected levels and message content."""
    configure_logging(enable_console=False)
    clear_stored_logs()

    logger = get_logger("pywellsfm.tests.logging")
    logger.info("Info message for retention test")
    logger.warning("Warning message for retention test")
    logger.error("Error message for retention test")

    entries = get_stored_logs()
    levels = [entry["level"] for entry in entries]

    assert "INFO" in levels
    assert "WARNING" in levels
    assert "ERROR" in levels


def test_clear_stored_logs_empties_memory_storage() -> None:
    """clear_stored_logs removes all retained entries."""
    configure_logging(enable_console=False)
    clear_stored_logs()

    logger = get_logger("pywellsfm.tests.logging")
    logger.info("Message that should be cleared")

    assert len(get_stored_logs()) >= 1

    clear_stored_logs()
    assert get_stored_logs() == []


def test_configure_logging_is_idempotent_for_handlers() -> None:
    """Repeated setup does not duplicate logger handlers."""
    logger = configure_logging(enable_console=True)
    before = len(logger.handlers)

    logger = configure_logging(enable_console=True)
    after = len(logger.handlers)

    assert before == after


def test_module_logging_paths_are_retained() -> None:
    """Messages emitted by library code are retained in storage."""
    configure_logging(enable_console=False)
    clear_stored_logs()

    well = Well(
        "TestWell",
        np.array([0.0, 0.0, 0.0], dtype=float),
        depth=100.0,
    )
    # Trigger invalid type handling in Well.addLog.
    well.addLog("invalid", object())  # type: ignore[arg-type]

    sim = AccommodationSimulator()
    sim.prepare()

    messages = get_stored_log_messages()

    assert any("Log type is not managed" in msg for msg in messages)
    assert any("Subsidence curve not set" in msg for msg in messages)
    assert any("Eustatic curve not set" in msg for msg in messages)


def test_no_print_calls_remain_in_src_tree() -> None:
    """Guard that source modules no longer use print calls."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "pywellsfm"
    for py_file in src_root.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "print(" not in content


def test_export_stored_logs_to_text_file(tmp_path: Path) -> None:
    """Retained logs can be exported as text."""
    configure_logging(enable_console=False)
    clear_stored_logs()

    logger = get_logger("pywellsfm.tests.logging")
    logger.info("First exportable message")
    logger.warning("Second exportable message")

    output = tmp_path / "logs" / "run.log"
    exported = export_stored_logs_to_text_file(output)

    assert exported == output
    assert output.exists()

    text = output.read_text(encoding="utf-8")
    assert "First exportable message" in text
    assert "Second exportable message" in text


def test_export_stored_logs_to_json_file(tmp_path: Path) -> None:
    """Retained logs can be exported as JSON records."""
    configure_logging(enable_console=False)
    clear_stored_logs()

    logger = get_logger("pywellsfm.tests.logging")
    logger.error("JSON export message")

    output = tmp_path / "logs.json"
    exported = export_stored_logs_to_json_file(output)

    assert exported == output
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) >= 1
    assert any(entry["message"] == "JSON export message" for entry in payload)


def test_export_stored_logs_invalid_format_raises(tmp_path: Path) -> None:
    """Unsupported export format raises ValueError."""
    configure_logging(enable_console=False)
    clear_stored_logs()
    get_logger("pywellsfm.tests.logging").info("Any message")

    with pytest.raises(ValueError):
        export_stored_logs(tmp_path / "logs.xyz", format="xml")
