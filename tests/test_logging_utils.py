# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Tests for package logging and in-memory log retention."""

import json
import logging
from pathlib import Path

import pytest

import pywellsfm
import pywellsfm.utils.logging_utils as logging_utils


def test_logging_retention_collects_info_warning_error() -> None:
    """Stored logs capture expected levels and message content."""
    pywellsfm.configure_logging(enable_console=False)
    pywellsfm.clear_stored_logs()

    logger = pywellsfm.get_logger("pywellsfm.tests.logging")
    logger.info("Info message for retention test")
    logger.warning("Warning message for retention test")
    logger.error("Error message for retention test")

    entries = pywellsfm.get_stored_logs()
    levels = [entry["level"] for entry in entries]

    assert "INFO" in levels
    assert "WARNING" in levels
    assert "ERROR" in levels


def test_clear_stored_logs_empties_memory_storage() -> None:
    """clear_stored_logs removes all retained entries."""
    pywellsfm.configure_logging(enable_console=False)
    pywellsfm.clear_stored_logs()

    logger = pywellsfm.get_logger("pywellsfm.tests.logging")
    logger.info("Message that should be cleared")

    assert len(pywellsfm.get_stored_logs()) >= 1

    pywellsfm.clear_stored_logs()
    assert pywellsfm.get_stored_logs() == []


def test_configure_logging_is_idempotent_for_handlers() -> None:
    """Repeated setup does not duplicate logger handlers."""
    logger = pywellsfm.configure_logging(enable_console=True)
    before = len(logger.handlers)

    logger = pywellsfm.configure_logging(enable_console=True)
    after = len(logger.handlers)

    assert before == after


def test_get_logger_variants_return_expected_names() -> None:
    """Logger naming rules return package or child loggers."""
    pywellsfm.configure_logging(enable_console=False)

    assert pywellsfm.get_logger().name == "pywellsfm"
    assert pywellsfm.get_logger("pywellsfm").name == "pywellsfm"
    assert pywellsfm.get_logger("pywellsfm.tests").name == "pywellsfm.tests"
    assert pywellsfm.get_logger("tests").name == "pywellsfm.tests"


def test_set_log_level_updates_logger_and_console_handler() -> None:
    """set_log_level updates both logger and console handler levels."""
    logger = pywellsfm.configure_logging(
        level=pywellsfm.INFO,
        enable_console=True,
    )

    pywellsfm.set_log_level(pywellsfm.ERROR)

    assert logger.level == pywellsfm.ERROR
    assert logging_utils._console_handler is not None
    assert logging_utils._console_handler.level == pywellsfm.ERROR


def test_readds_stored_handler_if_removed() -> None:
    """Stored handler is reattached when missing from logger handlers."""
    logger = pywellsfm.configure_logging(enable_console=False)
    assert logging_utils._stored_logs_handler is not None

    logger.removeHandler(logging_utils._stored_logs_handler)
    assert logging_utils._stored_logs_handler not in logger.handlers

    pywellsfm.configure_logging(enable_console=False)
    assert logging_utils._stored_logs_handler in logger.handlers


def test_clear_stored_logs_noop_when_handler_missing() -> None:
    """clear_stored_logs is safe when no handler exists."""
    old_handler = logging_utils._stored_logs_handler
    logging_utils._stored_logs_handler = None
    try:
        pywellsfm.clear_stored_logs()
    finally:
        logging_utils._stored_logs_handler = old_handler


def test_get_stored_logs_returns_empty_when_handler_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_stored_logs returns empty list when storage is unavailable."""
    monkeypatch.setattr(logging_utils, "configure_logging", lambda **_: None)
    monkeypatch.setattr(logging_utils, "_stored_logs_handler", None)

    assert pywellsfm.get_stored_logs() == []


def test_stored_handler_emit_error_calls_handle_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit forwards unexpected formatting errors to handleError."""
    handler = logging_utils.StoredLogsHandler()
    logger = pywellsfm.get_logger("tests.logging.emit")
    record = logger.makeRecord(
        logger.name,
        pywellsfm.INFO,
        __file__,
        10,
        "msg",
        args=(),
        exc_info=None,
        func=None,
        extra=None,
    )

    class _BrokenDatetime:
        @staticmethod
        def fromtimestamp(*args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    called = {"value": False}

    def _handle_error(_: logging.LogRecord) -> None:
        called["value"] = True

    monkeypatch.setattr(logging_utils, "datetime", _BrokenDatetime)
    monkeypatch.setattr(handler, "handleError", _handle_error)

    handler.emit(record)

    assert called["value"] is True


def test_no_print_calls_remain_in_src_tree() -> None:
    """Guard that source modules no longer use print calls."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "pywellsfm"
    for py_file in src_root.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "print(" not in content


def test_export_stored_logs_to_text_file(tmp_path: Path) -> None:
    """Retained logs can be exported as text."""
    pywellsfm.configure_logging(enable_console=False)
    pywellsfm.clear_stored_logs()

    logger = pywellsfm.get_logger("pywellsfm.tests.logging")
    logger.info("First exportable message")
    logger.warning("Second exportable message")

    output = tmp_path / "logs" / "run.log"
    exported = pywellsfm.export_stored_logs_to_text_file(output)

    assert exported == output
    assert output.exists()

    text = output.read_text(encoding="utf-8")
    assert "First exportable message" in text
    assert "Second exportable message" in text


def test_export_stored_logs_to_json_file(tmp_path: Path) -> None:
    """Retained logs can be exported as JSON records."""
    pywellsfm.configure_logging(enable_console=False)
    pywellsfm.clear_stored_logs()

    logger = pywellsfm.get_logger("pywellsfm.tests.logging")
    logger.error("JSON export message")

    output = tmp_path / "logs.json"
    exported = pywellsfm.export_stored_logs_to_json_file(output)

    assert exported == output
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) >= 1
    assert any(entry["message"] == "JSON export message" for entry in payload)


def test_export_stored_logs_invalid_format_raises(tmp_path: Path) -> None:
    """Unsupported export format raises ValueError."""
    pywellsfm.configure_logging(enable_console=False)
    pywellsfm.clear_stored_logs()
    pywellsfm.get_logger("pywellsfm.tests.logging").info("Any message")

    with pytest.raises(ValueError):
        pywellsfm.export_stored_logs(tmp_path / "logs.xyz", format="xml")


def test_export_stored_logs_json_append_raises(tmp_path: Path) -> None:
    """JSON export rejects append mode."""
    pywellsfm.configure_logging(enable_console=False)
    pywellsfm.clear_stored_logs()

    with pytest.raises(ValueError):
        pywellsfm.export_stored_logs(
            tmp_path / "logs.json",
            format="json",
            append=True,
        )
