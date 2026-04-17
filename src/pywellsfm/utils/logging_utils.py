# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Package logging helpers and in-memory log retention."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

# Internal module state for handlers and package logger name
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG


class StoredLogsHandler(logging.Handler):
    def __init__(self: Self) -> None:
        """In-memory log handler storing structured records."""
        super().__init__()
        self._records: list[dict[str, Any]] = []

    def emit(self: Self, record: logging.LogRecord) -> None:
        """Emit a log record.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        try:
            entry = {
                "created": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "name": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "pathname": record.pathname,
                "lineno": record.lineno,
            }
            self._records.append(entry)
        except Exception:
            self.handleError(record)

    def get_records(self: Self) -> list[dict[str, Any]]:
        """Return a copy of all retained records."""
        return list(self._records)

    def clear(self: Self) -> None:
        """Clear retained records."""
        self._records.clear()


_PACKAGE_LOGGER_NAME = "pywellsfm"
_stored_logs_handler: StoredLogsHandler | None = None
_console_handler: logging.Handler | None = None


def _default_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _ensure_handlers(level: int, enable_console: bool) -> logging.Logger:
    global _stored_logs_handler, _console_handler

    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if _stored_logs_handler is None:
        _stored_logs_handler = StoredLogsHandler()
        _stored_logs_handler.setLevel(logging.DEBUG)
        logger.addHandler(_stored_logs_handler)
    elif _stored_logs_handler not in logger.handlers:
        logger.addHandler(_stored_logs_handler)

    if enable_console:
        if _console_handler is None:
            _console_handler = logging.StreamHandler()
            _console_handler.setFormatter(_default_formatter())
            logger.addHandler(_console_handler)
        elif _console_handler not in logger.handlers:
            logger.addHandler(_console_handler)
        _console_handler.setLevel(level)
    elif _console_handler is not None and _console_handler in logger.handlers:
        logger.removeHandler(_console_handler)

    return logger


def configure_logging(
    *, level: int = INFO, enable_console: bool = True
) -> logging.Logger:
    """Configure package logging and return the package logger."""
    return _ensure_handlers(level=level, enable_console=enable_console)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a package logger or a package child logger."""
    configure_logging()

    if name is None:
        return logging.getLogger(_PACKAGE_LOGGER_NAME)

    if name == _PACKAGE_LOGGER_NAME or name.startswith(
        f"{_PACKAGE_LOGGER_NAME}."
    ):
        return logging.getLogger(name)

    child_name = f"{_PACKAGE_LOGGER_NAME}.{name}"
    return logging.getLogger(child_name)


def set_log_level(level: int) -> None:
    """Update the package logger level."""
    logger = configure_logging(level=level)
    logger.setLevel(level)
    if _console_handler is not None:
        _console_handler.setLevel(level)


def get_stored_logs() -> list[dict[str, Any]]:
    """Return retained logs as structured dictionaries."""
    configure_logging()
    if _stored_logs_handler is None:
        return []
    return _stored_logs_handler.get_records()


def get_stored_log_messages() -> list[str]:
    """Return retained logs as formatted single-line messages."""
    return [
        f"{entry['created']} | {entry['name']} | {entry['level']} | "
        f"{entry['message']}"
        for entry in get_stored_logs()
    ]


def clear_stored_logs() -> None:
    """Clear retained logs from in-memory storage."""
    if _stored_logs_handler is not None:
        _stored_logs_handler.clear()


def export_stored_logs(
    filepath: str | Path,
    *,
    format: str = "text",
    append: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Export retained logs to a file.

    Supported formats are:

    - text: one formatted message per line
    - json: structured list of records
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_format = format.strip().lower()
    if normalized_format == "text":
        mode = "a" if append else "w"
        lines = get_stored_log_messages()
        payload = "\n".join(lines)
        if payload and not payload.endswith("\n"):
            payload += "\n"
        with output_path.open(mode=mode, encoding=encoding) as f:
            f.write(payload)
        return output_path

    if normalized_format == "json":
        if append:
            raise ValueError("append=True is not supported for JSON export.")
        records = get_stored_logs()
        with output_path.open(mode="w", encoding=encoding) as f:
            json.dump(records, f, indent=2)
        return output_path

    raise ValueError("Unsupported export format. Use 'text' or 'json'.")


def export_stored_logs_to_text_file(
    filepath: str | Path,
    *,
    append: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Export retained logs to a text file."""
    return export_stored_logs(
        filepath,
        format="text",
        append=append,
        encoding=encoding,
    )


def export_stored_logs_to_json_file(
    filepath: str | Path,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Export retained logs to a JSON file."""
    return export_stored_logs(
        filepath,
        format="json",
        append=False,
        encoding=encoding,
    )
