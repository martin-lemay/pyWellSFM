# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pywellsfm.io._common import (
    ensure_dict,
    ensure_non_empty_list,
    is_url_ref,
    load_inline_or_url,
    reject_extra_keys,
    relpath_posix,
    require_url,
    resolve_ref_path,
)


def test_reject_extra_keys_accepts_schema_keywords() -> None:
    """Allow declared keys plus schema metadata keys."""
    obj: dict[str, Any] = {
        "name": "ok",
        "$schema": "x",
        "$id": "id",
        "description": "desc",
    }

    reject_extra_keys(obj=obj, allowed_keys={"name"}, ctx="payload")


def test_reject_extra_keys_raises_with_sorted_names() -> None:
    """Raise when unsupported keys are present."""
    obj = {"name": "ok", "zeta": 1, "alpha": 2}

    with pytest.raises(ValueError, match="alpha, zeta"):
        reject_extra_keys(obj=obj, allowed_keys={"name"}, ctx="payload")


def test_is_url_ref_only_for_single_url_key() -> None:
    """Recognize strict url reference objects only."""
    assert is_url_ref({"url": "a.json"}) is True
    assert is_url_ref({"url": "a.json", "x": 1}) is False
    assert is_url_ref({"x": 1}) is False


def test_require_url_rejects_non_ref_object() -> None:
    """Reject objects that are not strict url references."""
    with pytest.raises(ValueError, match="only a 'url' property"):
        require_url(obj={"url": "a.json", "x": 1}, ctx="ref")


@pytest.mark.parametrize("value", ["", "   ", 12, None])
def test_require_url_rejects_empty_or_non_string(value: object) -> None:
    """Reject missing, blank, or non-string url values."""
    with pytest.raises(ValueError, match="ref.url must be a non-empty string"):
        require_url(obj={"url": value}, ctx="ref")


def test_require_url_returns_raw_string() -> None:
    """Return the url string unchanged when valid."""
    assert require_url(obj={"url": "  a.json  "}, ctx="ref") == "  a.json  "


def test_resolve_ref_path_accepts_absolute_without_base_dir() -> None:
    """Keep absolute paths even when base_dir is missing."""
    absolute = Path(__file__).resolve()

    resolved = resolve_ref_path(
        base_dir=None,
        raw_url=str(absolute),
        ctx="curve",
    )
    assert resolved == absolute


def test_resolve_ref_path_resolves_relative_with_base_dir(
    tmp_path: Path,
) -> None:
    """Resolve relative paths from the provided base directory."""
    resolved = resolve_ref_path(
        base_dir=tmp_path,
        raw_url="sub/data.json",
        ctx="curve",
    )
    assert resolved == (tmp_path / "sub" / "data.json").resolve()


def test_resolve_ref_path_rejects_relative_without_base_dir() -> None:
    """Reject relative paths when base_dir is not set."""
    with pytest.raises(ValueError, match="Base directory must be provided"):
        resolve_ref_path(base_dir=None, raw_url="data.json", ctx="curve")


def test_load_inline_or_url_rejects_non_object() -> None:
    """Reject raw values that are not JSON objects."""
    with pytest.raises(ValueError, match="input must be an object"):
        load_inline_or_url(
            "x",
            base_dir=None,
            ctx="input",
            load_inline=lambda _: 0,
            load_file=lambda _: 1,
        )


def test_load_inline_or_url_uses_inline_loader() -> None:
    """Dispatch to inline loader for inline objects."""
    called: dict[str, Any] = {}

    def _load_inline(obj: dict[str, Any]) -> str:
        called["obj"] = obj
        return "inline"

    def _load_file(_: Path) -> str:
        raise AssertionError("file loader should not be used")

    out = load_inline_or_url(
        {"k": 1},
        base_dir=None,
        ctx="input",
        load_inline=_load_inline,
        load_file=_load_file,
    )
    assert out == "inline"
    assert called["obj"] == {"k": 1}


def test_load_inline_or_url_uses_file_loader(tmp_path: Path) -> None:
    """Dispatch to file loader for strict url references."""
    called: dict[str, Any] = {}

    def _load_inline(_: dict[str, Any]) -> str:
        raise AssertionError("inline loader should not be used")

    def _load_file(path: Path) -> str:
        called["path"] = path
        return "file"

    out = load_inline_or_url(
        {"url": "nested/input.json"},
        base_dir=tmp_path,
        ctx="input",
        load_inline=_load_inline,
        load_file=_load_file,
    )
    assert out == "file"
    assert called["path"] == (tmp_path / "nested" / "input.json").resolve()


def test_relpath_posix_returns_forward_slashes(tmp_path: Path) -> None:
    """Create POSIX-style relative paths."""
    root = tmp_path / "root"
    target = root / "a" / "b" / "c.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    assert relpath_posix(target, start=root) == "a/b/c.json"


def test_ensure_non_empty_list_accepts_non_empty_list() -> None:
    """Return a valid list unchanged."""
    data = [1, 2, 3]
    assert ensure_non_empty_list(data, ctx="vals") is data


@pytest.mark.parametrize("value", [None, "x", 1, [], {}])
def test_ensure_non_empty_list_rejects_invalid_values(value: object) -> None:
    """Reject empty or non-list values."""
    with pytest.raises(ValueError, match="vals must be a non-empty list"):
        ensure_non_empty_list(value, ctx="vals")


def test_ensure_dict_accepts_dict() -> None:
    """Return a dict value unchanged."""
    obj = {"a": 1}
    assert ensure_dict(obj, ctx="obj") is obj


@pytest.mark.parametrize("value", [None, "x", 1, [], ()])
def test_ensure_dict_rejects_non_dict_values(value: object) -> None:
    """Reject values that are not objects."""
    with pytest.raises(ValueError, match="obj must be an object"):
        ensure_dict(value, ctx="obj")
