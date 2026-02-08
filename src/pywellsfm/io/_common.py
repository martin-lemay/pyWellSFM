# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Internal helpers shared by I/O modules.

This module centralizes small, repetitive patterns:
- resolving relative paths against a base directory
- parsing schema-style {"url": "..."} references
- rejecting unexpected keys (additionalProperties=false)

Not part of the public API; callers should import from `pywellsfm.io.*`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def reject_extra_keys(
    *, obj: dict[str, Any], allowed_keys: set[str], ctx: str
) -> None:
    extra = set(obj.keys()) - allowed_keys
    if extra:
        raise ValueError(
            f"{ctx} contains unsupported properties: "
            + ", ".join(sorted(extra))
        )


def is_url_ref(obj: dict[str, Any]) -> bool:
    """Return True if obj is a strict schema ref object: {"url": "..."}."""
    return set(obj.keys()) == {"url"}


def require_url(*, obj: dict[str, Any], ctx: str) -> str:
    if not is_url_ref(obj):
        raise ValueError(
            f"{ctx} reference must be an object with only a 'url' property."
        )
    raw = obj.get("url")
    if not isinstance(raw, str) or raw.strip() == "":
        raise ValueError(f"{ctx}.url must be a non-empty string.")
    return raw


def resolve_ref_path(*, base_dir: Path | None, raw_url: str, ctx: str) -> Path:
    """Resolve an absolute or base_dir-relative path.

    - Absolute paths are accepted even when base_dir is None.
    - Relative paths require base_dir.
    """
    p = Path(raw_url)
    if p.is_absolute():
        return p
    if base_dir is None:
        raise ValueError(
            f"Base directory must be provided to resolve relative path for {ctx}."
        )
    return (base_dir / p).resolve()


def load_inline_or_url(
    raw: Any,
    *,
    base_dir: Path | None,
    ctx: str,
    load_inline: Callable[[dict[str, Any]], T],
    load_file: Callable[[Path], T],
) -> T:
    """Load either an inline JSON object or a {"url": "..."} reference."""
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx} must be an object.")

    if is_url_ref(raw):
        url = require_url(obj=raw, ctx=ctx)
        path = resolve_ref_path(base_dir=base_dir, raw_url=url, ctx=ctx)
        return load_file(path)

    return load_inline(raw)


def relpath_posix(target: Path, *, start: Path) -> str:
    """Return a relative path with forward slashes."""
    rel = target.resolve().relative_to(start.resolve())
    return rel.as_posix()


def ensure_non_empty_list(value: Any, *, ctx: str) -> list[Any]:
    if not isinstance(value, list) or len(value) < 1:
        raise ValueError(f"{ctx} must be a non-empty list.")
    return value


def ensure_dict(value: Any, *, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{ctx} must be an object.")
    return value
