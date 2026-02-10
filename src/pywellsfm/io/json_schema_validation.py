# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""JSON Schema validation helpers.

This module centralizes JSON Schema loading and validation so that domain/model
mapping code in I/O helpers stays focused on constructing model objects.

Schemas are expected in the repository-level `jsonSchemas/` directory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _json_schema_dir() -> Path:
    # repo-root/jsonSchemas (schema_validation.py is in src/pywellsfm/io)
    return Path(__file__).resolve().parents[3] / "jsonSchemas"


@lru_cache(maxsize=1)
def _json_schema_store() -> dict[str, Any]:
    """Load all schemas in jsonSchemas/ and index them by a few common keys.

    This supports offline resolution for $ref values that look like remote URLs
    (via $id) as well as short local-like names (e.g. "TabulatedFunction.json")
    """
    schema_dir = _json_schema_dir()
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")

    store: dict[str, Any] = {}
    for path in schema_dir.glob("*.json"):
        schema_obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(schema_obj, dict):
            continue

        schema_id = schema_obj.get("$id")
        if isinstance(schema_id, str) and schema_id.strip() != "":
            store[schema_id] = schema_obj

        # Also register a short alias that matches the common *.json naming.
        # Example: TabulatedFunctionSchema.json -> TabulatedFunctionSchema.json
        store[f"{path.stem}.json"] = schema_obj

        # And the filename itself, useful for relative $refs.
        store[path.name] = schema_obj

    return store


def _format_jsonschema_path(path_items: Any) -> str: # noqa: ANN401
    try:
        items = list(path_items)
    except TypeError:
        return "$"

    out = "$"
    for item in items:
        if isinstance(item, int):
            out += f"[{item}]"
        else:
            out += f".{item}"
    return out


def _iter_schema_errors(
        instance: Any, # noqa: ANN401
        schema_filename: str
    ) -> list[Any]:
    """Return jsonschema validation errors for instance (does not raise)."""
    try:
        from jsonschema import RefResolver
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jsonschema is required for schema validation. Install with: "
            "pip install jsonschema"
        ) from exc

    schema_path = _json_schema_dir() / schema_filename
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    ValidatorClass = validator_for(schema)
    ValidatorClass.check_schema(schema)

    resolver = RefResolver.from_schema(schema, store=_json_schema_store())
    validator = ValidatorClass(schema, resolver=resolver)
    return sorted(validator.iter_errors(instance), key=lambda e: str(e))


def _raise_first_schema_error(
    *,
    instance_path: str,
    schema_filename: str,
    errors: list[Any],
) -> None:
    if not errors:
        return

    first = errors[0]
    at = _format_jsonschema_path(getattr(first, "absolute_path", []))
    msg = getattr(first, "message", str(first))
    raise ValueError(
        f"{instance_path} does not conform to schema '{schema_filename}': "
        f"{msg} (at {at})"
    )


def validate_json_file_against_schema(
    filepath: str, schema_filename: str
) -> Any: # noqa: ANN401
    """Validate a JSON file against a schema in jsonSchemas/.

    Returns the parsed JSON object when valid.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    errors = _iter_schema_errors(data, schema_filename)
    _raise_first_schema_error(
        instance_path=filepath,
        schema_filename=schema_filename,
        errors=errors,
    )
    return data


def expect_format_version(
    data: Any, # noqa: ANN401
    *,
    expected_format: str,
    expected_version: str,
    kind: str = "JSON",
) -> None:
    """Validate the top-level format/version metadata of a JSON-like object.

    Many project file formats share the convention:

    - format: a string identifying the payload type
    - version: a string identifying the payload version

    :param Any data: The JSON-like object to validate. Must be a dict with
        "format" and "version" keys.
    :param str expected_format: Required value for the ``format`` field.
    :param str expected_version: Required value for the ``version`` field.
    :param str kind: Human-friendly label used in error messages.

    :raises ValueError: If data is not a dict or if format/version do not match
    """
    if not isinstance(data, dict):
        raise ValueError(f"{kind} JSON must be an object.")

    fmt = data.get("format")
    if fmt != expected_format:
        raise ValueError(
            f"Invalid {kind} format: expected '{expected_format}', "
            f"got '{fmt}'."
        )

    ver = data.get("version")
    if ver != expected_version:
        raise ValueError(
            f"Invalid {kind} version: expected '{expected_version}', "
            f"got '{ver}'."
        )


def validateFaciesModelJsonFile(filepath: str) -> dict[str, Any]:
    """Validate a facies model JSON file against FaciesModelSchema.json."""
    data = validate_json_file_against_schema(
        filepath, "FaciesModelSchema.json"
    )
    if not isinstance(data, dict):
        raise ValueError("Facies model JSON must be an object.")
    return data


def validateAccumulationModelJsonFile(filepath: str) -> dict[str, Any]:
    """Validate an accumulation model JSON file against schema."""
    data = validate_json_file_against_schema(
        filepath, "AccumulationModelSchema.json"
    )
    if not isinstance(data, dict):
        raise ValueError("Accumulation model JSON must be an object.")
    return data


def validateTabulatedFunctionJsonFile(filepath: str) -> dict[str, Any]:
    """Validate a TabulatedFunction JSON file against schema."""
    data = validate_json_file_against_schema(
        filepath, "TabulatedFunctionSchema.json"
    )
    if not isinstance(data, dict):
        raise ValueError("Tabulated function JSON must be an object.")
    return data


def validateScenarioJsonFile(filepath: str) -> dict[str, Any]:
    """Validate a top-level scenario/input JSON file against schema."""
    data = validate_json_file_against_schema(filepath, "ScenarioSchema.json")
    if not isinstance(data, dict):
        raise ValueError("Scenario JSON must be an object.")
    return data


def validateUncertaintyCurveJsonFile(filepath: str) -> dict[str, Any]:
    """Validate an UncertaintyCurve JSON file against schema."""
    data = validate_json_file_against_schema(
        filepath, "UncertaintyCurveSchema.json"
    )
    if not isinstance(data, dict):
        raise ValueError("Uncertainty curve JSON must be an object.")
    return data


def validateCurveJsonFile(filepath: str) -> dict[str, Any]:
    """Validate a Curve JSON file against schema."""
    data = validate_json_file_against_schema(filepath, "CurveSchema.json")
    if not isinstance(data, dict):
        raise ValueError("Curve JSON must be an object.")
    return data


def validateWellJsonFile(filepath: str) -> dict[str, Any]:
    """Validate a Well JSON file against schema."""
    data = validate_json_file_against_schema(filepath, "WellSchema.json")
    if not isinstance(data, dict):
        raise ValueError("Well JSON must be an object.")
    return data
