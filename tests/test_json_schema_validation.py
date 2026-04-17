# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pywellsfm.io.json_schema_validation as jsv


def test_format_jsonschema_path_variants() -> None:
    """Path formatter supports list-like paths and non-iterable fallback."""
    assert jsv._format_jsonschema_path(["a", 2, "b"]) == "$.a[2].b"
    assert jsv._format_jsonschema_path(("x", 1)) == "$.x[1]"
    assert jsv._format_jsonschema_path(42) == "$"


def test_raise_first_schema_error_no_errors_returns() -> None:
    """No-op behavior when validator returned no errors."""
    jsv._raise_first_schema_error(
        instance_path="dummy.json",
        schema_filename="SomeSchema.json",
        errors=[],
    )


def test_raise_first_schema_error_uses_absolute_path_and_message() -> None:
    """Raised message includes schema, location and underlying error text."""

    class DummyError:
        absolute_path = ["faciesModel", 0, "name"]
        message = "is a required property"

    with pytest.raises(ValueError) as exc:
        jsv._raise_first_schema_error(
            instance_path="facies.json",
            schema_filename="FaciesModelSchema.json",
            errors=[DummyError()],
        )

    msg = str(exc.value)
    assert "FaciesModelSchema.json" in msg
    assert "is a required property" in msg
    assert "$.faciesModel[0].name" in msg


def test_json_schema_store_raises_when_schema_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store loading fails early if schema directory does not exist."""
    jsv._json_schema_store.cache_clear()

    missing_dir = tmp_path / "missing_schemas"
    monkeypatch.setattr(jsv, "_json_schema_dir", lambda: missing_dir)

    with pytest.raises(FileNotFoundError):
        _ = jsv._json_schema_store()


def test_json_schema_store_indexes_id_stem_and_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store indexes valid schema dicts and ignores non-dict JSON files."""
    jsv._json_schema_store.cache_clear()

    schema_dir = tmp_path / "jsonSchemas"
    schema_dir.mkdir(parents=True)

    (schema_dir / "ArrayOnly.json").write_text("[1, 2, 3]", encoding="utf-8")
    (schema_dir / "MinimalSchema.json").write_text(
        json.dumps(
            {
                "$id": "https://example.org/MinimalSchema.json",
                "type": "object",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(jsv, "_json_schema_dir", lambda: schema_dir)
    store = jsv._json_schema_store()

    assert "https://example.org/MinimalSchema.json" in store
    assert "MinimalSchema.json" in store
    assert "ArrayOnly.json" not in store


def test_iter_schema_errors_raises_when_schema_file_missing() -> None:
    """Low-level iterator reports missing schema file explicitly."""
    with pytest.raises(FileNotFoundError):
        _ = jsv._iter_schema_errors({}, "__definitely_missing__.json")


def test_validate_json_file_against_schema_success_and_failure(
    tmp_path: Path,
) -> None:
    """Validation helper returns parsed data or raises with schema context."""
    facies_path = Path(__file__).parent / "data" / "well.json"
    valid_obj = jsv.validate_json_file_against_schema(
        str(facies_path),
        "WellSchema.json",
    )
    assert isinstance(valid_obj, dict)
    assert valid_obj["format"] == "pyWellSFM.WellData"

    invalid_path = tmp_path / "invalid_facies.json"
    invalid_path.write_text(
        json.dumps(
            {
                "format": "pyWellSFM.FaciesModelData",
                "version": "1.0",
                "faciesModel": [
                    {
                        "criteriaType": "sedimentological",
                        "criteria": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        _ = jsv.validate_json_file_against_schema(
            str(invalid_path),
            "WellSchema.json",
        )
    assert "does not conform to schema" in str(exc.value)


def test_expect_format_version_valid_and_invalid_cases() -> None:
    """Format/version checker enforces object shape and metadata values."""
    jsv.expect_format_version(
        {
            "format": "pyWellSFM.FaciesModelData",
            "version": "1.0",
        },
        expected_format="pyWellSFM.FaciesModelData",
        expected_version="1.0",
        kind="facies",
    )

    with pytest.raises(ValueError):
        jsv.expect_format_version(
            [],  # type: ignore[arg-type]
            expected_format="x",
            expected_version="1.0",
        )
    with pytest.raises(ValueError):
        jsv.expect_format_version(
            {"format": "wrong", "version": "1.0"},
            expected_format="right",
            expected_version="1.0",
            kind="curve",
        )
    with pytest.raises(ValueError):
        jsv.expect_format_version(
            {"format": "right", "version": "9.9"},
            expected_format="right",
            expected_version="1.0",
            kind="curve",
        )


def test_validate_wrapper_functions_with_real_and_mocked_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All wrapper validators are exercised for success and object checks."""
    well_path = Path(__file__).parent / "data" / "well.json"

    # Real success for wrappers with available sample files.
    well_obj = jsv.validateWellJsonFile(str(well_path))
    assert isinstance(well_obj, dict)

    # Mocked non-dict return to exercise each wrapper's type guard branch.
    monkeypatch.setattr(
        jsv,
        "validate_json_file_against_schema",
        lambda *_a, **_k: [],
    )

    wrappers = [
        jsv.validateAccumulationModelJsonFile,
        jsv.validateTabulatedFunctionJsonFile,
        jsv.validateScenarioJsonFile,
        jsv.validateUncertaintyCurveJsonFile,
        jsv.validateCurveJsonFile,
        jsv.validateWellJsonFile,
        jsv.validateEnvironmentConditionModelJsonFile,
        jsv.validateEnvironmentConditionsModelJsonFile,
        jsv.validateFaciesModelJsonFile,
    ]

    for fn in wrappers:
        with pytest.raises(ValueError):
            _ = fn("dummy.json")

    # Mocked dict return to exercise each wrapper success return branch.
    monkeypatch.setattr(
        jsv,
        "validate_json_file_against_schema",
        lambda *_a, **_k: {"ok": True},
    )
    for fn in wrappers:
        obj = fn("dummy.json")
        assert isinstance(obj, dict)
        assert obj["ok"] is True
