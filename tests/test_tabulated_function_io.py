# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pywellsfm.io.tabulated_function_io import (
    loadTabulatedFunctionFromFile,
    loadTabulatedFunctionFromJsonObj,
    saveTabulatedFunctionToCsv,
    saveTabulatedFunctionToJson,
    tabulatedFunctionToJsonObj,
)


def test_tabulatedFunctionToJsonObj_valid_payload() -> None:
    """Serialize arrays into the expected schema-compliant JSON object."""
    obj = tabulatedFunctionToJsonObj(
        abscissa_name="Age",
        ordinate_name="ReductionCoeff",
        x=np.array([0.0, 10.0], dtype=float),
        y=np.array([1.0, 0.5], dtype=float),
    )

    assert obj["format"] == "pyWellSFM.TabulatedFunctionData"
    assert obj["version"] == "1.0"
    assert obj["abscissaName"] == "Age"
    assert obj["ordinateName"] == "ReductionCoeff"
    assert obj["values"]["xValues"] == [0.0, 10.0]
    assert obj["values"]["yValues"] == [1.0, 0.5]


@pytest.mark.parametrize(
    "abscissa_name, ordinate_name",
    [
        ("", "Y"),
        ("X", ""),
        ("   ", "Y"),
        ("X", "   "),
        (1, "Y"),
        ("X", 2),
    ],
)
def test_tabulatedFunctionToJsonObj_rejects_invalid_axis_names(
    abscissa_name: object,
    ordinate_name: object,
) -> None:
    """Axis names must be non-empty strings."""
    with pytest.raises(ValueError):
        tabulatedFunctionToJsonObj(
            abscissa_name=abscissa_name,  # type: ignore[arg-type]
            ordinate_name=ordinate_name,  # type: ignore[arg-type]
            x=np.array([0.0], dtype=float),
            y=np.array([1.0], dtype=float),
        )


@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([], dtype=float), np.array([], dtype=float)),
        (np.array([0.0], dtype=float), np.array([], dtype=float)),
        (np.array([0.0, 1.0], dtype=float), np.array([1.0], dtype=float)),
    ],
)
def test_tabulatedFunctionToJsonObj_rejects_invalid_array_shapes(
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """x/y arrays must be non-empty and have the same length."""
    with pytest.raises(ValueError):
        tabulatedFunctionToJsonObj(
            abscissa_name="Age",
            ordinate_name="ReductionCoeff",
            x=x,
            y=y,
        )


def test_save_and_load_tabulated_function_json_round_trip(
    tmp_path: Path,
) -> None:
    """Saved JSON can be loaded back with metadata and values preserved."""
    out_path = tmp_path / "nested" / "tabulated.json"
    saveTabulatedFunctionToJson(
        abscissa_name="Age",
        ordinate_name="ReductionCoeff",
        x=np.array([0.0, 10.0], dtype=float),
        y=np.array([2.0, 1.0], dtype=float),
        filepath=str(out_path),
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["abscissaName"] == "Age"
    assert payload["ordinateName"] == "ReductionCoeff"

    abscissa_name, ordinate_name, x, y = loadTabulatedFunctionFromFile(out_path)
    assert abscissa_name == "Age"
    assert ordinate_name == "ReductionCoeff"
    assert np.allclose(x, np.array([0.0, 10.0], dtype=float))
    assert np.allclose(y, np.array([2.0, 1.0], dtype=float))


def test_saveTabulatedFunctionToJson_rejects_non_json_extension(
    tmp_path: Path,
) -> None:
    """Output extension for JSON writer must be .json."""
    with pytest.raises(ValueError):
        saveTabulatedFunctionToJson(
            abscissa_name="Age",
            ordinate_name="ReductionCoeff",
            x=np.array([0.0], dtype=float),
            y=np.array([1.0], dtype=float),
            filepath=str(tmp_path / "tabulated.txt"),
        )


def test_save_and_load_tabulated_function_csv_round_trip(
    tmp_path: Path,
) -> None:
    """CSV writer/loader round-trip preserves numeric values."""
    out_path = tmp_path / "nested" / "tabulated.csv"
    saveTabulatedFunctionToCsv(
        x=np.array([0.0, 5.0, 10.0], dtype=float),
        y=np.array([1.0, 0.8, 0.5], dtype=float),
        filepath=str(out_path),
    )

    assert out_path.exists()
    abscissa_name, ordinate_name, x, y = loadTabulatedFunctionFromFile(out_path)
    assert abscissa_name == "tabulated"
    assert ordinate_name == "ReductionCoeff"
    assert np.allclose(x, np.array([0.0, 5.0, 10.0], dtype=float))
    assert np.allclose(y, np.array([1.0, 0.8, 0.5], dtype=float))


def test_saveTabulatedFunctionToCsv_rejects_invalid_extension_and_shapes(
    tmp_path: Path,
) -> None:
    """CSV writer validates extension and x/y dimensions."""
    with pytest.raises(ValueError):
        saveTabulatedFunctionToCsv(
            x=np.array([0.0], dtype=float),
            y=np.array([1.0], dtype=float),
            filepath=str(tmp_path / "tabulated.json"),
        )

    with pytest.raises(ValueError):
        saveTabulatedFunctionToCsv(
            x=np.array([0.0, 1.0], dtype=float),
            y=np.array([1.0], dtype=float),
            filepath=str(tmp_path / "tabulated.csv"),
        )


def test_loadTabulatedFunctionFromJsonObj_rejects_invalid_structure() -> None:
    """Parser rejects malformed objects and invalid metadata."""
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj([])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.NotTabulated",
                "version": "1.0",
            }
        )

    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.TabulatedFunctionData",
                "version": "1.0",
                "abscissaName": "Age",
                "ordinateName": "Y",
                "values": {"xValues": [0], "yValues": ["not-a-number"]},
            }
        )


def test_loadTabulatedFunctionFromJsonObj_rejects_missing_axis_names() -> None:
    """Parser validates both abscissa and ordinate names explicitly."""
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.TabulatedFunctionData",
                "version": "1.0",
                "abscissaName": "",
                "ordinateName": "Y",
                "values": {"xValues": [0], "yValues": [1]},
            }
        )

    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.TabulatedFunctionData",
                "version": "1.0",
                "abscissaName": "X",
                "ordinateName": "",
                "values": {"xValues": [0], "yValues": [1]},
            }
        )


def test_loadTabulatedFunctionFromJsonObj_rejects_values_shape_types() -> None:
    """Parser rejects invalid values container/type combinations."""
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.TabulatedFunctionData",
                "version": "1.0",
                "abscissaName": "X",
                "ordinateName": "Y",
                "values": "not-an-object",
            }
        )

    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.TabulatedFunctionData",
                "version": "1.0",
                "abscissaName": "X",
                "ordinateName": "Y",
                "values": {"xValues": 1, "yValues": [1]},
            }
        )

    with pytest.raises(ValueError):
        loadTabulatedFunctionFromJsonObj(
            {
                "format": "pyWellSFM.TabulatedFunctionData",
                "version": "1.0",
                "abscissaName": "X",
                "ordinateName": "Y",
                "values": {"xValues": [0, 1], "yValues": [1]},
            }
        )


def test_loadTabulatedFunctionFromFile_rejects_missing_and_bad_input(
    tmp_path: Path,
) -> None:
    """Loader validates file existence, extension, and file content shape."""
    with pytest.raises(FileNotFoundError):
        loadTabulatedFunctionFromFile(tmp_path / "missing.json")

    bad_ext = tmp_path / "bad.txt"
    bad_ext.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromFile(bad_ext)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromFile(bad_json)

    bad_csv_cols = tmp_path / "bad_cols.csv"
    bad_csv_cols.write_text("1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromFile(bad_csv_cols)

    bad_csv_values = tmp_path / "bad_values.csv"
    bad_csv_values.write_text("1,a\n2,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        loadTabulatedFunctionFromFile(bad_csv_values)
