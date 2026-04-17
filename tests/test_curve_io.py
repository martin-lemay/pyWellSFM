# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.interpolate import interp1d

import pywellsfm.io.curve_io as cio
from pywellsfm.model.Curve import Curve
from pywellsfm.utils import PolynomialInterpolator
from pywellsfm.utils.interpolation import (
    LowerBoundInterpolator,
    UpperBoundInterpolator,
)
from pywellsfm.utils.logging_utils import clear_stored_logs, get_stored_logs


def _curve_linear() -> Curve:
    return Curve(
        "Depth",
        "Value",
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([0.0, 1.0, 4.0], dtype=float),
        "linear",
    )


def _curve_json_obj(
    *, interpolation_method: str | dict[str, Any] = "linear"
) -> dict[str, Any]:
    return {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": "Depth",
            "yAxisName": "Value",
            "interpolationMethod": interpolation_method,
            "data": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
        },
    }


def _ucurve_obj(curves: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "pyWellSFM.UncertaintyCurveData",
        "version": "1.0",
        "data": {"name": "U", "curves": curves},
    }


# Test of Curve IO functions
def test_loadCurvesFromCsv_reads_header_and_sorts(tmp_path: Path) -> None:
    """Test loadCurvesFromCsv function with 1 curve.

    :param Path tmp_path: path to temporary folder
    """
    csv_path = tmp_path / "my_curve.csv"
    csv_path.write_text(
        "Depth,Value\n2,20\n1,10\n3,30\n",
        encoding="utf-8",
    )

    curves = cio.loadCurvesFromCsv(csv_path)
    assert len(curves) == 1
    curve = curves[0]

    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert np.array_equal(curve._abscissa, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(curve._ordinate, np.array([10.0, 20.0, 30.0]))
    assert isinstance(curve._interpFunc, interp1d)
    assert curve._interpFunc._kind == "linear"


def test_loadCurvesFromCsv_reads_multiple_curves(tmp_path: Path) -> None:
    """Test loadCurvesFromCsv function with multiple curves.

    :param Path tmp_path: path to temporary folder
    """
    csv_path = tmp_path / "my_curve.csv"
    depth = [1, 2, 3]
    curve1 = [20, 100, 300]
    curve2 = [10, 50, 500]
    curve3 = [30, 150, 450]
    curvesAll = [curve1, curve2, curve3]
    csv_path.write_text(
        "Depth,Curve1,Curve2,Curve3\n"
        + f"{depth[0]},{curve1[0]},{curve2[0]},{curve3[0]}\n"
        + f"{depth[1]},{curve1[1]},{curve2[1]},{curve3[1]}\n"
        + f"{depth[2]},{curve1[2]},{curve2[2]},{curve3[2]}\n",
        encoding="utf-8",
    )

    curves = cio.loadCurvesFromCsv(csv_path)
    assert len(curves) == 3

    for u, curve in enumerate(curves):
        assert curve._xAxisName == "Depth"
        assert curve._yAxisName == f"Curve{u + 1}"
        expected_ordinate = np.array(curvesAll[u])
        assert np.array_equal(curve._abscissa, np.array(depth))
        assert np.array_equal(curve._ordinate, expected_ordinate)
        assert isinstance(curve._interpFunc, interp1d)
        assert curve._interpFunc._kind == "linear"


def test_loadCurvesFromCsv_drops_non_numeric_rows(tmp_path: Path) -> None:
    """Test loadCurvesFromCsv function.

    :param Path tmp_path: path to temporary folder
    """
    csv_path = tmp_path / "bad_rows.csv"
    csv_path.write_text(
        "X,Y\n0,0\noops,2\n2,4\n",
        encoding="utf-8",
    )

    curves = cio.loadCurvesFromCsv(csv_path)
    assert len(curves) == 1
    curve = curves[0]
    assert np.array_equal(curve._abscissa, np.array([0.0, 2.0]))
    assert np.array_equal(curve._ordinate, np.array([0.0, 4.0]))


def test_loadCurvesFromCsv_requires_two_columns(tmp_path: Path) -> None:
    """Test loadCurvesFromCsv function.

    :param Path tmp_path: path to temporary folder
    """
    csv_path = tmp_path / "one_col.csv"
    csv_path.write_text("X\n1\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 2 columns"):
        cio.loadCurvesFromCsv(csv_path)


def test_loadCurveFromJsonObj_linear() -> None:
    """Load a Curve from a CurveSchema.json-compatible object (linear)."""
    obj = {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": "Depth",
            "yAxisName": "Value",
            "interpolationMethod": "linear",
            "data": [
                {"x": 2, "y": 20},
                {"x": 1, "y": 10},
                {"x": 3, "y": 30},
            ],
        },
    }

    curve = cio.loadCurveFromJsonObj(obj)
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert np.array_equal(curve._abscissa, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(curve._ordinate, np.array([10.0, 20.0, 30.0]))
    assert isinstance(curve._interpFunc, interp1d)
    assert curve._interpFunc._kind == "linear"


def test_loadCurveFromJsonObj_polynomial() -> None:
    """Load a Curve from a CurveSchema.json-compatible object (polynomial)."""
    obj = {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": "Depth",
            "yAxisName": "Value",
            "interpolationMethod": {
                "name": "PolynomialInterpolator",
                "degree": 3,
                "nbPoints": 5,
            },
            "data": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
                {"x": 2.0, "y": 4.0},
                {"x": 3.0, "y": 9.0},
                {"x": 4.0, "y": 16.0},
            ],
        },
    }

    curve = cio.loadCurveFromJsonObj(obj)
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert isinstance(curve._interpFunc, PolynomialInterpolator)
    assert curve._interpFunc.deg == 3
    assert curve._interpFunc.nbPts >= 5


def test_loadCurveFromJsonObj_lowerBound() -> None:
    """Load a Curve from a CurveSchema.json-compatible object (lowerBound)."""
    obj = {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": "Depth",
            "yAxisName": "Value",
            "interpolationMethod": "LowerBound",
            "data": [
                {"x": 2, "y": 20},
                {"x": 1, "y": 10},
                {"x": 3, "y": 30},
            ],
        },
    }

    curve = cio.loadCurveFromJsonObj(obj)
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert np.array_equal(curve._abscissa, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(curve._ordinate, np.array([10.0, 20.0, 30.0]))
    assert isinstance(curve._interpFunc, LowerBoundInterpolator)


def test_loadCurveFromJsonObj_upperBound() -> None:
    """Load a Curve from a CurveSchema.json-compatible object (upperBound)."""
    obj = {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": "Depth",
            "yAxisName": "Value",
            "interpolationMethod": "UpperBound",
            "data": [
                {"x": 2, "y": 20},
                {"x": 1, "y": 10},
                {"x": 3, "y": 30},
            ],
        },
    }

    curve = cio.loadCurveFromJsonObj(obj)
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert np.array_equal(curve._abscissa, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(curve._ordinate, np.array([10.0, 20.0, 30.0]))
    assert isinstance(curve._interpFunc, UpperBoundInterpolator)


def test_loadCurveFromFile_json(tmp_path: Path) -> None:
    """Load Curve from a JSON file matching CurveSchema.json."""
    json_path = tmp_path / "curve.json"
    json_path.write_text(
        """
{
  \"format\": \"pyWellSFM.CurveData\",
  \"version\": \"1.0\",
  \"curve\": {
    \"xAxisName\": \"Depth\",
    \"yAxisName\": \"Value\",
    \"interpolationMethod\": \"linear\",
    \"data\": [
      {\"x\": 0, \"y\": 0},
      {\"x\": 1, \"y\": 1}
    ]
  }
}
""".strip(),
        encoding="utf-8",
    )

    curves = cio.loadCurvesFromFile(json_path)
    assert len(curves) == 1
    curve = curves[0]
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert curve.getValueAt(0.5) == pytest.approx(0.5)


def test_loadCurvesFromFile_csv(tmp_path: Path) -> None:
    """Load Curve from a CSV file with a header row."""
    csv_path = tmp_path / "curve.csv"
    csv_path.write_text(
        "Depth,Value\n0,0\n1,1\n",
        encoding="utf-8",
    )

    curves: list[Curve] = cio.loadCurvesFromFile(csv_path)
    assert len(curves) == 1
    curve: Curve = curves[0]
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert curve.getValueAt(0.25) == pytest.approx(0.25)


def test_loadCurvesFromFile_missing_raises(tmp_path: Path) -> None:
    """Missing file should raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        cio.loadCurvesFromFile(missing)


def test_loadCurvesFromFile_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported extension should raise ValueError."""
    p = tmp_path / "curve.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        cio.loadCurvesFromFile(p)


# Test UncertaintyCurve IO functions
def test_loadUncertaintyCurveFromJsonObj_single_curve() -> None:
    """Load an UncertaintyCurve from schema-compatible object."""
    obj = {
        "format": "pyWellSFM.UncertaintyCurveData",
        "version": "1.0",
        "data": {
            "name": "Bathymetry",
            "curves": [
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "Bathymetry",
                        "interpolationMethod": "linear",
                        "data": [
                            {"x": 0, "y": 10},
                            {"x": 1, "y": 20},
                        ],
                    },
                }
            ],
        },
    }

    u = cio.loadUncertaintyCurveFromJsonObj(obj)
    assert u.name == "Bathymetry"
    assert np.array_equal(u.getAbscissa(), np.array([0.0, 1.0]))
    assert np.array_equal(u.getMedianValues(), np.array([10.0, 20.0]))
    # With a single curve, min/max default to the median.
    assert np.array_equal(u.getMinValues(), np.array([10.0, 20.0]))
    assert np.array_equal(u.getMaxValues(), np.array([10.0, 20.0]))


def test_loadUncertaintyCurveFromJsonObj_three_curves_ordered() -> None:
    """Load an UncertaintyCurve from 3 curves."""
    obj = {
        "format": "pyWellSFM.UncertaintyCurveData",
        "version": "1.0",
        "data": {
            "name": "Accommodation",
            "curves": [
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "median",
                        "interpolationMethod": "linear",
                        "data": [{"x": 0, "y": 5}, {"x": 1, "y": 6}],
                    },
                },
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "min",
                        "interpolationMethod": "linear",
                        "data": [{"x": 0, "y": 2}, {"x": 1, "y": 3}],
                    },
                },
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "max",
                        "interpolationMethod": "linear",
                        "data": [{"x": 0, "y": 8}, {"x": 1, "y": 9}],
                    },
                },
            ],
        },
    }

    u = cio.loadUncertaintyCurveFromJsonObj(obj)
    assert u.name == "Accommodation"
    assert np.array_equal(u.getMedianValues(), np.array([5.0, 6.0]))
    assert np.array_equal(u.getMinValues(), np.array([2.0, 3.0]))
    assert np.array_equal(u.getMaxValues(), np.array([8.0, 9.0]))


def test_loadUncertaintyCurveFromJsonObj_abscissa_mismatch_raises() -> None:
    """Test loadUncertaintyCurveFromJsonObj with abscissa mismatch."""
    obj = {
        "format": "pyWellSFM.UncertaintyCurveData",
        "version": "1.0",
        "data": {
            "name": "Any",
            "curves": [
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "median",
                        "interpolationMethod": "linear",
                        "data": [{"x": 0, "y": 1}, {"x": 1, "y": 2}],
                    },
                },
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "min",
                        "interpolationMethod": "linear",
                        "data": [{"x": 0, "y": 0}, {"x": 2, "y": 0}],
                    },
                },
            ],
        },
    }

    with pytest.raises(ValueError, match="abscissa"):
        cio.loadUncertaintyCurveFromJsonObj(obj)


def test_loadUncertaintyCurveFromJsonObj_invalid_format_raises() -> None:
    """Test loadUncertaintyCurveFromJsonObj with invalid format."""
    obj = {
        "format": "pyWellSFM.NotUncertaintyCurve",
        "version": "1.0",
        "data": {"name": "X", "curves": []},
    }

    with pytest.raises(ValueError, match="Invalid uncertainty curve format"):
        cio.loadUncertaintyCurveFromJsonObj(obj)


def test_loadUncertaintyCurveFromCsv_no_header_4_columns_drops_nan_and_sorts(
    tmp_path: Path,
) -> None:
    """Load UncertaintyCurve from a headerless 4-column CSV."""
    csv_path = tmp_path / "bathy_unc.csv"
    csv_path.write_text(
        "0,1,4,20\n1,10,0,20\n0.5,5,0,10\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="UncertaintyCurve CSV must have a header row"
    ):
        cio.loadUncertaintyCurveFromCsv(csv_path)


def test_loadUncertaintyCurveFromCsv_header_two_columns_defaults(
    tmp_path: Path,
) -> None:
    """With only x,y columns, min/max default to median."""
    csv_path = tmp_path / "unc2.csv"
    csv_path.write_text(
        "Depth,Value\n2,20\n1,10\n",
        encoding="utf-8",
    )

    u = cio.loadUncertaintyCurveFromCsv(csv_path)
    assert np.array_equal(u.getAbscissa(), np.array([1.0, 2.0]))
    assert np.array_equal(u.getMedianValues(), np.array([10.0, 20.0]))
    assert np.array_equal(u.getMinValues(), np.array([10.0, 20.0]))
    assert np.array_equal(u.getMaxValues(), np.array([10.0, 20.0]))


def test_loadUncertaintyCurveFromCsv_three_columns_header_max_only(
    tmp_path: Path,
) -> None:
    """With 3 columns, a header containing 'ymax' sets only max values."""
    csv_path = tmp_path / "unc3.csv"
    csv_path.write_text(
        "Depth,median,ymax\n0,10,15\n1,20,30\n",
        encoding="utf-8",
    )

    u = cio.loadUncertaintyCurveFromCsv(csv_path)
    assert np.array_equal(u.getMedianValues(), np.array([10.0, 20.0]))
    # Only ymax provided; ymin defaults to median.
    assert np.array_equal(u.getMinValues(), np.array([10.0, 20.0]))
    assert np.array_equal(u.getMaxValues(), np.array([15.0, 30.0]))


def test_loadUncertaintyCurveFromCsv_invalid_column_count_raises(
    tmp_path: Path,
) -> None:
    """Invalid number of columns should raise ValueError."""
    p = tmp_path / "too_many.csv"
    p.write_text("a,b,c,d,e\n1,2,3,4,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2 to 4 columns"):
        cio.loadUncertaintyCurveFromCsv(p)


def test_loadUncertaintyCurveFromFile_csv(tmp_path: Path) -> None:
    """Load UncertaintyCurve from a CSV file with a header row."""
    csv_path = tmp_path / "unc.csv"
    csv_path.write_text(
        "Depth,median,ymin,ymax\n2,20,15,25\n1,10,5,12\n",
        encoding="utf-8",
    )

    u = cio.loadUncertaintyCurveFromFile(csv_path)
    # For CSV, current implementation uses the 1st column header as curve name.
    assert u.name == "Depth"
    assert np.array_equal(u.getAbscissa(), np.array([1.0, 2.0]))
    assert np.array_equal(u.getMedianValues(), np.array([10.0, 20.0]))
    assert np.array_equal(u.getMinValues(), np.array([5.0, 15.0]))
    assert np.array_equal(u.getMaxValues(), np.array([12.0, 25.0]))


def test_loadUncertaintyCurveFromFile_json(tmp_path: Path) -> None:
    """Load UncertaintyCurve from a JSON file matching schema."""
    json_path = tmp_path / "unc.json"
    json_path.write_text(
        """
{
    \"format\": \"pyWellSFM.UncertaintyCurveData\",
    \"version\": \"1.0\",
    \"data\": {
        \"name\": \"Bathymetry\",
        \"curves\": [
            {
                \"format\": \"pyWellSFM.CurveData\",
                \"version\": \"1.0\",
                \"curve\": {
                    \"xAxisName\": \"Depth\",
                    \"yAxisName\": \"median\",
                    \"interpolationMethod\": \"linear\",
                    \"data\": [
                        {\"x\": 0, \"y\": 10},
                        {\"x\": 1, \"y\": 20}
                    ]
                }
            }
        ]
    }
}
""".strip(),
        encoding="utf-8",
    )

    u = cio.loadUncertaintyCurveFromFile(json_path)
    assert u.name == "Bathymetry"
    assert np.array_equal(u.getAbscissa(), np.array([0.0, 1.0]))
    assert np.array_equal(u.getMedianValues(), np.array([10.0, 20.0]))


def test_loadUncertaintyCurveFromFile_missing_raises(tmp_path: Path) -> None:
    """Missing file should raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        cio.loadUncertaintyCurveFromFile(missing)


def test_loadUncertaintyCurveFromFile_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Unsupported extension should raise ValueError."""
    p = tmp_path / "unc.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        cio.loadUncertaintyCurveFromFile(p)


def test_curve_interpolation_method_branches() -> None:
    """Test of _curve_interpolation_method with different Curve types."""
    linear = _curve_linear()
    assert cio._curve_interpolation_method(linear) == "linear"

    poly = Curve(
        "Depth",
        "Value",
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=float),
        PolynomialInterpolator(),
        deg=3,
        nbPts=5,
    )
    poly_method = cio._curve_interpolation_method(poly)
    assert isinstance(poly_method, dict)
    assert poly_method["name"] == "PolynomialInterpolator"

    lb = Curve(
        "Depth",
        "Value",
        np.array([0.0, 1.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        LowerBoundInterpolator(),
    )
    assert cio._curve_interpolation_method(lb) == "LowerBound"

    ub = Curve(
        "Depth",
        "Value",
        np.array([0.0, 1.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        UpperBoundInterpolator(),
    )
    assert cio._curve_interpolation_method(ub) == "UpperBound"

    lb._interpFunc = object()
    assert cio._curve_interpolation_method(lb) == "LowerBound"


def test_curveToJsonObj_and_save_json_csv_dispatch(tmp_path: Path) -> None:
    """Test curveToJsonObj and saveCurve dispatch with various Curve types."""
    curve = _curve_linear()

    obj = cio.curveToJsonObj(curve)
    assert obj["curve"]["xAxisName"] == "Depth"
    assert obj["curve"]["yAxisName"] == "Value"
    assert len(obj["curve"]["data"]) == 3

    curve._xAxisName = ""
    curve._yAxisName = ""
    obj2 = cio.curveToJsonObj(curve)
    assert obj2["curve"]["xAxisName"] == "Age"
    assert obj2["curve"]["yAxisName"] == "Value"

    bad = _curve_linear()
    bad._abscissa = np.array([0.0, 1.0], dtype=float)
    bad._ordinate = np.array([1.0], dtype=float)
    with pytest.raises(ValueError, match="lengths differ"):
        cio.curveToJsonObj(bad)

    bad2 = _curve_linear()
    bad2._abscissa = np.array([0.0], dtype=float)
    bad2._ordinate = np.array([1.0], dtype=float)
    with pytest.raises(ValueError, match="at least 2 points"):
        cio.curveToJsonObj(bad2)

    json_path = tmp_path / "curve.json"
    csv_path = tmp_path / "curve.csv"

    cio.saveCurve(curve, str(json_path))
    cio.saveCurve(curve, str(csv_path), x_axis_name_default="X")
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["format"] == "pyWellSFM.CurveData"

    with pytest.raises(ValueError, match="Unsupported Curve output extension"):
        cio.saveCurve(curve, str(tmp_path / "curve.txt"))


def test_saveCurveToJson_and_saveCurveToCsv_errors(tmp_path: Path) -> None:
    """Test error branches of saveCurveToJson and saveCurveToCsv."""
    curve = _curve_linear()

    with pytest.raises(ValueError, match=".json extension"):
        cio.saveCurveToJson(curve, str(tmp_path / "curve.csv"))

    with pytest.raises(ValueError, match=".csv extension"):
        cio.saveCurveToCsv(curve, str(tmp_path / "curve.json"))

    bad = _curve_linear()
    bad._abscissa = np.array([], dtype=float)
    bad._ordinate = np.array([], dtype=float)
    with pytest.raises(ValueError, match="same non-zero x/y length"):
        cio.saveCurveToCsv(bad, str(tmp_path / "bad.csv"))


def test_load_single_curve_and_name_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _load_single_curve and name wrappers with various scenarios.

    :param Path tmp_path: path to temporary folder
    :param pytest.MonkeyPatch monkeypatch: pytest fixture for monkeypatching
    """
    c1 = _curve_linear()
    c2 = _curve_linear()
    monkeypatch.setattr(cio, "loadCurvesFromFile", lambda _p: [c1, c2])
    with pytest.raises(ValueError, match="exactly one curve"):
        cio._load_single_curve(str(tmp_path / "x.csv"), kind="Any")

    clear_stored_logs()
    monkeypatch.setattr(
        cio,
        "loadCurvesFromFile",
        lambda _p: [
            Curve(
                "Age",
                "BadName",
                np.array([0.0, 1.0], dtype=float),
                np.array([0.0, 1.0], dtype=float),
                "linear",
            )
        ],
    )

    subs = cio.loadSubsidenceCurve(str(tmp_path / "subs.csv"))
    eust = cio.loadEustaticCurve(str(tmp_path / "eus.csv"))
    assert subs._yAxisName == "Subsidence"
    assert eust._yAxisName == "Eustatism"

    logs = [r["message"] for r in get_stored_logs()]
    assert any("Subsidence curve yAxisName" in msg for msg in logs)
    assert any("Eustatic curve yAxisName" in msg for msg in logs)

    obj = _curve_json_obj()
    subs2 = cio.loadSubsidenceCurveFromJsonObj(obj)
    eust2 = cio.loadEustaticCurveFromJsonObj(obj)
    assert subs2._yAxisName == "Subsidence"
    assert eust2._yAxisName == "Eustatism"


def test_loadCurvesFromCsv_additional_branches(tmp_path: Path) -> None:
    """Test additional branches of loadCurvesFromCsv."""
    with pytest.raises(FileNotFoundError):
        cio.loadCurvesFromCsv(tmp_path / "missing.csv")

    unnamed = tmp_path / "unnamed.csv"
    unnamed.write_text(
        "Unnamed: 0,Unnamed: 1\n0,10\n1,20\n",
        encoding="utf-8",
    )
    curves = cio.loadCurvesFromCsv(unnamed)
    assert curves[0]._xAxisName == "x"
    assert curves[0]._yAxisName == "y1"

    bad_col = tmp_path / "all_bad_col.csv"
    bad_col.write_text("X,Y\n0,aa\n1,bb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one valid numeric"):
        cio.loadCurvesFromCsv(bad_col)


def test_loadCurveFromJsonObj_error_branches() -> None:
    """Test error branches of loadCurveFromJsonObj."""
    with pytest.raises(ValueError, match="'curve' object"):
        cio.loadCurveFromJsonObj(
            {"format": "pyWellSFM.CurveData", "version": "1.0"}
        )

    bad_x = _curve_json_obj()
    bad_x["curve"]["xAxisName"] = ""
    with pytest.raises(ValueError, match="xAxisName"):
        cio.loadCurveFromJsonObj(bad_x)

    bad_y = _curve_json_obj()
    bad_y["curve"]["yAxisName"] = ""
    with pytest.raises(ValueError, match="yAxisName"):
        cio.loadCurveFromJsonObj(bad_y)

    empty_method = _curve_json_obj(interpolation_method="")
    with pytest.raises(ValueError, match="non-empty"):
        cio.loadCurveFromJsonObj(empty_method)

    bad_dict_name = _curve_json_obj(interpolation_method={"name": ""})
    with pytest.raises(ValueError, match="non-empty"):
        cio.loadCurveFromJsonObj(bad_dict_name)

    bad_degree = _curve_json_obj(
        interpolation_method={
            "name": "PolynomialInterpolator",
            "degree": 0,
            "nbPoints": 5,
        }
    )
    with pytest.raises(ValueError, match="degree"):
        cio.loadCurveFromJsonObj(bad_degree)

    bad_nb_points = _curve_json_obj(
        interpolation_method={
            "name": "PolynomialInterpolator",
            "degree": 2,
            "nbPoints": 1,
        }
    )
    with pytest.raises(ValueError, match="nbPoints"):
        cio.loadCurveFromJsonObj(bad_nb_points)

    unsupported_name = _curve_json_obj(interpolation_method={"name": "Foo"})
    with pytest.raises(ValueError, match="Unsupported Curve interpolation"):
        cio.loadCurveFromJsonObj(unsupported_name)

    bad_type = _curve_json_obj(interpolation_method=42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a string or an object"):
        cio.loadCurveFromJsonObj(bad_type)

    data_not_list = _curve_json_obj()
    data_not_list["curve"]["data"] = 1
    with pytest.raises(ValueError, match="must be an array"):
        cio.loadCurveFromJsonObj(data_not_list)

    data_short = _curve_json_obj()
    data_short["curve"]["data"] = [{"x": 0, "y": 0}]
    with pytest.raises(ValueError, match="at least 2 points"):
        cio.loadCurveFromJsonObj(data_short)

    point_not_obj = _curve_json_obj()
    point_not_obj["curve"]["data"] = [1, {"x": 1, "y": 1}]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be an object"):
        cio.loadCurveFromJsonObj(point_not_obj)

    missing_xy = _curve_json_obj()
    missing_xy["curve"]["data"] = [{"x": 0}, {"x": 1, "y": 1}]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must contain 'x' and 'y'"):
        cio.loadCurveFromJsonObj(missing_xy)

    non_numeric = _curve_json_obj()
    non_numeric["curve"]["data"] = [{"x": "a", "y": 0}, {"x": 1, "y": 1}]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be numeric"):
        cio.loadCurveFromJsonObj(non_numeric)

    unique_x_lt_2 = _curve_json_obj()
    unique_x_lt_2["curve"]["data"] = [{"x": 1, "y": 0}, {"x": 1, "y": 1}]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least 2 unique x values"):
        cio.loadCurveFromJsonObj(unique_x_lt_2)


def test_loadCurvesFromFile_and_loadUncertaintyCurveFromFile_json_non_dict(
    tmp_path: Path,
) -> None:
    """Test load curves with JSON files that are not objects."""
    p_curve = tmp_path / "curve.json"
    p_curve.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        cio.loadCurvesFromFile(p_curve)

    p_unc = tmp_path / "unc.json"
    p_unc.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        cio.loadUncertaintyCurveFromFile(p_unc)


def test_loadUncertaintyCurveFromJsonObj_error_branches() -> None:
    """Test error branches of loadUncertaintyCurveFromJsonObj."""
    with pytest.raises(ValueError, match="'data' object"):
        cio.loadUncertaintyCurveFromJsonObj(
            {
                "format": "pyWellSFM.UncertaintyCurveData",
                "version": "1.0",
                "data": [],
            }
        )

    with pytest.raises(ValueError, match="non-empty string"):
        cio.loadUncertaintyCurveFromJsonObj(
            {
                "format": "pyWellSFM.UncertaintyCurveData",
                "version": "1.0",
                "data": {"name": "", "curves": []},
            }
        )

    with pytest.raises(ValueError, match="non-empty array"):
        cio.loadUncertaintyCurveFromJsonObj(
            {
                "format": "pyWellSFM.UncertaintyCurveData",
                "version": "1.0",
                "data": {"name": "U", "curves": []},
            }
        )

    curves4 = [_curve_json_obj() for _ in range(4)]
    with pytest.raises(ValueError, match="at most 3 curves"):
        cio.loadUncertaintyCurveFromJsonObj(_ucurve_obj(curves4))

    with pytest.raises(ValueError, match="must be an object"):
        cio.loadUncertaintyCurveFromJsonObj(_ucurve_obj([1]))  # type: ignore[list-item]

    c_a = _curve_json_obj()
    c_b = _curve_json_obj()
    c_b["curve"]["xAxisName"] = "Age"
    with pytest.raises(ValueError, match="same xAxisName"):
        cio.loadUncertaintyCurveFromJsonObj(_ucurve_obj([c_a, c_b]))

    c1 = _curve_json_obj()
    c1["curve"]["yAxisName"] = "min"
    c2 = _curve_json_obj()
    c2["curve"]["yAxisName"] = "min2"
    with pytest.raises(ValueError, match="multiple 'min' curves"):
        cio.loadUncertaintyCurveFromJsonObj(_ucurve_obj([c1, c2]))

    c3 = _curve_json_obj()
    c3["curve"]["yAxisName"] = "curve_a"
    c4 = _curve_json_obj()
    c4["curve"]["yAxisName"] = "curve_b"
    with pytest.raises(ValueError, match="With 2 curves"):
        cio.loadUncertaintyCurveFromJsonObj(_ucurve_obj([c3, c4]))


def test_loadUncertaintyCurveFromCsv_additional_branches(
    tmp_path: Path,
) -> None:
    """Test additional branches of loadUncertaintyCurveFromCsv."""
    p = tmp_path / "unc_invalid_rows.csv"
    p.write_text("Depth,median\na,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2 valid numeric"):
        cio.loadUncertaintyCurveFromCsv(p)

    p3 = tmp_path / "unc3_ambiguous.csv"
    p3.write_text("Depth,median,band\n0,10,8\n1,20,25\n", encoding="utf-8")
    u3 = cio.loadUncertaintyCurveFromCsv(p3)
    assert np.array_equal(u3.getMinValues(), np.array([8.0, 20.0]))
    assert np.array_equal(u3.getMaxValues(), np.array([10.0, 25.0]))

    p4 = tmp_path / "unc4_swapped.csv"
    p4.write_text(
        "Depth,median,ymax,ymin\n0,10,15,8\n1,20,30,18\n", encoding="utf-8"
    )
    u4 = cio.loadUncertaintyCurveFromCsv(p4)
    assert np.array_equal(u4.getMinValues(), np.array([8.0, 18.0]))
    assert np.array_equal(u4.getMaxValues(), np.array([15.0, 30.0]))

    p_nan = tmp_path / "unc4_nan.csv"
    p_nan.write_text(
        "Depth,median,ymin,ymax\n0,10,NaN,NaN\n1,20,18,22\n", encoding="utf-8"
    )
    u_nan = cio.loadUncertaintyCurveFromCsv(p_nan)
    assert np.array_equal(u_nan.getMinValues(), np.array([10.0, 18.0]))
    assert np.array_equal(u_nan.getMaxValues(), np.array([10.0, 22.0]))
