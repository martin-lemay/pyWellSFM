# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import os
import sys
from pathlib import Path

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)
print(sys.path)

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pytest
from scipy.interpolate import interp1d

from pywellsfm.io.curve_io import (
    loadCurveFromJsonObj,
    loadCurvesFromCsv,
    loadCurvesFromFile,
    loadUncertaintyCurveFromCsv,
    loadUncertaintyCurveFromFile,
    loadUncertaintyCurveFromJsonObj,
)
from pywellsfm.model import AccumulationCurve, Curve
from pywellsfm.utils import PolynomialInterpolator

# Test data
xAxisName: str = "Depth"
yAxisName: str = "Eustatism"
xmax = 2.0 * np.pi
nbPtsFunc: int = 101
abscissa = np.linspace(0, xmax, nbPtsFunc).astype(float)
ordinate = abscissa + 3 * np.sin(abscissa)

interpMethods: tuple[str, ...] = (
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
)
interpFunc = PolynomialInterpolator()

polyDeg = 3
polyNbPts = 5

# sin(x) between -1 and 1, so lxx between 0 and 2pi
lxx = xmax / 2.0 * (1 + np.sin(np.linspace(1, 99, 10)))
lyy = (
    4.352241,
    3.936097,
    3.298400,
    3.655341,
    1.872659,
    1.323882,
    4.716201,
    5.883149,
    2.272586,
    0.009962,
)

prec = 6
eps = pow(10, -prec)

# polynomial interpolation
lyy2 = (
    4.352095,
    3.937236,
    3.299449,
    3.654990,
    1.872134,
    1.324300,
    4.717376,
    5.882994,
    2.272068,
    0.009948,
)
coords = np.array(lxx)
coords = np.column_stack((coords, lyy2))

# Test Curve class


@pytest.mark.parametrize("interpMethod", interpMethods)
def test_Curve_init(interpMethod: str) -> None:
    """Test of Curve class."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate, interpMethod)
    assert curve._xAxisName == xAxisName, "X axis name is wrong"
    assert curve._yAxisName == yAxisName, "Y axis name is wrong"
    assert np.array_equal(curve._abscissa, abscissa), (
        "Curve abscissa array is wrong"
    )
    assert np.array_equal(curve._ordinate, ordinate), (
        "Curve ordinate array is wrong"
    )
    assert type(curve._interpFunc) is interp1d, "Interpolation method is wrong"


def test_Curve_init2() -> None:
    """Test of Curve class."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFunc,
        deg=polyDeg,
        nbPts=polyNbPts,
    )
    assert curve._xAxisName == xAxisName, "X axis name is wrong"
    assert curve._yAxisName == yAxisName, "Y axis name is wrong"
    assert np.array_equal(curve._abscissa, abscissa), (
        "Curve abscissa array is wrong"
    )
    assert np.array_equal(curve._ordinate, ordinate), (
        "Curve ordinate array is wrong"
    )
    assert type(curve._interpFunc) is PolynomialInterpolator, (
        "Interpolation method is wrong"
    )


def test_Curve_toDataFrame_default() -> None:
    """Test of toDataFrame method."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFunc,
        deg=polyDeg,
        nbPts=polyNbPts,
    )
    df = curve.toDataFrame()
    assert df.columns.to_list() == [xAxisName, yAxisName], (
        "DataFrame columns are wrong."
    )
    print(df)
    assert df.shape[0] == nbPtsFunc, "Number of points is wrong."
    assert (df[xAxisName].min() - np.min(abscissa)) < eps, "Minimum is wrong."
    assert (df[xAxisName].max() - np.max(abscissa)) < eps, "Maximum is wrong."


def test_Curve_toDataFrame() -> None:
    """Test of toDataFrame method."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFunc,
        deg=polyDeg,
        nbPts=polyNbPts,
    )
    df = curve.toDataFrame(
        fromX=1.0, toX=4.0, dx=0.1, columnNames=("Depth", "Subsidence")
    )
    assert df.columns.to_list() == ["Depth", "Subsidence"], (
        "DataFrame columns are wrong."
    )
    assert abs((df["Depth"][1] - df["Depth"][0]) - 0.1) < eps, (
        "Number of points is wrong."
    )
    assert abs(df["Depth"].min() - 1.0) < eps, "Minimum is wrong."
    assert abs(df["Depth"].max() - 4.0) < eps, "Maximum is wrong."


@dataclass(frozen=True)
class TestCase:
    """Test case."""

    __test__ = False
    xx: float
    yy: float
    interpMethod: Any


def __generate_test_data() -> Iterator[TestCase]:
    """Generate test cases.

    Yields:
        Iterator[ TestCase ]: iterator on test cases
    """
    for xx, yy in zip(lxx, lyy, strict=True):
        for interpMethod in interpMethods:
            yield TestCase(xx, yy, interpMethod)


@pytest.mark.parametrize("testCase", __generate_test_data())
def test_Curve_compute1(testCase: TestCase) -> None:
    """Test of Curve class."""
    curve = Curve(
        xAxisName, yAxisName, abscissa, ordinate, testCase.interpMethod
    )
    interp = interp1d(abscissa, ordinate, testCase.interpMethod)
    assert abs(curve.getValueAt(testCase.xx) - interp(testCase.xx)) < eps


@pytest.mark.parametrize("testCase", __generate_test_data())
def test_Curve_compute2(testCase: TestCase) -> None:
    """Test of Curve class."""
    curve = Curve(
        xAxisName, yAxisName, abscissa, ordinate, testCase.interpMethod
    )
    interp = interp1d(abscissa, ordinate, testCase.interpMethod)
    assert abs(curve(testCase.xx) - interp(testCase.xx)) < eps


@pytest.mark.parametrize("xx, yy", coords)
def test_Curve_compute3(xx: float, yy: float) -> None:
    """Test of Curve class."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFunc,
        deg=polyDeg,
        nbPts=polyNbPts,
    )
    assert abs(curve(xx) - yy) < eps


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

    curves = loadCurvesFromCsv(csv_path)
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

    curves = loadCurvesFromCsv(csv_path)
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

    curves = loadCurvesFromCsv(csv_path)
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
        loadCurvesFromCsv(csv_path)


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

    curve = loadCurveFromJsonObj(obj)
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
            "interpolationMethod": {"degree": 3, "nbPoints": 5},
            "data": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
                {"x": 2.0, "y": 4.0},
                {"x": 3.0, "y": 9.0},
                {"x": 4.0, "y": 16.0},
            ],
        },
    }

    curve = loadCurveFromJsonObj(obj)
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert isinstance(curve._interpFunc, PolynomialInterpolator)
    assert curve._interpFunc.deg == 3
    assert curve._interpFunc.nbPts >= 5


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

    curves = loadCurvesFromFile(json_path)
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

    curves: list[Curve] = loadCurvesFromFile(csv_path)
    assert len(curves) == 1
    curve: Curve = curves[0]
    assert curve._xAxisName == "Depth"
    assert curve._yAxisName == "Value"
    assert curve.getValueAt(0.25) == pytest.approx(0.25)


def test_loadCurvesFromFile_missing_raises(tmp_path: Path) -> None:
    """Missing file should raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        loadCurvesFromFile(missing)


def test_loadCurvesFromFile_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported extension should raise ValueError."""
    p = tmp_path / "curve.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        loadCurvesFromFile(p)


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

    u = loadUncertaintyCurveFromJsonObj(obj)
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

    u = loadUncertaintyCurveFromJsonObj(obj)
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
        loadUncertaintyCurveFromJsonObj(obj)


def test_loadUncertaintyCurveFromJsonObj_invalid_format_raises() -> None:
    """Test loadUncertaintyCurveFromJsonObj with invalid format."""
    obj = {
        "format": "pyWellSFM.NotUncertaintyCurve",
        "version": "1.0",
        "data": {"name": "X", "curves": []},
    }

    with pytest.raises(ValueError, match="Invalid uncertainty curve format"):
        loadUncertaintyCurveFromJsonObj(obj)


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
        loadUncertaintyCurveFromCsv(csv_path)


def test_loadUncertaintyCurveFromCsv_header_two_columns_defaults(
    tmp_path: Path,
) -> None:
    """With only x,y columns, min/max default to median."""
    csv_path = tmp_path / "unc2.csv"
    csv_path.write_text(
        "Depth,Value\n2,20\n1,10\n",
        encoding="utf-8",
    )

    u = loadUncertaintyCurveFromCsv(csv_path)
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

    u = loadUncertaintyCurveFromCsv(csv_path)
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
        loadUncertaintyCurveFromCsv(p)


def test_loadUncertaintyCurveFromFile_csv(tmp_path: Path) -> None:
    """Load UncertaintyCurve from a CSV file with a header row."""
    csv_path = tmp_path / "unc.csv"
    csv_path.write_text(
        "Depth,median,ymin,ymax\n2,20,15,25\n1,10,5,12\n",
        encoding="utf-8",
    )

    u = loadUncertaintyCurveFromFile(csv_path)
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

    u = loadUncertaintyCurveFromFile(json_path)
    assert u.name == "Bathymetry"
    assert np.array_equal(u.getAbscissa(), np.array([0.0, 1.0]))
    assert np.array_equal(u.getMedianValues(), np.array([10.0, 20.0]))


def test_loadUncertaintyCurveFromFile_missing_raises(tmp_path: Path) -> None:
    """Missing file should raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        loadUncertaintyCurveFromFile(missing)


def test_loadUncertaintyCurveFromFile_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Unsupported extension should raise ValueError."""
    p = tmp_path / "unc.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        loadUncertaintyCurveFromFile(p)


# Test AccumulationCurve class
def test_AccumulationCurve_init1() -> None:
    """Test of Curve class."""
    prodName = "Bathymetry"
    prodAbscissa = np.array([0.0, 10.0, 50.0])
    prodOrdinate = np.array([0.0, 1.0, 0.0])
    curve = AccumulationCurve(prodName, prodAbscissa, prodOrdinate)
    assert curve._xAxisName == prodName, "X axis name is wrong"
    assert curve._yAxisName == "ReductionCoeff", "Y axis name is wrong"
    assert np.array_equal(curve._abscissa, prodAbscissa), (
        "Curve abscissa array is wrong"
    )
    assert np.array_equal(curve._ordinate, prodOrdinate), (
        "Curve ordinate array is wrong"
    )
    assert curve._interpFunc._kind == "linear", "Interpolation method is wrong"


def test_AccumulationCurve_init2() -> None:
    """Test of Curve class."""
    prodName = "Bathymetry"
    prodAbscissa = np.array([0.0, 10.0, 50.0])
    prodOrdinate = np.array([0.0, 2.0, 0.0])
    with pytest.raises(
        AssertionError,
        match="Accumulation curve ordinates must be between 0 and 1.",
    ):
        AccumulationCurve(prodName, prodAbscissa, prodOrdinate)


if __name__ == "__main__":
    pytest.main(
        [
            os.path.abspath(__file__),
        ]
    )
