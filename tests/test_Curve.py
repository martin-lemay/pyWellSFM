# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import os
import sys

from pywellsfm.utils.interpolation import (
    LowerBoundInterpolator,
)

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)
print(sys.path)

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pytest
from scipy.interpolate import interp1d

from pywellsfm.model import AccumulationCurve, Curve, UncertaintyCurve
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
interpFuncPolynomial = PolynomialInterpolator()
interpFuncLowerBound = LowerBoundInterpolator()

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
coordsPolynom = np.column_stack((np.array(lxx), lyy2))

# lower bound "interpolation"
lyy3 = (
    4.335269,
    3.845454,
    3.191186,
    3.475562,
    1.861709,
    1.241210,
    4.725100,
    5.781522,
    2.304078,
    0.0,
)
coordsLowerBound = np.column_stack((np.array(lxx), lyy3))

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
        interpFuncPolynomial,
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


def test_Curve_init3() -> None:
    """Test of Curve class."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate)
    assert curve._xAxisName == xAxisName, "X axis name is wrong"
    assert curve._yAxisName == yAxisName, "Y axis name is wrong"
    assert np.array_equal(curve._abscissa, abscissa), (
        "Curve abscissa array is wrong"
    )
    assert np.array_equal(curve._ordinate, ordinate), (
        "Curve ordinate array is wrong"
    )
    assert isinstance(curve._interpFunc, LowerBoundInterpolator), (
        "Default interpolation method is wrong."
    )


def test_Curve_toDataFrame_default() -> None:
    """Test of toDataFrame method."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFuncPolynomial,
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
        interpFuncPolynomial,
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


@pytest.mark.parametrize("xx, yy", coordsPolynom)
def test_Curve_compute3(xx: float, yy: float) -> None:
    """Test of Curve class."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFuncPolynomial,
        deg=polyDeg,
        nbPts=polyNbPts,
    )
    assert abs(curve(xx) - yy) < eps


@pytest.mark.parametrize("xx, yy", coordsLowerBound)
def test_Curve_compute4(xx: float, yy: float) -> None:
    """Test of Curve class."""
    curve = Curve(
        xAxisName,
        yAxisName,
        abscissa,
        ordinate,
        interpFuncLowerBound,
        deg=polyDeg,
        nbPts=polyNbPts,
    )
    print(f"xx={xx}, yy={yy}, curve(xx)={curve(xx)}")
    assert abs(curve(xx) - yy) < eps


def test_Curve_init_invalid_interpolation_method() -> None:
    """Raise wrapped ValueError when scipy interp1d fails."""
    bad_abscissa = np.array([0.0, 1.0, 2.0])
    bad_ordinate = np.array([0.0, 1.0])
    with pytest.raises(
        ValueError,
        match="Invalid interpolation method: linear.",
    ):
        Curve(xAxisName, yAxisName, bad_abscissa, bad_ordinate, "linear")


def test_Curve_repr_matches_numpy_stack() -> None:
    """Return repr matching stacked abscissa and ordinate."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate)
    expected = str(np.column_stack((abscissa, ordinate)))
    assert repr(curve) == expected


def test_Curve_setSampledPoints_updates_arrays_and_bounds() -> None:
    """Update sampled arrays and cached bounds."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate)
    new_abscissa = np.array([-2.0, 0.0, 5.0])
    new_ordinate = np.array([10.0, 20.0, 30.0])
    curve.setSampledPoints(new_abscissa, new_ordinate)
    assert np.array_equal(curve._abscissa, new_abscissa)
    assert np.array_equal(curve._ordinate, new_ordinate)
    assert curve._minAbscissa == -2.0
    assert curve._maxAbscissa == 5.0


def test_Curve_setValueAt_raises_for_unknown_x() -> None:
    """Raise when setting ordinate for absent abscissa."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate)
    with pytest.raises(
        IndexError,
        match="Input abscissa 999.0 is not in the array of sampled points.",
    ):
        curve.setValueAt(999.0, 1.0)


def test_Curve_getValueAt_above_domain_returns_last() -> None:
    """Return last ordinate value above curve domain."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate)
    assert curve.getValueAt(curve._maxAbscissa + 1.0) == curve._ordinate[-1]


def test_Curve_setValueBetween_raises_on_vector_boolean() -> None:
    """Raise ValueError from invalid ndarray boolean operation."""
    curve = Curve(xAxisName, yAxisName, abscissa, ordinate)
    with pytest.raises(ValueError, match="truth value of an array"):
        curve.setValueBetween(0.25, 1.25, 8.0)


def test_Curve_addSampledPoint_updates_existing_x() -> None:
    """Update existing x without changing sampled array size."""
    curve = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    curve.addSampledPoint(1.0, 9.0)
    assert curve._abscissa.size == 2
    assert curve._ordinate[1] == 9.0


def test_Curve_getIndexOfX_returns_present_index() -> None:
    """Return exact sampled index when x exists."""
    curve = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    assert curve._getIndexOfX(1.0) == 1


def test_Curve_getValueAt_below_domain_returns_first() -> None:
    """Return first ordinate value below curve domain."""
    curve = Curve("x", "y", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    assert curve.getValueAt(-1.0) == 2.0


def test_Curve_setValueBetween_updates_single_point_with_large_tol() -> None:
    """Update point through setValueBetween when scalar boolean is valid."""
    curve = Curve("x", "y", np.array([0.5]), np.array([1.0]))
    curve.setValueBetween(0.0, 1.0, 7.0, tol=1.0)
    assert curve._ordinate[0] == 7.0


def test_Curve_toDataFrame_raises_on_inverted_bounds() -> None:
    """Raise when start abscissa is not lower than end abscissa."""
    curve = Curve("x", "y", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    with pytest.raises(
        ValueError,
        match="Start abscissa must be lower than end abscissa.",
    ):
        curve.toDataFrame(fromX=1.0, toX=1.0)


def test_Curve_toDataFrame_raises_on_non_positive_dx() -> None:
    """Raise when sampling step is non-positive."""
    curve = Curve("x", "y", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    with pytest.raises(
        ValueError,
        match="Sampling step must be strictly positive.",
    ):
        curve.toDataFrame(fromX=0.0, toX=1.0, dx=-1.0)


def test_UncertaintyCurve_setCurve_rejects_non_curve() -> None:
    """Reject non-Curve object when setting median curve."""
    base = Curve(xAxisName, yAxisName, abscissa, ordinate)
    unc = UncertaintyCurve("u", base)
    with pytest.raises(TypeError, match="Input curve must be of type Curve"):
        unc.setCurve("not_a_curve")  # type: ignore[arg-type]


def test_UncertaintyCurve_getMedianValues_raises_without_curve() -> None:
    """Raise when median curve is missing."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    unc._medianCurve = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Median curve is undefined."):
        unc.getMedianValues()


def test_UncertaintyCurve_getAbscissa_raises_without_curve() -> None:
    """Raise when median abscissa curve is missing."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    unc._medianCurve = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Median curve is undefined."):
        unc.getAbscissa()


def test_UncertaintyCurve_getMinValues_raises_without_curve() -> None:
    """Raise when minimum curve is missing."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    unc._minCurve = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Min curve is undefined."):
        unc.getMinValues()


def test_UncertaintyCurve_getMaxValues_raises_without_curve() -> None:
    """Raise when maximum curve is missing."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    unc._maxCurve = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Max curve is undefined."):
        unc.getMaxValues()


def test_UncertaintyCurve_setMinCurveValues_raises_on_wrong_size() -> None:
    """Raise when min values size does not match abscissa size."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    with pytest.raises(
        ValueError,
        match="Values size must match min curve abscissa size.",
    ):
        unc.setMinCurveValues(np.array([1.0]))


def test_UncertaintyCurve_setMaxCurveValues_raises_on_wrong_size() -> None:
    """Raise when max values size does not match abscissa size."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    with pytest.raises(
        ValueError,
        match="Values size must match max curve abscissa size.",
    ):
        unc.setMaxCurveValues(np.array([1.0]))


def test_UncertaintyCurve_setMinCurveValues_raises_without_curve() -> None:
    """Raise when minimum curve is missing during update."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    unc._minCurve = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Min curve is undefined."):
        unc.setMinCurveValues(np.array([1.0, 2.0]))


def test_UncertaintyCurve_setMaxCurveValues_raises_without_curve() -> None:
    """Raise when maximum curve is missing during update."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    unc._maxCurve = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Max curve is undefined."):
        unc.setMaxCurveValues(np.array([1.0, 2.0]))


def test_UncertaintyCurve_addSampledPoint_defaults_to_median() -> None:
    """Use median value as default bounds when ymin and ymax are NaN."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    unc = UncertaintyCurve("u", base)
    unc.addSampledPoint(2.0, 6.0)
    assert unc.getMedianValues()[-1] == 6.0
    assert unc.getMinValues()[-1] == 6.0
    assert unc.getMaxValues()[-1] == 6.0


def test_UncertaintyCurve_setSampledPoints_default_nan_bounds() -> None:
    """Set median values and keep min/max values as NaN by default."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    x = np.array([0.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 5.0])
    unc.setSampledPoints(x, y)
    assert np.array_equal(unc.getAbscissa(), x)
    assert np.array_equal(unc.getMedianValues(), y)
    assert np.isnan(unc.getMinValues()).all()
    assert np.isnan(unc.getMaxValues()).all()


def test_UncertaintyCurve_setSampledPoints_sets_min_and_max() -> None:
    """Set min and max arrays when flags are enabled."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    unc = UncertaintyCurve("u", base)
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([2.0, 4.0, 6.0])
    unc.setSampledPoints(x, y, setAsMinValues=True, setAsMaxValues=True)
    assert np.array_equal(unc.getMinValues(), y)
    assert np.array_equal(unc.getMaxValues(), y)


def test_UncertaintyCurve_getRangeAt_returns_min_median_max() -> None:
    """Return triplet from min, median and max curves at abscissa."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    unc = UncertaintyCurve("u", base)
    unc.setMinCurveValues(np.array([1.0, 3.0]))
    unc.setMaxCurveValues(np.array([3.0, 5.0]))
    assert unc.getRangeAt(0.5) == (2.0, 2.0, 2.0)


def test_UncertaintyCurve_addSampledPoint_uses_explicit_bounds() -> None:
    """Add point using explicit minimum and maximum values."""
    base = Curve("x", "y", np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    unc = UncertaintyCurve("u", base)
    unc.addSampledPoint(2.0, 6.0, ymin=5.0, ymax=7.0)
    assert unc.getMedianValues()[-1] == 6.0
    assert unc.getMinValues()[-1] == 5.0
    assert unc.getMaxValues()[-1] == 7.0


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
        ValueError,
        match="Accumulation curve ordinates must be between 0 and 1.",
    ):
        AccumulationCurve(prodName, prodAbscissa, prodOrdinate)


if __name__ == "__main__":
    pytest.main(
        [
            os.path.abspath(__file__),
        ]
    )
