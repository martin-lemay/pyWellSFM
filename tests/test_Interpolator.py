# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import os
import sys

m_path = os.path.join(os.getcwd(), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

import random as rd
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pytest

from pywellsfm.utils import (
    LinearInterpolator,
    LowerBoundInterpolator,
    PolynomialInterpolator,
    UpperBoundInterpolator,
)
from pywellsfm.utils.interpolation import Interpolator

nbPtsFunc: int = 101
nbDeg: int = 5
nbNbPts: int = 10
xmax = 2.0 * np.pi

# x from 0 to 2pi
x = np.linspace(0, xmax, nbPtsFunc)

# sin(x) between -1 and 1, so lxx between 0 and 2pi
lxx = xmax / 2.0 * (1 + np.sin(np.linspace(0, 100, nbDeg * nbNbPts)))


dataFile = os.path.join(
    os.path.dirname(__file__), "data/test_PolynomialInterpolatorExp.txt"
)

prec = 6

with open(dataFile, "r") as fin:
    line = fin.readline()
    values = line.split(",")
lyy = [float(val) for val in values]
eps = pow(10, -prec)


@dataclass(frozen=True)
class TestCase:
    """Test case."""

    __test__ = False
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    deg: int
    nbPts: int
    xx: float
    yy: float


def __generate_test_data_polynomial() -> Iterator[TestCase]:
    """Generate test cases for polynomial interpolation.

    Yields:
        Iterator[ TestCase ]: iterator on test cases
    """
    i = -1
    y = 5 * x + np.sin(x)
    for deg in np.arange(nbDeg):
        for nbPts in np.arange(nbNbPts):
            i += 1
            yield TestCase(x, y, int(deg), int(nbPts), lxx[i], lyy[i])


@pytest.mark.parametrize("testCase", __generate_test_data_polynomial())
def test_LinearInterpolator_init(testCase: TestCase) -> None:
    """Test of LinearInterpolator class."""
    interp = LinearInterpolator()
    interp.initialize(testCase.x, testCase.y)
    assert np.array_equal(interp.x, testCase.x), "x array is wrong"
    assert np.array_equal(interp.y, testCase.y), "y array is wrong"


def test_LinearInterpolator_compute() -> None:
    """Test of LinearInterpolator.compute method."""
    xmin, xmax = -10, 10
    lx = np.linspace(xmin, xmax, 101)
    interp = LinearInterpolator()
    interp.initialize(lx, lx)
    for _ in range(10):
        xx = xmin + (xmax - xmin) * float(rd.randrange(100)) / 100.0
        yy = np.round(interp(xx), prec)
        yyExp = np.round(xx, prec)
        assert abs(yy - yyExp) < eps, "Evaluated value is wrong."


def test_LinearInterpolator_outside_domain_returns_edges() -> None:
    """Linear interpolator clamps values outside the domain."""
    interp = LinearInterpolator()
    interp.initialize(np.array([0.0, 1.0]), np.array([10.0, 20.0]))

    assert interp(-0.1) == 10.0
    assert interp(1.1) == 20.0


def test_LowerBoundInterpolator_compute_and_str() -> None:
    """Lower bound interpolator handles boundaries and formatting."""
    interp = LowerBoundInterpolator()
    interp.initialize(np.array([0.0, 1.0, 2.0]), np.array([2.0, 4.0, 6.0]))

    assert str(interp) == "Lower bound interpolator"
    assert interp(-1.0) == 2.0
    assert interp(1.2) == 4.0


def test_UpperBoundInterpolator_compute_and_str() -> None:
    """Upper bound interpolator handles boundaries and formatting."""
    interp = UpperBoundInterpolator()
    interp.initialize(np.array([0.0, 1.0, 2.0]), np.array([2.0, 4.0, 6.0]))

    assert str(interp) == "Upper bound interpolator"
    assert interp(3.0) == 6.0
    assert interp(0.4) == 4.0


@pytest.mark.parametrize("testCase", __generate_test_data_polynomial())
def test_PolynomialInterpolator_init(testCase: TestCase) -> None:
    """Test of PolynomialInterpolator class."""
    interp = PolynomialInterpolator()
    interp.initialize(testCase.x, testCase.y)
    interp.setAdditionalArgs(deg=testCase.deg, nbPts=testCase.nbPts)
    assert np.array_equal(interp.x, testCase.x), "x array is wrong"
    assert np.array_equal(interp.y, testCase.y), "y array is wrong"
    assert np.array_equal(interp.deg, testCase.deg), "deg is wrong"
    nbPts = (
        testCase.nbPts
        if 2 * testCase.nbPts > testCase.deg
        else int(testCase.deg / 2) + 1
    )
    assert np.array_equal(interp.nbPts, nbPts), "nbPts is wrong"


@pytest.mark.parametrize("deg, nbPts", ((1, 3), (2, 5), (3, 8), (4, 10)))
def test_PolynomialInterpolator_Parabol(deg: int, nbPts: int) -> None:
    """Test of PolynomialInterpolator.compute method."""
    xmin, xmax = -10, 10
    lx = np.linspace(xmin, xmax, 101)
    ly = lx**deg
    interp = PolynomialInterpolator()
    interp.initialize(lx, ly)
    interp.setAdditionalArgs(deg=deg, nbPts=nbPts)
    for _ in range(10):
        xx = xmin + (xmax - xmin) * float(rd.randrange(100)) / 100.0
        yy = np.round(interp(xx), prec)
        yyExp = np.round(xx**deg, prec)
        assert abs(yy - yyExp) < eps, "Evaluated value is wrong."


def test_PolynomialInterpolator_boundary_and_str_paths() -> None:
    """Polynomial interpolator covers boundary paths and string output."""
    lx = np.linspace(0.0, 10.0, 11)
    ly = lx**2
    interp = PolynomialInterpolator()
    interp.initialize(lx, ly)
    interp.setAdditionalArgs(deg=2, nbPts=3)

    assert str(interp) == "Polynomial interpolator"
    assert interp(-0.01) == ly[0]
    assert interp(10.01) == ly[-1]
    assert np.isfinite(interp(0.1))
    assert np.isfinite(interp(9.9))


def test_Interpolator_base_methods() -> None:
    """Protocol helper methods sort inputs and raise by default."""
    interp = LinearInterpolator()
    x = np.array([2.0, 0.0, 1.0])
    y = np.array([20.0, 0.0, 10.0])

    Interpolator.initialize(interp, x, y)
    Interpolator.setAdditionalArgs(interp, custom=1)

    assert interp.name == "LinearInterpolator"
    assert np.array_equal(interp.x, np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(interp.y, np.array([0.0, 10.0, 20.0]))
    with pytest.raises(NotImplementedError):
        Interpolator.__call__(interp, 1.0)  # type: ignore[abstract]


@pytest.mark.parametrize(
    "InterpolatorClass",
    [
        LinearInterpolator,
        LowerBoundInterpolator,
        UpperBoundInterpolator,
        PolynomialInterpolator,
    ],
)
def test_initialize_caches_x_bounds(InterpolatorClass: type) -> None:
    """All interpolators cache _x_min and _x_max after initialize."""
    interp = InterpolatorClass()
    x = np.array([3.0, 1.0, 2.0])
    y = np.array([30.0, 10.0, 20.0])
    interp.initialize(x, y)

    assert interp._x_min == 1.0
    assert interp._x_max == 3.0


def test_cached_bounds_match_unsorted_input() -> None:
    """Cached bounds reflect the sorted array, not the input order."""
    interp = LinearInterpolator()
    x = np.array([5.0, -2.0, 10.0, 0.0])
    y = np.array([50.0, -20.0, 100.0, 0.0])
    interp.initialize(x, y)

    assert interp._x_min == -2.0
    assert interp._x_max == 10.0
    assert interp.x[0] == -2.0
    assert interp.x[-1] == 10.0


@pytest.mark.parametrize("testCase", __generate_test_data_polynomial())
def test_PolynomialInterpolator_compute(testCase: TestCase) -> None:
    """Test of PolynomialInterpolator varying degree and number of points."""
    interp = PolynomialInterpolator()
    interp.initialize(testCase.x, testCase.y)
    interp.setAdditionalArgs(deg=testCase.deg, nbPts=testCase.nbPts)
    yy = np.round(interp(testCase.xx), prec)
    assert abs(yy - testCase.yy) < eps, "Evaluated value is wrong."


if __name__ == "__main__":
    pytest.main(
        [
            os.path.abspath(__file__),
        ]
    )
