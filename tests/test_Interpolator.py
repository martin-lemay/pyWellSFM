# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

import os
import sys

m_path = os.path.join(os.getcwd(), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)
print(sys.path)

import random as rd
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pytest

from pywellsfm.utils import LinearInterpolator, PolynomialInterpolator

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
def test_PolynomialInterpolator_Parabol(deg, nbPts) -> None:
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


@pytest.mark.parametrize("testCase", __generate_test_data_polynomial())
def test_PolynomialInterpolator_compute(testCase: TestCase) -> None:
    """Test of PolynomialInterpolator.compute varying degree and number of points."""
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
