# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.interpolate as interp

# class ExtrapolationMethod(StrEnum):
#     def _constant(at, abcissa, ordinate):
#         return ordinate[0] if at < np.min(abcissa) else ordinate[-1]

#     #: use the first/last value of the array
#     CONSTANT = _constant
#     #: mirror the array
#     MIRROR = "Mirror"
#     #: use linear interpolation from the 2 first/last values of the array
#     PROPAGATE = "Propagate"
#     #: no extrapolation, return nan if outside of the array
#     NONE = "None"


class Curve:
    def __init__(
        self: Self,
        xAxisName: str,
        yAxisName: str,
        abscissa: npt.NDArray[np.float64],
        ordinate: npt.NDArray[np.float64],
        interpolationFunction: str | Any = "linear",
        **args: Any,
    ) -> None:
        """Defines a curve by a list of abscissa and a list of ordinate coordinates.

        :param str xAxisName: x axis name
        :param str yAxisName: y axis name
        :param npt.NDArray[np.float64] abscissa: abscissa values
        :param npt.NDArray[np.float64] ordinate: ordinate values
        :param str | "function" interpolationFunction: name of interpolation method
            according to scipy.interpolate.interp1d method, defaults to "linear", or
            a class inherited from `pywellsfm.utils.helpers.Interpolator` to compute
            the interpolation (see for instance
            `pywellsfm.utils.helpers.PolynomialInterpolator`)
        """
        self._xAxisName: str = xAxisName
        self._yAxisName: str = yAxisName
        self._abscissa: npt.NDArray[np.float64] = abscissa
        self._ordinate: npt.NDArray[np.float64] = ordinate
        self._interpFunc: Any
        if type(interpolationFunction) is str:
            self._interpFunc = interp.interp1d(
                self._abscissa,
                self._ordinate,
                kind=interpolationFunction,
                bounds_error=False,
            )
        else:
            self._interpFunc = interpolationFunction
            self._interpFunc.initialize(self._abscissa, self._ordinate)
            self._interpFunc.setAdditionalArgs(**args)
        # self._extrapMethod: ExtrapolationMethod = extrapolationMethod
        self._minAbscissa = np.nan
        self._maxAbscissa = np.nan
        self._updateBounds()

        self._call = self.__class__.getValueAt

    def __repr__(self: Self) -> str:
        """Repr method.

        :return str: repr string
        """
        return str(np.column_stack((self._abscissa, self._ordinate)))

    def _updateBounds(self: Self) -> None:
        """Update minimum and maximum abscissa."""
        self._minAbscissa = np.min(self._abscissa)
        self._maxAbscissa = np.max(self._abscissa)

    def __call__(self, at: float) -> float:
        """Get the value at the given coordinate.

        :param float at: input coordinate
        :return float: output value
        """
        return self.getValueAt(at)

    # TODO: add unit test
    def setSampledPoints(
        self: Self,
        abscissa: npt.NDArray[np.float64],
        ordinate: npt.NDArray[np.float64],
    ) -> None:
        """Set abscissa and ordinate values of sampled points.

        :param npt.NDArray[np.float64] abscissa: abscissa values
        :param npt.NDArray[np.float64] ordinate: ordinate values
        """
        self._abscissa = abscissa
        self._ordinate = ordinate
        self._updateBounds()

    # TODO: add unit test
    def addSampledPoint(
        self: Self, x: float, y: float, tol: float = 1e-6
    ) -> None:
        """Add a sampled point to the curve.

        :param float x: abscissa value
        :param float y: ordinate value
        :param float tol: tolerance value to determine if x is already in the array of
            abscissa. Defaults to 1e-6.
        """
        # set the value to existing abscissa if it exists
        if self._getIndexOfX(x, tol) >= 0:
            self.setValueAt(x, y, tol)
            return
        # add the new x value
        self._abscissa = np.append(self._abscissa, [x])
        sortIndexes = np.argsort(self._abscissa)
        self._abscissa = np.take_along_axis(self._abscissa, sortIndexes)
        self._ordinate = np.take_along_axis(
            np.append(self._ordinate, [y]), sortIndexes
        )
        self._updateBounds()

    # TODO: add unit test
    def setValueBetween(
        self: Self, xmin: float, xmax: float, y: float, tol: float = 1e-6
    ) -> None:
        """Set a constant value between xmin and xmax abscissa.

        :param float xmin: minimum abscissa value
        :param float xmax: maximum abscissa value
        :param float y: ordinate value
        :param float tol: tolerance value to determine if x is already in the array of
            abscissa. Defaults to 1e-6.
        """
        if self._getIndexOfX(xmin, tol) < 0:
            self.addSampledPoint(xmin, y, tol)
        if self._getIndexOfX(xmax, tol) < 0:
            self.addSampledPoint(xmax, y, tol)
        for index in np.nonzero(
            (self._abscissa > xmin) and (self._abscissa < xmax)
        ):
            self._ordinate[index] = y

    def _getIndexOfX(self: Self, x: float, tol: float = 1e-6) -> int:
        """Get the index of x is in the abscissa values.

        :param float x: value to check for
        :param float tol: Tolerance, defaults to 1e-6.
        :return int: index of x in abscissa array, or -1 if absent.
        """
        diff: npt.NDArray[np.bool_] = np.abs(self._abscissa - x) < tol
        if np.any(diff):
            return int(np.nonzero(diff)[0])
        return -1

    def setValueAt(self: Self, x: float, y: float, tol: float = 1e-6) -> None:
        """Set the value at the given abscissa.

        :param float x: abscissa value
        :param float y: ordinate value
        :param float tol: tolerance value to determine if x is already in the array of
            abscissa. Defaults to 1e-6.
        """
        index: int = self._getIndexOfX(x, tol)
        if index < 0:
            raise IndexError(
                f"Input abscissa {x} is not in the array of sampled points."
            )
        self._ordinate[index] = y

    def getValueAt(self: Self, at: float) -> float:
        """Get the value at the given coordinate.

        Returns the first/last value if input coordinate is outside of the domain the
        curve is defined.

        :param float at: input coordinate
        :return float: output value
        """
        if at < self._minAbscissa:
            return float(self._ordinate[0])
        if at > self._maxAbscissa:
            return float(self._ordinate[-1])
        return self._interpFunc(at)

    # TODO: to add unit test
    def toDataFrame(
        self: Self,
        fromX: float = -np.inf,
        toX: float = np.inf,
        dx: float = np.inf,
        columnNames: tuple[str, str] = ("", ""),
    ) -> pd.DataFrame:
        """Create a DataFrame from the curve.

        By default the dataframe contains the curve over the whole definition domain
        sampled by the same number of points as the curve.

        :param float fromX: starting abscissa, defaults to -inf
        :param float toX: end abscissa, defaults to +inf
        :param float dx: sampling step, defaults to +inf
        :param tuple[str, str] columnNames: names of dataframe columns,
            defaults to ("x", "y")
        :return pd.DataFrame: 2 column dataframe
        """
        fromX0 = fromX if np.isfinite(fromX) else self._minAbscissa
        toX0 = toX if np.isfinite(toX) else self._maxAbscissa
        assert fromX0 < toX0, "Start abscissa must be lower than end abscissa."
        nx = (
            int((toX0 - fromX0) / dx) + 1
            if np.isfinite(dx)
            else self._abscissa.size
        )
        assert dx > 0, "Sampling step must be strictly positive."
        lx = np.linspace(fromX0, toX0, nx)
        ly = [self.getValueAt(x) for x in lx]
        columnNames0 = list(columnNames)
        if len(columnNames[0]) == 0:
            columnNames0[0] = self._xAxisName
        if len(columnNames[1]) == 0:
            columnNames0[1] = self._yAxisName
        return pd.DataFrame(np.column_stack((lx, ly)), columns=columnNames0)

    # def _extrapolateValueAt(self: Self, at: float) -> float:
    #     assert (at < self._minAbscissa) | (at > self._maxAbscissa), (
    #         "Input abscissa must be lower (or higher) than the mnimum (or maximum)"
    #         + " for extrapolation."
    #     )
    #     match self._extrapMethod:
    #         case ExtrapolationMethod.CONSTANT:
    #             return (
    #                 self._ordinate[0] if at < self._minAbscissa else self._ordinate[-1]
    #             )
    #         case ExtrapolationMethod.MIRROR:
    #             return np.nan
    #         case ExtrapolationMethod.PROPAGATE:
    #             return np.nan
    #         case ExtrapolationMethod.NONE:
    #             return np.nan

    def copy(self: Self) -> Self:
        """Create a copy of self.

        Return Curve: a copy of the current Curve object.
        """
        interpFunc = (
            self._interpFunc
            if not isinstance(self._interpFunc, interp.interp1d)
            else "linear"
        )
        return self.__class__(
            self._xAxisName,
            self._yAxisName,
            np.copy(self._abscissa),
            np.copy(self._ordinate),
            interpFunc,
        )


class AccumulationCurve(Curve):
    def __init__(
        self: Self,
        envFactorName: str,
        abscissa: npt.NDArray[np.float64],
        ordinate: npt.NDArray[np.float64],
    ) -> None:
        """Defines a accumulation curve.

        An accumulation curve defines the reduction coefficients according environment
        conditions.
        The curve uses a linear interpolation function between given points.

        :param str envFactorName: environmental factor name
        :param npt.NDArray[np.float64] abscissa: abscissa values
        :param npt.NDArray[np.float64] ordinate: ordinate values. Must be between 0 and
            1.
        :param str | "function" interpolationFunction: name of interpolation method
            according to scipy.interpolate.interp1d method, defaults to "linear", or
            a class inherited from `pywellsfm.utils.helpers.Interpolator` to compute
            the interpolation (see for instance
            `pywellsfm.utils.helpers.PolynomialInterpolator`)
        """
        assert (np.min(ordinate) >= 0.0) and (np.max(ordinate) <= 1.0), (
            "Accumulation curve ordinates must be between 0 and 1."
        )
        super().__init__(
            envFactorName, "ReductionCoeff", abscissa, ordinate, "linear"
        )


# TODO: add unit tests
class UncertaintyCurve:
    def __init__(
        self: Self,
        curveName: str,
        curve: Curve,
    ) -> None:
        """Defines a curve associated to an uncertainty range.

        :param str curveName: curve name
        :param Curve curve: median curve.
        """
        self.name: str = curveName
        self._medianCurve: Curve
        self._minCurve: Curve
        self._maxCurve: Curve
        self.setCurve(curve)

    def getAbscissa(self: Self) -> npt.NDArray[np.float64]:
        """Get abscissa values of the curves.

        :return npt.NDArray[np.float64]: abscissa array
        """
        assert self._medianCurve is not None, "Median curve is undefined."
        return self._medianCurve._abscissa

    def getMedianValues(self: Self) -> npt.NDArray[np.float64]:
        """Get abscissa values of the curves.

        :return npt.NDArray[np.float64]: abscissa array
        """
        assert self._medianCurve is not None, "Median curve is undefined."
        return self._medianCurve._ordinate

    def getMinValues(self: Self) -> npt.NDArray[np.float64]:
        """Get abscissa values of the curves.

        :return npt.NDArray[np.float64]: abscissa array
        """
        assert self._minCurve is not None, "Median curve is undefined."
        return self._minCurve._ordinate

    def getMaxValues(self: Self) -> npt.NDArray[np.float64]:
        """Get abscissa values of the curves.

        :return npt.NDArray[np.float64]: abscissa array
        """
        assert self._maxCurve is not None, "Median curve is undefined."
        return self._maxCurve._ordinate

    def setCurve(
        self: Self,
        curve: Curve,
    ) -> None:
        """Set the median curve.

        : param Curve (optional) curve: median curve.
        : raises TypeError: if input object is not of type Curve.
        """
        if not isinstance(curve, Curve):
            raise TypeError("Input curve must be of type Curve")
        self._medianCurve = curve
        self._minCurve = curve.copy()
        self._maxCurve = curve.copy()

    def setMinCurveValues(self: Self, values: npt.NDArray[np.float64]) -> None:
        """Set the minimum curve values.

        :param npt.NDArray[np.float64] values: minimum values
        """
        assert self._minCurve is not None, "Min curve is undefined."
        assert values.size == self._minCurve._abscissa.size
        self._minCurve._ordinate = values

    def setMaxCurveValues(self: Self, values: npt.NDArray[np.float64]) -> None:
        """Set the maximum curve values.

        :param npt.NDArray[np.float64] values: maximum values
        """
        assert self._maxCurve is not None, "Max curve is undefined."
        assert values.size == self._maxCurve._abscissa.size
        self._maxCurve._ordinate = values

    def setSampledPoints(
        self: Self,
        abscissa: npt.NDArray[np.float64],
        ordinate: npt.NDArray[np.float64],
        setAsMinValues: bool = False,
        setAsMaxValues: bool = False,
    ) -> None:
        """Set abscissa and ordinate values of sampled points.

        The ordinate is set to the median curve, but it can also be set to minimum and
        maximum curves by setting setAsMinValues and setAsMaxValues to True
        respectively.

        :param npt.NDArray[np.float64] abscissa: abscissa values
        :param npt.NDArray[np.float64] ordinate: ordinate values
        :param bool setAsMinValues: if True, set ordinate array to minimum curve.
            Defaults to False.
        :param bool setAsMaxValues: if True, set ordinate array to maximum curve.
            Defaults to False.
        """
        self._medianCurve.setSampledPoints(abscissa, ordinate)
        ordinate2 = np.full_like(ordinate, np.nan)
        if setAsMinValues:
            self._minCurve.setSampledPoints(abscissa, ordinate)
        else:
            self._minCurve.setSampledPoints(abscissa, ordinate2)
        if setAsMaxValues:
            self._maxCurve.setSampledPoints(abscissa, ordinate)
        else:
            self._maxCurve.setSampledPoints(abscissa, ordinate2)

    def addSampledPoint(
        self: Self,
        x: float,
        y: float,
        ymin: float = np.nan,
        ymax: float = np.nan,
    ) -> None:
        """Add a sampled point to the curves.

        If no values are given for ymin and/or ymax, y value is set to minimum and
        maximum curves.

        :param float x: abscissa value
        :param float y: median value
        :param float ymin: minimum value. Defaults to np.nan.
        :param float ymax: maximum value. Defaults to np.nan.
        """
        self._medianCurve.addSampledPoint(x, y)
        ymin1: float = ymin if np.isfinite(ymin) else y
        self._minCurve.addSampledPoint(x, ymin1)
        ymax1: float = ymax if np.isfinite(ymax) else y
        self._maxCurve.addSampledPoint(x, ymax1)

    def getRangeAt(self: Self, abscissa: float) -> tuple[float, float, float]:
        """Get the range of values for a given abscissa.

        :param float abscissa: input abscissa
        :return tuple[float, float, float]: tuple containing the minimum, median and
            maximum values
        """
        return (
            self._minCurve(abscissa),
            self._medianCurve(abscissa),
            self._maxCurve(abscissa),
        )
