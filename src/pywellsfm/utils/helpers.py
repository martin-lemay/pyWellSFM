# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt


class Interpolator(Protocol):
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self: Self) -> None:
        """Interpolator parent class for a 1D increasing function."""
        self.x: npt.NDArray[np.float64] = np.empty((0,), dtype=float)
        self.y: npt.NDArray[np.float64] = np.empty((0,), dtype=float)

    def initialize(
        self: Self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> None:
        """Initialize function sampled points.

        Input points are sorted in ascending order.

        :param npt.NDArray[np.float64] x: sampled x coordinates. Must contains at least
            2 elements.
        :param npt.NDArray[np.float64] y: sampled y coordinates. Must contains at least
            2 elements and have the same size as x.
        """
        indices = np.argsort(x)
        self.x = x[indices]
        self.y = y[indices]

    def setAdditionalArgs(self: Self, **args: Any) -> None:
        """Function to add any additional parameters the interpolator would require."""
        pass

    def __call__(self: Self, xx: float) -> float:
        """Get the value at the given coordinate from interpolator.

        :param float xx: input coordinate
        :return float: output value
        """
        raise NotImplementedError


class LinearInterpolator(Interpolator):
    def __init__(self: Self) -> None:
        """Linear interpolator for a 1D increasing function."""
        self.x: npt.NDArray[np.float64] = np.empty((0,), dtype=float)
        self.y: npt.NDArray[np.float64] = np.empty((0,), dtype=float)

    def __str__(self: Self) -> str:
        """Overload of __str__ method.

        :return str: description
        """
        return "Linear interpolator"

    def __call__(self: Self, xx: float) -> float:
        """Get the value at the given coordinate from linear interpolation.

        Returns the first/last value if evaluated point is outside the definition
        domain.

        :param float xx: input coordinate
        :return float: output value
        """
        if xx <= np.min(self.x):
            return self.y[0]
        if xx >= np.max(self.x):
            return self.y[-1]

        # find the interval
        idx = np.searchsorted(self.x, xx) - 1
        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        y0 = self.y[idx]
        y1 = self.y[idx + 1]

        # linear interpolation
        return y0 + (y1 - y0) * (xx - x0) / (x1 - x0)


class PolynomialInterpolator(Interpolator):
    def __init__(self: Self) -> None:
        """Polynomial interpolator for a 1D increasing function."""
        self.x: npt.NDArray[np.float64] = np.empty((0,), dtype=float)
        self.y: npt.NDArray[np.float64] = np.empty((0,), dtype=float)
        self.deg: int = 0
        self.nbPts: int = 0

    def initialize(
        self: Self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        """Initialization method.

        :param npt.NDArray[np.float64] x: sampled x coordinates. Must contains at least
            2 elements.
        :param npt.NDArray[np.float64] y: sampled y coordinates. Must contains at least
            2 elements and have the same size as x.
        :param int deg: polynom degree, defaults to 1
        :param int nbPts: number of points at each side of evaluated value to compute
            the polynom fit. 2*nbPts must be strictly greater than the polynom degree.
            Defaults to 1.
        """
        assert x.size > 1, "x array must contains at least 2 elements."
        assert x.size == y.size, "x and y arrays must have the same size."
        indices = np.argsort(x)
        self.x = x[indices]
        self.y = y[indices]

    def setAdditionalArgs(self: Self, **args: Any) -> None:
        """Set additional inputs for the Interpolator.

        :param Any **args: dictionnary of additional arguments
        """
        self.deg = args["deg"]
        nbPtsTmp: int = args["nbPts"]
        self.nbPts = (
            nbPtsTmp if 2 * nbPtsTmp > self.deg else int(self.deg / 2) + 1
        )

    def __str__(self: Self) -> str:
        """Overload of __str__ method.

        :return str: description
        """
        return "Polynomial interpolator"

    def __call__(self: Self, xx: float) -> float:
        """Get the value at the given coordinate from polynomial interpolation.

        If evaluated point is close to domain boundaries, polynom is fitted using the
        first/last 2*nbPts points. Returns the first/last value if evaluated point is
        outside the definition domain.

        :param float xx: input coordinate
        :return float: output value
        """
        if xx < np.min(self.x):
            return self.y[0]
        if xx > np.max(self.x):
            return self.y[-1]
        # get the indexes
        indices = np.argsort(np.abs(self.x - xx))
        start, end = indices[:2]
        if start > end:
            start, end = end, start
        start -= self.nbPts - 1
        end += self.nbPts
        deg = self.deg
        if start < 0:
            start = 0
            end = start + 2 * self.nbPts
        if end >= self.x.size:
            end = self.x.size - 1
            start = end - 2 * self.nbPts
        p = np.polyfit(self.x[start:end], self.y[start:end], deg)
        return float(np.polyval(p, xx))


# TODO: add function to convoluate step signal

# window_length = 2.5  # metres.

# N = int(window_length / step)
# boxcar = 100 * np.ones(N) / N

# z = np.linspace(start, stop, L.size)
# prop = np.convolve(L, boxcar, mode='same')
