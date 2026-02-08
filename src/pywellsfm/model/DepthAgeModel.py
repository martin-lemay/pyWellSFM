# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from pywellsfm.model.Curve import Curve
from pywellsfm.model.Marker import Marker

# ..WARNING:: consider only 1 depth per age (no stratigraphic layer duplication)


class DepthAgeModel:
    def __init__(
        self: Self, interpolationMethod: str | Any = "linear", **args: Any
    ) -> None:
        """Defines depth-age model and conversion methods.

        : param str | Any, optional interpolationMethod: Interpolation methods between
            time markers. Defaults to "linear".
        : param dict[Any, Any] **args: any other arguments.
        """
        self._xaxisName: str = "Age"
        self._yaxisName: str = "Depth"
        self.depthAgeCurve: Curve | None = None
        self.markers: set[Marker] = set()
        self.interpolationMethod: str | Any = interpolationMethod
        self.interpArgs: Any = args

    def setMarkers(self: Self, markers: set[Marker]) -> None:
        """Set time markers.

        : param set[Marker] markers: list of markers
        """
        ageDepths = np.array(
            [(marker.age, marker.depth) for marker in markers]
        )
        self.updateCurve(ageDepths)

    def updateCurve(self: Self, ageDepths: npt.NDArray[np.float64]) -> None:
        """Update depth-age curve.

        : param npt.NDArray[np.float64] ageDepths: array of shape (n,2) with age in
            first column and depth in second column.
        """
        self.depthAgeCurve = Curve(
            self._xaxisName,
            self._yaxisName,
            ageDepths[:, 0],
            ageDepths[:, 1],
            self.interpolationMethod,
            **self.interpArgs,
        )

    def addMarker(self: Self, marker: Marker) -> None:
        """Add a single time marker.

        : param Marker marker: marker
        """
        if self.depthAgeCurve is not None:
            self.depthAgeCurve.addSampledPoint(marker.age, marker.depth)
        else:
            self.setMarkers(
                {
                    marker,
                }
            )

    def getDepth(self: Self, age: float) -> float:
        """Get the depth from a given age.

        : param float age: input age

        : return float: output depth
        """
        if self.depthAgeCurve is None:
            return np.nan
        return self.depthAgeCurve(age)

    def getAge(self: Self, depth: float) -> tuple[float] | None:
        """Get the age(s) from a given depth.

        A given depth may correspond to multiple ages.

        : param float depth: input depth
        : return tuple[float]: output ages.
        """
        # if exact depth in markers, check for multiple age values
        if self.depthAgeCurve is None:
            return None
        diff: npt.NDArray[np.bool] = (
            np.abs(self.depthAgeCurve._ordinate - depth) < 0.001
        )
        if any(diff):
            return tuple(self.depthAgeCurve._ordinate[diff.nonzero()])

        # else interpolation of the age from markers
        df: pd.DataFrame = self.depthAgeCurve.toDataFrame()
        invCurve = Curve(
            self._yaxisName,
            self._xaxisName,
            df[self._yaxisName].to_numpy(),
            df[self._xaxisName].to_numpy(),
            self.interpolationMethod,
            **self.interpArgs,
        )
        return (invCurve(depth),)

    def convertContinuousLogToDepth(self: Self, inputLog: Curve) -> Curve:
        return None

    def convertContinuousLogToAge(self: Self, inputLog: Curve) -> Curve:
        return None

    def convertDiscreteLogToDepth(self: Self, inputLog: Curve) -> Curve:
        return None

    def convertDiscreteLogToAge(self: Self, inputLog: Curve) -> Curve:
        return None
