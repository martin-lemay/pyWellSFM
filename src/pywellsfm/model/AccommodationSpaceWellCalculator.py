# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Optional, Self

import numpy as np
import numpy.typing as npt
from striplog import Interval, Striplog

from pywellsfm.model.Curve import Curve, UncertaintyCurve
from pywellsfm.model.Facies import FaciesCriteria, SedimentaryFacies
from pywellsfm.model.Marker import Marker
from pywellsfm.model.Well import Well


class AccommodationSpaceWellCalculator:
    def __init__(
        self: Self,
        well: Well,
        faciesList: list[SedimentaryFacies],
    ) -> None:
        """Class to compute accommodation space curve.

        Accommodation space curve represents the variation of accommodation from the
        base of the start of a sequence.

        :param Well well: input well
        :param list[SedimentaryFacies] faciesList: list of sedimentary facies with
            depositional conditions to get the bathymetry from facies log.
        """
        #: input well
        self._well: Well = well
        #: dictionary of facies with environment conditions
        self._faciesDict: dict[str, SedimentaryFacies] = {
            facies.name: facies for facies in faciesList
        }
        self._bathymetryComputed: bool = False
        self._accommodationComputed: bool = False
        #: bathymetry per interval
        self._bathymetryStepCurve: Optional[npt.NDArray[np.float64]] = None
        #: output bathymetry curve with uncertainties
        self.bathymetryCurve: UncertaintyCurve
        #: accommodation per interval
        self._accommodationStepCurve: Optional[npt.NDArray[np.float64]] = None
        #: output accommodation variation curve with uncertainties
        self.accommodationChangeCurve: UncertaintyCurve
        #: output cummulative accommodation curve with uncertainties from base to top
        self.accommodationCurve: UncertaintyCurve
        #: epsilon for depth around the markers
        self._eps: float = 0.001  # 1mm
        self._initCurves()

    def _initCurves(self: Self) -> None:
        abscissa = np.array([0.0, self._well.depth])
        ordinate = np.full_like(abscissa, np.nan)
        self.bathymetryCurve = UncertaintyCurve(
            "Bathymetry", Curve("Depth", "Bathymetry", abscissa, ordinate)
        )
        self.accommodationChangeCurve = UncertaintyCurve(
            "AccommodationChange",
            Curve("Depth", "AccommodationChange", abscissa, ordinate),
        )
        self.accommodationCurve = UncertaintyCurve(
            "Accommodation", Curve("Depth", "Accommodation", abscissa, ordinate)
        )

    def computeAccommodationCurve(
        self: Self,
        faciesLogName: str,
        fromMarker: Optional[Marker] = None,
        toMarker: Optional[Marker] = None,
        accommodationAtBase: float = 0.0,
    ) -> UncertaintyCurve:
        """Compute accommodation space along the well.

        :param str faciesLogName: name of the sedimentary facies log
        :param float step: step between continuous log samples
        :param Marker fromMarker: base marker where to start calculation. If no
            marker is given, calculation starts from the base of the well.
            Defaults to None.
        :param Marker toMarker: to marker where to stop calculation. If no marker is
            given, calculation stops at the top of the well. Defaults to None.
        :param float accommodationAtBase: accommodation at the base marker.
            Defaults to 0.
        :return UncertaintyCurve: accommodation curve
        """
        assert isinstance(self._well.getDepthLog(faciesLogName), Striplog), (
            f"The discrete log {faciesLogName} does not exist in the well "
            + f"{self._well.name}."
        )
        faciesLog: Striplog = self._well.getDepthLog(faciesLogName)
        baseDepth: float = (
            fromMarker.depth if fromMarker is not None else faciesLog.stop.z
        )
        topDepth: float = toMarker.depth if toMarker is not None else faciesLog.start.z

        # compute bathymetry curve if it is not defined
        if self._bathymetryStepCurve is None:
            self.computeBathymetryCurve(faciesLogName, fromMarker, toMarker)

        # Accommodation array: depth, acco min 1, acco min 2, acco max 1, acco max 2
        accoArray: npt.NDArray[np.float64] = self._computeAccommodationArray(
            faciesLog, baseDepth, topDepth, accommodationAtBase
        )
        # compute as min=max(min1, min2), max = min(max1, max2), mean from min and max
        for row in accoArray:
            depth = row[0]
            accoMin = np.max(row[1:3])
            accoMax = np.min(row[3:])
            accoMed = np.mean((accoMin, accoMax))
            self.accommodationCurve.addSampledPoint(depth, accoMed, accoMin, accoMax)

        return self.accommodationCurve

    def _computeAccommodationArray(
        self: Self,
        faciesLog: Striplog,
        baseDepth: float,
        topDepth: float,
        accommodationAtBase: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Compute apparent accommodation space array along the well.

        Array is composed of as many rows as the number of limits of interval
        (i.e., len(faciesLog) + 1) and columns are:

        * depth where accommodation is computed
        * minimum accommodation computed from the bathymetry right below the depth
        * minimum accommodation computed from the bathymetry right above the depth
        * maximum accommodation computed from the bathymetry right below the depth
        * maximum accommodation computed from the bathymetry right above the depth

        :param str faciesLog: sedimentary facies log
        :param float baseDepth: depth where to start calculation.
        :param float topDepth: depth to stop calculation.
        :param float accommodationAtBase: cummulative accommodation at the base depth.
            Defaults to 0.
        :return npt.NDArray[np.float64]: accommodation array
        """
        # compute bathymetry curve if it is not defined
        if self._bathymetryStepCurve is None:
            self._computeBathymetryStepCurve(faciesLog, baseDepth, topDepth)

        # get bathymetry at the base and computation depth
        depthBase: float = baseDepth
        bathyAtBase: float = 0.0
        lastIndex: int = len(faciesLog)
        for row in self._bathymetryStepCurve[::-1]:
            if row[0] > baseDepth:
                lastIndex -= 1
                continue
            bathyAtBase = row[2:]
            depthBase = row[0]
            break

        assert np.isfinite(bathyAtBase[0]), "Bathymetry at the base is undefined."

        # accommodation array: depth, acco min 1, acco min 2, acco max 1, acco max 2
        accoArray: npt.NDArray[np.float64] = np.full(
            (self._bathymetryStepCurve.shape[0] + 1, 5), np.nan
        )
        interval: Interval
        for i, interval in enumerate(faciesLog):
            # skip stratas below the base
            if interval.base.z > depthBase:
                continue
            # bathymetry of the interval
            bathyInterval = self._bathymetryStepCurve[i, 2:]
            # compute accommodation at top
            thickness: float
            if i == 0:
                thickness = depthBase - interval.top.z
                # bathymetry of the interval
                bathyInterval = self._bathymetryStepCurve[i, 2:]
                # compute accommodation from bathy of the interval
                acco0: tuple[float, float] = self._computeAccommodationValue(
                    thickness, bathyAtBase, bathyInterval
                )
                # store the results
                accoArray[i, 0] = interval.top.z

                # min accommodation
                accoArray[i, 1:3] = (acco0[0], acco0[0])
                # max accommodation
                accoArray[i, 3:] = (acco0[1], acco0[1])

            # compute accommodation at the base
            thickness = depthBase - interval.base.z

            # bathymetry of above interval
            bathyAboveInterval = bathyInterval
            if (i < len(faciesLog) - 1) and (i < lastIndex):
                bathyAboveInterval = self._bathymetryStepCurve[i + 1, 2:]

            # compute accommodation
            # computed from bathy of the interval
            print(bathyAtBase, bathyInterval)
            acco1: tuple[float, float] = self._computeAccommodationValue(
                thickness, bathyAtBase, bathyInterval
            )
            # computed from the bathy of the facies above
            acco2: tuple[float, float] = self._computeAccommodationValue(
                thickness, bathyAtBase, bathyAboveInterval
            )

            # store the results
            accoArray[i + 1, 0] = interval.base.z
            # min accommodation
            accoArray[i + 1, 1:3] = (acco1[0], acco2[0])
            # max accommodation
            accoArray[i + 1, 3:] = (acco1[1], acco2[1])

        accoArray[-1, 1:] = 0.0

        # add initial accommodation value
        accoArray[:, 1:] += accommodationAtBase
        return accoArray

    def _computeAccommodationValue(
        self: Self,
        thickness: float,
        bathyBase: tuple[float, float],
        bathyTop: tuple[float, float],
    ) -> tuple[float, float]:
        """Compute the accommodation according to thickness and bathymetry ranges.

        :param float thickness: interval thickness
        :param tuple[float, float] bathyBase: bathymetry at the base of the interval
        :param tuple[float, float] bathyTop: bathymetry at the top of the interval
        :return tuple[float, float]: accommodation variation from base to top
        """
        # minimum bathymetry variation: consider bathy is max at base marker and
        #  min at top marker
        deltaBathyMin: float = bathyTop[0] - bathyBase[1]
        # maximum bathymetry variation: consider bathy is min at base marker and
        # max at top marker
        deltaBathyMax: float = bathyTop[1] - bathyBase[0]
        accoMin: float = thickness + deltaBathyMin
        accoMax: float = thickness + deltaBathyMax
        if accoMin > accoMax:
            accoMin, accoMax = accoMax, accoMin
        return accoMin, accoMax

    def computeAccommodationCurve0(
        self: Self,
        faciesLogName: str,
        fromMarker: Marker = None,
        toMarker: Marker = None,
    ) -> UncertaintyCurve:
        """Compute accommodation space along the well.

        :param str faciesLogName: name of the sedimentary facies log
        :param float step: step between continuous log samples
        :param Marker fromMarker: base marker where to start calculation. If no
            marker is given, calculation starts from the base of the well.
            Defaults to None.
        :param Marker toMarker: to marker where to stop calculation. If no marker is
            given, calculation stops at the top of the well. Defaults to None.
        :return UncertaintyCurve: accommodation curve
        """
        assert isinstance(self._well.getDepthLog(faciesLogName), Striplog), (
            f"The discrete log {faciesLogName} does not exist in the well "
            + f"{self._well.name}."
        )
        faciesLog: Striplog = self._well.getDepthLog(faciesLogName)
        baseDepth: float = (
            fromMarker.depth if fromMarker is not None else faciesLog.stop.z
        )
        topDepth: float = toMarker.depth if toMarker is not None else faciesLog.start.z
        # compute accommodation step curve
        if self._accommodationStepCurve is None:
            self._computeAccommodationStepCurve(faciesLog, baseDepth, topDepth)
        # store uncertainty accommodation change curve
        self._convertIntervalCurve2UncertaintyCurve(
            self._accommodationStepCurve, self.accommodationChangeCurve
        )

        # compute cumulative accommodation
        accommodationCumulStepCurve = np.copy(self._accommodationStepCurve)
        for i in (2, 3):
            accommodationCumulStepCurve[:, i][::-1] = np.cumsum(
                accommodationCumulStepCurve[:, i][::-1]
            )
        self._convertIntervalCurve2UncertaintyCurve(
            accommodationCumulStepCurve, self.accommodationCurve
        )
        return self.accommodationCurve

    def computeBathymetryCurve(
        self: Self,
        faciesLogName: str,
        fromMarker: Marker = None,
        toMarker: Marker = None,
    ) -> UncertaintyCurve:
        """Compute the bathymetry along the well.

        :param str faciesLogName: name of the sedimentary facies log
        :param float step: step between continuous log samples
        :param Marker fromMarker: base marker where to start calculation. If no
            marker is given, calculation starts from the base of the well.
            Defaults to None.
        :param Marker toMarker: to marker where to stop calculation. If no marker is
            given, calculation stops at the top of the well. Defaults to None.
        :return UncertaintyCurve: bathymetry curve
        """
        assert isinstance(self._well.getDepthLog(faciesLogName), Striplog), (
            f"The discrete log {faciesLogName} does not exist in the well "
            + f"{self._well.name}."
        )
        faciesLog: Striplog = self._well.getDepthLog(faciesLogName)
        baseDepth: float = (
            fromMarker.depth if fromMarker is not None else faciesLog.stop.z
        )
        topDepth: float = toMarker.depth if toMarker is not None else faciesLog.start.z
        self._computeBathymetryStepCurve(faciesLog, baseDepth, topDepth)

        self._convertIntervalCurve2UncertaintyCurve(
            self._bathymetryStepCurve, self.bathymetryCurve
        )
        self._bathymetryComputed = True
        return self.bathymetryCurve

    def _getBathymetryRangeFromFaciesName(
        self: Self, faciesName: str
    ) -> tuple[float, float]:
        """Get the bathymetry range from the facies name.

        :param str faciesName: facies name
        :raises ValueError: if the facies name is not in the list or the bathymetry
            conditions is undefined for a given facies.
        :return tuple[float, float]: bathymetry minimum and maximum values.
        """
        facies: SedimentaryFacies = self._faciesDict.get(faciesName, None)
        if facies is None:
            raise ValueError(
                f"Facies {faciesName} is not in the facies list. "
                + "Bathymetry curve cannot be computed."
            )
        bathyRange: FaciesCriteria = facies.getEnvironmentCondition("Bathymetry")
        if bathyRange is None:
            raise ValueError(
                f"Bathymetry is undefined for the facies {faciesName}. "
                + "Bathymetry curve cannot be computed."
            )
        return (bathyRange.minRange, bathyRange.maxRange)

    def _computeBathymetryStepCurve(
        self: Self,
        faciesLog: Striplog,
        baseDepth: float,
        topDepth: float,
    ) -> npt.NDArray[np.float64]:
        self._bathymetryStepCurve = np.full((len(faciesLog), 4), np.nan)
        # add epsilon because if Interval.completely_contains returns True only if
        # limits are not equal
        eps: float = 1e-6
        computedInterval = Interval(topDepth - eps, baseDepth + eps)
        interval: Interval
        nbIntervals: int = 0
        for i, interval in enumerate(faciesLog):
            # ..WARNING:: assume that interval coordinates are in MD
            if not computedInterval.completely_contains(interval):
                continue
            faciesName: str = interval.primary["lithology"]
            bathRange = self._getBathymetryRangeFromFaciesName(faciesName)
            self._bathymetryStepCurve[i, 0] = interval.base.z
            self._bathymetryStepCurve[i, 1] = interval.top.z
            self._bathymetryStepCurve[i, 2:] = bathRange
            nbIntervals += 1

        return self._bathymetryStepCurve

    def _computeAccommodationStepCurve(
        self: Self, faciesLog: Striplog, baseDepth: float, topDepth: float
    ) -> npt.NDArray[np.float64]:
        # compute bathymetry per interval if not computed yet
        if self._bathymetryStepCurve is None:
            self._computeBathymetryStepCurve(faciesLog, baseDepth, topDepth)

        self._accommodationStepCurve = np.full_like(self._bathymetryStepCurve, np.nan)
        interval: Interval
        for i, interval in enumerate(faciesLog):
            # bathymetry at the top of the interval
            (bathyMinEnd, bathyMaxEnd) = self._bathymetryStepCurve[i, 2:]
            # bathymetry at the top of the interval (=base of above interval, except for
            # the last interval where we assume no variations)
            bathyMinStart, bathyMaxStart = bathyMinEnd, bathyMaxEnd
            if i > 0:  # self._bathymetryStepCurve.shape[0] - 1:
                (bathyMinStart, bathyMaxStart) = self._bathymetryStepCurve[i - 1, 2:]
            # minimum bathymetry variation: consider bathy is max at bottom interval and
            #  min at current interval
            deltaBathyMin: float = bathyMinEnd - bathyMaxStart
            # minimum bathymetry variation: consider bathy is min at bottom interval and
            # max at current interval
            deltaBathyMax: float = bathyMaxEnd - bathyMinStart
            accoMin: float = interval.thickness + deltaBathyMin
            accoMax: float = interval.thickness + deltaBathyMax
            if accoMin > accoMax:
                accoMin, accoMax = accoMax, accoMin
            self._accommodationStepCurve[i, :2] = self._bathymetryStepCurve[i, :2]
            self._accommodationStepCurve[i, 2:] = (accoMin, accoMax)
        return self._accommodationStepCurve

    def _convertIntervalCurve2UncertaintyCurve(
        self: Self,
        stepCurve: npt.NDArray[np.float64],
        uncertaintyCurve: UncertaintyCurve,
    ) -> None:
        for row in stepCurve:
            med: float = (row[2] + row[3]) / 2.0
            uncertaintyCurve.addSampledPoint(row[0] - self._eps, med, row[2], row[3])
            uncertaintyCurve.addSampledPoint(row[1] + self._eps, med, row[2], row[3])
