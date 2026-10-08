# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from enum import StrEnum
from typing import Optional, Self, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import lsq_linear
from striplog import Interval, Striplog

from .Curve import Curve, UncertaintyCurve
from .Facies import FaciesCriteria, SedimentaryFacies
from .Marker import Marker
from .Well import Well


class AccommodationBoundaryRule(StrEnum):
    """Rule to combine accommodation estimates at facies boundaries.

    At each boundary between two facies intervals, accommodation is estimated
    twice: from the water depth range of the interval below and from the
    water depth range of the interval above the boundary.
    """

    #: combined span of both estimates (min of minimums, max of maximums).
    #: Conservative: it does not assume that the water depth at the boundary
    #: lies in both facies ranges.
    UNION = "union"
    #: overlap of both estimates (max of minimums, min of maximums). It
    #: assumes that the water depth at the boundary lies in both facies
    #: ranges; the range may become empty when they do not overlap.
    INTERSECTION = "intersection"


class AccommodationEstimateMethod(StrEnum):
    """Method to compute the best estimate of accommodation in its range.

    The accommodation range at each facies boundary is set by the water
    depth ranges of the facies and by the boundary rule. The best estimate is
    the median curve of the output accommodation curve.
    """

    #: middle of the accommodation range at each boundary. Simple, but jumps
    #: at every facies change since each facies has its own range.
    MIDPOINT = "midpoint"
    #: most likely water depth of the facies (mode of the ``waterDepth``
    #: criteria, middle of the range for a uniform distribution). At each
    #: boundary, the modes of both facies are averaged and limited to the
    #: water depth range of the boundary.
    MODE = "mode"
    #: smoothest accommodation curve inside the range. Water depth at each
    #: boundary is estimated in the water depth range of the facies by
    #: penalizing the curvature of accommodation and the distance to the
    #: facies mode, while the water depth at the base is a single datum
    #: shared by the whole curve.
    SMOOTH = "smooth"


def _combineRanges(
    ranges: npt.NDArray[np.float64], boundaryRule: AccommodationBoundaryRule
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Combine the two ranges estimated at each facies boundary.

    :param npt.NDArray[np.float64] ranges: array with columns min from the
        interval, min from the adjacent interval, max from the interval, max
        from the adjacent interval
    :param AccommodationBoundaryRule boundaryRule: combination rule
    :return tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: minimum
        and maximum values
    """
    if boundaryRule == AccommodationBoundaryRule.UNION:
        return (
            np.minimum(ranges[:, 0], ranges[:, 1]),
            np.maximum(ranges[:, 2], ranges[:, 3]),
        )
    return (
        np.maximum(ranges[:, 0], ranges[:, 1]),
        np.minimum(ranges[:, 2], ranges[:, 3]),
    )


class AccommodationSpaceWellCalculator:
    def __init__(
        self: Self,
        well: Well,
        faciesList: list[SedimentaryFacies],
    ) -> None:
        """Class to compute accommodation space curve.

        Accommodation space curve represents the variation of accommodation
        from the base of the start of a sequence. Apparent accommodation at
        a depth z is the deposited thickness since the base plus the change
        of water depth:

        .. math::

            A(z) = h(z) + w(z) - w_0

        where water depth :math:`w` is only known as the range of the facies
        (``waterDepth`` criteria). The output curve is an
        :class:`UncertaintyCurve`:

        * minimum and maximum curves bracket the accommodation; their width
          is set by the water depth ranges of the facies and by the
          :class:`AccommodationBoundaryRule`;
        * the median curve is the best estimate inside this range, computed
          with the :class:`AccommodationEstimateMethod`.

        Example::

            calculator = AccommodationSpaceWellCalculator(well, faciesList)
            curve = calculator.computeAccommodationCurve(
                "lithology",
                estimateMethod=AccommodationEstimateMethod.SMOOTH,
                smoothingLength=10.0,
            )

        :param Well well: input well
        :param list[SedimentaryFacies] faciesList: list of sedimentary facies
            with depositional conditions to get the waterDepth from facies log.
        """
        #: input well
        self._well: Well = well
        #: dictionary of facies with environment conditions
        self._faciesDict: dict[str, SedimentaryFacies] = {
            facies.name: facies for facies in faciesList
        }
        self._waterDepthComputed: bool = False
        self._accommodationComputed: bool = False
        #: waterDepth per interval
        self._waterDepthStepCurve: Optional[npt.NDArray[np.float64]] = None
        #: output waterDepth curve with uncertainties
        self.waterDepthCurve: UncertaintyCurve
        #: accommodation per interval
        self._accommodationStepCurve: Optional[npt.NDArray[np.float64]] = None
        #: output accommodation variation curve with uncertainties
        self.accommodationChangeCurve: UncertaintyCurve
        #: output cummulative accommodation curve with uncertainties
        self.accommodationCurve: UncertaintyCurve
        #: water depth ranges at each facies boundary: depth, min from the
        #: interval, min from the adjacent interval, max from the interval,
        #: max from the adjacent interval
        self._boundaryWaterDepthArray: Optional[npt.NDArray[np.float64]] = None
        #: water depth range at the base of the accommodation calculation
        self._baseWaterDepth: tuple[float, float] = (np.nan, np.nan)
        #: most likely water depth (mode) per interval of the facies log
        self._waterDepthModes: Optional[npt.NDArray[np.float64]] = None
        #: water depth modes at each facies boundary: mode of the interval,
        #: mode of the adjacent interval
        self._boundaryWaterDepthModeArray: Optional[
            npt.NDArray[np.float64]
        ] = None
        #: most likely water depth at the base of the calculation
        self._baseWaterDepthMode: float = np.nan
        #: epsilon for depth around the markers
        self._eps: float = 0.001  # 1mm
        self._initCurves()

    def _initCurves(self: Self) -> None:
        abscissa = np.array([0.0, self._well.depth])
        ordinate = np.full_like(abscissa, np.nan)
        self.waterDepthCurve = UncertaintyCurve(
            "waterDepth", Curve("Depth", "waterDepth", abscissa, ordinate)
        )
        self.accommodationChangeCurve = UncertaintyCurve(
            "AccommodationChange",
            Curve("Depth", "AccommodationChange", abscissa, ordinate),
        )
        self.accommodationCurve = UncertaintyCurve(
            "Accommodation",
            Curve("Depth", "Accommodation", abscissa, ordinate),
        )

    def getInitialwaterDepth(self: Self) -> float:
        """Get the initial waterDepth.

        Take the middle value of the range as initial waterDepth get from
        facies log.

        :return float: initial waterDepth value.
        """
        wdRange: tuple[float, float] = self._getInitialwaterDepthRange()
        return 0.5 * (wdRange[0] + wdRange[1])

    # Helper functions
    def _getInitialwaterDepthRange(self: Self) -> tuple[float, float]:
        """Get the initial waterDepth range.

        The waterDepth is retreive from the facies log at and facies
        conditions.

        :return tuple[float, float]: min and max initial waterDepth values.
        """
        faciesLogNames: set[str] = self._well.getDiscreteLogNames()
        if len(faciesLogNames) == 0:
            raise ValueError(
                f"No discrete log found in well '{self._well.name}' "
                "to get initial waterDepth."
            )
        faciesLog: Striplog = cast(
            Striplog, self._well.getDepthLog(faciesLogNames.pop())
        )
        if faciesLog is None:
            raise ValueError(
                f"Facies log not found in well '{self._well.name}' to "
                "get initial waterDepth."
            )

        interval: Interval = cast(Interval, faciesLog[-1])
        faciesName: str = interval.primary["lithology"]  # type: ignore
        waterDepthRange = self._getWaterDepthRangeFromFaciesName(faciesName)
        if waterDepthRange is None:
            raise ValueError(
                f"waterDepth condition not found for facies {faciesName}."
            )
        return waterDepthRange

    def computeAccommodationCurve(
        self: Self,
        faciesLogName: str,
        fromMarker: Optional[Marker] = None,
        toMarker: Optional[Marker] = None,
        accommodationAtBase: float = 0.0,
        waterDepthAtBase: float | tuple[float, float] | None = None,
        boundaryRule: AccommodationBoundaryRule = (
            AccommodationBoundaryRule.UNION
        ),
        estimateMethod: AccommodationEstimateMethod = (
            AccommodationEstimateMethod.MIDPOINT
        ),
        smoothingLength: float = 10.0,
    ) -> UncertaintyCurve:
        r"""Compute accommodation space along the well.

        Apparent accommodation since the base is the deposited thickness plus
        the change of water depth, water depth being known as the range of
        each facies. The water depth at the base is a datum for the whole
        curve: its uncertainty propagates to every sample. Starting from a
        facies with a narrow water depth range, or providing the water depth
        at the base when it is known independently, sharpens the curve.

        Minimum and maximum curves are the range of accommodation at each
        facies boundary. The median curve is the best estimate in this range:

        * ``MIDPOINT``: middle of the range at each boundary;
        * ``MODE``: water depth at each boundary is the most likely water
          depth of the facies, i.e., the mode of the ``waterDepth`` criteria
          (see :class:`FaciesCriteria`), or the middle of its range if no
          mode is given (uniform distribution). At a boundary, the modes of
          the facies below and above are averaged, and the result is limited
          to the water depth range of the boundary. The water depth at the
          base is the mode of the basal facies, or the middle of
          ``waterDepthAtBase``. With uniform distributions, it differs from
          ``MIDPOINT`` only at boundaries, where the middle of the combined
          range is not the mean of the two middles;
        * ``SMOOTH``: water depth at each boundary :math:`w_k` and at the
          base :math:`w_0` are estimated together, bounded by their ranges,
          by minimizing

          .. math::

              \int \left(\frac{h_{ref}}{h(z)}\right)^2
              \left(w(z) - m(z)\right)^2 dz
              + \left(\frac{L}{2\pi}\right)^4 \int A''(z)^2 dz

          where :math:`m` is the water depth of the ``MODE`` method,
          :math:`h` the half width of the water depth range, :math:`h_{ref}`
          the median half width and
          :math:`L` the smoothing length. Accommodation variations with a
          wavelength shorter than :math:`L` are damped, longer ones are kept.
          The estimate always lies in the accommodation range. Since the
          thickness is linear with depth, :math:`A'' = w''`: smoothing in
          depth does not constrain the datum :math:`w_0`, which stays close
          to its mode unless the bounds are reached.

        :param str faciesLogName: name of the sedimentary facies log
        :param Marker fromMarker: base marker where to start calculation. If no
            marker is given, calculation starts from the base of the well.
            Defaults to None.
        :param Marker toMarker: to marker where to stop calculation. If no
            marker is given, calculation stops at the top of the well.
            Defaults to None.
        :param float accommodationAtBase: accommodation at the base marker.
            Defaults to 0.
        :param float | tuple[float, float] | None waterDepthAtBase: water
            depth (value or (min, max) range) at the base where calculation
            starts. If None, the water depth range of the facies at the base
            is used. Defaults to None.
        :param AccommodationBoundaryRule boundaryRule: rule to combine the
            two accommodation estimates at facies boundaries. Defaults to
            AccommodationBoundaryRule.UNION (combined span).
        :param AccommodationEstimateMethod estimateMethod: method to compute
            the best estimate (median curve) in the accommodation range.
            Defaults to AccommodationEstimateMethod.MIDPOINT.
        :param float smoothingLength: wavelength (same unit as depth) below
            which accommodation variations are damped. Used by the SMOOTH
            method only. Defaults to 10.
        :return UncertaintyCurve: accommodation curve
        """
        if not isinstance(self._well.getDepthLog(faciesLogName), Striplog):
            raise ValueError(
                f"The discrete log {faciesLogName} does not exist in the well "
                + f"{self._well.name}."
            )
        faciesLog: Striplog = cast(
            Striplog, self._well.getDepthLog(faciesLogName)
        )
        baseDepth: float = (
            fromMarker.depth if fromMarker is not None else faciesLog.stop.z
        )
        topDepth: float = (
            toMarker.depth if toMarker is not None else faciesLog.start.z
        )
        boundaryRule = AccommodationBoundaryRule(boundaryRule)
        estimateMethod = AccommodationEstimateMethod(estimateMethod)
        if (
            estimateMethod == AccommodationEstimateMethod.SMOOTH
            and not smoothingLength > 0.0
        ):
            raise ValueError("smoothingLength must be strictly positive.")

        # compute waterDepth curve if it is not defined
        if self._waterDepthStepCurve is None:
            self.computeWaterDepthCurve(faciesLogName, fromMarker, toMarker)

        # Accommodation array: depth, acco min 1, acco min 2, acco max 1,
        # acco max 2
        accoArray: npt.NDArray[np.float64] = self._computeAccommodationArray(
            faciesLog,
            baseDepth,
            topDepth,
            accommodationAtBase,
            waterDepthAtBase,
        )
        # combine the estimates from the intervals below and above each
        # boundary
        accoMin, accoMax = _combineRanges(accoArray[:, 1:], boundaryRule)
        accoMed = 0.5 * (accoMin + accoMax)
        if estimateMethod == AccommodationEstimateMethod.MODE:
            accoMed = self._computeModeAccommodation(
                boundaryRule, accommodationAtBase
            )
        elif estimateMethod == AccommodationEstimateMethod.SMOOTH:
            accoMed = self._computeSmoothAccommodation(
                boundaryRule, accommodationAtBase, smoothingLength
            )
        for depth, med, amin, amax in zip(
            accoArray[:, 0], accoMed, accoMin, accoMax, strict=True
        ):
            self.accommodationCurve.addSampledPoint(
                float(depth), float(med), float(amin), float(amax)
            )

        return self.accommodationCurve

    def _boundaryWaterDepthNodes(
        self: Self, boundaryRule: AccommodationBoundaryRule
    ) -> tuple[
        npt.NDArray[np.intp],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Get water depth range and mode at each facies boundary.

        Nodes are the boundaries with a defined depth, sorted by increasing
        depth: the last node is the base, whose water depth is the datum.

        :param AccommodationBoundaryRule boundaryRule: rule to combine the
            water depth ranges at facies boundaries.
        :return tuple: rows of the boundary arrays, depth, minimum, maximum
            and mode of water depth at each node.
        """
        wdArray = cast(npt.NDArray[np.float64], self._boundaryWaterDepthArray)
        modeArray = cast(
            npt.NDArray[np.float64], self._boundaryWaterDepthModeArray
        )
        rows = np.flatnonzero(np.isfinite(wdArray[:, 0]))
        rows = rows[np.argsort(wdArray[rows, 0])]
        depth = wdArray[rows, 0]
        wdMin, wdMax = _combineRanges(wdArray[rows, 1:], boundaryRule)
        # empty intersection: water depth lies between both ranges
        wdMin, wdMax = np.minimum(wdMin, wdMax), np.maximum(wdMin, wdMax)
        wdMode = np.mean(modeArray[rows], axis=1)
        if rows.size > 0:
            wdMin[-1], wdMax[-1] = self._baseWaterDepth
            wdMode[-1] = self._baseWaterDepthMode
        wdMode = np.clip(wdMode, wdMin, wdMax)
        return rows, depth, wdMin, wdMax, wdMode

    def _computeModeAccommodation(
        self: Self,
        boundaryRule: AccommodationBoundaryRule,
        accommodationAtBase: float,
    ) -> npt.NDArray[np.float64]:
        """Compute accommodation from the most likely facies water depth.

        :param AccommodationBoundaryRule boundaryRule: rule to combine the
            water depth ranges at facies boundaries.
        :param float accommodationAtBase: accommodation at the base.
        :return npt.NDArray[np.float64]: accommodation at each row of the
            boundary water depth array (NaN where the depth is undefined).
        """
        wdArray = cast(npt.NDArray[np.float64], self._boundaryWaterDepthArray)
        result = np.full(wdArray.shape[0], np.nan)
        rows, depth, _, _, wdMode = self._boundaryWaterDepthNodes(boundaryRule)
        if rows.size > 0:
            thickness = depth[-1] - depth
            result[rows] = (
                accommodationAtBase + thickness + wdMode - wdMode[-1]
            )
        return result

    def _computeSmoothAccommodation(
        self: Self,
        boundaryRule: AccommodationBoundaryRule,
        accommodationAtBase: float,
        smoothingLength: float,
    ) -> npt.NDArray[np.float64]:
        """Compute the smoothest accommodation inside its range.

        See :meth:`computeAccommodationCurve` for the formulation. The
        problem is a linear least squares problem bounded by the water depth
        ranges, solved with :func:`scipy.optimize.lsq_linear`.

        :param AccommodationBoundaryRule boundaryRule: rule to combine the
            water depth ranges at facies boundaries.
        :param float accommodationAtBase: accommodation at the base.
        :param float smoothingLength: wavelength below which accommodation
            variations are damped.
        :return npt.NDArray[np.float64]: accommodation at each row of the
            boundary water depth array (NaN where the depth is undefined).
        """
        wdArray = cast(npt.NDArray[np.float64], self._boundaryWaterDepthArray)
        result = np.full(wdArray.shape[0], np.nan)
        # water depth bounds and mode per node, sorted by increasing depth;
        # the base node (last one) is the datum w0
        rows, depth, wdMin, wdMax, wdMode = self._boundaryWaterDepthNodes(
            boundaryRule
        )
        nbNodes = rows.size
        if nbNodes < 2:
            result[rows] = accommodationAtBase
            return result

        # data term: distance to the mode, weighted by the inverse of the
        # half width of the range, integrated along depth
        halfWidth = 0.5 * (wdMax - wdMin)
        informative = (
            np.isfinite(halfWidth) & (halfWidth > 0.0) & np.isfinite(wdMode)
        )
        refHalfWidth = (
            float(np.median(halfWidth[informative]))
            if np.any(informative)
            else 1.0
        )
        nodeLength = np.zeros(nbNodes)
        nodeLength[:-1] += 0.5 * np.diff(depth)
        nodeLength[1:] += 0.5 * np.diff(depth)
        dataWeight = np.zeros(nbNodes)
        dataWeight[informative] = (
            np.sqrt(nodeLength[informative])
            * refHalfWidth
            / halfWidth[informative]
        )
        dataMatrix = np.diag(dataWeight)
        dataRhs = dataWeight * np.where(informative, wdMode, 0.0)

        # smoothing term: change of slope of accommodation between
        # consecutive segments, integrated along depth. Thickness is
        # linear with depth, so only water depth contributes.
        slope = np.zeros((nbNodes - 1, nbNodes))
        dz = np.maximum(np.diff(depth), 1e-12)
        idx = np.arange(nbNodes - 1)
        slope[idx, idx] = -1.0 / dz
        slope[idx, idx + 1] = 1.0 / dz
        curvature = np.diff(slope, axis=0)
        curvatureLength = 0.5 * (depth[2:] - depth[:-2])
        smoothWeight = (smoothingLength / (2.0 * np.pi)) ** 2 / np.sqrt(
            np.maximum(curvatureLength, 1e-12)
        )
        smoothMatrix = smoothWeight[:, None] * curvature

        # fixed water depth (e.g., given base water depth): lsq_linear
        # requires strictly ordered bounds
        lower = np.where(np.isfinite(wdMin), wdMin, -np.inf)
        upper = np.where(np.isfinite(wdMax), wdMax, np.inf)
        upper = np.maximum(upper, lower + 1e-9)
        solution = lsq_linear(
            np.vstack((dataMatrix, smoothMatrix)),
            np.concatenate((dataRhs, np.zeros(smoothMatrix.shape[0]))),
            bounds=(lower, upper),
            method="bvls",
        )
        waterDepth = solution.x
        thickness = depth[-1] - depth
        result[rows] = (
            accommodationAtBase + thickness + waterDepth - waterDepth[-1]
        )
        return result

    def _computeAccommodationArray(
        self: Self,
        faciesLog: Striplog,
        baseDepth: float,
        topDepth: float,
        accommodationAtBase: float = 0.0,
        waterDepthAtBase: float | tuple[float, float] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute apparent accommodation space array along the well.

        Array is composed of as many rows as the number of limits of interval
        (i.e., len(faciesLog) + 1) and columns are:

        * depth where accommodation is computed
        * minimum accommodation computed from the waterDepth right below the
          depth
        * minimum accommodation computed from the waterDepth right above the
          depth
        * maximum accommodation computed from the waterDepth right below the
          depth
        * maximum accommodation computed from the waterDepth right above the
          depth

        :param str faciesLog: sedimentary facies log
        :param float baseDepth: depth where to start calculation.
        :param float topDepth: depth to stop calculation.
        :param float accommodationAtBase: cummulative accommodation at the
            base depth. Defaults to 0.
        :param float | tuple[float, float] | None waterDepthAtBase: water
            depth (value or (min, max) range) at the base depth. If None, the
            water depth range of the facies at the base is used.
        :return npt.NDArray[np.float64]: accommodation array
        """
        # compute waterDepth curve if it is not defined
        if self._waterDepthStepCurve is None:
            self._computeWaterDepthStepCurve(faciesLog, baseDepth, topDepth)

        # get waterDepth at the base and computation depth
        depthBase: float = baseDepth
        baseWaterDepth: tuple[float, float] = (0.0, 0.0)
        lastIndex: int = len(faciesLog)
        for row in self._waterDepthStepCurve[::-1]:  # type: ignore
            if row[0] > baseDepth:
                lastIndex -= 1
                continue
            baseWaterDepth = row[2:]
            depthBase = row[0]
            break

        modes = cast(npt.NDArray[np.float64], self._waterDepthModes)
        baseMode: float = (
            float(modes[lastIndex - 1]) if lastIndex > 0 else np.nan
        )

        # water depth at the base given by the user overrides the facies one
        if waterDepthAtBase is not None:
            if isinstance(waterDepthAtBase, (int, float)):
                baseWaterDepth = (
                    float(waterDepthAtBase),
                    float(waterDepthAtBase),
                )
            else:
                baseWaterDepth = (
                    float(min(waterDepthAtBase)),
                    float(max(waterDepthAtBase)),
                )
            baseMode = 0.5 * (baseWaterDepth[0] + baseWaterDepth[1])
        if not np.isfinite(baseWaterDepth[0]):
            raise ValueError("waterDepth at the base is undefined.")

        # accommodation array: depth, acco min 1, acco min 2, acco max 1,
        # acco max 2
        accoArray: npt.NDArray[np.float64] = np.full(
            (self._waterDepthStepCurve.shape[0] + 1, 5),  # type: ignore
            np.nan,
        )
        # same layout for the water depth ranges at each boundary
        wdArray: npt.NDArray[np.float64] = np.full_like(accoArray, np.nan)
        # water depth modes: interval, adjacent interval
        modeArray: npt.NDArray[np.float64] = np.full(
            (accoArray.shape[0], 2), np.nan
        )
        interval: Interval
        for i, interval in enumerate(faciesLog):
            # skip stratas below the base
            if interval.base.z > depthBase:
                continue
            # waterDepth of the interval
            waterDepthInterval = self._waterDepthStepCurve[i, 2:]  # type: ignore
            # compute accommodation at top
            thickness: float
            if i == 0:
                thickness = depthBase - interval.top.z
                # waterDepth of the interval
                waterDepthInterval = self._waterDepthStepCurve[i, 2:]  # type: ignore
                # compute accommodation from water depth of the interval
                acco0: tuple[float, float] = self._computeAccommodationValue(
                    thickness,
                    baseWaterDepth,
                    tuple(waterDepthInterval.tolist()),
                )
                # store the results
                accoArray[i, 0] = interval.top.z

                # min accommodation
                accoArray[i, 1:3] = (acco0[0], acco0[0])
                # max accommodation
                accoArray[i, 3:] = (acco0[1], acco0[1])
                wdArray[i, 0] = interval.top.z
                wdArray[i, 1:3] = waterDepthInterval[0]
                wdArray[i, 3:] = waterDepthInterval[1]
                modeArray[i] = modes[i]

            # compute accommodation at the base
            thickness = depthBase - interval.base.z

            # waterDepth of above interval
            waterDepthAboveInterval = waterDepthInterval
            adjacentIndex = i
            if (i < len(faciesLog) - 1) and (i < lastIndex):
                waterDepthAboveInterval = self._waterDepthStepCurve[i + 1, 2:]  # type: ignore
                adjacentIndex = i + 1

            # compute accommodation
            # computed from waterDepth of the interval
            acco1: tuple[float, float] = self._computeAccommodationValue(
                thickness, baseWaterDepth, tuple(waterDepthInterval.tolist())
            )
            # computed from the waterDepth of the facies above
            acco2: tuple[float, float] = self._computeAccommodationValue(
                thickness,
                baseWaterDepth,
                tuple(waterDepthAboveInterval.tolist()),
            )

            # store the results
            accoArray[i + 1, 0] = interval.base.z
            # min accommodation
            accoArray[i + 1, 1:3] = (acco1[0], acco2[0])
            # max accommodation
            accoArray[i + 1, 3:] = (acco1[1], acco2[1])
            wdArray[i + 1, 0] = interval.base.z
            wdArray[i + 1, 1:3] = (
                waterDepthInterval[0],
                waterDepthAboveInterval[0],
            )
            wdArray[i + 1, 3:] = (
                waterDepthInterval[1],
                waterDepthAboveInterval[1],
            )
            modeArray[i + 1] = (modes[i], modes[adjacentIndex])

        # set initial accommodation at the base to 0 (no uncertainty here)
        accoArray[-1, 1:] = 0.0

        # add initial accommodation value
        accoArray[:, 1:] += accommodationAtBase
        self._boundaryWaterDepthArray = wdArray
        self._boundaryWaterDepthModeArray = modeArray
        self._baseWaterDepthMode = baseMode
        self._baseWaterDepth = (
            float(baseWaterDepth[0]),
            float(baseWaterDepth[1]),
        )
        return accoArray

    def _computeAccommodationValue(
        self: Self,
        thickness: float,
        waterDepthBase: tuple[float, float],
        waterDepthTop: tuple[float, float],
    ) -> tuple[float, float]:
        """Compute the accommodation according to thickness and waterDepth.

        :param float thickness: interval thickness
        :param tuple[float, float] waterDepthBase: waterDepth at the base of
            the interval
        :param tuple[float, float] waterDepthTop: waterDepth at the top of the
            interval
        :return tuple[float, float]: accommodation variation from base to top
        """
        # minimum waterDepth variation: consider waterDepth is max at base
        # marker and min at top marker
        deltaWaterDepthMin: float = waterDepthTop[0] - waterDepthBase[1]
        # maximum waterDepth variation: consider waterDepth is min at base
        # marker and max at top marker
        deltaWaterDepthMax: float = waterDepthTop[1] - waterDepthBase[0]
        accoMin: float = thickness + deltaWaterDepthMin
        accoMax: float = thickness + deltaWaterDepthMax
        if accoMin > accoMax:
            accoMin, accoMax = accoMax, accoMin
        return accoMin, accoMax

    def computeAccommodationCurve0(
        self: Self,
        faciesLogName: str,
        fromMarker: Optional[Marker] = None,
        toMarker: Optional[Marker] = None,
    ) -> UncertaintyCurve:
        """Compute accommodation space along the well.

        :param str faciesLogName: name of the sedimentary facies log
        :param float step: step between continuous log samples
        :param Marker fromMarker: base marker where to start calculation. If no
            marker is given, calculation starts from the base of the well.
            Defaults to None.
        :param Marker toMarker: to marker where to stop calculation. If no
            marker is given, calculation stops at the top of the well.
            Defaults to None.
        :return UncertaintyCurve: accommodation curve
        """
        if not isinstance(self._well.getDepthLog(faciesLogName), Striplog):
            raise ValueError(
                f"The discrete log {faciesLogName} does not exist in the well "
                + f"{self._well.name}."
            )
        faciesLog: Striplog = cast(
            Striplog, self._well.getDepthLog(faciesLogName)
        )
        baseDepth: float = (
            fromMarker.depth if fromMarker is not None else faciesLog.stop.z
        )
        topDepth: float = (
            toMarker.depth if toMarker is not None else faciesLog.start.z
        )
        # compute accommodation step curve
        if self._accommodationStepCurve is None:
            self._computeAccommodationStepCurve(faciesLog, baseDepth, topDepth)
        # store uncertainty accommodation change curve
        self._convertIntervalCurve2UncertaintyCurve(
            self._accommodationStepCurve,  # type: ignore[arg-type]
            self.accommodationChangeCurve,
        )

        # compute cumulative accommodation
        accommodationCumulStepCurve = np.copy(self._accommodationStepCurve)  # type: ignore[arg-type]
        for i in (2, 3):
            accommodationCumulStepCurve[:, i][::-1] = np.cumsum(
                accommodationCumulStepCurve[:, i][::-1]
            )
        self._convertIntervalCurve2UncertaintyCurve(
            accommodationCumulStepCurve, self.accommodationCurve
        )
        return self.accommodationCurve

    def computeWaterDepthCurve(
        self: Self,
        faciesLogName: str,
        fromMarker: Optional[Marker] = None,
        toMarker: Optional[Marker] = None,
    ) -> UncertaintyCurve:
        """Compute the waterDepth along the well.

        :param str faciesLogName: name of the sedimentary facies log
        :param float step: step between continuous log samples
        :param Marker fromMarker: base marker where to start calculation. If no
            marker is given, calculation starts from the base of the well.
            Defaults to None.
        :param Marker toMarker: to marker where to stop calculation. If no
            marker is given, calculation stops at the top of the well.
            Defaults to None.
        :return UncertaintyCurve: waterDepth curve
        """
        if not isinstance(self._well.getDepthLog(faciesLogName), Striplog):
            raise ValueError(
                f"The discrete log {faciesLogName} does not exist in the well "
                + f"{self._well.name}."
            )
        faciesLog: Striplog = cast(
            Striplog, self._well.getDepthLog(faciesLogName)
        )
        baseDepth: float = (
            fromMarker.depth if fromMarker is not None else faciesLog.stop.z
        )
        topDepth: float = (
            toMarker.depth if toMarker is not None else faciesLog.start.z
        )
        self._computeWaterDepthStepCurve(faciesLog, baseDepth, topDepth)

        if self._waterDepthStepCurve is None:
            raise ValueError("waterDepth step curve is not computed.")
        self._convertIntervalCurve2UncertaintyCurve(
            self._waterDepthStepCurve, self.waterDepthCurve
        )
        self._waterDepthComputed = True
        return self.waterDepthCurve

    def _getWaterDepthRangeFromFaciesName(
        self: Self, faciesName: str
    ) -> tuple[float, float]:
        """Get the waterDepth range from the facies name.

        :param str faciesName: facies name
        :raises ValueError: if the facies name is not in the list or the
            waterDepth conditions is undefined for a given facies.
        :return tuple[float, float]: waterDepth minimum and maximum values.
        """
        criteria = self._getWaterDepthCriteriaFromFaciesName(faciesName)
        return (criteria.minRange, criteria.maxRange)

    def _getWaterDepthCriteriaFromFaciesName(
        self: Self, faciesName: str
    ) -> FaciesCriteria:
        """Get the waterDepth criteria from the facies name.

        :param str faciesName: facies name
        :raises ValueError: if the facies name is not in the list or the
            waterDepth conditions is undefined for a given facies.
        :return FaciesCriteria: waterDepth criteria of the facies.
        """
        facies: Optional[SedimentaryFacies] = self._faciesDict.get(
            faciesName, None
        )
        if facies is None:
            raise ValueError(
                f"Facies {faciesName} is not in the facies list. "
                + "waterDepth curve cannot be computed."
            )
        waterDepthRange: Optional[FaciesCriteria] = facies.getCriteria(
            "waterDepth"
        )
        if waterDepthRange is None:
            raise ValueError(
                f"waterDepth is undefined for the facies {faciesName}. "
                + "waterDepth curve cannot be computed."
            )
        return waterDepthRange

    def _computeWaterDepthStepCurve(
        self: Self,
        faciesLog: Striplog,
        baseDepth: float,
        topDepth: float,
    ) -> npt.NDArray[np.float64]:
        self._waterDepthStepCurve = np.full((len(faciesLog), 4), np.nan)
        self._waterDepthModes = np.full(len(faciesLog), np.nan)
        # add epsilon because if Interval.completely_contains returns True only
        # if limits are not equal
        eps: float = 1e-6
        computedInterval = Interval(topDepth - eps, baseDepth + eps)
        interval: Interval
        nbIntervals: int = 0
        for i, interval in enumerate(faciesLog):
            # ..WARNING:: assume that interval coordinates are in MD
            if not computedInterval.completely_contains(interval):
                continue
            if interval.primary is None:
                raise ValueError("Interval primary attribute is None.")
            faciesName: str = interval.primary["lithology"]
            criteria = self._getWaterDepthCriteriaFromFaciesName(faciesName)
            self._waterDepthStepCurve[i, 0] = interval.base.z
            self._waterDepthStepCurve[i, 1] = interval.top.z
            self._waterDepthStepCurve[i, 2:] = (
                criteria.minRange,
                criteria.maxRange,
            )
            self._waterDepthModes[i] = criteria.getMode()
            nbIntervals += 1

        return self._waterDepthStepCurve

    def _computeAccommodationStepCurve(
        self: Self, faciesLog: Striplog, baseDepth: float, topDepth: float
    ) -> npt.NDArray[np.float64]:
        # compute waterDepth per interval if not computed yet
        if self._waterDepthStepCurve is None:
            self._computeWaterDepthStepCurve(faciesLog, baseDepth, topDepth)

        self._accommodationStepCurve = np.full_like(
            self._waterDepthStepCurve, np.nan
        )
        interval: Interval
        for i, interval in enumerate(faciesLog):
            # waterDepth at the top of the interval
            (wdMinEnd, wdMaxEnd) = self._waterDepthStepCurve[i, 2:]  # type: ignore
            # waterDepth at the top of the interval (=base of above interval,
            # except for the last interval where we assume no variations)
            wdMinStart, wdMaxStart = wdMinEnd, wdMaxEnd
            if i > 0:  # self._waterDepthStepCurve.shape[0] - 1:
                (wdMinStart, wdMaxStart) = self._waterDepthStepCurve[  # type: ignore
                    i - 1, 2:
                ]
            # minimum waterDepth variation: consider water Depth is max at
            # bottom interval and min at current interval
            deltaBathyMin: float = wdMinEnd - wdMaxStart
            # minimum waterDepth variation: consider water Depth is min at
            # bottom interval and max at current interval
            deltaBathyMax: float = wdMaxEnd - wdMinStart
            accoMin: float = interval.thickness + deltaBathyMin
            accoMax: float = interval.thickness + deltaBathyMax
            if accoMin > accoMax:
                accoMin, accoMax = accoMax, accoMin
            self._accommodationStepCurve[i, :2] = self._waterDepthStepCurve[  # type: ignore
                i, :2
            ]
            self._accommodationStepCurve[i, 2:] = (accoMin, accoMax)
        return self._accommodationStepCurve

    def computeWaterDepthThicknessRatioCurve(
        self: Self,
        faciesLogName: str,
    ) -> UncertaintyCurve:
        """Compute water depth / thickness ratio curve.

        Uses the deepest interval as the reference for water depth
        delta computation.

        :param str faciesLogName: name of the sedimentary facies log
        :raises RuntimeError: if water depth step curve has not been
            computed yet.
        :return UncertaintyCurve: ratio curve with min/max values.
        """
        if self._waterDepthStepCurve is None:
            raise RuntimeError(
                "water depth step curve must be computed before"
                " computing the WD/thickness ratio. Call"
                " computeAccommodationCurve() first."
            )
        faciesLog: Striplog = cast(
            Striplog, self._well.getDepthLog(faciesLogName)
        )
        if faciesLog is None:
            raise ValueError(
                f"The discrete log {faciesLogName} does not"
                f" exist in the well {self._well.name}."
            )

        # Reference: deepest interval (last row)
        bathy0 = self._waterDepthStepCurve[-1]

        # Collect ratio step curve rows: [base_depth, top_depth, min, max]
        ratio_step: list[tuple[float, float, float, float]] = []
        interval: Interval
        for i, interval in enumerate(faciesLog):
            row = self._waterDepthStepCurve[i]
            if not np.isfinite(row[2]):
                continue
            thickness = abs(interval.top.middle - interval.base.middle)
            if thickness == 0:
                continue
            dbmin = row[2] - bathy0[3]
            dbmax = row[3] - bathy0[2]
            ratio_min = dbmin / thickness
            ratio_max = dbmax / thickness
            if ratio_min > ratio_max:
                ratio_min, ratio_max = ratio_max, ratio_min
            ratio_step.append(
                (float(row[0]), float(row[1]), ratio_min, ratio_max)
            )

        if not ratio_step:
            abscissa = np.array([0.0, self._well.depth])
            ordinate = np.full_like(abscissa, np.nan)
            return UncertaintyCurve(
                "WDThicknessRatio",
                Curve("Depth", "WDThicknessRatio", abscissa, ordinate),
            )

        # Build flat arrays: 2 depth samples per interval (base-eps, top+eps)
        xs: list[float] = []
        ys_med: list[float] = []
        ys_min: list[float] = []
        ys_max: list[float] = []
        for base_d, top_d, r_min, r_max in ratio_step:
            med = (r_min + r_max) / 2.0
            xs.extend([base_d - self._eps, top_d + self._eps])
            ys_med.extend([med, med])
            ys_min.extend([r_min, r_min])
            ys_max.extend([r_max, r_max])

        abscissa = np.array(xs)
        sort_idx = np.argsort(abscissa)
        abscissa = abscissa[sort_idx]
        ys_med_arr = np.array(ys_med)[sort_idx]
        ys_min_arr = np.array(ys_min)[sort_idx]
        ys_max_arr = np.array(ys_max)[sort_idx]

        median_curve = Curve("Depth", "WDThicknessRatio", abscissa, ys_med_arr)
        ratio_curve = UncertaintyCurve("WDThicknessRatio", median_curve)
        ratio_curve.setMinCurveValues(ys_min_arr)
        ratio_curve.setMaxCurveValues(ys_max_arr)

        return ratio_curve

    def _convertIntervalCurve2UncertaintyCurve(
        self: Self,
        stepCurve: npt.NDArray[np.float64],
        uncertaintyCurve: UncertaintyCurve,
    ) -> None:
        for row in stepCurve:
            med: float = (row[2] + row[3]) / 2.0
            uncertaintyCurve.addSampledPoint(
                row[0] - self._eps, med, row[2], row[3]
            )
            uncertaintyCurve.addSampledPoint(
                row[1] + self._eps, med, row[2], row[3]
            )
