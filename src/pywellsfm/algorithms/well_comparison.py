# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Tools to compare observed and simulated wells.

This module gathers the building blocks needed to confront simulated wells
with observations:

- sampling and resampling of discrete logs (e.g., to degrade a simulated log
  to the resolution of an observed one),
- conversion of simulation outputs (``FSSimulator.outputs``) from the time
  domain to the depth domain,
- misfit measures between discrete logs and between marker depths,
- conversion of a depositional environment model into the sedimentary facies
  expected by :class:`AccommodationSpaceWellCalculator`.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr
from striplog import Component, Interval, Striplog

from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironmentModel,
)
from pywellsfm.model.Facies import (
    FaciesCriteria,
    FaciesCriteriaType,
    SedimentaryFacies,
)

__all__ = [
    "discreteLogLabelsAt",
    "faciesFromDepositionalEnvironments",
    "faciesMismatchFraction",
    "markerDepthRmse",
    "resampleStriplog",
    "simulatedAgesToDepths",
    "simulatedStepEndDepths",
]


def _sortedIntervals(
    log: Striplog, component: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
    """Get interval tops, bases and labels sorted by increasing depth.

    :param Striplog log: discrete log in depth domain.
    :param str component: name of the component attribute.
    :return tuple: tops, bases and labels.
    """
    intervals = sorted(log, key=lambda iv: iv.top.z)
    tops = np.array([iv.top.z for iv in intervals], dtype=np.float64)
    bases = np.array([iv.base.z for iv in intervals], dtype=np.float64)
    labels: list[str] = []
    for iv in intervals:
        try:
            value = iv.primary[component] if iv.primary is not None else ""
        except (KeyError, AttributeError):
            value = ""
        labels.append("" if value is None else str(value))
    return tops, bases, labels


def discreteLogLabelsAt(
    log: Striplog,
    depths: npt.NDArray[np.float64],
    component: str = "lithology",
) -> npt.NDArray[np.str_]:
    """Sample the labels of a discrete log at given depths.

    :param Striplog log: discrete log in depth domain.
    :param npt.NDArray[np.float64] depths: depths where to sample the log.
    :param str component: name of the component attribute holding the
        label. Default is "lithology".
    :return npt.NDArray[np.str_]: label at each depth, or an empty string
        where the log is undefined.
    """
    depths = np.atleast_1d(np.asarray(depths, dtype=np.float64))
    tops, bases, labels = _sortedIntervals(log, component)
    out = np.full(depths.shape, "", dtype=object)
    if tops.size == 0:
        return out.astype(str)
    idx = np.searchsorted(tops, depths, side="right") - 1
    valid = (idx >= 0) & (idx < tops.size)
    valid[valid] &= depths[valid] <= bases[idx[valid]]
    labelArray = np.array(labels, dtype=object)
    out[valid] = labelArray[idx[valid]]
    return out.astype(str)


def resampleStriplog(
    log: Striplog,
    step: float,
    top: float | None = None,
    base: float | None = None,
    component: str = "lithology",
) -> Striplog:
    """Resample a discrete log on a regular grid.

    Each bin of thickness *step* takes the label covering the largest part
    of the bin. Adjacent bins with the same label are then merged. This is
    useful to degrade a high-resolution simulated log to the resolution of
    an observed log (e.g., core description or electrofacies).

    :param Striplog log: discrete log in depth domain.
    :param float step: bin thickness (m).
    :param float | None top: top depth of the resampled log. Defaults to the
        top of the log.
    :param float | None base: base depth of the resampled log. Defaults to
        the base of the log.
    :param str component: name of the component attribute holding the
        label. Default is "lithology".
    :return Striplog: resampled log.
    """
    if step <= 0:
        raise ValueError("step must be > 0.")
    tops, bases, labels = _sortedIntervals(log, component)
    if tops.size == 0:
        raise ValueError("Cannot resample an empty log.")
    top = float(tops.min()) if top is None else float(top)
    base = float(bases.max()) if base is None else float(base)
    if base <= top:
        raise ValueError("base must be deeper than top.")

    edges = np.arange(top, base, step, dtype=np.float64)
    edges = np.append(edges, base)
    binned: list[tuple[float, float, str]] = []
    for z0, z1 in zip(edges[:-1], edges[1:], strict=True):
        overlap = np.clip(
            np.minimum(bases, z1) - np.maximum(tops, z0), 0, None
        )
        if not np.any(overlap > 0):
            continue
        cover: dict[str, float] = {}
        for label, ov in zip(labels, overlap, strict=True):
            if ov > 0:
                cover[label] = cover.get(label, 0.0) + float(ov)
        label = max(cover, key=lambda k: cover[k])
        if binned and binned[-1][2] == label and binned[-1][1] == z0:
            binned[-1] = (binned[-1][0], z1, label)
        else:
            binned.append((z0, z1, label))
    return Striplog(
        [
            Interval(z0, z1, components=[Component({component: label})])
            for z0, z1, label in binned
        ]
    )


def faciesMismatchFraction(
    observedLog: Striplog,
    observedBase: float,
    simulatedLog: Striplog,
    simulatedBase: float,
    step: float = 0.1,
    height: float | None = None,
    component: str = "lithology",
) -> float:
    """Fraction of mismatching labels between two logs aligned at the base.

    Both logs are sampled at the same heights above their respective base
    (e.g., a datum marker), every *step* metres. When *height* is None, the
    comparison extends to the top of the thicker log and missing samples
    (above the top of the thinner log) count as mismatches, so that the
    measure also penalizes thickness differences.

    :param Striplog observedLog: observed discrete log (depth domain).
    :param float observedBase: depth of the datum in the observed well.
    :param Striplog simulatedLog: simulated discrete log (depth domain).
    :param float simulatedBase: depth of the datum in the simulated well.
    :param float step: sampling step (m). Default is 0.1 m.
    :param float | None height: height above the datum over which logs are
        compared. Default is None.
    :param str component: name of the component attribute holding the
        label. Default is "lithology".
    :return float: mismatch fraction between 0 (identical) and 1.
    """
    if step <= 0:
        raise ValueError("step must be > 0.")
    if height is None:
        obsTops, _, _ = _sortedIntervals(observedLog, component)
        simTops, _, _ = _sortedIntervals(simulatedLog, component)
        height = max(
            observedBase - float(obsTops.min()),
            simulatedBase - float(simTops.min()),
        )
    heights = np.arange(0.5 * step, height, step, dtype=np.float64)
    if heights.size == 0:
        return float("nan")
    obs = discreteLogLabelsAt(observedLog, observedBase - heights, component)
    sim = discreteLogLabelsAt(simulatedLog, simulatedBase - heights, component)
    mismatch = (obs != sim) | (obs == "") | (sim == "")
    return float(np.mean(mismatch))


def simulatedStepEndDepths(
    dataset: xr.Dataset,
    realization: int,
    baseDepth: float,
) -> npt.NDArray[np.float64]:
    """Depth of the top of the deposits at the end of each time step.

    In ``FSSimulator.outputs``, ``thickness_cumul``, ``accommodation``,
    ``sea_level``, ``subsidence`` and ``basement`` are given at the end of
    each step (age ``time - dt``), while ``waterDepth``, rates and
    ``environment`` are given at the start of each step (age ``time``).

    :param xr.Dataset dataset: simulation outputs.
    :param int realization: realization index.
    :param float baseDepth: depth of the base of the simulated column in the
        well (e.g., the oldest marker depth).
    :return npt.NDArray[np.float64]: depth at the end of each step.
    """
    cumul = dataset["thickness_cumul"].isel(realization=realization).values
    return baseDepth - np.asarray(cumul, dtype=np.float64)


def simulatedAgesToDepths(
    dataset: xr.Dataset,
    realization: int,
    ages: Sequence[float] | npt.NDArray[np.float64],
    baseDepth: float,
) -> npt.NDArray[np.float64]:
    """Depth of isochronous surfaces in a simulated well.

    Depths are linearly interpolated between time steps. This is the depth
    at which a marker of a given age would be picked in the simulated well.

    :param xr.Dataset dataset: simulation outputs.
    :param int realization: realization index.
    :param Sequence[float] ages: ages of the surfaces (Myr).
    :param float baseDepth: depth of the base of the simulated column.
    :return npt.NDArray[np.float64]: depth of each surface.
    """
    times = np.asarray(dataset["time"].values, dtype=np.float64)
    dts = np.asarray(dataset["dt"].values, dtype=np.float64)
    cumul = np.asarray(
        dataset["thickness_cumul"].isel(realization=realization).values,
        dtype=np.float64,
    )
    # cumulated thickness at step boundaries, from oldest to youngest
    boundaryAges = np.concatenate(([times[0]], times - dts))
    boundaryThickness = np.concatenate(([0.0], cumul))
    # np.interp needs increasing abscissa: ages decrease with time
    thickness = np.interp(
        np.asarray(ages, dtype=np.float64),
        boundaryAges[::-1],
        boundaryThickness[::-1],
    )
    return baseDepth - thickness


def markerDepthRmse(
    observedDepths: Sequence[float] | npt.NDArray[np.float64],
    simulatedDepths: Sequence[float] | npt.NDArray[np.float64],
) -> float:
    """Root mean square difference between observed and simulated depths.

    :param Sequence[float] observedDepths: observed marker depths.
    :param Sequence[float] simulatedDepths: simulated depths of the same
        markers, in the same order.
    :return float: root mean square error (m).
    """
    obs = np.asarray(observedDepths, dtype=np.float64)
    sim = np.asarray(simulatedDepths, dtype=np.float64)
    if obs.shape != sim.shape:
        raise ValueError("Depth arrays must have the same shape.")
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def faciesFromDepositionalEnvironments(
    depositionalEnvironmentModel: DepositionalEnvironmentModel,
    minWaterDepth: float = -10.0,
    maxWaterDepth: float = 1000.0,
) -> list[SedimentaryFacies]:
    """Build sedimentary facies from depositional environments.

    Each environment becomes a sedimentary facies of the same name whose
    water-depth criterion is the environment water-depth range, clipped to
    [minWaterDepth, maxWaterDepth] so that open ranges (e.g., continent,
    basin) remain usable for accommodation computation.

    :param DepositionalEnvironmentModel depositionalEnvironmentModel:
        depositional environment model.
    :param float minWaterDepth: lower bound of water-depth ranges (m).
    :param float maxWaterDepth: upper bound of water-depth ranges (m).
    :return list[SedimentaryFacies]: facies usable by
        :class:`AccommodationSpaceWellCalculator`.
    """
    faciesList: list[SedimentaryFacies] = []
    for env in depositionalEnvironmentModel.environments:
        lo = float(np.clip(env.waterDepth_min, minWaterDepth, maxWaterDepth))
        hi = float(np.clip(env.waterDepth_max, minWaterDepth, maxWaterDepth))
        faciesList.append(
            SedimentaryFacies(
                env.name,
                {
                    FaciesCriteria(
                        "waterDepth",
                        lo,
                        hi,
                        FaciesCriteriaType.SEDIMENTOLOGICAL,
                    )
                },
            )
        )
    return faciesList
