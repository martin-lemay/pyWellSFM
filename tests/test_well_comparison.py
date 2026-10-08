# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
import os
import sys

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

import numpy as np
import pytest
import xarray as xr
from striplog import Component, Interval, Striplog

from pywellsfm.algorithms import (
    discreteLogLabelsAt,
    faciesFromDepositionalEnvironments,
    faciesMismatchFraction,
    markerDepthRmse,
    resampleStriplog,
    simulatedAgesToDepths,
    simulatedStepEndDepths,
)
from pywellsfm.model import CarbonateOpenRampDepositionalEnvironmentModel


def _striplog(rows: list[tuple[float, float, str]]) -> Striplog:
    """Build a striplog from (top, base, label) rows."""
    return Striplog(
        [
            Interval(top, base, components=[Component({"lithology": label})])
            for top, base, label in rows
        ]
    )


@pytest.fixture
def log() -> Striplog:
    """Simple 3-bed log between 10 and 14 m."""
    return _striplog([(10.0, 11.0, "a"), (11.0, 13.0, "b"), (13.0, 14.0, "c")])


@pytest.fixture
def dataset() -> xr.Dataset:
    """Minimal simulation output with 3 steps of 1 Myr."""
    return xr.Dataset(
        {
            "dt": (("time",), np.array([1.0, 1.0, 1.0])),
            "thickness_cumul": (
                ("realization", "time"),
                np.array([[2.0, 3.0, 6.0]]),
            ),
        },
        coords={"time": np.array([10.0, 9.0, 8.0]), "realization": [0]},
    )


def test_labels_at_depths(log: Striplog) -> None:
    """Labels are sampled at depths; outside the log gives empty labels."""
    labels = discreteLogLabelsAt(log, np.array([9.0, 10.5, 12.0, 13.5, 15]))
    assert labels.tolist() == ["", "a", "b", "c", ""]


def test_resample_majority_and_merge(log: Striplog) -> None:
    """Bins take the majority label and adjacent bins are merged."""
    resampled = resampleStriplog(log, 1.5)
    rows = [(iv.top.z, iv.base.z, iv.primary["lithology"]) for iv in resampled]
    # bins: [10,11.5]->a(1)/b(.5)=a ; [11.5,13]->b ; [13,14]->c
    assert rows == [(10.0, 11.5, "a"), (11.5, 13.0, "b"), (13.0, 14.0, "c")]


def test_resample_invalid_inputs(log: Striplog) -> None:
    """Invalid steps or bounds raise ValueError."""
    with pytest.raises(ValueError):
        resampleStriplog(log, 0.0)
    with pytest.raises(ValueError):
        resampleStriplog(log, 1.0, top=14.0, base=10.0)


def test_mismatch_identical_and_shifted(log: Striplog) -> None:
    """Identical logs match; thickness differences count as mismatch."""
    assert faciesMismatchFraction(log, 14.0, log, 14.0) == 0.0
    thinner = _striplog([(11.0, 12.0, "a"), (12.0, 14.0, "b")])
    # heights 0-1: c vs b (mismatch), 1-2: b vs b, 2-3: b vs a
    # (mismatch), 3-4: a vs missing (mismatch)
    value = faciesMismatchFraction(log, 14.0, thinner, 14.0, step=0.5)
    assert value == pytest.approx(0.75)
    limited = faciesMismatchFraction(
        log, 14.0, thinner, 14.0, step=0.5, height=3.0
    )
    assert limited == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError):
        faciesMismatchFraction(log, 14.0, log, 14.0, step=0.0)


def test_step_end_depths(dataset: xr.Dataset) -> None:
    """Step-end depths stack thickness upward from the base."""
    depths = simulatedStepEndDepths(dataset, 0, 100.0)
    np.testing.assert_allclose(depths, [98.0, 97.0, 94.0])


def test_ages_to_depths(dataset: xr.Dataset) -> None:
    """Isochrons are interpolated between step boundaries."""
    depths = simulatedAgesToDepths(dataset, 0, [10.0, 9.5, 8.0, 7.0], 100.0)
    np.testing.assert_allclose(depths, [100.0, 99.0, 97.0, 94.0])


def test_marker_rmse() -> None:
    """RMSE of depth differences."""
    assert markerDepthRmse([1.0, 2.0], [1.0, 4.0]) == pytest.approx(
        np.sqrt(2.0)
    )
    with pytest.raises(ValueError):
        markerDepthRmse([1.0], [1.0, 2.0])


def test_facies_from_environments() -> None:
    """Environments become facies with clipped water-depth criteria."""
    model = CarbonateOpenRampDepositionalEnvironmentModel()
    facies = faciesFromDepositionalEnvironments(model, -5.0, 300.0)
    byName = {f.name: f for f in facies}
    assert set(byName) == {e.name for e in model.environments}
    continent = byName["Continent"].getCriteria("waterDepth")
    assert continent is not None
    assert (continent.minRange, continent.maxRange) == (-5.0, -2.0)
    basin = byName["Basin"].getCriteria("waterDepth")
    assert basin is not None
    assert basin.maxRange == 300.0
