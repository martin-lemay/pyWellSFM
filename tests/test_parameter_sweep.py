# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
import os
import sys

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

from typing import Any

import numpy as np
import pytest

from pywellsfm.algorithms import parameterGrid, runParameterSweep
from pywellsfm.model import (
    AccumulationModel,
    AccumulationModelElementOptimum,
    Curve,
    Marker,
    RealizationData,
    Scenario,
    SubsidenceType,
    Well,
)
from pywellsfm.simulator import FSSimulator


def test_parameter_grid() -> None:
    """Grid lists all combinations in row-major order."""
    grid = parameterGrid({"a": [1, 2], "b": ["x", "y"]})
    assert grid == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert parameterGrid(None) == [{}]
    assert parameterGrid({}) == [{}]


def _build(
    scenarioParams: dict[str, Any], realizationParams: list[dict[str, Any]]
) -> FSSimulator:
    """Build a constant-rate simulator: one realization per subsidence."""
    rate = scenarioParams["rate"]
    model = AccumulationModel(
        "constant", {"A": AccumulationModelElementOptimum("A", rate)}
    )
    scenario = Scenario("sweep", model, None)
    realizations = []
    for params in realizationParams:
        well = Well("W", np.zeros(3), 100.0)
        well.addMarkers([Marker("Base", 100.0, 1.0), Marker("Top", 0.0, 0.0)])
        subs = params.get("subsidence", 0.0)
        realizations.append(
            RealizationData(
                well,
                10.0,
                None,
                Curve(
                    "Age",
                    "Subsidence",
                    np.array([0.0, 2.0]),
                    np.array([subs, subs]),
                ),
                SubsidenceType.RATE,
            )
        )
    return FSSimulator(scenario, realizations)


def _evaluate(
    simulator: FSSimulator, r: int, params: dict[str, Any]
) -> dict[str, float]:
    """Return total thickness of a realization."""
    assert simulator.outputs is not None
    total = simulator.outputs["thickness_cumul"].isel(realization=r).values
    return {"thickness": float(total[-1])}


def test_run_parameter_sweep() -> None:
    """Sweep returns one row per combination with metrics and runtime."""
    table = runParameterSweep(
        {"rate": [1.0, 2.0]},
        _build,
        _evaluate,
        realizationGrid={"subsidence": [0.0, 5.0]},
    )
    assert len(table) == 4
    assert list(table["rate"]) == [1.0, 1.0, 2.0, 2.0]
    np.testing.assert_allclose(table["thickness"], table["rate"] * 1.0)
    assert (table["runtime_s"] > 0).all()
    np.testing.assert_allclose(
        table["runtime_per_realization_s"], table["runtime_s"] / 2
    )


def test_run_parameter_sweep_checks_realization_count() -> None:
    """A builder returning the wrong number of realizations fails."""

    def badBuild(
        scenarioParams: dict[str, Any], realizationParams: list[dict[str, Any]]
    ) -> FSSimulator:
        return _build(scenarioParams, realizationParams[:1])

    with pytest.raises(ValueError):
        runParameterSweep(
            {"rate": [1.0]},
            badBuild,
            _evaluate,
            realizationGrid={"subsidence": [0.0, 5.0]},
        )
