# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import numpy as np
import pytest

from pywellsfm.model.Curve import Curve
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelConstant,
    EnvironmentConditionModelCurve,
    EnvironmentConditionsModel,
)
from pywellsfm.simulator.EnvironmentConditionSimulator import (
    EnvironmentConditionSimulator,
)


def _build_env_model() -> DepositionalEnvironmentModel:
    """Build a model with deterministic environment conditions."""
    shelf_env = DepositionalEnvironment(
        name="Shelf",
        waterDepthModel=EnvironmentConditionModelConstant("waterDepth", 5.0),
        envConditionsModel=EnvironmentConditionsModel(
            [EnvironmentConditionModelConstant("energy", 0.75)]
        ),
    )

    temp_curve = Curve(
        "age",
        "temperature",
        np.array([0.0, 10.0], dtype=np.float64),
        np.array([20.0, 30.0], dtype=np.float64),
        "linear",
    )
    basin_env = DepositionalEnvironment(
        name="Basin",
        waterDepthModel=EnvironmentConditionModelConstant("waterDepth", 60.0),
        envConditionsModel=EnvironmentConditionsModel(
            [EnvironmentConditionModelCurve("temperature", temp_curve)]
        ),
    )

    return DepositionalEnvironmentModel("demo", [shelf_env, basin_env])


def test_prepare_raises_when_environment_model_missing() -> None:
    """Raise when prepare is called without an environment model."""
    sim = EnvironmentConditionSimulator()

    with pytest.raises(ValueError, match="Environment model is not set"):
        sim.prepare()


def test_prepare_succeeds_when_environment_model_is_set() -> None:
    """Prepare should pass when an environment model was provided."""
    model = _build_env_model()
    sim = EnvironmentConditionSimulator()

    sim.setEnvironmentModel(model)
    sim.prepare()

    assert sim.environmentModel is model


def test_compute_raises_when_environment_model_missing() -> None:
    """Raise when computing conditions without an environment model."""
    sim = EnvironmentConditionSimulator()

    with pytest.raises(ValueError, match="Environment model is not set"):
        sim.computeEnvironmentalConditions("Shelf", 10.0, 1.0)


def test_compute_returns_depth_and_model_conditions() -> None:
    """Return water depth and deterministic conditions for known env."""
    sim = EnvironmentConditionSimulator()
    sim.setEnvironmentModel(_build_env_model())

    out = sim.computeEnvironmentalConditions("Shelf", 10, 2.0)

    assert out["waterDepth"] == 10.0
    assert out["energy"] == pytest.approx(0.75)


def test_compute_returns_depth_only_for_missing_environment() -> None:
    """Return only water depth when environment name is unknown."""
    sim = EnvironmentConditionSimulator()
    sim.setEnvironmentModel(_build_env_model())

    out = sim.computeEnvironmentalConditions("Unknown", 15.0, 3.0)

    assert out == {"waterDepth": 15.0}


def test_compute_uses_case_insensitive_name_and_age() -> None:
    """Use age input when environment condition depends on age."""
    sim = EnvironmentConditionSimulator()
    sim.setEnvironmentModel(_build_env_model())

    out = sim.computeEnvironmentalConditions("basin", 45.0, 5.0)

    assert out["waterDepth"] == 45.0
    assert out["temperature"] == pytest.approx(25.0)
