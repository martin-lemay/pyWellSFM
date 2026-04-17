# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402, D103, E501

from __future__ import annotations

import os
import sys

import pytest

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.model.DepositionalEnvironment import (
    CarbonateOpenRampDepositionalEnvironmentModel,
    CarbonateProtectedRampDepositionalEnvironmentModel,
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelConstant,
    EnvironmentConditionModelUniform,
    EnvironmentConditionsModel,
)


def _make_environment(
    name: str,
    min_depth: float,
    max_depth: float,
    distality: float | None = None,
) -> DepositionalEnvironment:
    return DepositionalEnvironment(
        name=name,
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth",
            min_depth,
            max_depth,
        ),
        distality=distality,
    )


#############################################################################
#                       Tests for DepositionEnvironment                     #
#############################################################################


def test_depositional_environment_equality_hash_and_repr() -> None:
    """Checks equality, hash and repr for matching environments."""
    env1 = _make_environment("OuterRamp", 20.0, 50.0, distality=2.0)
    env2 = _make_environment("OuterRamp", 20.0, 50.0, distality=2.0)

    assert env1 == env2
    assert hash(env1) == hash(env2)
    assert repr(env1) == "OuterRamp"


def test_depositional_environment_waterdepth_helpers() -> None:
    """Checks water depth helper properties."""
    env = _make_environment("Shore", 0.0, 10.0)

    assert env.waterDepth_range == (0.0, 10.0)
    assert env.waterDepth_min == 0.0
    assert env.waterDepth_max == 10.0
    assert env.waterDepth_rangeRef == 5.0
    assert env.waterDepth_rangeWidth == 10.0


def test_get_environment_conditions() -> None:
    """Returns configured environment conditions."""
    env = DepositionalEnvironment(
        name="Lagoon",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", 2.0, 10.0
        ),
        envConditionsModel=EnvironmentConditionsModel(
            [
                EnvironmentConditionModelConstant("energy", 0.1),
                EnvironmentConditionModelConstant("temperature", 27.0),
            ]
        ),
    )

    values = env.getEnvironmentConditions(waterDepth=5.0, age=0.0)
    assert values["energy"] == 0.1
    assert values["temperature"] == 27.0


def test_depositional_environment_eq_returns_false_for_other_type() -> None:
    """Returns false when compared to a non environment object."""
    env = _make_environment("OuterRamp", 20.0, 50.0, distality=2.0)

    assert env != "OuterRamp"


#############################################################################
#                     Tests for DepositionalEnvironmentModel                #
#############################################################################


def test_depositional_environment_model_equality_respects_content() -> None:
    """Compares model equality by values, independent from order."""
    left = DepositionalEnvironmentModel(
        name="M",
        environments=[
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 20.0),
        ],
    )
    right_same_content_different_order = DepositionalEnvironmentModel(
        name="M",
        environments=[
            _make_environment("B", 10.0, 20.0),
            _make_environment("A", 0.0, 10.0),
        ],
    )
    right_different = DepositionalEnvironmentModel(
        name="M",
        environments=[
            _make_environment("A", 0.0, 10.0),
            _make_environment("C", 20.0, 30.0),
        ],
    )

    assert left == right_same_content_different_order
    assert left != right_different


def test_depositional_environment_model_eq_other_type_and_name() -> None:
    """Returns false for other type and different model name."""
    env = _make_environment("A", 0.0, 10.0)
    model = DepositionalEnvironmentModel(name="M", environments=[env])
    same_env_other_name = DepositionalEnvironmentModel(
        name="N",
        environments=[_make_environment("A", 0.0, 10.0)],
    )

    assert model != "M"
    assert model != same_env_other_name


def test_depositional_environment_model_add_get_exists_and_duplicate() -> None:
    """Adds, checks, gets and rejects duplicate environments."""
    env = _make_environment("Lagoon", 0.0, 10.0)
    model = DepositionalEnvironmentModel(name="M", environments=[])

    model.addEnvironment(env)
    assert model.getEnvironmentCount() == 1
    assert model.environmentExists("Lagoon")
    assert model.getEnvironmentByName("Lagoon") is env

    model.addEnvironment(_make_environment("Lagoon", 0.0, 10.0))
    assert model.getEnvironmentCount() == 1


def test_depositional_environment_model_add_set_and_remove() -> None:
    """Adds a set then removes matching environment names."""
    model = DepositionalEnvironmentModel(name="M", environments=[])
    env_set = {
        _make_environment("A", 0.0, 10.0),
        _make_environment("B", 10.0, 20.0),
    }

    model.addEnvironment(env_set)
    assert model.getEnvironmentCount() == 2
    assert model.environmentExists("A")
    assert model.environmentExists("B")

    model.removeEnvironment({"A", "B"})
    assert model.getEnvironmentCount() == 0
    assert model.isEmpty()


def test_remove_environment_unknown_name_keeps_collection() -> None:
    """Keeps environments unchanged when name is absent."""
    model = DepositionalEnvironmentModel(
        name="M",
        environments=[
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 20.0),
        ],
    )

    model.removeEnvironment("C")
    assert model.getEnvironmentCount() == 2
    assert model.environmentExists("A")
    assert model.environmentExists("B")


def test_depositional_environment_model_clear_and_type_errors() -> None:
    """Raises type errors and clears all environments."""
    model = DepositionalEnvironmentModel(
        name="M",
        environments=[_make_environment("A", 0.0, 10.0)],
    )

    with pytest.raises(TypeError):
        model.addEnvironment("invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.removeEnvironment(123)  # type: ignore[arg-type]

    model.clearAllEnvironments()
    assert model.isEmpty()


def test_get_environment_by_name_returns_none_when_missing() -> None:
    """Returns none when environment name does not exist."""
    model = DepositionalEnvironmentModel(
        name="M",
        environments=[_make_environment("A", 0.0, 10.0)],
    )

    assert model.getEnvironmentByName("B") is None


#############################################################################
#             Tests for derived DepositionalEnvironmentModel classes        #
#############################################################################


def test_carbonate_open_ramp_default_environments() -> None:
    """Builds open ramp model with expected default environments."""
    model = CarbonateOpenRampDepositionalEnvironmentModel()

    assert model.name == "Carbonate Open Ramp"
    assert model.getEnvironmentCount() == 8
    assert model.environmentExists("SupraTidal")
    assert model.environmentExists("InnerRampUpperShoreface")
    assert model.environmentExists("Basin")

    supraTidal = model.getEnvironmentByName("SupraTidal")
    basin = model.getEnvironmentByName("Basin")
    assert supraTidal is not None
    assert basin is not None
    assert supraTidal.waterDepth_range == (-2.0, 0.0)
    assert basin.waterDepth_range == (1000.0, 10000.0)


def test_carbonate_protected_ramp_default_environments() -> None:
    """Builds protected ramp model with expected defaults."""
    model = CarbonateProtectedRampDepositionalEnvironmentModel()

    assert model.name == "Carbonate Protected Ramp"
    assert model.getEnvironmentCount() == 11
    assert model.environmentExists("Lagoon")
    assert model.environmentExists("ReefCrest")
    assert model.environmentExists("Basin")

    lagoon = model.getEnvironmentByName("Lagoon")
    fore_reef = model.getEnvironmentByName("ForeReef")
    assert lagoon is not None
    assert fore_reef is not None
    assert lagoon.waterDepth_range == (2.0, 10.0)
    assert fore_reef.waterDepth_range == (1.0, 20.0)
