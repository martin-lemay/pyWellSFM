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
    """Builds the protected (rimmed) platform model with expected defaults."""
    model = CarbonateProtectedRampDepositionalEnvironmentModel()

    assert model.name == "Carbonate Protected Ramp"
    assert [e.name for e in model.environments] == [
        "Continent",
        "TidalFlat",
        "Lagoon",
        "Buildup",
        "BackReef",
        "ReefFlat",
        "ForeReef",
        "OuterPlatform",
        "Basin",
    ]
    expected = {
        "Continent": (-10000.0, -1.0),
        "TidalFlat": (-1.0, 1.0),
        "Lagoon": (1.0, 10.0),
        "Buildup": (1.0, 10.0),
        "BackReef": (0.0, 10.0),
        "ReefFlat": (0.0, 8.0),
        "ForeReef": (5.0, 40.0),
        "OuterPlatform": (30.0, 200.0),
        "Basin": (200.0, 10000.0),
    }
    for env in model.environments:
        assert env.waterDepth_range == expected[env.name]
    distalities = [e.distality for e in model.environments]
    assert distalities == [-1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    weights = {e.name: e.weight for e in model.environments}
    assert weights.pop("Buildup") == pytest.approx(0.1)
    assert set(weights.values()) == {1.0}

    lagoon = model.getEnvironmentByName("Lagoon")
    assert lagoon is not None
    assert set(lagoon.envConditionsModel.environmentConditionNames) == {
        "energy",
        "salinity",
    }


def test_carbonate_protected_ramp_parameters() -> None:
    """Water depth limits of the protected platform are parameterized."""
    model = CarbonateProtectedRampDepositionalEnvironmentModel(
        tidal_range=4.0,
        lagoon_max_waterDepth=15.0,
        reef_flat_max_waterDepth=6.0,
        fairweather_wave_breaking_waterDepth=3.0,
        fairweather_wave_base_waterDepth=20.0,
        storm_wave_base_waterDepth=50.0,
        shelf_break_waterDepth=150.0,
        buildup_weight=0.3,
    )
    ranges = {e.name: e.waterDepth_range for e in model.environments}
    assert ranges["TidalFlat"] == (-2.0, 2.0)
    assert ranges["Lagoon"] == (2.0, 15.0)
    assert ranges["Buildup"] == (2.0, 15.0)
    assert ranges["BackReef"] == (0.0, 15.0)
    assert ranges["ReefFlat"] == (0.0, 6.0)
    assert ranges["ForeReef"] == (3.0, 50.0)
    assert ranges["OuterPlatform"] == (20.0, 150.0)
    assert ranges["Basin"] == (150.0, 10000.0)
    buildup = model.getEnvironmentByName("Buildup")
    assert buildup is not None
    assert buildup.weight == pytest.approx(0.3)


def test_carbonate_presets_share_wave_base_defaults() -> None:
    """Open and protected presets use the same default wave bases."""
    openRamp = CarbonateOpenRampDepositionalEnvironmentModel()
    protected = CarbonateProtectedRampDepositionalEnvironmentModel()
    outerRamp = openRamp.getEnvironmentByName("OuterRamp")
    outerPlatform = protected.getEnvironmentByName("OuterPlatform")
    lowerShoreface = openRamp.getEnvironmentByName("InnerRampLowerShoreface")
    foreReef = protected.getEnvironmentByName("ForeReef")
    buildup = openRamp.getEnvironmentByName("Buildup")
    assert outerRamp is not None and outerPlatform is not None
    assert lowerShoreface is not None and foreReef is not None
    assert buildup is not None
    # fair-weather wave base (30 m)
    assert outerRamp.waterDepth_min == outerPlatform.waterDepth_min == 30.0
    assert lowerShoreface.waterDepth_max == 30.0
    # storm wave base (40 m)
    assert foreReef.waterDepth_max == buildup.waterDepth_max == 40.0


def test_environment_weight() -> None:
    """Environment weight defaults to 1 and must be non-negative."""
    env = _make_environment("A", 0.0, 10.0)
    assert env.weight == 1.0
    weighted = DepositionalEnvironment(
        "B",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", 0.0, 10.0
        ),
        weight=0.2,
    )
    assert weighted.weight == pytest.approx(0.2)
    with pytest.raises(ValueError, match="non-negative"):
        DepositionalEnvironment(
            "C",
            waterDepthModel=EnvironmentConditionModelUniform(
                "waterDepth", 0.0, 10.0
            ),
            weight=-1.0,
        )


def test_open_ramp_preset_water_depth_ranges_are_contiguous() -> None:
    """Open ramp environments (except buildups) tile the water depth axis."""
    model = CarbonateOpenRampDepositionalEnvironmentModel(
        shelf_break_waterDepth=150.0
    )
    envs = sorted(
        (e for e in model.environments if e.name != "Buildup"),
        key=lambda e: e.waterDepth_min,
    )
    for shallower, deeper in zip(envs[:-1], envs[1:], strict=True):
        assert shallower.waterDepth_max == deeper.waterDepth_min
    outerRamp = model.getEnvironmentByName("OuterRamp")
    assert outerRamp is not None
    assert outerRamp.waterDepth_range == (30.0, 150.0)
