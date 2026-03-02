# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.io.depositional_environment_model_io import (  # noqa: E402
    depositionalEnvironmentModelToJsonObj,
    loadDepositionalEnvironmentModel,
    loadDepositionalEnvironmentModelFromJsonObj,
    saveDepositionalEnvironmentModel,
)
from pywellsfm.model.Curve import Curve  # noqa: E402
from pywellsfm.model.DepositionalEnvironment import (  # noqa: E402
    CarbonateOpenRampDepositionalEnvironmentModel,
    CarbonateProtectedRampDepositionalEnvironmentModel,
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)

#############################################################################
#                  Tests for deposition environment model I/O.              #
#############################################################################


def _curve_json() -> dict[str, object]:
    return {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": "bathymetry",
            "yAxisName": "energy",
            "interpolationMethod": "linear",
            "data": [
                {"x": 10.0, "y": 0.0},
                {"x": 20.0, "y": 0.3},
                {"x": 30.0, "y": 0.9},
            ],
        },
    }


def test_loadDepositionalEnvironmentModelFromJsonObj_inline_curve() -> None:
    """Loads model JSON with inline property curve."""
    payload = {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "MyModel",
        "environments": [
            {
                "name": "Lagoon",
                "bathymetry_range": [0.0, 10.0],
                "distality": 0.2,
                "other_property_ranges": {
                    "bathymetry": [20.0, 30.0],
                    "energy": [0.0, 1.0],
                },
                "property_curves": {
                    "energy_vs_bathymetry": _curve_json(),
                },
            }
        ],
    }

    model = loadDepositionalEnvironmentModelFromJsonObj(payload)

    assert model.name == "MyModel"
    assert model.getEnvironmentCount() == 1

    env = model.getEnvironmentByName("Lagoon")
    assert env is not None
    assert env.bathymetry_range == (0.0, 10.0)
    assert env.distality == 0.2
    assert env.other_property_ranges["bathymetry"] == (20.0, 30.0)
    assert env.other_property_ranges["energy"] == (0.0, 1.0)
    assert "energy_vs_bathymetry" in env.property_curves

    curve = env.property_curves["energy_vs_bathymetry"]
    assert curve._xAxisName == "bathymetry"
    assert curve._yAxisName == "energy"
    assert np.isclose(curve.getValueAt(25.0), 0.6)


def test_loadDepositionalEnvironmentModel_curve_reference(
    tmp_path: Path,
) -> None:
    """Loads model JSON where property curve is a URL reference."""
    curve_path = tmp_path / "energy_curve.json"
    curve_path.write_text(json.dumps(_curve_json()), encoding="utf-8")

    payload = {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "RefModel",
        "environments": [
            {
                "name": "InnerRamp",
                "bathymetry_range": [0.0, 5.0],
                "other_property_ranges": {
                    "bathymetry": [20.0, 30.0],
                    "energy": [0.0, 1.0],
                },
                "property_curves": {
                    "energy_vs_bathymetry": {"url": "energy_curve.json"},
                },
            }
        ],
    }

    model = loadDepositionalEnvironmentModelFromJsonObj(
        payload,
        base_dir=str(tmp_path),
    )

    env = model.getEnvironmentByName("InnerRamp")
    assert env is not None
    assert "energy_vs_bathymetry" in env.property_curves
    assert np.isclose(
        env.property_curves["energy_vs_bathymetry"].getValueAt(30.0),
        0.9,
    )


def test_save_and_loadDepositionalEnvironmentModel_roundtrip(
    tmp_path: Path,
) -> None:
    """Round-trips model through save/load and preserves key values."""
    curve = Curve(
        "bathymetry",
        "energy",
        np.asarray([20.0, 30.0], dtype=float),
        np.asarray([0.1, 0.9], dtype=float),
        "linear",
    )
    environment = DepositionalEnvironment(
        name="OuterRamp",
        bathymetry_range=(20.0, 50.0),
        other_property_ranges={
            "bathymetry": (10.0, 20.0),
            "energy": (0.0, 0.2),
        },
        distality=2.0,
    )
    environment.setPropertyCurve(curve)

    model = DepositionalEnvironmentModel(
        name="RoundtripModel",
        environments=[environment],
    )

    payload = depositionalEnvironmentModelToJsonObj(model)
    assert payload["format"] == "pyWellSFM.DepositionalEnvironmentModelSchema"
    assert payload["version"] == "1.0"
    assert payload["name"] == "RoundtripModel"

    out_path = tmp_path / "deposition_model.json"
    saveDepositionalEnvironmentModel(model, str(out_path))
    loaded = loadDepositionalEnvironmentModel(str(out_path))

    loaded_env = loaded.getEnvironmentByName("OuterRamp")
    assert loaded_env is not None
    assert loaded_env.bathymetry_range == (20.0, 50.0)
    assert loaded_env.distality == 2.0
    assert loaded_env.other_property_ranges["bathymetry"] == (10.0, 20.0)
    assert loaded_env.other_property_ranges["energy"] == (0.0, 0.2)
    assert "energy_vs_bathymetry" in loaded_env.property_curves


#############################################################################
#                       Tests for DepositionEnvironment                     #
#############################################################################


def _energy_vs_bathymetry_curve() -> Curve:
    return Curve(
        "bathymetry",
        "energy",
        np.asarray([0.0, 10.0], dtype=float),
        np.asarray([0.0, 1.0], dtype=float),
        "linear",
    )


def test_depositional_environment_constructor_normalizes_ranges() -> None:
    """Constructor stores sorted bounds for bathymetry and properties."""
    env = DepositionalEnvironment(
        name="Lagoon",
        bathymetry_range=(10.0, 0.0),
        other_property_ranges={
            "energy": (1.0, 0.0),
            "temperature": (30.0, 20.0),
        },
        distality=0.3,
    )

    assert env.bathymetry_range == (0.0, 10.0)
    assert env.other_property_ranges["energy"] == (0.0, 1.0)
    assert env.other_property_ranges["temperature"] == (20.0, 30.0)
    assert env.distality == 0.3


def test_depositional_environment_equality_hash_and_repr() -> None:
    """Objects compare by content and hash/repr rely on name."""
    env1 = DepositionalEnvironment(
        name="OuterRamp",
        bathymetry_range=(20.0, 50.0),
        other_property_ranges={"energy": (0.0, 0.2)},
        distality=2.0,
    )
    env2 = DepositionalEnvironment(
        name="OuterRamp",
        bathymetry_range=(20.0, 50.0),
        other_property_ranges={"energy": (0.0, 0.2)},
        distality=2.0,
    )

    assert env1 == env2
    assert hash(env1) == hash(env2)
    assert repr(env1) == "OuterRamp"


def test_depositional_environment_mid_and_width_helpers() -> None:
    """Mid-point and width helpers return expected values."""
    env = DepositionalEnvironment(
        name="Shore",
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={"energy": (0.2, 0.8)},
    )

    assert np.isclose(env.bathymetry_mid, 5.0)
    assert np.isclose(env.bathymetry_range_width, 10.0)
    assert np.isclose(env.getPropertyMid("energy"), 0.5)
    assert np.isclose(env.getPropertyRangeWidth("energy"), 0.6)


def test_depositional_environment_property_helpers_raise_for_missing() -> None:
    """Missing property access raises ValueError."""
    env = DepositionalEnvironment(
        name="Basin", bathymetry_range=(100.0, 200.0)
    )

    with pytest.raises(ValueError):
        env.getPropertyMid("energy")
    with pytest.raises(ValueError):
        env.getPropertyRangeWidth("energy")


def test_depositional_environment_set_other_property_range() -> None:
    """SetOtherPropertyRange adds/overwrites property bounds."""
    env = DepositionalEnvironment(name="Lagoon", bathymetry_range=(0.0, 5.0))

    env.setOtherPropertyRange("energy", (0.0, 0.4))
    assert env.other_property_ranges["energy"] == (0.0, 0.4)

    env.setOtherPropertyRange("energy", (0.1, 0.5))
    assert env.other_property_ranges["energy"] == (0.1, 0.5)


def test_depositional_environment_set_curve_and_get_value() -> None:
    """Property curve can be stored and queried through helper."""
    env = DepositionalEnvironment(
        name="InnerRamp",
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={
            "bathymetry": (0.0, 10.0),
            "energy": (0.0, 1.0),
        },
    )
    env.setPropertyCurve(_energy_vs_bathymetry_curve())

    assert "energy_vs_bathymetry" in env.property_curves
    assert np.isclose(
        env.getValueFromCurveAt("bathymetry", "energy", 5.0), 0.5
    )
    assert np.isclose(
        env.getValueFromCurveAt("bathymetry", "energy", -10.0),
        0.0,
    )
    assert np.isclose(
        env.getValueFromCurveAt("bathymetry", "energy", 20.0),
        1.0,
    )


def test_depositional_environment_curve_value_fallbacks() -> None:
    """Edge cases for zero-width y and x property ranges are handled."""
    env_y_width_zero = DepositionalEnvironment(
        name="FlatEnergy",
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={
            "bathymetry": (0.0, 10.0),
            "energy": (0.4, 0.4),
        },
    )
    env_y_width_zero.setPropertyCurve(_energy_vs_bathymetry_curve())
    assert np.isclose(
        env_y_width_zero.getValueFromCurveAt("bathymetry", "energy", 7.0),
        0.4,
    )

    env_x_width_zero = DepositionalEnvironment(
        name="FlatBathymetry",
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={
            "bathymetry": (5.0, 5.0),
            "energy": (0.2, 0.6),
        },
    )
    env_x_width_zero.setPropertyCurve(_energy_vs_bathymetry_curve())
    assert np.isclose(
        env_x_width_zero.getValueFromCurveAt("bathymetry", "energy", 7.0),
        0.4,
    )


def test_depositional_environment_get_value_from_curve_errors() -> None:
    """Missing curve or missing property definitions raise ValueError."""
    env_no_curve = DepositionalEnvironment(
        name="NoCurve",
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={
            "bathymetry": (0.0, 10.0),
            "energy": (0.0, 1.0),
        },
    )
    with pytest.raises(ValueError):
        env_no_curve.getValueFromCurveAt("bathymetry", "energy", 5.0)

    env_missing_property = DepositionalEnvironment(
        name="MissingProperty",
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={"bathymetry": (0.0, 10.0)},
    )
    curve = Curve(
        "bathymetry",
        "salinity",
        np.asarray([0.0, 10.0], dtype=float),
        np.asarray([20.0, 40.0], dtype=float),
        "linear",
    )
    env_missing_property.setPropertyCurve(curve)
    with pytest.raises(ValueError):
        env_missing_property.getValueFromCurveAt("bathymetry", "salinity", 5.0)


#############################################################################
#                     Tests for DepositionalEnvironmentModel                #
#############################################################################


def _make_environment(name: str) -> DepositionalEnvironment:
    return DepositionalEnvironment(
        name=name,
        bathymetry_range=(0.0, 10.0),
        other_property_ranges={"energy": (0.0, 1.0)},
    )


def test_depositional_environment_model_add_get_exists_and_duplicate() -> None:
    """Model adds environments, detects duplicates and retrieves by name."""
    env = _make_environment("Lagoon")
    model = DepositionalEnvironmentModel(name="M", environments=[])

    model.addEnvironment(env)
    assert model.getEnvironmentCount() == 1
    assert model.environmentExists("Lagoon")
    assert model.getEnvironmentByName("Lagoon") is env

    model.addEnvironment(_make_environment("Lagoon"))
    assert model.getEnvironmentCount() == 1


def test_depositional_environment_model_add_set_and_remove() -> None:
    """Model supports batch add/remove with sets."""
    model = DepositionalEnvironmentModel(name="M", environments=[])
    env_set = {_make_environment("A"), _make_environment("B")}

    model.addEnvironment(env_set)
    assert model.getEnvironmentCount() == 2
    assert model.environmentExists("A")
    assert model.environmentExists("B")

    model.removeEnvironment({"A", "B"})
    assert model.getEnvironmentCount() == 0
    assert model.isEmpty()


def test_depositional_environment_model_remove_by_name_noop_when_missing() -> (
    None
):
    """Removing missing name is a no-op."""
    model = DepositionalEnvironmentModel(
        name="M",
        environments=[_make_environment("A")],
    )

    model.removeEnvironment("NotThere")
    assert model.getEnvironmentCount() == 1


def test_depositional_environment_model_clear_and_get_missing() -> None:
    """Clearing empties model and missing lookup returns None."""
    model = DepositionalEnvironmentModel(
        name="M",
        environments=[_make_environment("A"), _make_environment("B")],
    )

    assert model.getEnvironmentByName("C") is None
    model.clearAllEnvironments()
    assert model.isEmpty()
    assert model.getEnvironmentCount() == 0


def test_depositional_environment_model_type_errors() -> None:
    """Type checking for add/remove methods raises TypeError."""
    model = DepositionalEnvironmentModel(name="M", environments=[])

    with pytest.raises(TypeError):
        model.addEnvironment("invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.removeEnvironment(123)  # type: ignore[arg-type]


#############################################################################
#             Tests for derived DepositionalEnvironmentModel classes        #
#############################################################################


def test_carbonate_open_ramp_default_environments() -> None:
    """Open ramp model builds expected default environment set."""
    model = CarbonateOpenRampDepositionalEnvironmentModel()

    assert model.name == "Carbonate Open Ramp"
    assert model.getEnvironmentCount() == 7
    assert model.environmentExists("Sabkha")
    assert model.environmentExists("InnerRampUpperShoreface")
    assert model.environmentExists("Basin")

    sabkha = model.getEnvironmentByName("Sabkha")
    basin = model.getEnvironmentByName("Basin")
    assert sabkha is not None
    assert basin is not None
    assert sabkha.bathymetry_range == (-2.0, 0.0)
    assert basin.bathymetry_range == (1000.0, 10000.0)
    assert basin.other_property_ranges["energy"] == (0.0, 0.0)


def test_carbonate_open_ramp_parameters_update_ranges() -> None:
    """Open ramp constructor parameters propagate to environment ranges."""
    model = CarbonateOpenRampDepositionalEnvironmentModel(
        tidal_range=3.0,
        fairweather_wave_breaking_bathymetry=6.0,
        fairweather_wave_base_bathymetry=30.0,
        storm_wave_base_bathymetry=60.0,
        shelf_break_bathymetry=220.0,
        slope_toe_max_bathymetry=1200.0,
    )

    sabkha = model.getEnvironmentByName("Sabkha")
    upper_shoreface = model.getEnvironmentByName("InnerRampUpperShoreface")
    lower_shoreface = model.getEnvironmentByName("InnerRampLowerShoreface")
    outer_ramp = model.getEnvironmentByName("OuterRamp")
    shelf_slope = model.getEnvironmentByName("ShelfSlope")

    assert sabkha is not None
    assert upper_shoreface is not None
    assert lower_shoreface is not None
    assert outer_ramp is not None
    assert shelf_slope is not None

    assert sabkha.bathymetry_range == (-3.0, 0.0)
    assert upper_shoreface.bathymetry_range == (0.0, 6.0)
    assert lower_shoreface.bathymetry_range == (6.0, 30.0)
    assert outer_ramp.bathymetry_range == (30.0, 60.0)
    assert shelf_slope.bathymetry_range == (220.0, 1200.0)


def test_carbonate_protected_ramp_default_environments() -> None:
    """Protected ramp model builds expected default environment set."""
    model = CarbonateProtectedRampDepositionalEnvironmentModel()

    assert model.name == "Carbonate Protected Ramp"
    assert model.getEnvironmentCount() == 10
    assert model.environmentExists("Lagoon")
    assert model.environmentExists("ReefCrest")
    assert model.environmentExists("Basin")

    lagoon = model.getEnvironmentByName("Lagoon")
    fore_reef = model.getEnvironmentByName("ForeReef")
    assert lagoon is not None
    assert fore_reef is not None
    assert lagoon.bathymetry_range == (2.0, 10.0)
    assert fore_reef.bathymetry_range == (1.0, 20.0)
    assert fore_reef.other_property_ranges["energy"] == (0.2, 0.7)


def test_carbonate_protected_ramp_parameters_update_ranges() -> None:
    """Protected ramp constructor parameters propagate to key ranges."""
    model = CarbonateProtectedRampDepositionalEnvironmentModel(
        tidal_range=4.0,
        lagoon_max_bathymetry=12.0,
        fairweather_wave_base_bathymetry=25.0,
        storm_wave_base_bathymetry=70.0,
        shelf_break_bathymetry=250.0,
        slope_toe_max_bathymetry=1400.0,
    )

    sabkha = model.getEnvironmentByName("Sabkha")
    lagoon = model.getEnvironmentByName("Lagoon")
    fore_reef = model.getEnvironmentByName("ForeReef")
    outer_ramp = model.getEnvironmentByName("OuterRamp")
    shelf_slope = model.getEnvironmentByName("ShelfSlope")

    assert sabkha is not None
    assert lagoon is not None
    assert fore_reef is not None
    assert outer_ramp is not None
    assert shelf_slope is not None

    assert sabkha.bathymetry_range == (-4.0, 0.0)
    assert lagoon.bathymetry_range == (2.0, 12.0)
    assert fore_reef.bathymetry_range == (1.0, 25.0)
    assert outer_ramp.bathymetry_range == (25.0, 70.0)
    assert shelf_slope.bathymetry_range == (250.0, 1400.0)
