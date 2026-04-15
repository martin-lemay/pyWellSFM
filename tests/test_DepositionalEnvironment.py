# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402, D103, E501 # test-module import order/docstrings/line-length

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.io.depositional_environment_model_io import (
    depositionalEnvironmentModelToJsonObj,
    loadDepositionalEnvironmentModel,
    loadDepositionalEnvironmentModelFromJsonObj,
    saveDepositionalEnvironmentModel,
)
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
#                  Tests for deposition environment model I/O.              #
#############################################################################


def test_load_depositional_environment_model_from_json_obj() -> None:
    payload = {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "MyModel",
        "environments": [
            {
                "name": "Lagoon",
                "waterDepthModel": {
                    "format": "pyWellSFM.EnvironmentConditionModelData",
                    "version": "1.0",
                    "model": {
                        "modelType": "Uniform",
                        "minValue": 0.0,
                        "maxValue": 10.0,
                    },
                },
                "distality": 0.2,
                "environmentConditionsModel": {
                    "format": "pyWellSFM.EnvironmentConditionsModelData",
                    "version": "1.0",
                    "environmentConditions": {
                        "energy": {
                            "format": "pyWellSFM.EnvironmentConditionModelData",
                            "version": "1.0",
                            "model": {
                                "modelType": "Constant",
                                "value": 0.25,
                            },
                        }
                    },
                },
            }
        ],
    }

    model = loadDepositionalEnvironmentModelFromJsonObj(payload)

    assert model.name == "MyModel"
    assert model.getEnvironmentCount() == 1

    env = model.getEnvironmentByName("Lagoon")
    assert env is not None
    assert env.waterDepth_range == (0.0, 10.0)
    assert env.distality == 0.2
    assert env.envConditionsModel.environmentConditionNames == ["energy"]


def test_load_rejects_legacy_environment_fields() -> None:
    payload = {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "LegacyLike",
        "environments": [
            {
                "name": "Lagoon",
                "waterDepth_range": [0.0, 10.0],
            }
        ],
    }

    with pytest.raises(ValueError, match="unsupported properties"):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    environment = DepositionalEnvironment(
        name="OuterRamp",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", 20.0, 50.0
        ),
        envConditionsModel=EnvironmentConditionsModel(
            [
                EnvironmentConditionModelUniform("energy", 0.0, 0.2),
            ]
        ),
        distality=2.0,
    )

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
    assert loaded_env.waterDepth_range == (20.0, 50.0)
    assert loaded_env.distality == 2.0
    assert loaded_env.envConditionsModel.environmentConditionNames == [
        "energy"
    ]


def test_duplicate_environment_names_are_rejected() -> None:
    payload = {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "DupModel",
        "environments": [
            {
                "name": "A",
                "waterDepthModel": {
                    "format": "pyWellSFM.EnvironmentConditionModelData",
                    "version": "1.0",
                    "model": {
                        "modelType": "Constant",
                        "value": 1.0,
                    },
                },
            },
            {
                "name": "A",
                "waterDepthModel": {
                    "format": "pyWellSFM.EnvironmentConditionModelData",
                    "version": "1.0",
                    "model": {
                        "modelType": "Constant",
                        "value": 2.0,
                    },
                },
            },
        ],
    }

    with pytest.raises(ValueError, match="Duplicate environment name"):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


#############################################################################
#                       Tests for DepositionEnvironment                     #
#############################################################################


def test_depositional_environment_equality_hash_and_repr() -> None:
    env1 = _make_environment("OuterRamp", 20.0, 50.0, distality=2.0)
    env2 = _make_environment("OuterRamp", 20.0, 50.0, distality=2.0)

    assert env1 == env2
    assert hash(env1) == hash(env2)
    assert repr(env1) == "OuterRamp"


def test_depositional_environment_waterdepth_helpers() -> None:
    env = _make_environment("Shore", 0.0, 10.0)

    assert env.waterDepth_range == (0.0, 10.0)
    assert env.waterDepth_min == 0.0
    assert env.waterDepth_max == 10.0
    assert env.waterDepth_rangeRef == 5.0
    assert env.waterDepth_rangeWidth == 10.0


def test_get_environment_conditions() -> None:
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


#############################################################################
#                     Tests for DepositionalEnvironmentModel                #
#############################################################################


def test_depositional_environment_model_equality_respects_content() -> None:
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


def test_depositional_environment_model_add_get_exists_and_duplicate() -> None:
    env = _make_environment("Lagoon", 0.0, 10.0)
    model = DepositionalEnvironmentModel(name="M", environments=[])

    model.addEnvironment(env)
    assert model.getEnvironmentCount() == 1
    assert model.environmentExists("Lagoon")
    assert model.getEnvironmentByName("Lagoon") is env

    model.addEnvironment(_make_environment("Lagoon", 0.0, 10.0))
    assert model.getEnvironmentCount() == 1


def test_depositional_environment_model_add_set_and_remove() -> None:
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


def test_depositional_environment_model_clear_and_type_errors() -> None:
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


#############################################################################
#             Tests for derived DepositionalEnvironmentModel classes        #
#############################################################################


def test_carbonate_open_ramp_default_environments() -> None:
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


def test_saved_payload_has_no_legacy_fields(tmp_path: Path) -> None:
    model = DepositionalEnvironmentModel(
        name="NoLegacy",
        environments=[_make_environment("A", 0.0, 1.0)],
    )
    path = tmp_path / "model.json"
    saveDepositionalEnvironmentModel(model, str(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    env_obj = raw["environments"][0]
    assert "waterDepthModel" in env_obj
    assert "waterDepth_range" not in env_obj
    assert "other_property_ranges" not in env_obj
    assert "property_curves" not in env_obj
