# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pywellsfm.io.depositional_environment_model_io import (
    depositionalEnvironmentModelToJsonObj,
    loadDepositionalEnvironmentModel,
    loadDepositionalEnvironmentModelFromJsonObj,
    saveDepositionalEnvironmentModel,
)
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelUniform,
    EnvironmentConditionsModel,
)


def _base_payload() -> dict[str, Any]:
    return {
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
                            "format": (
                                "pyWellSFM.EnvironmentConditionModelData"
                            ),
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


def _make_model() -> DepositionalEnvironmentModel:
    environment = DepositionalEnvironment(
        name="OuterRamp",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", 20.0, 50.0
        ),
        envConditionsModel=EnvironmentConditionsModel(
            [EnvironmentConditionModelUniform("energy", 0.0, 0.2)]
        ),
        distality=2.0,
    )
    return DepositionalEnvironmentModel(
        name="RoundtripModel",
        environments=[environment],
    )


def test_load_depositional_environment_model_from_json_obj() -> None:
    """Load a valid depositional environment model from JSON."""
    model = loadDepositionalEnvironmentModelFromJsonObj(_base_payload())

    assert model.name == "MyModel"
    assert model.getEnvironmentCount() == 1

    env = model.getEnvironmentByName("Lagoon")
    assert env is not None
    assert env.waterDepth_range == (0.0, 10.0)
    assert env.distality == 0.2
    assert env.envConditionsModel.environmentConditionNames == ["energy"]


def test_load_rejects_legacy_environment_fields() -> None:
    """Reject legacy fields not allowed by the current schema."""
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
    """Round-trip model save/load while preserving core values."""
    model = _make_model()

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
    """Reject duplicated environment names in one model."""
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


def test_load_rejects_non_object_environment_item() -> None:
    """Reject non-object entries in the environments array."""
    payload = _base_payload()
    payload["environments"] = ["invalid"]

    with pytest.raises(
        ValueError,
        match=r"environments\[0\] must be an object",
    ):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_load_rejects_non_object_water_depth_model() -> None:
    """Reject waterDepthModel when it is not an object."""
    payload = _base_payload()
    payload["environments"][0]["waterDepthModel"] = 10

    with pytest.raises(ValueError, match="waterDepthModel must be an object"):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_load_rejects_non_statistical_water_depth_model() -> None:
    """Reject waterDepthModel that is not a statistical model."""
    payload = _base_payload()
    payload["environments"][0]["waterDepthModel"] = {
        "format": "pyWellSFM.EnvironmentConditionModelData",
        "version": "1.0",
        "model": {
            "modelType": "Combination",
            "models": [
                {"modelType": "Constant", "value": 1.0},
                {"modelType": "Constant", "value": 2.0},
            ],
        },
    }

    with pytest.raises(
        ValueError,
        match="must resolve to a statistical environment condition model",
    ):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_load_rejects_non_numeric_distality() -> None:
    """Reject non-numeric distality values."""
    payload = _base_payload()
    payload["environments"][0]["distality"] = "far"

    with pytest.raises(ValueError, match="distality must be a number"):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_load_rejects_non_object_environment_conditions_model() -> None:
    """Reject environmentConditionsModel when not an object."""
    payload = _base_payload()
    payload["environments"][0]["environmentConditionsModel"] = "invalid"

    with pytest.raises(
        ValueError,
        match="environmentConditionsModel must be an object",
    ):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_load_rejects_empty_name() -> None:
    """Reject an empty top-level model name."""
    payload = _base_payload()
    payload["name"] = ""

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_load_rejects_empty_environments() -> None:
    """Reject an empty environments collection during load."""
    payload = _base_payload()
    payload["environments"] = []

    with pytest.raises(
        ValueError,
        match="environments must be a non-empty list",
    ):
        loadDepositionalEnvironmentModelFromJsonObj(payload)


def test_serialize_rejects_empty_model_name() -> None:
    """Reject serialization when model name is empty."""
    model = _make_model()
    model.name = ""

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        depositionalEnvironmentModelToJsonObj(model)


def test_serialize_rejects_empty_environments() -> None:
    """Reject serialization when environment list is empty."""
    model = _make_model()
    model.environments = []

    with pytest.raises(
        ValueError,
        match="environments must be a non-empty list",
    ):
        depositionalEnvironmentModelToJsonObj(model)


def test_load_rejects_non_json_file_extension(tmp_path: Path) -> None:
    """Reject loading files that are not JSON."""
    path = tmp_path / "model.txt"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected '.json', got '.txt'"):
        loadDepositionalEnvironmentModel(str(path))


def test_save_rejects_non_json_file_extension(tmp_path: Path) -> None:
    """Reject saving models to non-JSON extensions."""
    model = _make_model()

    with pytest.raises(ValueError, match="must have a .json extension"):
        saveDepositionalEnvironmentModel(model, str(tmp_path / "model.txt"))


def test_saved_payload_has_no_legacy_fields(tmp_path: Path) -> None:
    """Ensure saved payload does not contain removed legacy fields."""
    model = DepositionalEnvironmentModel(
        name="NoLegacy",
        environments=[
            DepositionalEnvironment(
                name="A",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, 1.0
                ),
            )
        ],
    )
    path = tmp_path / "model.json"
    saveDepositionalEnvironmentModel(model, str(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    env_obj = raw["environments"][0]
    assert "waterDepthModel" in env_obj
    assert "waterDepth_range" not in env_obj
    assert "other_property_ranges" not in env_obj
    assert "property_curves" not in env_obj
