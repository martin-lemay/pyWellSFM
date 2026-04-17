# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pywellsfm.io.environment_condition_model_io as env_io
from pywellsfm.io import (
    loadDepositionalEnvironmentModelFromJsonObj,
    loadEnvironmentConditionsModel,
    saveEnvironmentConditionsModel,
    validateEnvironmentConditionsModelJsonFile,
)
from pywellsfm.io.curve_io import curveToJsonObj
from pywellsfm.model.Curve import Curve
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelBase,
    EnvironmentConditionModelCombination,
    EnvironmentConditionModelConstant,
    EnvironmentConditionModelCurve,
    EnvironmentConditionModelGaussian,
    EnvironmentConditionModelTriangular,
    EnvironmentConditionModelUniform,
    EnvironmentConditionsModel,
)


def _write_json(tmp_path: Path, payload: object, filename: str) -> str:
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _wrap_model(model_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "pyWellSFM.EnvironmentConditionModelData",
        "version": "1.0",
        "model": model_obj,
    }


def test_save_load_EnvironmentConditionsModel_round_trip_inline(
    tmp_path: Path,
) -> None:
    """Round-trip a model containing all supported model types."""
    np.random.seed(0)

    oxygen_curve = Curve(
        "waterDepth",
        "oxygen",
        np.array([0.0, 10.0], dtype=float),
        np.array([100.0, 50.0], dtype=float),
        "linear",
    )

    rate_curve = Curve(
        "waterDepth",
        "sedimentationRate",
        np.array([0.0, 10.0], dtype=float),
        np.array([10.0, 20.0], dtype=float),
        "linear",
    )

    model = EnvironmentConditionsModel(
        [
            EnvironmentConditionModelConstant("temperature", 20.0),
            EnvironmentConditionModelUniform("energy", 0.5, 1.5),
            EnvironmentConditionModelTriangular("salinity", 35.0, 30.0, 40.0),
            EnvironmentConditionModelGaussian(
                "turbidity", 10.0, stdDev=2.0, minValue=0.0, maxValue=100.0
            ),
            EnvironmentConditionModelCurve("oxygen", oxygen_curve),
            EnvironmentConditionModelCombination(
                [
                    EnvironmentConditionModelCurve(
                        "sedimentationRate", rate_curve
                    ),
                    EnvironmentConditionModelGaussian(
                        "sedimentationRate",
                        1.0,
                        stdDev=0.1,
                        minValue=0.5,
                        maxValue=1.5,
                    ),
                ]
            ),
        ]
    )

    out_json = tmp_path / "env_conditions.json"
    saveEnvironmentConditionsModel(model, str(out_json))

    validateEnvironmentConditionsModelJsonFile(str(out_json))

    reloaded = loadEnvironmentConditionsModel(str(out_json))
    assert set(reloaded.environmentConditionNames) == set(
        model.environmentConditionNames
    )

    # Smoke: dependency resolution should work for nested Curve in Combination
    np.random.seed(0)
    values = reloaded.getEnvironmentConditionsAt(waterDepth=5.0)
    assert "temperature" in values
    assert "oxygen" in values
    assert "sedimentationRate" in values


def test_loadEnvironmentConditionsModel_curve_by_url(tmp_path: Path) -> None:
    """Loading supports Curve models referencing a curve file by url."""
    curve_obj = curveToJsonObj(
        Curve(
            "waterDepth",
            "oxygen",
            np.array([0.0, 10.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            "linear",
        ),
        y_axis_name="oxygen",
        x_axis_name_default="waterDepth",
    )
    curve_path = tmp_path / "oxygen_curve.json"
    curve_path.write_text(json.dumps(curve_obj, indent=2), encoding="utf-8")

    payload: dict[str, Any] = {
        "format": "pyWellSFM.EnvironmentConditionsModelData",
        "version": "1.0",
        "environmentConditions": {
            "oxygen": {
                "format": "pyWellSFM.EnvironmentConditionModelData",
                "version": "1.0",
                "model": {
                    "modelType": "Curve",
                    "curve": {"url": "oxygen_curve.json"},
                },
            }
        },
    }

    json_path = _write_json(tmp_path, payload, "env_curve_url.json")
    validateEnvironmentConditionsModelJsonFile(json_path)

    model = loadEnvironmentConditionsModel(json_path)
    vals = model.getEnvironmentConditionsAt(waterDepth=5.0)
    assert vals["oxygen"] == pytest.approx(0.5)


def test_loadDepositionalEnvironmentModel_embedded_models() -> None:
    """Embedded model objects in depositional schema are parsed correctly."""
    payload: dict[str, Any] = {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "TestDepModel",
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
                                "value": 0.5,
                            },
                        }
                    },
                },
            }
        ],
    }

    model = loadDepositionalEnvironmentModelFromJsonObj(payload)
    env = model.getEnvironmentByName("Lagoon")
    assert env is not None
    vals = env.getEnvironmentConditions(waterDepth=5.0, age=0.0)
    assert vals["energy"] == pytest.approx(0.5)


def test_combination_multiple_curve_dependencies_rejected() -> None:
    """Combination with multiple different curve xAxisName is ambiguous."""
    m = EnvironmentConditionsModel(
        [
            EnvironmentConditionModelCombination(
                [
                    EnvironmentConditionModelCurve(
                        "x",
                        Curve(
                            "waterDepth",
                            "x",
                            np.array([0.0, 1.0], dtype=float),
                            np.array([0.0, 1.0], dtype=float),
                            "linear",
                        ),
                    ),
                    EnvironmentConditionModelCurve(
                        "x",
                        Curve(
                            "age",
                            "x",
                            np.array([0.0, 1.0], dtype=float),
                            np.array([0.0, 1.0], dtype=float),
                            "linear",
                        ),
                    ),
                ]
            )
        ]
    )

    with pytest.raises(ValueError) as exc:
        m.getEnvironmentConditionsAt(waterDepth=1.0, age=1.0)
    assert "ambiguous" in str(exc.value).lower()


def test_save_environment_conditions_model_requires_json_extension() -> None:
    """Saving only supports .json files."""
    model = EnvironmentConditionsModel(
        [EnvironmentConditionModelConstant("temperature", 1.0)]
    )
    with pytest.raises(ValueError, match="\\.json"):
        saveEnvironmentConditionsModel(model, "out.txt")


def test_load_environment_conditions_model_requires_json_extension() -> None:
    """Loading only supports .json files."""
    with pytest.raises(ValueError, match="\\.json"):
        loadEnvironmentConditionsModel("input.txt")


def test_load_environment_conditions_model_rejects_non_object(
    tmp_path: Path,
) -> None:
    """Top-level payload must be an object."""
    path = tmp_path / "not_an_object.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        loadEnvironmentConditionsModel(str(path))


def test_env_model_to_json_rejects_non_inline_curve_mode() -> None:
    """Serializer rejects non-inline curves_mode."""
    model = EnvironmentConditionModelConstant("temperature", 20.0)
    with pytest.raises(ValueError, match="curves_mode='inline'"):
        env_io.environmentConditionModelToJsonObj(model, curves_mode="url")


def test_env_model_to_json_rejects_unsupported_type() -> None:
    """Serializer raises for unsupported model types."""

    class DummyModel(EnvironmentConditionModelBase):
        def getEnvironmentConditionAt(
            self, relatedCondition: float | None = None
        ) -> float:
            return 0.0

    with pytest.raises(ValueError, match="Unsupported"):
        env_io.environmentConditionModelToJsonObj(DummyModel("x"))


@pytest.mark.parametrize(
    ("model_obj", "msg"),
    [
        ({"modelType": " "}, "non-empty string"),
        ({"modelType": "Constant", "value": "bad"}, "must be numeric"),
        (
            {"modelType": "Uniform", "minValue": "x", "maxValue": 2.0},
            "must be numeric",
        ),
        (
            {"modelType": "Uniform", "minValue": 3.0, "maxValue": 2.0},
            "must be <= maxValue",
        ),
        (
            {
                "modelType": "Triangular",
                "minValue": "x",
                "modeValue": 1.0,
                "maxValue": 2.0,
            },
            "must be numeric",
        ),
        (
            {
                "modelType": "Triangular",
                "minValue": 0.0,
                "modeValue": 3.0,
                "maxValue": 2.0,
            },
            "requires minValue <= modeValue <= maxValue",
        ),
        ({"modelType": "Gaussian", "meanValue": "x"}, "must be numeric"),
        (
            {"modelType": "Gaussian", "meanValue": 1.0, "stdDev": "x"},
            "must be numeric",
        ),
        (
            {"modelType": "Gaussian", "meanValue": 1.0, "stdDev": -1.0},
            "must be >= 0",
        ),
        (
            {
                "modelType": "Gaussian",
                "meanValue": 1.0,
                "minValue": 2.0,
                "maxValue": 1.0,
            },
            "must be <= maxValue",
        ),
        (
            {
                "modelType": "Curve",
                "curve": {"format": "not-a-curve", "curve": {}},
            },
            "must be either an inline",
        ),
        ({"modelType": "Nope"}, "must be one of"),
    ],
)
def test_load_env_condition_model_validation_errors(
    model_obj: dict[str, Any], msg: str
) -> None:
    """Invalid model payloads raise explicit validation errors."""
    wrapped = _wrap_model(model_obj)
    with pytest.raises(ValueError, match=msg):
        env_io.loadEnvironmentConditionModelFromJsonObj(
            wrapped,
            condition_name="temperature",
            base_dir=None,
            ctx="testCtx",
        )


def test_curve_url_csv_with_multiple_curves_is_rejected(
    tmp_path: Path,
) -> None:
    """Curve URL must resolve to exactly one curve."""
    csv_path = tmp_path / "two_curves.csv"
    csv_path.write_text(
        "waterDepth,oxygen,energy\n0,1,2\n10,3,4\n",
        encoding="utf-8",
    )

    wrapped = _wrap_model(
        {
            "modelType": "Curve",
            "curve": {"url": "two_curves.csv"},
        }
    )
    with pytest.raises(ValueError, match="exactly one curve"):
        env_io.loadEnvironmentConditionModelFromJsonObj(
            wrapped,
            condition_name="oxygen",
            base_dir=tmp_path,
            ctx="testCtx",
        )


def test_load_environment_conditions_model_requires_non_empty_map() -> None:
    """EnvironmentConditions payload must be non-empty."""
    payload = {
        "format": "pyWellSFM.EnvironmentConditionsModelData",
        "version": "1.0",
        "environmentConditions": {},
    }
    with pytest.raises(ValueError, match="non-empty object"):
        env_io.loadEnvironmentConditionsModelFromJsonObj(
            payload, base_dir=None
        )


def test_load_environment_conditions_model_rejects_empty_keys() -> None:
    """Condition names must be non-empty strings."""
    payload = {
        "format": "pyWellSFM.EnvironmentConditionsModelData",
        "version": "1.0",
        "environmentConditions": {
            "": _wrap_model({"modelType": "Constant", "value": 1.0})
        },
    }
    with pytest.raises(ValueError, match="non-empty strings"):
        env_io.loadEnvironmentConditionsModelFromJsonObj(
            payload, base_dir=None
        )
