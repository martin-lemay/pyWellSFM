# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pywellsfm.io import (
    loadDepositionalEnvironmentModelFromJsonObj,
    loadEnvironmentConditionsModel,
    saveEnvironmentConditionsModel,
    validateEnvironmentConditionsModelJsonFile,
)
from pywellsfm.io.curve_io import curveToJsonObj
from pywellsfm.model.Curve import Curve
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelCombination,
    EnvironmentConditionModelConstant,
    EnvironmentConditionModelCurve,
    EnvironmentConditionModelGaussian,
    EnvironmentConditionModelTriangular,
    EnvironmentConditionModelUniform,
    EnvironmentConditionsModel,
)


def _write_json(tmp_path: Path, payload: Any, filename: str) -> str:  # noqa: ANN401
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


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
