# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for DepositionalEnvironmentModel variants.

This module contains serialization/deserialization helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pywellsfm.simulator.DepositionalEnvironmentSimulator import (
    DepositionalEnvironmentSimulator,
    DESimulatorParameters,
)
from pywellsfm.utils import IntervalDistanceMethod

from ._common import (
    reject_extra_keys,
)
from .depositional_environment_model_io import (
    depositionalEnvironmentModelToJsonObj,
    loadDepositionalEnvironmentModelFromJsonObj,
)
from .json_schema_validation import expect_format_version


def loadSimulatorParametersFromJsonObj(
    obj: dict[str, Any],
) -> DESimulatorParameters:
    """Parse simulator parameters from JSON object."""
    reject_extra_keys(
        obj=obj,
        allowed_keys={
            "waterDepth_sigma",
            "waterDepth_weight",
            "transition_sigma",
            "transition_weight",
            "trend_sigma",
            "trend_window",
            "trend_weight",
            "interval_distance_method",
        },
        ctx="DESimulation.params",
    )

    kwargs: dict[str, Any] = {}

    for key in ["waterDepth_sigma", "transition_sigma", "trend_sigma"]:
        val = obj.get(key)
        if val is not None:
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"DESimulation.params.{key} must be " + "a number."
                )
            kwargs[key] = float(val)

    trend_window_raw = obj.get("trend_window")
    if trend_window_raw is not None:
        if not isinstance(trend_window_raw, int):
            raise ValueError(
                "DESimulation.params.trend_window must be " + "an integer."
            )
        kwargs["trend_window"] = trend_window_raw

    for key in ["waterDepth_weight", "transition_weight", "trend_weight"]:
        val = obj.get(key)
        if val is not None:
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"DESimulation.params.{key} must be a number."
                )
            kwargs[key] = float(val)

    interval_method_raw = obj.get("interval_distance_method")
    if interval_method_raw is not None:
        if not isinstance(interval_method_raw, str):
            raise ValueError(
                "DESimulation.params.interval_distance_method must be "
                + "a string."
            )
        try:
            kwargs["interval_distance_method"] = IntervalDistanceMethod(
                interval_method_raw
            )
        except ValueError as exc:
            allowed = ", ".join([m.value for m in IntervalDistanceMethod])
            raise ValueError(
                "DESimulation.params.interval_distance_method must be one of: "
                f"{allowed}."
            ) from exc

    return DESimulatorParameters(**kwargs)


def simulatorParametersToJsonObj(
    params: DESimulatorParameters,
) -> dict[str, Any]:
    """Serialize simulator parameters to JSON object."""
    return {
        "waterDepth_sigma": float(params.waterDepth_sigma),
        "waterDepth_weight": float(params.waterDepth_weight),
        "transition_sigma": float(params.transition_sigma),
        "transition_weight": float(params.transition_weight),
        "trend_sigma": float(params.trend_sigma),
        "trend_window": int(params.trend_window),
        "trend_weight": float(params.trend_weight),
        "interval_distance_method": str(params.interval_distance_method),
    }


def loadSimulatorWeights(
    obj: dict[str, Any],
) -> dict[str, float]:
    """Parse simulator weights from JSON object."""
    reject_extra_keys(
        obj=obj,
        allowed_keys=set(obj.keys()),
        ctx="DESimulation.weights",
    )

    weights: dict[str, float] = {}
    for env_name, weight in obj.items():
        if not isinstance(env_name, str):
            raise ValueError("DESimulation.weights keys must be strings.")
        if not env_name:
            raise ValueError(
                "DESimulation.weights keys must be non-empty " + "strings."
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"DESimulation.weights['{env_name}'] must be a number."
            )
        if weight <= 0:
            raise ValueError(
                f"DESimulation.weights['{env_name}'] must be > 0."
            )
        weights[env_name] = float(weight)
    return weights


def loadDepositionalEnvironmentSimulationFromJsonObj(
    obj: dict[str, Any],
    *,
    base_dir: str | None = None,
) -> DepositionalEnvironmentSimulator:
    """Parse a DepositionalEnvironmentSimulator JSON object."""
    expect_format_version(
        obj,
        expected_format="pyWellSFM.DESimulationSchema",
        expected_version="1.0",
        kind="deposition environment simulation",
    )

    reject_extra_keys(
        obj=obj,
        allowed_keys={
            "format",
            "version",
            "depositionalEnvironmentModel",
            "weights",
            "params",
        },
        ctx="DESimulation",
    )

    environmentModel_obj = obj.get("depositionalEnvironmentModel")
    if not isinstance(environmentModel_obj, dict):
        raise ValueError(
            "DESimulation.depositionalEnvironmentModel must be " + "an object."
        )
    environmentModel = loadDepositionalEnvironmentModelFromJsonObj(
        environmentModel_obj,
        base_dir=base_dir,
    )

    weights_obj = obj.get("weights")
    weights: dict[str, float] | None = None
    if weights_obj is not None:
        weights = loadSimulatorWeights(weights_obj)
        env_names = {env.name for env in environmentModel.environments}
        missing = env_names.difference(weights.keys())
        unknown = set(weights.keys()).difference(env_names)
        if missing:
            raise ValueError(
                "DESimulation.weights is missing keys for environments: "
                f"{sorted(missing)}"
            )
        if unknown:
            raise ValueError(
                "DESimulation.weights contains unknown environments: "
                f"{sorted(unknown)}"
            )

    params_obj = obj.get("params")
    params: DESimulatorParameters | None = None
    if params_obj is not None:
        if not isinstance(params_obj, dict):
            raise ValueError("DESimulation.params must be an object.")
        params = loadSimulatorParametersFromJsonObj(params_obj)

    return DepositionalEnvironmentSimulator(
        depositionalEnvironmentModel=environmentModel,
        weights=weights,
        params=params,
    )


def loadDepositionalEnvironmentSimulation(
    filepath: str,
) -> DepositionalEnvironmentSimulator:
    """Load DepositionalEnvironmentSimulator from a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Unsupported depositional environment simulation file format. "
            f"Expected '.json', got '{path.suffix}'."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return loadDepositionalEnvironmentSimulationFromJsonObj(
        data,
        base_dir=str(path.resolve().parent),
    )


def depositionalEnvironmentSimulationToJsonObj(
    simulator: DepositionalEnvironmentSimulator,
) -> dict[str, Any]:
    """Serialize DepositionalEnvironmentSimulator to JSON object."""
    payload: dict[str, Any] = {
        "format": "pyWellSFM.DESimulationSchema",
        "version": "1.0",
        "depositionalEnvironmentModel": depositionalEnvironmentModelToJsonObj(
            simulator.depositionalEnvironmentModel
        ),
    }

    payload["weights"] = simulator._weights

    params = simulator.params
    default_params = DESimulatorParameters()
    if params != default_params:
        payload["params"] = simulatorParametersToJsonObj(params)

    return payload


def saveDepositionalEnvironmentSimulation(
    simulator: DepositionalEnvironmentSimulator,
    filepath: str,
) -> None:
    """Save DepositionalEnvironmentSimulator to a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "DepositionalEnvironmentSimulation output file must have a .json "
            "extension."
        )

    payload = depositionalEnvironmentSimulationToJsonObj(simulator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
