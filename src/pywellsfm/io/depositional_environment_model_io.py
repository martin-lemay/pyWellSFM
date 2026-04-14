# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for DepositionalEnvironmentModel variants.

This module contains serialization/deserialization helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pywellsfm.io._common import reject_extra_keys
from pywellsfm.io.environment_condition_model_io import (
    environmentConditionModelToJsonObj,
    environmentConditionsModelToJsonObj,
    loadEnvironmentConditionModelFromJsonObj,
    loadEnvironmentConditionsModelFromJsonObj,
)
from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelStats,
    EnvironmentConditionsModel,
)


def _loadDepositionalEnvironmentFromJsonObj(
    obj: dict[str, Any],
    *,
    base_dir: Path | None,
    index: int,
) -> DepositionalEnvironment:
    """Parse one DepositionalEnvironment JSON object."""
    ctx = f"DepositionalEnvironmentModel.environments[{index}]"

    reject_extra_keys(
        obj=obj,
        allowed_keys={
            "name",
            "description",
            "waterDepthModel",
            "distality",
            "environmentConditionsModel",
        },
        ctx=ctx,
    )

    name = obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError(f"{ctx}.name must be a non-empty string.")

    waterDepthModelObj = obj.get("waterDepthModel")
    if not isinstance(waterDepthModelObj, dict):
        raise ValueError(f"{ctx}.waterDepthModel must be an object.")
    waterDepthModel = cast(
        EnvironmentConditionModelStats,
        loadEnvironmentConditionModelFromJsonObj(
            waterDepthModelObj,
            condition_name="waterDepth",
            base_dir=base_dir,
            ctx=f"{ctx}.waterDepthModel",
        ),
    )
    if not isinstance(waterDepthModel, EnvironmentConditionModelStats):
        raise ValueError(
            f"{ctx}.waterDepthModel must resolve to a statistical "
            + "environment condition model."
        )

    distality_raw = obj.get("distality")
    distality: float | None
    if distality_raw is None:
        distality = None
    elif isinstance(distality_raw, (int, float)):
        distality = float(distality_raw)
    else:
        raise ValueError(f"{ctx}.distality must be a number when provided.")

    environmentConditionsModelObj = obj.get("environmentConditionsModel")
    environmentConditionsModel: EnvironmentConditionsModel | None = None
    if environmentConditionsModelObj is not None:
        if not isinstance(environmentConditionsModelObj, dict):
            raise ValueError(
                f"{ctx}.environmentConditionsModel must be an object."
            )
        environmentConditionsModel = loadEnvironmentConditionsModelFromJsonObj(
            environmentConditionsModelObj, base_dir=base_dir
        )

    environment = DepositionalEnvironment(
        name=name,
        waterDepthModel=waterDepthModel,
        envConditionsModel=environmentConditionsModel,
        distality=distality,
    )
    return environment


def loadDepositionalEnvironmentModelFromJsonObj(
    obj: dict[str, Any],
    base_dir: str | None = None,
) -> DepositionalEnvironmentModel:
    """Parse a DepositionalEnvironmentModel JSON object.

    The JSON must match `jsonSchemas/DepositionalEnvironmentModelSchema.json`.
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.DepositionalEnvironmentModelSchema",
        expected_version="1.0",
        kind="deposition environment model",
    )

    reject_extra_keys(
        obj=obj,
        allowed_keys={"format", "version", "name", "environments"},
        ctx="DepositionalEnvironmentModel",
    )

    name = obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError(
            "DepositionalEnvironmentModel.name must be a non-empty string."
        )

    environments_obj = obj.get("environments")
    if not isinstance(environments_obj, list) or len(environments_obj) < 1:
        raise ValueError(
            "DepositionalEnvironmentModel.environments must be a non-empty "
            "list."
        )

    seen_names: set[str] = set()
    environments: list[DepositionalEnvironment] = []
    base_path = Path(base_dir) if base_dir is not None else None
    for idx, env_raw in enumerate(environments_obj):
        if not isinstance(env_raw, dict):
            raise ValueError(
                "DepositionalEnvironmentModel.environments[{0}] must be an "
                "object.".format(idx)
            )

        environment = _loadDepositionalEnvironmentFromJsonObj(
            env_raw,
            base_dir=base_path,
            index=idx,
        )

        if environment.name in seen_names:
            raise ValueError(
                "Duplicate environment name '{0}' in "
                "DepositionalEnvironmentModel.environments.".format(
                    environment.name
                )
            )
        seen_names.add(environment.name)
        environments.append(environment)

    return DepositionalEnvironmentModel(
        name=name,
        environments=environments,
    )


def loadDepositionalEnvironmentModel(
    filepath: str,
) -> DepositionalEnvironmentModel:
    """Load a DepositionalEnvironmentModel from a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Unsupported depositional environment model file format. "
            f"Expected '.json', got '{path.suffix}'."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return loadDepositionalEnvironmentModelFromJsonObj(
        data,
        base_dir=str(path.resolve().parent),
    )


def _depositionalEnvironmentToJsonObj(
    environment: DepositionalEnvironment,
) -> dict[str, Any]:
    """Serialize one DepositionalEnvironment to JSON object."""
    env_obj: dict[str, Any] = {
        "name": str(environment.name),
        "waterDepthModel": environmentConditionModelToJsonObj(
            environment.waterDepthModel
        ),
    }

    if environment.distality is not None:
        env_obj["distality"] = float(environment.distality)

    if (
        environment.envConditionsModel is not None
        and len(environment.envConditionsModel.environmentConditionNames) > 0
    ):
        env_obj["environmentConditionsModel"] = (
            environmentConditionsModelToJsonObj(environment.envConditionsModel)
        )

    return env_obj


def depositionalEnvironmentModelToJsonObj(
    model: DepositionalEnvironmentModel,
) -> dict[str, Any]:
    """Serialize a DepositionalEnvironmentModel to JSON object."""
    if not isinstance(model.name, str) or model.name.strip() == "":
        raise ValueError(
            "DepositionalEnvironmentModel.name must be a non-empty string."
        )

    if not isinstance(model.environments, list) or len(model.environments) < 1:
        raise ValueError(
            "DepositionalEnvironmentModel.environments must be a "
            + "non-empty list."
        )

    return {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": str(model.name),
        "environments": [
            _depositionalEnvironmentToJsonObj(environment)
            for environment in model.environments
        ],
    }


def saveDepositionalEnvironmentModel(
    model: DepositionalEnvironmentModel,
    filepath: str,
) -> None:
    """Save a DepositionalEnvironmentModel to a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "DepositionalEnvironmentModel output file must have a .json "
            "extension."
        )

    payload = depositionalEnvironmentModelToJsonObj(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
