# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for DepositionalEnvironmentModel variants.

This module contains serialization/deserialization helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pywellsfm.io._common import (
    load_inline_or_url,
    reject_extra_keys,
)
from pywellsfm.io.curve_io import (
    curveToJsonObj,
    loadCurveFromJsonObj,
    loadCurvesFromFile,
)
from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.model.Curve import Curve
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)


def _loadPropertyRangeFromJsonObj(
    value: Any,  # noqa: ANN401
    *,
    ctx: str,
) -> tuple[float, float]:
    """Parse a [min, max] numeric range encoded as a 2-item array."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{ctx} must be an array of two numbers [min, max].")
    min_raw, max_raw = value[0], value[1]
    if not isinstance(min_raw, (int, float)) or not isinstance(
        max_raw, (int, float)
    ):
        raise ValueError(f"{ctx} must contain only numbers.")
    return (float(min_raw), float(max_raw))


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
            "waterDepth_range",
            "distality",
            "other_property_ranges",
            "property_curves",
        },
        ctx=ctx,
    )

    name = obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError(f"{ctx}.name must be a non-empty string.")

    waterDepth_range = _loadPropertyRangeFromJsonObj(
        obj.get("waterDepth_range"),
        ctx=f"{ctx}.waterDepth_range",
    )

    distality_raw = obj.get("distality")
    distality: float | None
    if distality_raw is None:
        distality = None
    elif isinstance(distality_raw, (int, float)):
        distality = float(distality_raw)
    else:
        raise ValueError(f"{ctx}.distality must be a number when provided.")

    other_property_ranges_obj = obj.get("other_property_ranges")
    other_property_ranges: dict[str, tuple[float, float]] = {}
    if other_property_ranges_obj is not None:
        if not isinstance(other_property_ranges_obj, dict):
            raise ValueError(f"{ctx}.other_property_ranges must be an object.")
        for prop_name, range_obj in other_property_ranges_obj.items():
            if not isinstance(prop_name, str) or prop_name.strip() == "":
                raise ValueError(
                    f"{ctx}.other_property_ranges keys must be non-empty "
                    + "strings."
                )
            other_property_ranges[prop_name] = _loadPropertyRangeFromJsonObj(
                range_obj,
                ctx=f"{ctx}.other_property_ranges['{prop_name}']",
            )

    environment = DepositionalEnvironment(
        name=name,
        waterDepth_range=waterDepth_range,
        other_property_ranges=other_property_ranges,
        distality=distality,
    )

    property_curves_obj = obj.get("property_curves")
    if property_curves_obj is not None:
        if not isinstance(property_curves_obj, dict):
            raise ValueError(f"{ctx}.property_curves must be an object.")

        for curve_name, curve_raw in property_curves_obj.items():
            curve_ctx = f"{ctx}.property_curves['{curve_name}']"

            def _load_inline_curve(
                curve_json: dict[str, Any],
                curve_ctx: str = curve_ctx,
            ) -> Curve:
                if curve_json.get("format") != "pyWellSFM.CurveData":
                    raise ValueError(
                        f"{curve_ctx} must be either an inline CurveSchema "
                        "object (format='pyWellSFM.CurveData') or a "
                        + "{'url': ...} reference."
                    )
                return loadCurveFromJsonObj(curve_json)

            def _load_curve_file(
                path: Path,
                curve_ctx: str = curve_ctx,
            ) -> Curve:
                curves = loadCurvesFromFile(path)
                if len(curves) != 1:
                    raise ValueError(
                        f"{curve_ctx}.url must point to a file containing "
                        "exactly one curve."
                    )
                return curves[0]

            curve = load_inline_or_url(
                curve_raw,
                base_dir=base_dir,
                ctx=curve_ctx,
                load_inline=_load_inline_curve,
                load_file=_load_curve_file,
            )
            environment.setPropertyCurve(curve)

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
        "waterDepth_range": [
            float(environment.waterDepth_range[0]),
            float(environment.waterDepth_range[1]),
        ],
    }

    if environment.distality is not None:
        env_obj["distality"] = float(environment.distality)

    if len(environment.other_property_ranges) > 0:
        env_obj["other_property_ranges"] = {
            str(prop_name): [float(prop_range[0]), float(prop_range[1])]
            for prop_name, prop_range in sorted(
                environment.other_property_ranges.items(),
                key=lambda kv: kv[0],
            )
        }

    if len(environment.property_curves) > 0:
        env_obj["property_curves"] = {
            str(curve_name): curveToJsonObj(curve)
            for curve_name, curve in sorted(
                environment.property_curves.items(),
                key=lambda kv: kv[0],
            )
        }

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
