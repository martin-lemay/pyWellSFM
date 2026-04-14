# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for EnvironmentConditionModel* and EnvironmentConditionsModel.

JSON format is defined by `jsonSchemas/EnvironmentConditionsModelSchema.json`.

The top-level payload stores a mapping from condition name to a model
configuration (discriminated by `modelType`).

Curve-based models support both inline CurveSchema objects and `{ "url": ... }`
references resolved relative to the environment-conditions JSON file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pywellsfm.io._common import (
    ensure_dict,
    ensure_non_empty_list,
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


def environmentConditionModelToJsonObj(
    model: EnvironmentConditionModelBase,
    *,
    curves_mode: str = "inline",
) -> dict[str, Any]:
    """Serialize an EnvironmentConditionModel* to a schema-shaped JSON object.

    Note: saving curves by url is not implemented yet (inline-only), but
    loading supports both inline and url references.
    """
    envCondModel = _environmentConditionModelDefToJsonObj(
        model,
        curves_mode=curves_mode,
    )

    return {
        "format": "pyWellSFM.EnvironmentConditionModelData",
        "version": "1.0",
        "model": envCondModel,
    }


def _environmentConditionModelDefToJsonObj(
    model: EnvironmentConditionModelBase,
    *,
    curves_mode: str = "inline",
) -> dict[str, Any]:
    """Serialize only the model definition object ({modelType, ...})."""
    envCondModel: dict[str, Any]
    mode = curves_mode.lower().strip()
    if mode != "inline":
        raise ValueError(
            "Only curves_mode='inline' is supported when saving environment "
            + "condition models."
        )

    if isinstance(model, EnvironmentConditionModelConstant):
        envCondModel = {"modelType": "Constant", "value": float(model.value)}
    elif isinstance(model, EnvironmentConditionModelUniform):
        envCondModel = {
            "modelType": "Uniform",
            "minValue": float(model.minValue),
            "maxValue": float(model.maxValue),
        }
    elif isinstance(model, EnvironmentConditionModelTriangular):
        envCondModel = {
            "modelType": "Triangular",
            "minValue": float(model.minValue),
            "modeValue": float(model.mode),
            "maxValue": float(model.maxValue),
        }
    elif isinstance(model, EnvironmentConditionModelGaussian):
        envCondModel = {
            "modelType": "Gaussian",
            "meanValue": float(model.meanValue),
            "stdDev": float(model.stdDev),
        }
        if math.isfinite(float(model.minValue)):
            envCondModel["minValue"] = float(model.minValue)
        if math.isfinite(float(model.maxValue)):
            envCondModel["maxValue"] = float(model.maxValue)
    elif isinstance(model, EnvironmentConditionModelCurve):
        envCondModel = {
            "modelType": "Curve",
            "curve": curveToJsonObj(
                model.curve,
                y_axis_name=str(model.envConditionName),
                x_axis_name_default=str(
                    getattr(model.curve, "_xAxisName", "x")
                ),
            ),
        }
    elif isinstance(model, EnvironmentConditionModelCombination):
        envCondModel = {
            "modelType": "Combination",
            "models": [
                _environmentConditionModelDefToJsonObj(
                    m,
                    curves_mode=curves_mode,
                )
                for m in model.models
            ],
        }
    else:
        raise ValueError(
            f"Unsupported environment condition model type: {type(model)!r}"
        )
    return envCondModel


def environmentConditionsModelToJsonObj(
    model: EnvironmentConditionsModel,
    *,
    curves_mode: str = "inline",
) -> dict[str, Any]:
    """Serialize an EnvironmentConditionsModel to JSON schema payload."""
    env_conditions = {
        name: environmentConditionModelToJsonObj(m, curves_mode=curves_mode)
        for name, m in sorted(model.envConditionModels.items())
    }
    return {
        "format": "pyWellSFM.EnvironmentConditionsModelData",
        "version": "1.0",
        "environmentConditions": env_conditions,
    }


def saveEnvironmentConditionsModel(
    model: EnvironmentConditionsModel,
    filepath: str,
    *,
    indent: int = 2,
    curves_mode: str = "inline",
) -> None:
    """Save an EnvironmentConditionsModel to a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Environment conditions output file must have a .json extension."
        )

    payload = environmentConditionsModelToJsonObj(
        model, curves_mode=curves_mode
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=int(indent), ensure_ascii=False),
        encoding="utf-8",
    )


def _load_single_curve_from_file(path: Path, *, ctx: str) -> Curve:
    loaded = loadCurvesFromFile(path)
    if len(loaded) != 1:
        raise ValueError(f"{ctx} curve file must contain exactly one curve.")
    return loaded[0]


def loadEnvironmentConditionModelFromJsonObj(
    obj: dict[str, Any],
    *,
    condition_name: str,
    base_dir: Path | None,
    ctx: str,
) -> EnvironmentConditionModelBase:
    """Parse a wrapped model object into an EnvironmentConditionModel*."""
    expect_format_version(
        obj,
        expected_format="pyWellSFM.EnvironmentConditionModelData",
        expected_version="1.0",
        kind="environment condition model",
    )
    reject_extra_keys(
        obj=obj,
        allowed_keys={"format", "version", "model"},
        ctx=ctx,
    )
    model_obj = ensure_dict(obj.get("model"), ctx=f"{ctx}.model")

    return _loadEnvironmentConditionModelDefFromJsonObj(
        model_obj,
        condition_name=condition_name,
        base_dir=base_dir,
        ctx=f"{ctx}.model",
    )


def _loadEnvironmentConditionModelDefFromJsonObj(
    model_obj: dict[str, Any],
    *,
    condition_name: str,
    base_dir: Path | None,
    ctx: str,
) -> EnvironmentConditionModelBase:
    """Parse only the model definition object ({modelType, ...})."""
    model_ctx = ctx

    model_type = model_obj.get("modelType")
    if not isinstance(model_type, str) or model_type.strip() == "":
        raise ValueError(f"{model_ctx}.modelType must be a non-empty string.")
    model_type = model_type.strip()

    if model_type == "Constant":
        reject_extra_keys(
            obj=model_obj,
            allowed_keys={"modelType", "value"},
            ctx=model_ctx,
        )
        try:
            value = float(model_obj["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{model_ctx}.value must be numeric.") from exc
        return EnvironmentConditionModelConstant(condition_name, value)

    if model_type == "Uniform":
        reject_extra_keys(
            obj=model_obj,
            allowed_keys={"modelType", "minValue", "maxValue"},
            ctx=model_ctx,
        )
        try:
            min_value = float(model_obj["minValue"])
            max_value = float(model_obj["maxValue"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_ctx}.minValue/maxValue must be numeric."
            ) from exc
        if min_value > max_value:
            raise ValueError(f"{model_ctx}.minValue must be <= maxValue.")
        return EnvironmentConditionModelUniform(
            condition_name, min_value, max_value
        )

    if model_type == "Triangular":
        reject_extra_keys(
            obj=model_obj,
            allowed_keys={"modelType", "minValue", "modeValue", "maxValue"},
            ctx=model_ctx,
        )
        try:
            min_value = float(model_obj["minValue"])
            mode_value = float(model_obj["modeValue"])
            max_value = float(model_obj["maxValue"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_ctx}.minValue/modeValue/maxValue must be numeric."
            ) from exc
        if not (min_value <= mode_value <= max_value):
            raise ValueError(
                f"{model_ctx} requires minValue <= modeValue <= maxValue."
            )
        return EnvironmentConditionModelTriangular(
            condition_name,
            mode_value,
            min_value,
            max_value,
        )

    if model_type == "Gaussian":
        reject_extra_keys(
            obj=model_obj,
            allowed_keys={
                "modelType",
                "meanValue",
                "stdDev",
                "minValue",
                "maxValue",
            },
            ctx=model_ctx,
        )
        try:
            mean_value = float(model_obj["meanValue"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_ctx}.meanValue must be numeric."
            ) from exc

        stddev_raw = model_obj.get("stdDev")
        stddev = None
        if stddev_raw is not None:
            try:
                stddev = float(stddev_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{model_ctx}.stdDev must be numeric."
                ) from exc
            if stddev < 0:
                raise ValueError(f"{model_ctx}.stdDev must be >= 0.")

        min_value_raw = model_obj.get("minValue")
        max_value_raw = model_obj.get("maxValue")
        min_f = (
            -float(np.inf) if min_value_raw is None else float(min_value_raw)
        )
        max_f = (
            float(np.inf) if max_value_raw is None else float(max_value_raw)
        )
        if min_f > max_f:
            raise ValueError(f"{model_ctx}.minValue must be <= maxValue.")

        return EnvironmentConditionModelGaussian(
            condition_name,
            mean_value,
            stdDev=stddev,
            minValue=min_f,
            maxValue=max_f,
        )

    if model_type == "Curve":
        reject_extra_keys(
            obj=model_obj,
            allowed_keys={"modelType", "curve"},
            ctx=model_ctx,
        )
        curve_raw = model_obj.get("curve")

        def _load_inline_curve(curve_json: dict[str, Any]) -> Curve:
            if curve_json.get("format") != "pyWellSFM.CurveData":
                raise ValueError(
                    f"{model_ctx}.curve must be either an inline "
                    + "CurveSchema object "
                    "(format='pyWellSFM.CurveData') or a {'url': ...} "
                    + "reference."
                )
            return loadCurveFromJsonObj(curve_json)

        def _load_curve_file(path: Path) -> Curve:
            return _load_single_curve_from_file(path, ctx=f"{ctx}.curve")

        curve = load_inline_or_url(
            curve_raw,
            base_dir=base_dir,
            ctx=f"{model_ctx}.curve",
            load_inline=_load_inline_curve,
            load_file=_load_curve_file,
        )
        return EnvironmentConditionModelCurve(condition_name, curve)

    if model_type == "Combination":
        reject_extra_keys(
            obj=model_obj,
            allowed_keys={"modelType", "models"},
            ctx=model_ctx,
        )
        models_raw = ensure_non_empty_list(
            model_obj.get("models"),
            ctx=f"{model_ctx}.models",
        )
        submodels: list[EnvironmentConditionModelBase] = []
        for i, sub_raw in enumerate(models_raw):
            sub_obj = ensure_dict(sub_raw, ctx=f"{model_ctx}.models[{i}]")
            sub = _loadEnvironmentConditionModelDefFromJsonObj(
                sub_obj,
                condition_name=condition_name,
                base_dir=base_dir,
                ctx=f"{model_ctx}.models[{i}]",
            )
            if getattr(sub, "envConditionName", None) != condition_name:
                raise ValueError(
                    f"{model_ctx}.models[{i}] targets condition "
                    + f"'{sub.envConditionName}', expected '{condition_name}'."
                )
            submodels.append(sub)
        return EnvironmentConditionModelCombination(submodels)

    raise ValueError(
        f"{model_ctx}.modelType must be one of Constant, Uniform, Triangular, "
        f"Gaussian, Curve, Combination. Got '{model_type}'."
    )


def loadEnvironmentConditionsModelFromJsonObj(
    obj: dict[str, Any],
    *,
    base_dir: Path | None,
) -> EnvironmentConditionsModel:
    """Load an EnvironmentConditionsModel from a JSON object."""
    expect_format_version(
        obj,
        expected_format="pyWellSFM.EnvironmentConditionsModelData",
        expected_version="1.0",
        kind="environment conditions model",
    )

    reject_extra_keys(
        obj=obj,
        allowed_keys={"format", "version", "environmentConditions"},
        ctx="EnvironmentConditionsModel",
    )

    raw_models = obj.get("environmentConditions")
    if not isinstance(raw_models, dict) or len(raw_models) < 1:
        raise ValueError(
            "EnvironmentConditionsModel.environmentConditions must be a "
            + "non-empty object."
        )

    env_model = EnvironmentConditionsModel()
    for cond_name, raw in sorted(raw_models.items()):
        if not isinstance(cond_name, str) or cond_name.strip() == "":
            raise ValueError(
                "EnvironmentConditionsModel.environmentConditions keys must "
                + "be non-empty strings."
            )
        model_obj = ensure_dict(
            raw,
            ctx=f"environmentConditions['{cond_name}']",
        )
        model = loadEnvironmentConditionModelFromJsonObj(
            model_obj,
            condition_name=cond_name,
            base_dir=base_dir,
            ctx=f"environmentConditions['{cond_name}']",
        )
        env_model.addEnvironmentConditionModel(cond_name, model)

    return env_model


def loadEnvironmentConditionsModel(
    filepath: str,
) -> EnvironmentConditionsModel:
    """Load an EnvironmentConditionsModel from a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Environment conditions input file must have a .json extension."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Environment conditions JSON must be an object.")
    return loadEnvironmentConditionsModelFromJsonObj(
        data, base_dir=path.parent
    )
