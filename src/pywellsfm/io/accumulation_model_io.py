# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for AccumulationModel variants.

This module contains serialization/deserialization helpers for:

- Gaussian accumulation models
- Environment optimum accumulation models

These functions were originally implemented in `pywellsfm.io.ioHelpers`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pywellsfm.io._common import load_inline_or_url
from pywellsfm.io.curve_io import (
    curveToJsonObj,
    loadCurveFromJsonObj,
    loadCurvesFromFile,
)
from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.model.AccumulationModel import (
    AccumulationModel,
    AccumulationModelElementBase,
    AccumulationModelElementEnvironmentOptimum,
    AccumulationModelElementGaussian,
)
from pywellsfm.model.Curve import AccumulationCurve


def _loadAccumulationCurveFromCurveJsonObj(
    curve_obj: dict[str, Any],
) -> AccumulationCurve:
    """Load an AccumulationCurve from a CurveSchema-compliant JSON object."""
    curve = loadCurveFromJsonObj(curve_obj)
    try:
        return AccumulationCurve(
            curve._xAxisName,
            curve._abscissa,
            curve._ordinate,
        )
    except AssertionError as exc:
        raise ValueError(
            "AccumulationCurve ordinate values must be between 0 and 1."
        ) from exc


def accumulationModelGaussianToJsonObj(
    accumulationModel: AccumulationModel,
) -> dict[str, Any]:
    """Serialize an AccumulationModel containing Gaussian element models.

    The output matches `jsonSchemas/AccumulationModelSchema.json`.
    """
    for element_name, element_model in accumulationModel.elements.items():
        if not isinstance(element_model, AccumulationModelElementGaussian):
            raise ValueError(
                "accumulationModelGaussianToJsonObj requires all elements "
                f"to be Gaussian. Found '{element_name}'="
                f"{element_model.__class__.__name__}."
            )
    return accumulationModelToJsonObj(accumulationModel)


def accumulationModelEnvironmentOptimumToJsonObjInline(
    accumulationModel: AccumulationModel,
) -> dict[str, Any]:
    """Serialize an AccumulationModel containing EnvironmentOptimum elements.

    Curves are embedded inline as CurveSchema-compliant objects.
    """
    for element_name, element_model in accumulationModel.elements.items():
        if not isinstance(
            element_model, AccumulationModelElementEnvironmentOptimum
        ):
            raise ValueError(
                "accumulationModelEnvironmentOptimumToJsonObjInline requires "
                "all elements to be EnvironmentOptimum. Found "
                f"'{element_name}'={element_model.__class__.__name__}."
            )
    return accumulationModelToJsonObj(accumulationModel)


def accumulationModelToJsonObj(model: AccumulationModel) -> dict[str, Any]:
    """Serialize an AccumulationModel to JSON.

    Output follows `jsonSchemas/AccumulationModelSchema.json`:

    - accumulationModel.elements is a mapping: elementName -> element model
    - per-element model discriminated by model.modelType
    """
    elements_payload: dict[str, Any] = {}
    for element_name in sorted(model.elements.keys()):
        element_model = model.elements[element_name]
        element_entry: dict[str, Any] = {
            "accumulationRate": float(element_model.accumulationRate),
        }

        if isinstance(element_model, AccumulationModelElementGaussian):
            element_entry["model"] = {
                "modelType": "Gaussian",
                "stddevFactor": float(element_model.std_dev_factor),
            }
        elif isinstance(
            element_model, AccumulationModelElementEnvironmentOptimum
        ):
            curves = getattr(element_model, "accumulationCurves", {})
            if not isinstance(curves, dict):
                raise ValueError(
                    "EnvironmentOptimum element model accumulationCurves must "
                    "be a dict."
                )
            curve_objs: list[dict[str, Any]] = []
            for curve_name in sorted(curves.keys()):
                curve = curves[curve_name]
                curve_objs.append(
                    curveToJsonObj(
                        curve,
                        y_axis_name="ReductionCoeff",
                        x_axis_name_default=str(
                            getattr(curve, "_xAxisName", "")
                        ),
                    )
                )
            element_entry["model"] = {
                "modelType": "EnvironmentOptimum",
                "accumulationCurves": curve_objs,
            }
        else:
            raise ValueError(
                "Unsupported element accumulation model type: "
                f"{element_model.__class__.__name__}"
            )

        elements_payload[str(element_name)] = element_entry

    return {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": str(model.name),
            "elements": elements_payload,
        },
    }


def loadAccumulationModel(filepath: str) -> AccumulationModel:
    """Load accumulation model from file.

    The file can be a csv or a json file. Csv file supports only Gaussian
    element models. The json file conforms to the schema in
    `jsonSchemas/AccumulationModelSchema.json`.

    Parameters:
        filepath: Path to accumulation model file.

    Returns:
        An accumulation model.
    """
    path = Path(filepath)
    if path.suffix.lower() == ".csv":
        return loadAccumulationModelGaussianFromCsv(filepath)
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return loadAccumulationModelFromJsonObj(
            data,
            base_dir=str(path.resolve().parent),
        )
    raise ValueError(
        "Unsupported accumulation model file format. Supported formats are: "
        f".csv, .json Got '{path.suffix}'."
    )


def loadAccumulationModelFromJsonObj(
    obj: dict[str, Any],
    base_dir: str | None = None,
) -> AccumulationModel:
    """Parse an AccumulationModel JSON object into an AccumulationModel.

    The JSON must match the on-disk format used by this project:

    - format: "pyWellSFM.AccumulationModelData"
    - version: "1.0"

    .. Note::

        EnvironmentOptimum model curves can be embedded objects or relative
        paths When paths are used, they are resolved against ``base_dir``.

    :param dict[str, Any] obj: Parsed accumulation model JSON object.
    :return AccumulationModel: Loaded accumulation model.
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.AccumulationModelData",
        expected_version="1.0",
        kind="accumulation model",
    )

    model_obj = obj.get("accumulationModel")
    if not isinstance(model_obj, dict):
        raise ValueError("'accumulationModel' must be an object.")

    name = model_obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("accumulationModel.name must be a non-empty string.")

    elements_obj = model_obj.get("elements")
    if not isinstance(elements_obj, dict) or len(elements_obj) < 1:
        raise ValueError(
            "accumulationModel.elements must be a non-empty object (mapping)."
        )

    element_models: dict[str, AccumulationModelElementBase] = {}
    for element_name, element_def in elements_obj.items():
        if not isinstance(element_name, str) or element_name.strip() == "":
            raise ValueError(
                "accumulationModel.elements keys must be non-empty strings."
            )
        if not isinstance(element_def, dict):
            raise ValueError(
                "accumulationModel.elements['{0}'] must be an object.".format(
                    element_name
                )
            )
        element_models[element_name] = (
            _loadElementAccumulationModelFromJsonObj(
                element_name,
                element_def,
                base_dir=base_dir,
            )
        )

    return AccumulationModel(
        name=name,
        elementAccumulationModels=element_models,
    )


def loadAccumulationModelGaussianFromCsv(
    filepath: str,
) -> AccumulationModel:
    """Load Gaussian accumulation model from csv file.

    The csv file must contain the following columns:

    - name: name of the element
    - accumulationRate: mean accumulation rate (m/My)
    - stddevFactor: standard deviation factor (multiplied by mean to get
      stddev)

    :params str filepath: Path to accumulation model csv file.
    :returns AccumulationModel: AccumulationModel with Gaussian element models.
    """
    data = pd.read_csv(filepath)
    element_models: dict[str, AccumulationModelElementBase] = {}
    for _, row in data.iterrows():
        name = row["name"]
        mean = float(row["accumulationRate"])
        stddev_factor = float(row["stddevFactor"])
        element_models[str(name)] = AccumulationModelElementGaussian(
            str(name),
            float(mean),
            std_dev_factor=float(stddev_factor),
        )
    return AccumulationModel(
        name="GaussianAccumulationModel",
        elementAccumulationModels=element_models,
    )


def saveAccumulationModelGaussianToCsv(
    accumulationModel: AccumulationModel, filepath: str
) -> None:
    """Save Gaussian accumulation model to csv file."""
    rows = []
    for element_name in sorted(accumulationModel.elements.keys()):
        element_model = accumulationModel.elements[element_name]
        if not isinstance(element_model, AccumulationModelElementGaussian):
            raise ValueError(
                "saveAccumulationModelGaussianToCsv requires all elements to "
                f"be Gaussian. Found '{element_name}'="
                f"{element_model.__class__.__name__}."
            )
        stddev_factor = float(element_model.std_dev_factor)
        rows.append(
            {
                "name": str(element_name),
                "accumulationRate": float(element_model.accumulationRate),
                "stddevFactor": stddev_factor,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def _loadElementAccumulationModelFromJsonObj(
    element_name: str,
    obj: dict[str, Any],
    base_dir: str | None = None,
) -> AccumulationModelElementBase:
    """Parse a single element entry from AccumulationModelSchema.json."""
    rate = obj.get("accumulationRate")
    if not isinstance(rate, (int, float)):
        raise ValueError(
            "accumulationModel.elements['{0}'].accumulationRate must be a "
            "number.".format(element_name)
        )

    model_obj = obj.get("model")
    if not isinstance(model_obj, dict):
        raise ValueError(
            f"accumulationModel.elements['{element_name}'].model must "
            + "be an object."
        )

    model_type = model_obj.get("modelType")
    if model_type == "Gaussian":
        stddevFactor = model_obj.get("stddevFactor")
        if not isinstance(stddevFactor, (int, float)):
            raise ValueError(
                "accumulationModel.elements['{0}'].model.stddevFactor must "
                "be a number.".format(element_name)
            )
        return AccumulationModelElementGaussian(
            element_name,
            float(rate),
            std_dev_factor=float(stddevFactor),
        )

    if model_type == "EnvironmentOptimum":
        curves_obj = model_obj.get("accumulationCurves")
        if not isinstance(curves_obj, list) or len(curves_obj) < 1:
            raise ValueError(
                "accumulationModel.elements['{0}'].model.accumulationCurves "
                "must be a non-empty array.".format(element_name)
            )

        curves: dict[str, AccumulationCurve] = {}
        base_path = Path(base_dir) if base_dir is not None else None

        for idx, curve_raw in enumerate(curves_obj):
            ctx = (
                "accumulationModel.elements['{0}'].model.accumulationCurves"
                "[{1}]".format(element_name, idx)
            )

            def _load_inline_curve(
                curve_json: dict[str, Any],
                ctx: str = ctx,
            ) -> list[AccumulationCurve]:
                # Inline curve must be a CurveSchema-compliant object.
                if curve_json.get("format") != "pyWellSFM.CurveData":
                    raise ValueError(
                        f"{ctx} must be either an inline CurveSchema object "
                        "(format='pyWellSFM.CurveData') or a {'url': ...} "
                        + "reference."
                    )
                return [_loadAccumulationCurveFromCurveJsonObj(curve_json)]

            def _load_curve_file(
                path: Path, ctx: str = ctx
            ) -> list[AccumulationCurve]:
                loaded = loadCurvesFromFile(path)
                out: list[AccumulationCurve] = []
                for c in loaded:
                    try:
                        out.append(cast(AccumulationCurve, c))
                    except AssertionError as exc:
                        raise ValueError(
                            "AccumulationCurve ordinate values must be "
                            + "between 0 and 1."
                        ) from exc
                assert len(out) == 1, (
                    "No or multiple curves loaded from "
                    + "file; expected exactly one."
                )
                return out

            loaded_curves = load_inline_or_url(
                curve_raw,
                base_dir=base_path,
                ctx=ctx,
                load_inline=_load_inline_curve,
                load_file=_load_curve_file,
            )

            for curve in loaded_curves:
                curve_name = getattr(curve, "_xAxisName", None)
                if not isinstance(curve_name, str) or curve_name.strip() == "":
                    raise ValueError(
                        f"{ctx} loaded an accumulation curve with an empty "
                        + "xAxisName."
                    )
                if curve_name in curves:
                    existing = curves[curve_name]
                    if (
                        existing._abscissa.shape != curve._abscissa.shape
                        or not np.allclose(existing._abscissa, curve._abscissa)
                        or not np.allclose(existing._ordinate, curve._ordinate)
                    ):
                        raise ValueError(
                            "Conflicting definitions for accumulation curve "
                            f"'{curve_name}' in element '{element_name}'."
                        )
                curves[curve_name] = curve

        return AccumulationModelElementEnvironmentOptimum(
            element_name,
            float(rate),
            accumulationCurves=curves,
        )

    raise ValueError(
        "accumulationModel.elements['{0}'].model.modelType must be 'Gaussian' "
        "or 'EnvironmentOptimum'. Got '{1}'.".format(
            element_name,
            model_type,
        )
    )


def saveAccumulationModelGaussianToJson(
    accumulationModel: AccumulationModel, filepath: str
) -> None:
    """Save Gaussian accumulation model to json file."""
    payload = accumulationModelGaussianToJsonObj(accumulationModel)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def saveAccumulationModelEnvironmentOptimumToJson(
    accumulationModel: AccumulationModel,
    filepath: str,
    *,
    curves_mode: str = "inline",
    curves_dir: str | None = None,
    curves_format: str = "json",
) -> None:
    """Save EnvironmentOptimum accumulation model to json.

    Only `curves_mode='inline'` is supported with the current JSON schema.
    """
    if curves_dir is not None or curves_format is not None:
        # Kept for signature compatibility; unused.
        pass

    mode = curves_mode.lower().strip()
    if mode != "inline":
        raise ValueError(
            "Only curves_mode='inline' is supported by "
            "AccumulationModelSchema.json."
        )

    payload = accumulationModelEnvironmentOptimumToJsonObjInline(
        accumulationModel
    )
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def saveAccumulationModel(
    model: AccumulationModel,
    filepath: str,
    *,
    curves_mode: str = "inline",
    curves_dir: str | None = None,
    curves_format: str = "json",
) -> None:
    """Save an accumulation model to `.json` or `.csv`.

    - `.json`: always supported (schema-compliant)
    - `.csv`: supported only when all elements are Gaussian
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".json":
        payload = accumulationModelToJsonObj(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    if ext == ".csv":
        return saveAccumulationModelGaussianToCsv(model, filepath)

    raise ValueError("Unsupported accumulation model output extension.")
