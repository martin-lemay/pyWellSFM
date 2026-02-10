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
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pywellsfm.io._common import resolve_ref_path
from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.io.tabulated_function_io import (
    loadTabulatedFunctionFromFile,
    loadTabulatedFunctionFromJsonObj,
    saveTabulatedFunctionToCsv,
    saveTabulatedFunctionToJson,
    tabulatedFunctionToJsonObj,
)
from pywellsfm.model.AccumulationModel import (
    AccumulationModelBase,
    AccumulationModelEnvironmentOptimum,
    AccumulationModelGaussian,
)
from pywellsfm.model.Curve import AccumulationCurve
from pywellsfm.model.Element import Element


def accumulationModelGaussianToJsonObj(
    accumulationModel: AccumulationModelGaussian,
) -> dict[str, Any]:
    """Serialize a Gaussian accumulation model to a JSON object."""
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": accumulationModel.name,
            "modelType": "Gaussian",
            "elements": [],
        },
    }

    for element in sorted(accumulationModel.elements, key=lambda e: e.name):
        stddev_factor = accumulationModel.std_dev_factors.get(
            element.name, accumulationModel.defaultStdDev
        )
        payload["accumulationModel"]["elements"].append(
            {
                "name": element.name,
                "accumulationRate": float(element.accumulationRate),
                "stddevFactor": float(stddev_factor),
            }
        )

    return payload


def accumulationModelEnvironmentOptimumToJsonObjInline(
    accumulationModel: AccumulationModelEnvironmentOptimum,
) -> dict[str, Any]:
    """Serialize an EnvironmentOptimum accumulation model to a JSON object.

    Curves are embedded inline as TabulatedFunction objects.
    """
    curve_names = sorted(accumulationModel.prodCurves.keys())
    inline_entries: list[dict[str, Any]] = []
    for curve_name in curve_names:
        curve = accumulationModel.prodCurves[curve_name]
        inline_entries.append(
            tabulatedFunctionToJsonObj(
                abscissa_name=str(getattr(curve, "_xAxisName", "")),
                ordinate_name=str(getattr(curve, "_yAxisName", "")),
                x=np.asarray(getattr(curve, "_abscissa", []), dtype=float),
                y=np.asarray(getattr(curve, "_ordinate", []), dtype=float),
            )
        )

    elements_sorted = sorted(accumulationModel.elements, key=lambda e: e.name)
    elements_payload: list[dict[str, Any]] = []
    for element in elements_sorted:
        elements_payload.append(
            {
                "name": element.name,
                "accumulationRate": float(element.accumulationRate),
                "accumulationCurves": inline_entries,
            }
        )

    return {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": accumulationModel.name,
            "modelType": "EnvironmentOptimum",
            "elements": elements_payload,
        },
    }


def accumulationModelToJsonObj(model: AccumulationModelBase) -> dict[str, Any]:
    """Serialize an accumulation model to JSON."""
    if isinstance(model, AccumulationModelGaussian):
        return accumulationModelGaussianToJsonObj(model)
    if isinstance(model, AccumulationModelEnvironmentOptimum):
        return accumulationModelEnvironmentOptimumToJsonObjInline(model)
    raise ValueError(
        "Unsupported accumulation model type for JSON export: "
        f"{model.__class__.__name__}"
    )


def loadAccumulationModel(filepath: str) -> AccumulationModelBase:
    """Load accumulation model from file.

    The file can be a csv or a json file. Csv file supports only Gaussian
    accumulation model. The json file conforms to the schema in
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

        base_dir = path.resolve().parent
        return loadAccumulationModelFromJsonObj(
            data,
            base_dir=base_dir,
        )
    raise ValueError(
        "Unsupported accumulation model file format. Supported formats are: "
        f".csv, .json Got '{path.suffix}'."
    )


def loadAccumulationModelFromJsonObj(
    obj: dict[str, Any], *, base_dir: Path | None = None
) -> AccumulationModelBase:
    """Parse an AccumulationModel JSON object into an AccumulationModel.

    The JSON must match the on-disk format used by this project:

    - format: "pyWellSFM.AccumulationModelData"
    - version: "1.0"

    .. Note::

        EnvironmentOptimum model curves can be embedded objects or relative paths
        When paths are used, they are resolved against ``base_dir``.


    :param dict[str, Any] obj: Parsed accumulation model JSON object.
    :param Path | None base_dir: Directory to resolve relative curve references
    :return AccumulationModelBase: Loaded accumulation model.
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
    if not isinstance(elements_obj, list) or len(elements_obj) < 1:
        raise ValueError(
            "accumulationModel.elements must be a non-empty list."
        )

    model_type = model_obj.get("modelType")
    if model_type not in ("Gaussian", "EnvironmentOptimum"):
        raise ValueError(
            "accumulationModel.modelType must be one of: 'Gaussian', "
            f"'EnvironmentOptimum'. Got '{model_type}'."
        )

    model: AccumulationModelBase
    if model_type == "Gaussian":
        model = AccumulationModelGaussian(name=name)
        _loadAccumulationModelGaussianFromJsonObj(elements_obj, model)
    elif model_type == "EnvironmentOptimum":
        model = AccumulationModelEnvironmentOptimum(name=name)
        _loadAccumulationModelEnvironmentOptimumFromJsonObj(
            elements_obj, model, base_dir
        )
    else:
        raise ValueError(
            "accumulationModel.modelType must be one of: 'Gaussian',"
            f"'EnvironmentOptimum'. Got '{model_type}'."
        )
    return model


def loadAccumulationModelGaussianFromCsv(
    filepath: str,
) -> AccumulationModelGaussian:
    """Load Gaussian accumulation model from csv file.

    The csv file must contain the following columns:

    - name: name of the element
    - mean: mean accumulation rate (m/My)
    - stddevFactor: standard deviation factor (multiplied by mean to get
      stddev)

    :params str filepath: Path to accumulation model csv file.
    :returns AccumulationModelGaussian: A Gaussian accumulation model.
    """
    data = pd.read_csv(filepath)
    elements = set()
    stddev_factors = {}
    for _, row in data.iterrows():
        name = row["name"]
        mean = float(row["mean"])
        stddev_factors[name] = float(row["stddevFactor"])
        element = Element(name=name, accumulationRate=mean)
        elements.add(element)
    return AccumulationModelGaussian(
        name="GaussianAccumulationModel",
        elements=elements,
        std_dev_factors=stddev_factors,
    )


def saveAccumulationModelGaussianToCsv(
    accumulationModel: AccumulationModelGaussian, filepath: str
) -> None:
    """Save Gaussian accumulation model to csv file."""
    rows = []
    for element in sorted(accumulationModel.elements, key=lambda e: e.name):
        stddev_factor = accumulationModel.std_dev_factors.get(
            element.name, accumulationModel.defaultStdDev
        )
        rows.append(
            {
                "name": element.name,
                "mean": element.accumulationRate,
                "stddevFactor": stddev_factor,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def _loadAccumulationModelGaussianFromJsonObj(
    elements_obj: list[Any], model: AccumulationModelGaussian
) -> None:
    """Load Gaussian accumulation model from json file.

    The json file conforms to the schema in
    `jsonSchemas/AccumulationModelSchema.json`.

    :params list[Any] elements_obj: Parsed accumulation model elements JSON
        object.
    :params AccumulationModelGaussian model: An empty Gaussian accumulation
        model to populate.
    """
    if not isinstance(elements_obj, list) or len(elements_obj) < 1:
        raise ValueError(
            "accumulationModel.elements must be a non-empty list."
        )

    seen_names: set[str] = set()
    for idx, elt in enumerate(elements_obj):
        if not isinstance(elt, dict):
            raise ValueError(
                f"accumulationModel.elements[{idx}] must be an object."
            )

        elt_name = elt.get("name")
        if not isinstance(elt_name, str) or elt_name.strip() == "":
            raise ValueError(
                f"accumulationModel.elements[{idx}].name must be a non-empty" +
                "string."
            )
        if elt_name in seen_names:
            raise ValueError(
                f"Duplicate element name '{elt_name}' in " +
                "accumulationModel.elements."
            )
        seen_names.add(elt_name)

        mean = elt.get("accumulationRate")
        if not isinstance(mean, (int, float)):
            raise ValueError(
                f"accumulationModel.elements[{idx}].accumulationRate must be" +
                " a number."
            )

        stddev_factor = elt.get("stddevFactor")
        if not isinstance(stddev_factor, (int, float)):
            raise ValueError(
                f"accumulationModel.elements[{idx}].stddevFactor must be" +
                " a number."
            )

        model.addElement(
            Element(name=elt_name, accumulationRate=float(mean)),
            std_dev_factor=float(stddev_factor),
        )


def _loadAccumulationModelEnvironmentOptimumFromJsonObj(
    elements_obj: list[Any],
    model: AccumulationModelEnvironmentOptimum,
    base_dir: Path | None,
) -> None:
    """Load environment optimum accumulation model from json file.

    The json file conforms to the schema in
    `jsonSchemas/AccumulationModelSchema.json`.

    :params Path | None base_dir: Base directory for relative paths in
        accumulation model json file.
    :params list[Any] elements_obj: Parsed accumulation model elements JSON
        object.
    :params AccumulationModelEnvironmentOptimum model: An empty environment
        optimum accumulation model to populate.
    """
    if not isinstance(elements_obj, list) or len(elements_obj) < 1:
        raise ValueError(
            "accumulationModel.elements must be a non-empty list."
        )

    seen_names: set[str] = set()
    for idx, elt in enumerate(elements_obj):
        if not isinstance(elt, dict):
            raise ValueError(
                f"accumulationModel.elements[{idx}] must be an object."
            )

        elt_name = elt.get("name")
        if not isinstance(elt_name, str) or elt_name.strip() == "":
            raise ValueError(
                f"accumulationModel.elements[{idx}].name must be a " +
                "non-empty string."
            )
        if elt_name in seen_names:
            raise ValueError(
                f"Duplicate element name '{elt_name}' in " +
                "accumulationModel.elements."
            )
        seen_names.add(elt_name)

        rate = elt.get("accumulationRate")
        if not isinstance(rate, (int, float)):
            raise ValueError(
                f"accumulationModel.elements[{idx}].accumulationRate must " +
                "be a number."
            )
        model.addElement(Element(name=elt_name, accumulationRate=float(rate)))

        curves_obj = elt.get("accumulationCurves")
        if not isinstance(curves_obj, list) or len(curves_obj) < 1:
            raise ValueError(
                f"accumulationModel.elements[{idx}].accumulationCurves must " +
                "be a non-empty list."
            )

        # NOTE: Current implementation stores curves globally (not per-element)
        for jdx, curve_def in enumerate(curves_obj):
            if isinstance(curve_def, dict):
                abscissa_name, _ord_name, x, y = (
                    loadTabulatedFunctionFromJsonObj(curve_def)
                )
            elif isinstance(curve_def, str) and curve_def.strip() != "":
                curve_path = resolve_ref_path(
                    base_dir=base_dir,
                    raw_url=curve_def,
                    ctx=(f"accumulationModel.elements[{idx}]" +
                    f".accumulationCurves[{jdx}]"),
                )
                abscissa_name, _ord_name, x, y = loadTabulatedFunctionFromFile(
                    curve_path
                )
            else:
                raise ValueError(
                    f"accumulationModel.elements[{idx}].accumulationCurves"
                    f"[{jdx}] must be an object or non-empty string."
                )

            new_curve = AccumulationCurve(abscissa_name, x, y)
            if abscissa_name in model.prodCurves:
                existing = model.prodCurves[abscissa_name]
                if (
                    (existing._abscissa.shape != new_curve._abscissa.shape)
                    or (
                        not np.allclose(
                            existing._abscissa, new_curve._abscissa
                        )
                    )
                    or (
                        not np.allclose(
                            existing._ordinate, new_curve._ordinate
                        )
                    )
                ):
                    raise ValueError(
                        "Conflicting definitions for accumulation curve "
                        f"'{abscissa_name}'."
                    )
            else:
                model.addAccumulationCurve(new_curve)


def saveAccumulationModelGaussianToJson(
    accumulationModel: AccumulationModelGaussian, filepath: str
) -> None:
    """Save Gaussian accumulation model to json file."""
    payload = accumulationModelGaussianToJsonObj(accumulationModel)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def saveAccumulationModelEnvironmentOptimumToJson(
    accumulationModel: AccumulationModelEnvironmentOptimum,
    filepath: str,
    *,
    curves_mode: str = "inline",
    curves_dir: str | None = None,
    curves_format: str = "json",
) -> None:
    """Save EnvironmentOptimum accumulation model to json.

    Supports two curve serialization modes:

    - curves_mode="inline": embed TabulatedFunction objects in the model JSON.
    - curves_mode="external": write curve files and reference them by relative
      paths.
    """
    out_path = Path(filepath)
    out_dir = out_path.parent
    mode = curves_mode.lower().strip()
    fmt = curves_format.lower().strip()

    if mode == "inline":
        payload = accumulationModelEnvironmentOptimumToJsonObjInline(
            accumulationModel
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    if mode != "external":
        raise ValueError("curves_mode must be 'inline' or 'external'.")

    curves_out_dir = Path(curves_dir) if curves_dir is not None else out_dir
    curves_out_dir.mkdir(parents=True, exist_ok=True)

    curve_names = sorted(accumulationModel.prodCurves.keys())
    external_entries: list[str] = []

    for curve_name in curve_names:
        curve = accumulationModel.prodCurves[curve_name]
        if fmt == "json":
            curve_path = curves_out_dir / f"{curve_name}.json"
            saveTabulatedFunctionToJson(
                abscissa_name=str(getattr(curve, "_xAxisName", "")),
                ordinate_name=str(getattr(curve, "_yAxisName", "")),
                x=np.asarray(getattr(curve, "_abscissa", []), dtype=float),
                y=np.asarray(getattr(curve, "_ordinate", []), dtype=float),
                filepath=str(curve_path),
                indent=2,
            )
        elif fmt == "csv":
            curve_path = curves_out_dir / f"{curve_name}.csv"
            saveTabulatedFunctionToCsv(
                x=np.asarray(getattr(curve, "_abscissa", []), dtype=float),
                y=np.asarray(getattr(curve, "_ordinate", []), dtype=float),
                filepath=str(curve_path),
            )
        else:
            raise ValueError("curves_format must be 'json' or 'csv'.")

        rel = os.path.relpath(curve_path, start=out_dir)
        external_entries.append(rel.replace("\\", "/"))

    elements_sorted = sorted(accumulationModel.elements, key=lambda e: e.name)
    elements_payload: list[dict[str, Any]] = []
    for element in elements_sorted:
        elements_payload.append(
            {
                "name": element.name,
                "accumulationRate": float(element.accumulationRate),
                "accumulationCurves": external_entries,
            }
        )

    payload1: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": accumulationModel.name,
            "modelType": "EnvironmentOptimum",
            "elements": elements_payload,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload1, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def saveAccumulationModel(
    model: AccumulationModelBase,
    filepath: str,
    *,
    curves_mode: str = "inline",
    curves_dir: str | None = None,
    curves_format: str = "json",
) -> None:
    """Save an accumulation model to `.json` or `.csv`.

    - Gaussian: `.csv` or `.json`
    - EnvironmentOptimum: `.json` only (optionally external curves)
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if isinstance(model, AccumulationModelGaussian):
        if ext == ".csv":
            return saveAccumulationModelGaussianToCsv(model, filepath)
        if ext == ".json":
            return saveAccumulationModelGaussianToJson(model, filepath)
        raise ValueError(
            "Gaussian accumulation model output must be .csv or .json"
        )

    if isinstance(model, AccumulationModelEnvironmentOptimum):
        if ext != ".json":
            raise ValueError(
                "EnvironmentOptimum accumulation model output must be a " +
                ".json file."
            )
        return saveAccumulationModelEnvironmentOptimumToJson(
            model,
            filepath,
            curves_mode=curves_mode,
            curves_dir=curves_dir,
            curves_format=curves_format,
        )

    raise ValueError(
        "Unsupported accumulation model type for saving: " +
        f"{model.__class__.__name__}"
    )
