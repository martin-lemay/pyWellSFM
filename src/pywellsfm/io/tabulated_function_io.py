# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pywellsfm.io.json_schema_validation import expect_format_version


def tabulatedFunctionToJsonObj(
    *,
    abscissa_name: str,
    ordinate_name: str,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Serialize a tabulated function to a JSON object.

    The returned object conforms to the on-disk format used by this project:
    - format: "pyWellSFM.TabulatedFunctionData"
    - version: "1.0"
    """
    if not isinstance(abscissa_name, str) or abscissa_name.strip() == "":
        raise ValueError("TabulatedFunction.abscissaName must be a non-empty string.")
    if not isinstance(ordinate_name, str) or ordinate_name.strip() == "":
        raise ValueError("TabulatedFunction.ordinateName must be a non-empty string.")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 1 or x_arr.size != y_arr.size:
        raise ValueError("TabulatedFunction x/y must have same non-zero length.")

    return {
        "format": "pyWellSFM.TabulatedFunctionData",
        "version": "1.0",
        "abscissaName": str(abscissa_name),
        "ordinateName": str(ordinate_name),
        "values": {
            "xValues": [float(v) for v in x_arr.tolist()],
            "yValues": [float(v) for v in y_arr.tolist()],
        },
    }


def saveTabulatedFunctionToJson(
    *,
    abscissa_name: str,
    ordinate_name: str,
    x: np.ndarray,
    y: np.ndarray,
    filepath: str,
    indent: int = 2,
) -> None:
    """Save a tabulated function to a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Tabulated function output file must have a .json extension.")
    payload = tabulatedFunctionToJsonObj(
        abscissa_name=abscissa_name,
        ordinate_name=ordinate_name,
        x=x,
        y=y,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=int(indent), ensure_ascii=False),
        encoding="utf-8",
    )


def saveTabulatedFunctionToCsv(
    *,
    x: np.ndarray,
    y: np.ndarray,
    filepath: str,
) -> None:
    """Save a tabulated function to a `.csv` file (two columns x,y, no header)."""
    path = Path(filepath)
    if path.suffix.lower() != ".csv":
        raise ValueError("Tabulated function output file must have a .csv extension.")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 1 or x_arr.size != y_arr.size:
        raise ValueError("TabulatedFunction x/y must have same non-zero length.")
    df = pd.DataFrame({"x": x_arr, "y": y_arr})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=False)


def loadTabulatedFunctionFromJsonObj(
    obj: dict[str, Any],
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Parse a TabulatedFunction JSON object into arrays.

    Expects the object format used throughout this project:
    - format: "pyWellSFM.TabulatedFunctionData"
    - version: "1.0"

    Returns (abscissa_name, ordinate_name, x, y).
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.TabulatedFunctionData",
        expected_version="1.0",
        kind="tabulated function",
    )

    abscissa_name = obj.get("abscissaName")
    ordinate_name = obj.get("ordinateName")
    if not isinstance(abscissa_name, str) or abscissa_name.strip() == "":
        raise ValueError("TabulatedFunction.abscissaName must be a non-empty string.")
    if not isinstance(ordinate_name, str) or ordinate_name.strip() == "":
        raise ValueError("TabulatedFunction.ordinateName must be a non-empty string.")

    values = obj.get("values")
    if not isinstance(values, dict):
        raise ValueError("TabulatedFunction.values must be an object.")
    x_values = values.get("xValues")
    y_values = values.get("yValues")
    if not isinstance(x_values, list) or not isinstance(y_values, list):
        raise ValueError("TabulatedFunction.values.xValues and yValues must be arrays.")
    if len(x_values) < 1 or len(y_values) < 1 or len(x_values) != len(y_values):
        raise ValueError(
            "TabulatedFunction.values.xValues and yValues must have same non-zero "
            "length."
        )

    try:
        x = np.array([float(v) for v in x_values], dtype=float)
        y = np.array([float(v) for v in y_values], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("TabulatedFunction xValues/yValues must be numeric.") from exc

    return abscissa_name, ordinate_name, x, y


def loadTabulatedFunctionFromFile(
    path: Path,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Load a tabulated function from a `.json` or `.csv` file.

    - `.json`: must match the TabulatedFunction JSON structure.
    - `.csv`: expects two numeric columns (x,y), no header.

    Returns (abscissa_name, ordinate_name, x, y).
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    if ext == ".json":
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Tabulated function JSON must be an object.")
        return loadTabulatedFunctionFromJsonObj(obj)

    if ext == ".csv":
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 2:
            raise ValueError(
                "Tabulated function CSV must have at least 2 columns (x,y)."
            )
        try:
            x = df.iloc[:, 0].astype(float).to_numpy()
            y = df.iloc[:, 1].astype(float).to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Tabulated function CSV x/y columns must be numeric."
            ) from exc

        if x.size < 1 or y.size < 1 or x.size != y.size:
            raise ValueError(
                "Tabulated function CSV must have same non-zero x/y length."
            )

        # CSV contains no metadata; infer name from filename stem.
        return path.stem, "ReductionCoeff", x, y

    raise ValueError(f"Unsupported tabulated function file extension '{path.suffix}'.")
