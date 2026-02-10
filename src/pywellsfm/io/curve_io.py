# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for Curve and UncertaintyCurve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.model.Curve import (
    Curve,
    UncertaintyCurve,
)
from pywellsfm.utils import PolynomialInterpolator


def _curve_interpolation_method(curve: Curve) -> str | dict[str, int]:
    """Best-effort extraction of the interpolation method from a Curve."""
    interp_func = getattr(curve, "_interpFunc", None)
    kind = getattr(interp_func, "kind", None)
    if isinstance(kind, str) and kind.strip() != "":
        return kind

    degree = getattr(interp_func, "deg", None)
    nb_points = getattr(interp_func, "nbPts", None)
    if isinstance(degree, int) and isinstance(nb_points, int):
        return {"degree": int(degree), "nbPoints": int(nb_points)}

    return "linear"


def curveToJsonObj(
    curve: Curve,
    *,
    y_axis_name: str | None = None,
    x_axis_name_default: str = "Age",
) -> dict[str, Any]:
    """Serialize a Curve to JSON matching `jsonSchemas/CurveSchema.json`."""
    x_axis_name = getattr(curve, "_xAxisName", None)
    if not isinstance(x_axis_name, str) or x_axis_name.strip() == "":
        x_axis_name = str(x_axis_name_default)

    y_name = (
        y_axis_name
        if y_axis_name is not None
        else getattr(curve, "_yAxisName", None)
    )
    if not isinstance(y_name, str) or y_name.strip() == "":
        y_name = "Value"

    abscissa = np.asarray(getattr(curve, "_abscissa", []), dtype=float)
    ordinate = np.asarray(getattr(curve, "_ordinate", []), dtype=float)
    if abscissa.size != ordinate.size:
        raise ValueError(
            f"Curve abscissa/ordinate lengths differ: {abscissa.size} != " +
            f"{ordinate.size}"
        )
    if abscissa.size < 2:
        raise ValueError("Curve must have at least 2 points")

    data = [
        {"x": float(x), "y": float(y)}
        for x, y in zip(abscissa.tolist(), ordinate.tolist(), strict=False)
    ]

    return {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": str(x_axis_name),
            "yAxisName": str(y_name),
            "interpolationMethod": _curve_interpolation_method(curve),
            "data": data,
        },
    }


def saveCurveToJson(
    curve: Curve,
    filepath: str,
    *,
    y_axis_name: str | None = None,
    x_axis_name_default: str = "Age",
    indent: int = 2,
) -> None:
    """Save a Curve to a `.json` file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Curve output file must have a .json extension.")
    payload = curveToJsonObj(
        curve,
        y_axis_name=y_axis_name,
        x_axis_name_default=x_axis_name_default,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=int(indent), ensure_ascii=False),
        encoding="utf-8",
    )


def saveCurveToCsv(
    curve: Curve,
    filepath: str,
    *,
    y_axis_name: str | None = None,
    x_axis_name_default: str = "x",
) -> None:
    """Save a Curve to a `.csv` file with header (xAxisName,yAxisName)."""
    path = Path(filepath)
    if path.suffix.lower() != ".csv":
        raise ValueError("Curve output file must have a .csv extension.")

    x_axis_name = getattr(curve, "_xAxisName", None)
    if not isinstance(x_axis_name, str) or x_axis_name.strip() == "":
        x_axis_name = str(x_axis_name_default)
    y_name = (
        y_axis_name
        if y_axis_name is not None
        else getattr(curve, "_yAxisName", None)
    )
    if not isinstance(y_name, str) or y_name.strip() == "":
        y_name = "y"

    x = np.asarray(getattr(curve, "_abscissa", []), dtype=float)
    y = np.asarray(getattr(curve, "_ordinate", []), dtype=float)
    if x.size != y.size or x.size < 1:
        raise ValueError("Curve must have same non-zero x/y length.")

    df = pd.DataFrame({str(x_axis_name): x, str(y_name): y})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def saveCurve(
    curve: Curve,
    filepath: str,
    *,
    y_axis_name: str | None = None,
    x_axis_name_default: str = "Age",
) -> None:
    """Save a Curve to a `.json` or `.csv` file."""
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".json":
        return saveCurveToJson(
            curve,
            filepath,
            y_axis_name=y_axis_name,
            x_axis_name_default=x_axis_name_default,
        )
    if ext == ".csv":
        return saveCurveToCsv(
            curve,
            filepath,
            y_axis_name=y_axis_name,
            x_axis_name_default=x_axis_name_default,
        )
    raise ValueError("Unsupported Curve output extension. Use .json or .csv.")


def _load_single_curve(filepath: str, *, kind: str) -> Curve:
    curves = loadCurvesFromFile(Path(filepath))
    if len(curves) != 1:
        raise ValueError(f"{kind} file must contain exactly one curve.")
    return curves[0]


def loadSubsidenceCurve(filepath: str) -> Curve:
    """Load a subsidence Curve from a `.json` or `.csv` file."""
    curve = _load_single_curve(filepath, kind="Subsidence curve")
    return _updateSubsidenceCurveName(curve)


def loadSubsidenceCurveFromJsonObj(obj: dict[str, Any]) -> Curve:
    """Load a subsidence Curve from an embedded Curve JSON object."""
    curve = loadCurveFromJsonObj(obj)
    return _updateSubsidenceCurveName(curve)


def _updateSubsidenceCurveName(curve: Curve) -> Curve:
    """Normalize an already-loaded Curve to a subsidence curve."""
    if curve._yAxisName.lower() != "subsidence":
        print(
            f"Warning: Subsidence curve yAxisName is '{curve._yAxisName}', "
            "expected 'Subsidence'. Curve was renamed 'Subsidence' but check "
            "your input file if this is unexpected."
        )
    curve._yAxisName = "Subsidence"
    return curve


def loadEustaticCurve(filepath: str) -> Curve:
    """Load an eustatic Curve from a `.json` or `.csv` file."""
    curve = _load_single_curve(filepath, kind="Eustatic curve")
    return _updateEustaticCurveName(curve)


def loadEustaticCurveFromJsonObj(obj: dict[str, Any]) -> Curve:
    """Load an eustatic Curve from an embedded Curve JSON object."""
    curve = loadCurveFromJsonObj(obj)
    return _updateEustaticCurveName(curve)


def _updateEustaticCurveName(curve: Curve) -> Curve:
    """Normalize an already-loaded Curve to an eustatic curve."""
    if curve._yAxisName.lower() != "eustatism":
        print(
            f"Warning: Eustatic curve yAxisName is '{curve._yAxisName}', "
            "expected 'Eustatic'. Curve was renamed 'Eustatic' but check your "
            "inputfile if this is unexpected."
        )
    curve._yAxisName = "Eustatism"
    return curve


def loadCurvesFromCsv(path: Path) -> list[Curve]:
    """Load one or multiple Curve from a CSV file.

    The csv file is expected to have a row header defining axis names and at
    least two columns (the first one defines the x values and the other ones
    the y values of the curves):

    - depth: depth value
    - Curve 1 values: the value of the first curve at the given depth.
    - Curve 2 values: the value of the second curve at the given depth.
    - ...

    :param Path path: Path to the CSV file.
    :returns list[Curve]: Loaded curves.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(
            "Curve CSV must have at least 2 columns (x,y) with a header row."
        )

    # Axis names come from the header row when available.
    x_axis_name = str(df.columns[0]).strip() if df.columns.size >= 1 else "x"
    if x_axis_name == "" or x_axis_name.lower().startswith("unnamed"):
        x_axis_name = "x"
    x_series = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    # get the curves
    curves: list[Curve] = []
    for col_idx in range(1, df.shape[1]):
        y_axis_name = str(df.columns[col_idx]).strip()
        if y_axis_name == "" or y_axis_name.lower().startswith("unnamed"):
            y_axis_name = f"y{col_idx}"

        y_series = pd.to_numeric(df.iloc[:, col_idx], errors="coerce")
        clean = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
        if clean.shape[0] < 1:
            raise ValueError(
                f"Curve CSV column {col_idx} must contain at least one valid "
                "numeric (x,y) row."
            )

        # Ensure increasing x values for interpolation.
        clean = clean.sort_values("x", kind="mergesort")
        # If x values are duplicated, keep the last occurrence.
        clean = clean.drop_duplicates(subset=["x"], keep="last")

        x = clean["x"].to_numpy(dtype=float)
        y = clean["y"].to_numpy(dtype=float)
        if x.size < 1 or x.size != y.size:
            raise ValueError(
                f"Curve CSV column {col_idx} must have same non-zero x/y "
                "length."
            )

        # Default interpolation: linear.
        curves.append(Curve(x_axis_name, y_axis_name, x, y, "linear"))

    print("Interpolation method set to 'linear' by default")
    return curves


def loadCurveFromJsonObj(obj: dict[str, Any]) -> Curve:
    """Parse a Curve JSON object into a Curve.

    CurveSchema.json expects as an example:

    .. code-block:: json

        {
          "format": "pyWellSFM.CurveData",
          "version": "1.0",
          "curve": {
            "xAxisName": "Depth",
            "yAxisName": "Value",
            "interpolationMethod": "linear",
            "data": [
              {"x": 0, "y": 0},
              {"x": 1, "y": 1}
            ]
          }
        }

    :returns Curve: Curve object.
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.CurveData",
        expected_version="1.0",
        kind="curve",
    )

    curve_obj = obj.get("curve")
    if not isinstance(curve_obj, dict):
        raise ValueError("Curve JSON must contain a 'curve' object.")

    x_axis_name = curve_obj.get("xAxisName")
    y_axis_name = curve_obj.get("yAxisName")
    if not isinstance(x_axis_name, str) or x_axis_name.strip() == "":
        raise ValueError("Curve.curve.xAxisName must be a non-empty string.")
    if not isinstance(y_axis_name, str) or y_axis_name.strip() == "":
        raise ValueError("Curve.curve.yAxisName must be a non-empty string.")

    interpolation_method = curve_obj.get("interpolationMethod")
    interpolation_function: str | Any
    interpolation_args: dict[str, Any] = {}
    if isinstance(interpolation_method, str):
        if interpolation_method.strip() == "":
            raise ValueError(
                "Curve.curve.interpolationMethod must be non-empty."
            )
        interpolation_function = interpolation_method
    elif isinstance(interpolation_method, dict):
        degree = interpolation_method.get("degree")
        nb_points = interpolation_method.get("nbPoints")
        if not isinstance(degree, int) or degree < 1:
            raise ValueError(
                "Curve.curve.interpolationMethod.degree must be an" \
                " integer >= 1."
            )
        if not isinstance(nb_points, int) or nb_points < 2:
            raise ValueError(
                "Curve.curve.interpolationMethod.nbPoints must be an" \
                " integer >= 2."
            )
        interpolation_function = PolynomialInterpolator()
        # PolynomialInterpolator expects 'deg' and 'nbPts'.
        interpolation_args = {"deg": degree, "nbPts": nb_points}
    else:
        raise ValueError(
            "Curve.curve.interpolationMethod must be a string or an object."
        )

    data = curve_obj.get("data")
    if not isinstance(data, list):
        raise ValueError("Curve.curve.data must be an array.")
    if len(data) < 2:
        raise ValueError("Curve.curve.data must contain at least 2 points.")

    x_values: list[float] = []
    y_values: list[float] = []
    for i, point in enumerate(data):
        if not isinstance(point, dict):
            raise ValueError(f"Curve.curve.data[{i}] must be an object.")
        if "x" not in point or "y" not in point:
            raise ValueError(
                f"Curve.curve.data[{i}] must contain 'x' and 'y'."
            )
        try:
            x_values.append(float(point["x"]))
            y_values.append(float(point["y"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Curve.curve.data[{i}] x/y values must be numeric."
            ) from exc

    # Sort by x and drop duplicate x (keep last).
    pairs = np.column_stack(
        (np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float))
    )
    sort_idx = np.argsort(pairs[:, 0], kind="mergesort")
    pairs = pairs[sort_idx]
    # keep last occurrence for duplicated x
    _, unique_last_idx = np.unique(pairs[:, 0], return_index=True)
    # np.unique returns first index; to keep last, we compute mask manually
    # by scanning from end
    mask = np.ones(pairs.shape[0], dtype=bool)
    seen: set[float] = set()
    for j in range(pairs.shape[0] - 1, -1, -1):
        xj = float(pairs[j, 0])
        if xj in seen:
            mask[j] = False
        else:
            seen.add(xj)
    pairs = pairs[mask]

    x = pairs[:, 0].astype(float)
    y = pairs[:, 1].astype(float)
    if x.size < 2:
        raise ValueError("Curve must contain at least 2 unique x values.")

    return Curve(
        x_axis_name,
        y_axis_name,
        x,
        y,
        interpolation_function,
        **interpolation_args,
    )


def loadCurvesFromFile(
    path: Path,
) -> list[Curve]:
    """Load one or multiple Curve from a `.json` or `.csv` file.

    json files contain a single Curve object, csv files may contain multiple
    curves.

    - `.json`: must match the Curve JSON structure.
    - `.csv`: expects two numeric columns (x,y), 1 row header defining axis
      names.

    :returns list[Curve]: list of curve objects.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    if ext == ".json":
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Tabulated function JSON must be an object.")
        return [loadCurveFromJsonObj(obj)]
    elif ext == ".csv":
        return loadCurvesFromCsv(path)

    raise ValueError(
        f"Unsupported file extension '{ext}' for tabulated function."
    )


def loadUncertaintyCurveFromFile(
    path: Path,
) -> UncertaintyCurve:
    """Load an UncertaintyCurve from a `.json` or `.csv` file.

    - `.json`: must match the Curve JSON structure.
    - `.csv`: expects two to four numeric columns (x,y, ymin, ymax), 1 row
      header defining axis names.

    :returns Curve: curve object.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    if ext == ".json":
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Tabulated function JSON must be an object.")
        return loadUncertaintyCurveFromJsonObj(obj)
    elif ext == ".csv":
        return loadUncertaintyCurveFromCsv(path)

    raise ValueError(
        f"Unsupported file extension '{ext}' for tabulated function."
    )


def loadUncertaintyCurveFromJsonObj(obj: dict[str, Any]) -> UncertaintyCurve:
    """Parse an UncertaintyCurve JSON object into an UncertaintyCurve.

    Schema in jsonSchemas/UncertaintyCurveSchema.json uses:

    - format: "pyWellSFM.UncertaintyCurveData"
    - version: "1.0"
    - data: {"name": str, "curves": [<CurveSchema objects>]}

    :returns UncertaintyCurve: uncertainty curve object.
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.UncertaintyCurveData",
        expected_version="1.0",
        kind="uncertainty curve",
    )

    data = obj.get("data")
    if not isinstance(data, dict):
        raise ValueError(
            "Uncertainty curve JSON must contain a 'data' object."
        )

    curve_name = data.get("name")
    if not isinstance(curve_name, str) or curve_name.strip() == "":
        raise ValueError(
            "UncertaintyCurve.data.name must be a non-empty string."
        )

    curves_obj = data.get("curves")
    if not isinstance(curves_obj, list) or len(curves_obj) < 1:
        raise ValueError(
            "UncertaintyCurve.data.curves must be a non-empty array."
        )
    if len(curves_obj) > 3:
        raise ValueError(
            "UncertaintyCurve.data.curves supports at most 3 curves"
            " (median/min/max)."
        )

    parsed_curves: list[Curve] = []
    for i, cobj in enumerate(curves_obj):
        if not isinstance(cobj, dict):
            raise ValueError(
                f"UncertaintyCurve.data.curves[{i}] must be an object."
            )
        parsed_curves.append(loadCurveFromJsonObj(cobj))

    # Validate x-axis consistency across curves.
    x_axis_name = parsed_curves[0]._xAxisName
    abscissa = parsed_curves[0]._abscissa
    for i, c in enumerate(parsed_curves[1:], start=1):
        if c._xAxisName != x_axis_name:
            raise ValueError(
                "All curves in an UncertaintyCurve must share the same "
                f"xAxisName. Got '{x_axis_name}' and '{c._xAxisName}' at "
                f"index {i}."
            )
        if c._abscissa.shape != abscissa.shape or not np.allclose(
            c._abscissa, abscissa, equal_nan=True
        ):
            raise ValueError(
                "All curves in an UncertaintyCurve must share the same" \
                " abscissa values."
            )

    def _infer_role(curve: Curve) -> str | None:
        y = str(getattr(curve, "_yAxisName", "")).strip().lower()
        if any(k in y for k in ("median", "mean", "mid")):
            return "median"
        if any(k in y for k in ("min", "lower", "ymin")):
            return "min"
        if any(k in y for k in ("max", "upper", "ymax")):
            return "max"
        return None

    roles: dict[str, Curve] = {}
    unlabeled: list[Curve] = []
    for c in parsed_curves:
        role = _infer_role(c)
        if role is None:
            unlabeled.append(c)
            continue
        if role in roles:
            raise ValueError(
                f"UncertaintyCurve.data.curves has multiple '{role}' curves."
            )
        roles[role] = c

    # Determine median/min/max mapping.
    median_curve: Curve
    min_curve: Curve | None = None
    max_curve: Curve | None = None

    if "median" in roles:
        median_curve = roles["median"]
    elif len(parsed_curves) == 1:
        median_curve = parsed_curves[0]
    elif len(parsed_curves) == 3 and not roles:
        # Convention: [median, min, max]
        median_curve = parsed_curves[0]
        min_curve = parsed_curves[1]
        max_curve = parsed_curves[2]
    else:
        # Fallback: take first curve as the median.
        median_curve = parsed_curves[0]

    if min_curve is None and "min" in roles:
        min_curve = roles["min"]
    if max_curve is None and "max" in roles:
        max_curve = roles["max"]

    if len(parsed_curves) == 2 and (min_curve is None or max_curve is None):
        raise ValueError(
            "With 2 curves, yAxisName must identify 'min' and 'max'\n"
            "(e.g. contains 'min'/'max')."
        )

    ucurve = UncertaintyCurve(curve_name, median_curve)
    if min_curve is not None:
        ucurve.setMinCurveValues(min_curve._ordinate)
    if max_curve is not None:
        ucurve.setMaxCurveValues(max_curve._ordinate)
    return ucurve


def loadUncertaintyCurveFromCsv(
    path: Path, delimiter: str = ","
) -> UncertaintyCurve:
    """Load an UncertaintyCurve from a CSV file.

    The csv file is expected to have a row header defining axis names and two
    to four columns:

    - depth: depth value
    - value: the value of the curve at the given depth.
    - ymin (optional): minimum value at the given depth.
    - ymax (optional): maximum value at the given depth.

    Depending on csv file format, the UncertaintyCurve object is populated as
    follows:

    - if 1 column, set ymin=ymax=value by default
    - if 2 columns, search for min or max keywords in header to set ymin/ymax.
      Otherwise, compare the values to the mean, if values < mean, then set
      ymin=value, ymax=mean, else set ymin=mean, ymax=value
    - if 3 columns, search for min or max keywords in header to set ymin/ymax,
      or compare values to mean as above.

    :param Path path: Path to the CSV file.
    :returns UncertaintyCurve: Loaded uncertainty curve.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    # CSVs found in the wild may or may not have a header. Check first line for
    # alphabetic characters to detect presence of header.
    with path.open(encoding="utf-8") as f:
        first_line = f.readline()
        print(first_line)
        if not any(ch.isalpha() for ch in first_line):
            raise ValueError(
                "UncertaintyCurve CSV must have a header row defining" \
                " axis names."
            )

    df = pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
    )

    if df.shape[1] < 2 or df.shape[1] > 4:
        raise ValueError(
            "UncertaintyCurve CSV must have 2 to 4 columns "
            "(x, y[, ymin[, ymax]])."
        )

    x_axis_name = str(df.columns[0]).strip() or path.stem
    y_axis_name = str(df.columns[1]).strip() or "median"
    if x_axis_name.lower().startswith("unnamed"):
        x_axis_name = path.stem
    if y_axis_name.lower().startswith("unnamed"):
        y_axis_name = "median"

    x_series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y_series = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    clean = pd.DataFrame({"x": x_series, "y": y_series})

    def _is_min(col_name: str) -> bool:
        n = col_name.strip().lower()
        return any(k in n for k in ("min", "lower", "ymin"))

    def _is_max(col_name: str) -> bool:
        n = col_name.strip().lower()
        return any(k in n for k in ("max", "upper", "ymax"))

    ymin_present = False
    ymax_present = False
    if df.shape[1] == 3:
        third_series = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        third_name = str(df.columns[2])
        if _is_max(third_name) and not _is_min(third_name):
            clean["ymax"] = third_series
            ymax_present = True
        elif _is_min(third_name) and not _is_max(third_name):
            clean["ymin"] = third_series
            ymin_present = True
        else:
            # Infer from median values.
            median_values = clean["y"]
            ymin_values = pd.Series(
                np.where(
                    third_series <= median_values, third_series, median_values
                )
            )
            ymax_values = pd.Series(
                np.where(
                    third_series >= median_values, third_series, median_values
                )
            )
            clean["ymin"] = ymin_values
            clean["ymax"] = ymax_values
            ymin_present = True
            ymax_present = True
    elif df.shape[1] == 4:
        third_series = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        fourth_series = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        third_name = str(df.columns[2])
        fourth_name = str(df.columns[3])

        # Default convention: x, median, ymin, ymax
        ymin_s = third_series
        ymax_s = fourth_series
        if _is_max(third_name) and _is_min(fourth_name):
            # Swap if headers clearly indicate reversed order.
            ymin_s, ymax_s = fourth_series, third_series

        clean["ymin"] = ymin_s
        clean["ymax"] = ymax_s
        ymin_present = True
        ymax_present = True

    # Drop rows where x or median is not numeric.
    clean = clean.dropna(subset=["x", "y"])
    if clean.shape[0] < 2:
        raise ValueError(
            "UncertaintyCurve CSV must contain at least 2 valid numeric "
            "(x, y) rows."
        )

    # Ensure increasing x values for interpolation.
    clean = clean.sort_values("x", kind="mergesort")
    clean = clean.drop_duplicates(subset=["x"], keep="last")

    x = clean["x"].to_numpy(dtype=float)
    y = clean["y"].to_numpy(dtype=float)
    if x.size < 2 or x.size != y.size:
        raise ValueError(
            "UncertaintyCurve CSV must have at least 2 unique x values."
        )

    print("Interpolation method set to 'linear' by default")
    median_curve = Curve(x_axis_name, y_axis_name, x, y, "linear")
    ucurve = UncertaintyCurve(x_axis_name, median_curve)

    if ymin_present and "ymin" in clean.columns:
        ymin = pd.to_numeric(clean["ymin"], errors="coerce").to_numpy(
            dtype=float
        )
        if ymin.size == y.size:
            ymin = np.where(np.isfinite(ymin), ymin, y)
            ucurve.setMinCurveValues(ymin)
    if ymax_present and "ymax" in clean.columns:
        ymax = pd.to_numeric(clean["ymax"], errors="coerce").to_numpy(
            dtype=float
        )
        if ymax.size == y.size:
            ymax = np.where(np.isfinite(ymax), ymax, y)
            ucurve.setMaxCurveValues(ymax)

    return ucurve
