# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for Well.

Currently supports loading a well from a JSON file that conforms to
`jsonSchemas/WellSchema.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd
from striplog import Component, Interval, Striplog

from pywellsfm.io._common import require_url, resolve_ref_path
from pywellsfm.io.curve_io import (
    curveToJsonObj,
    loadCurveFromJsonObj,
    loadCurvesFromFile,
)
from pywellsfm.io.json_schema_validation import (
    expect_format_version,
)
from pywellsfm.model.Curve import Curve
from pywellsfm.model.Marker import Marker, StratigraphicSurfaceType
from pywellsfm.model.Well import Well


def wellToJsonObj(well: Well) -> dict[str, Any]:
    """Serialize a Well to JSON matching `jsonSchemas/WellSchema.json`."""

    def _as_float(value: Any) -> float: # noqa: ANN401
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"Value '{value}' is not a number") from exc

    def _depth_from_position(pos: Any) -> float: # noqa: ANN401
        if hasattr(pos, "z"):
            return _as_float(pos.z)
        return _as_float(pos)

    def _lithology_from_interval(interval: Interval) -> str:
        try:
            comps = getattr(interval, "components", None)
            if comps:
                for comp in comps:
                    if isinstance(comp, dict):
                        for key in ("lithology", "Lithology"):
                            if key in comp and str(comp[key]).strip() != "":
                                return str(comp[key]).strip()
                    else:
                        try:
                            val = comp.get("lithology")  # type: ignore[attr-defined]
                            if val is not None and str(val).strip() != "":
                                return str(val).strip()
                        except Exception:
                            pass
        except Exception:
            pass
        return "Unknown"

    def _striplog_to_json(log_name: str, striplog: Striplog) -> dict[str, Any]:
        intervals_out: list[dict[str, Any]] = []
        itv: Interval
        for itv in striplog:
            top = _depth_from_position(itv.top)
            base = _depth_from_position(itv.base)
            litho = _lithology_from_interval(itv)
            if str(litho).strip() == "":
                litho = "Unknown"
            intervals_out.append(
                {
                    "top": float(top),
                    "base": float(base),
                    "lithology": str(litho),
                }
            )

        if len(intervals_out) < 1:
            raise ValueError(
                f"Striplog '{log_name}' must contain at least 1 interval"
            )

        return {"name": str(log_name), "intervals": intervals_out}

    head = np.asarray(well.wellHeadCoords, dtype=float)
    if head.size != 3:
        raise ValueError("Well.wellHeadCoords must be a 3-element array")

    well_obj: dict[str, Any] = {
        "name": str(well.name),
        "location": {
            "x": float(head[0]),
            "y": float(head[1]),
            "z": float(head[2]),
        },
        "depth": float(well.depth),
    }

    well_path = getattr(well, "_wellPath", None)
    if well_path is not None:
        arr = np.asarray(well_path, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 2:
            well_obj["wellPath"] = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in arr
            ]

    markers = list(well.getMarkers())
    if markers:
        markers_out: list[dict[str, Any]] = []
        for m in markers:
            st = getattr(
                m, "stratigraphicType", StratigraphicSurfaceType.UNKNOWN
            )
            st_val = (
                st.value
                if isinstance(st, StratigraphicSurfaceType)
                else str(st)
            )
            markers_out.append(
                {
                    "name": str(m.name),
                    "depth": float(m.depth),
                    "age": float(m.age),
                    "stratigraphicType": str(st_val),
                }
            )
        markers_out.sort(key=lambda d: (d["depth"], d["name"]))
        well_obj["markers"] = markers_out

    logs = getattr(well, "_logs", {})
    if isinstance(logs, dict) and logs:
        striplogs_out: list[dict[str, Any]] = []
        curves_out: list[dict[str, Any]] = []

        for log_name in sorted(logs.keys(), key=lambda s: str(s)):
            log = logs[log_name]
            if isinstance(log, Striplog):
                striplogs_out.append(_striplog_to_json(str(log_name), log))
            elif isinstance(log, Curve):
                curves_out.append(
                    curveToJsonObj(
                        log,
                        y_axis_name=str(log_name),
                        x_axis_name_default="Depth",
                    )
                )

        if striplogs_out:
            well_obj["striplogs"] = striplogs_out
        if curves_out:
            well_obj["continuousLogs"] = curves_out

    return {"format": "pyWellSFM.WellData", "version": "1.0", "well": well_obj}


def _parse_stratigraphic_type(raw: Any) -> StratigraphicSurfaceType: # noqa: ANN401
    print("Debug: Parsing stratigraphic type from raw value:", raw)
    if not isinstance(raw, str) or raw.strip() == "":
        return StratigraphicSurfaceType.UNKNOWN

    value = raw.strip()
    # Try matching enum values.
    for member in StratigraphicSurfaceType:
        if member.value.lower() == value.lower():
            return member

    print(
        f"Warning: unknown StratigraphicSurfaceType value '{value}', "
        "defaulting to 'Unknown'."
    )
    return StratigraphicSurfaceType.UNKNOWN


def _load_striplog_from_json_obj(obj: dict[str, Any]) -> tuple[str, Striplog]:
    name = obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Striplog.name must be a non-empty string.")

    intervals_raw = obj.get("intervals")
    if not isinstance(intervals_raw, list):
        raise ValueError(f"Striplog '{name}' intervals must be an array.")
    if len(intervals_raw) < 1:
        raise ValueError(
            f"Striplog '{name}' intervals must contain at least 1 item."
        )

    intervals: list[Interval] = []
    for i, itv in enumerate(intervals_raw):
        if not isinstance(itv, dict):
            raise ValueError(
                f"Striplog '{name}' intervals[{i}] must be an object."
            )

        top = itv.get("top")
        base = itv.get("base")
        litho = itv.get("lithology")
        if not isinstance(top, (int, float)) or not isinstance(
            base, (int, float)
        ):
            raise ValueError(
                f"Striplog '{name}' intervals[{i}].top/base must be numbers."
            )
        if not isinstance(litho, str) or litho.strip() == "":
            raise ValueError(
                f"Striplog '{name}' intervals[{i}].lithology must be a "
                "non-empty string."
            )

        if float(base) < float(top):
            raise ValueError(
                f"Striplog '{name}' intervals[{i}] base must be >= top."
            )

        comp = Component({"lithology": litho})
        intervals.append(Interval(float(top), float(base), components=[comp]))

    return name, Striplog(intervals)


def _load_striplog_from_csv(path: Path, delimiter: str = ",") -> Striplog:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(str(path), delimiter=delimiter, engine="python")
    if (
        "top" not in df.columns
        or "base" not in df.columns
        or "lithology" not in df.columns
    ):
        raise ValueError(
            "Striplog CSV must contain 'top', 'base', and 'lithology' columns."
        )

    intervals: list[Interval] = []
    for _, row in df.iterrows():
        top = float(row["top"])
        base = float(row["base"])
        comp_dict = {
            str(key): row[key]
            for key in row.index
            if str(key).lower() not in ("top", "base", "lithology")
        }
        intervals.append(
            Interval(top, base, components=[Component(comp_dict)])
        )

    return Striplog(intervals)


def loadWell(filepath: str) -> Well:
    """Load a Well from a file.

    Supported file formats are:

    - json file matching `jsonSchemas/WellSchema.json`.
    - las file matching LAS 2.0 format.

    .. Note::

        - `well.location` defines the well head (x,y,z).
        - `continuousLogs` items use the Curve schema and are stored under the
          log name derived from `curve.yAxisName`.
        - `striplogs` items are stored under the striplog `name`.

    :param str filepath: path to the well file
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    if ext == ".json":
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Tabulated function JSON must be an object.")
        return loadWellFromJsonObj(obj, base_dir=path.resolve().parent)
    if ext == ".las":
        return loadWellFromLasFile(str(path))

    raise ValueError("Well file must be either a .json or .las file.")


def loadWellFromJsonObj(
    obj: dict[str, Any], base_dir: Path | None = None
) -> Well:
    """Load a Well from a JSON file matching `jsonSchemas/WellSchema.json`.

    .. Note::

        - `well.location` defines the well head (x,y,z).
        - `continuousLogs` items use the Curve schema and are stored under the
          log name derived from `curve.yAxisName`.
        - `striplogs` items are stored under the striplog `name`.
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.WellData",
        expected_version="1.0",
        kind="well",
    )

    well_obj = obj.get("well")
    if not isinstance(well_obj, dict):
        raise ValueError("Well.well must be an object.")

    name = well_obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Well.well.name must be a non-empty string.")

    loc = well_obj.get("location")
    if not isinstance(loc, dict):
        raise ValueError("Well.well.location must be an object.")
    x = loc.get("x")
    y = loc.get("y")
    z = loc.get("z")
    if (
        not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not isinstance(z, (int, float))
    ):
        raise ValueError("Well.well.location.x/y/z must be numbers.")

    depth = well_obj.get("depth")
    if not isinstance(depth, (int, float)):
        raise ValueError("Well.well.depth must be a number.")

    well_head = np.asarray([float(x), float(y), float(z)], dtype=float)
    well = Well(name=str(name), wellHeadCoords=well_head, depth=float(depth))

    # Optional well path
    well_path_raw = well_obj.get("wellPath")
    if isinstance(well_path_raw, list):
        coords: list[list[float]] = []
        for i, pt in enumerate(well_path_raw):
            if not isinstance(pt, dict):
                raise ValueError(f"Well.well.wellPath[{i}] must be an object.")
            px, py, pz = pt.get("x", 0), pt.get("y", 0), pt.get("z", 0)
            if not all(isinstance(v, (int, float)) for v in (px, py, pz)):
                raise ValueError(
                    f"Well.well.wellPath[{i}].x/y/z must be numeric values."
                )
            coords.append([float(px), float(py), float(pz)])
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            raise ValueError(
                "Well.well.wellPath must be an array of 2+ (x,y,z) points"
            )
        well.setWellPath(arr)

    # Optional markers
    markers_raw = well_obj.get("markers")
    if isinstance(markers_raw, list):
        marker_set: set[Marker] = set()
        for i, m in enumerate(markers_raw):
            if not isinstance(m, dict):
                raise ValueError(f"Well.well.markers[{i}] must be an object.")

            m_name = m.get("name")
            m_depth = m.get("depth")
            m_age = m.get("age")
            m_type = m.get("stratigraphicType")

            if not isinstance(m_name, str) or m_name.strip() == "":
                raise ValueError(
                    f"Well.well.markers[{i}].name must be a string."
                )
            if not isinstance(m_depth, (int, float)):
                raise ValueError(
                    f"Well.well.markers[{i}].depth must be a number."
                )
            if not isinstance(m_age, (int, float)):
                raise ValueError(
                    f"Well.well.markers[{i}].age must be a number."
                )

            marker_set.add(
                Marker(
                    name=str(m_name),
                    depth=float(m_depth),
                    age=float(m_age),
                    stratigraphicType=_parse_stratigraphic_type(m_type),
                )
            )
        if marker_set:
            well.addMarkers(marker_set)

    # Optional striplogs
    striplogs_raw = well_obj.get("striplogs")
    if isinstance(striplogs_raw, list):
        for i, sl in enumerate(striplogs_raw):
            if isinstance(sl, dict):
                sl_name, striplog = _load_striplog_from_json_obj(sl)
                well.addLog(sl_name, striplog)
            elif isinstance(sl, str):
                path = resolve_ref_path(
                    base_dir=base_dir,
                    raw_url=sl,
                    ctx=f"Well.well.striplogs[{i}]",
                )
                striplog = _load_striplog_from_csv(path)
                # Use filename stem as fallback log name.
                well.addLog(path.stem, striplog)
            else:
                raise ValueError(
                    f"Well.well.striplogs[{i}] must be an object or a "
                    "string path."
                )

    # Optional continuous logs
    cont_raw = well_obj.get("continuousLogs")
    if isinstance(cont_raw, list):
        for i, item in enumerate(cont_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Well.well.continuousLogs[{i}] must be a Curve "
                    "JSON object."
                )
            curve = loadCurveFromJsonObj(item)
            # Store using yAxisName (log name) from the JSON object.
            curve_obj = item.get("curve")
            log_name = None
            if isinstance(curve_obj, dict):
                log_name = curve_obj.get("yAxisName")
            if not isinstance(log_name, str) or log_name.strip() == "":
                log_name = getattr(curve, "_yAxisName", f"log_{i}")
            well.addLog(str(log_name), curve)
    elif isinstance(cont_raw, dict):
        # External curve file
        url = require_url(obj=cont_raw, ctx="Well.well.continuousLogs")
        curve_path = resolve_ref_path(
            base_dir=base_dir,
            raw_url=url,
            ctx="Well.well.continuousLogs",
        )
        curves = loadCurvesFromFile(curve_path)
        for curve in curves:
            log_name = getattr(curve, "_yAxisName", curve_path.stem)
            xaxisName = (
                curve._xAxisName if hasattr(curve, "_xAxisName") else "Depth"
            )
            if xaxisName.lower() == "depth":
                well.addLog(str(log_name), curve)
            elif xaxisName.lower() == "age":
                well.addAgeLog(str(log_name), curve)
            else:
                print(
                    f"Warning: Curve xAxisName '{xaxisName}' not recognized, "
                    "adding as depth log."
                )
                well.addLog(str(log_name), curve)
    return well


def loadWellFromLasFile(filepath: str) -> Well:
    """Load a Well from a LAS file.

    Uses `lasio` to parse well header and curves.

    Populates:

    - `Well.name`
    - `Well.wellHeadCoords` (defaults to [0,0,0] if not found)
    - `Well.depth` (prefers STOP if present, else max depth index)
    - `Well._wellPath` (vertical path with 2 points)
    - `Well._logs` (continuous logs as `Curve`, keyed by LAS mnemonic)

    :param str filepath: path to the LAS file
    :raises FileNotFoundError: if file does not exist
    :raises ImportError: if `lasio` is not installed
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(str(path))

    las = lasio.read(str(path), ignore_header_errors=True)

    def _get_header_value(*keys: str) -> Any: # noqa: ANN401
        for key in keys:
            try:
                if hasattr(las, "well") and key in las.well:
                    return las.well[key].value
            except Exception:
                pass
            try:
                if hasattr(las, "params") and key in las.params:
                    return las.params[key].value
            except Exception:
                pass
        return None

    def _to_float(value: Any) -> float | None: # noqa: ANN401
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_non_empty_str(value: Any) -> str | None: # noqa: ANN401
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return None
        return s

    # --- Name ---
    name = (
        _to_non_empty_str(_get_header_value("WELL", "WELLNAME", "WELL NAME"))
        or _to_non_empty_str(_get_header_value("UWI", "API"))
        or path.stem
    )

    # --- Well head coordinates ---
    x0 = _to_float(_get_header_value("X", "XCOORD", "EASTING", "LONG", "LON"))
    y0 = _to_float(_get_header_value("Y", "YCOORD", "NORTHING", "LAT"))
    # Elevation/KB/GL are various variants; store as z if present.
    z0 = _to_float(
        _get_header_value("Z", "ELEV", "ELEVATION", "KB", "GL", "DATUM")
    )
    well_head = np.asarray(
        [
            x0 if x0 is not None else 0.0,
            y0 if y0 is not None else 0.0,
            z0 if z0 is not None else 0.0,
        ],
        dtype=float,
    )

    # --- Depth ---
    stop = _to_float(_get_header_value("STOP", "STRT", "TD", "TDEP"))
    index = np.asarray(getattr(las, "index", []), dtype=float)
    index = index[np.isfinite(index)]
    if index.size < 1:
        raise ValueError("LAS file contains no valid depth/index values.")
    index_max = float(np.nanmax(index))

    # Prefer STOP/TD when available but ensure depth is at least max index.
    depth = (
        float(stop) if stop is not None and np.isfinite(stop) else index_max
    )
    depth = max(depth, index_max)

    well = Well(name=str(name), wellHeadCoords=well_head, depth=float(depth))

    # --- Well path ---
    # Keep convention aligned with JSON loader tests: z increases with depth.
    path_arr = np.asarray(
        [well_head, well_head + np.asarray([0.0, 0.0, float(depth)])],
        dtype=float,
    )
    well.setWellPath(path_arr)

    # --- Logs ---
    # LAS index is the abscissa (depth). Curves are ordinates.
    x_raw = np.asarray(getattr(las, "index", []), dtype=float)
    if x_raw.size < 2:
        # If no index, attempt to use first curve as index (rare).
        raise ValueError(
            "LAS file must contain at least 2 depth/index samples."
        )

    null_value = getattr(las, "null", None)
    for curve_info in getattr(las, "curves", []):
        mnemonic = getattr(curve_info, "mnemonic", None)
        mnemonic_str = _to_non_empty_str(mnemonic)
        if mnemonic_str is None:
            continue

        # Skip the index curve itself.
        if mnemonic_str.strip().upper() in {"DEPT", "DEPTH", "TIME"}:
            continue

        try:
            y_raw = np.asarray(las[mnemonic_str], dtype=float)
        except Exception:
            continue

        if y_raw.size != x_raw.size:
            continue

        x = x_raw.copy()
        y = y_raw.copy()

        # Apply LAS null value if present.
        if null_value is not None:
            nv = _to_float(null_value)
            if nv is not None and np.isfinite(nv):
                y = np.where(y == nv, np.nan, y) # type: ignore[arg-type]

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            continue

        # Ensure increasing x and drop duplicated x (keep last occurrence).
        sort_idx = np.argsort(x, kind="mergesort")
        x = x[sort_idx]
        y = y[sort_idx]
        keep = np.ones(x.size, dtype=bool)
        seen: set[float] = set()
        for i in range(x.size - 1, -1, -1):
            xi = float(x[i])
            if xi in seen:
                keep[i] = False
            else:
                seen.add(xi)
        x = x[keep]
        y = y[keep]
        if x.size < 2:
            continue

        curve = Curve(
            "Depth", mnemonic_str, x.astype(float), y.astype(float), "linear"
        )
        well.addLog(mnemonic_str, curve)

    return well


def saveWellToJson(well: Well, filepath: str) -> None:
    """Save a Well to a JSON file.

    Output json file follows the schema defined by
    `jsonSchemas/WellSchema.json`.

    :param Well well: well object to save
    :param str filepath: path to output well file
    """
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Well output file must have a .json extension.")

    out = wellToJsonObj(well)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=4, ensure_ascii=False), encoding="utf-8"
    )


def saveWell(well: Well, filepath: str) -> None:
    """Save a Well to a file.

    Supported output formats are:

    - json file following the schema defined by `jsonSchemas/WellSchema.json`.

    :param Well well: well object to save
    :param str filepath: path to output well file
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".json":
        return saveWellToJson(well, filepath)
    else:
        raise ValueError("Well output file must have a .json extension.")
