# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Unit tests for Well and Well IO."""

from __future__ import annotations

import os
import sys

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from striplog import Component, Interval, Striplog

import pywellsfm.io.well_io as wio
from pywellsfm.io import loadWell
from pywellsfm.io.json_schema_validation import validateWellJsonFile
from pywellsfm.model import Curve
from pywellsfm.model.Marker import (
    Marker,
    StratigraphicSurfaceType,
)
from pywellsfm.model.Well import Well


def _write_json(tmp_path: Path, payload: dict[str, Any], filename: str) -> str:
    """Helper to write a temporary json file.

    :param Path tmp_path: path to temporary directory
    :param dict[str, Any] payload: data to write to the json file
    :param str filename: name of the json file
    :return str: path to the written json file
    """
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def test_loadWell_minimal(tmp_path: Path) -> None:
    """Test of loadWellFromJson with minimal information.

    :param Path tmp_path: temporary path fixture
    """
    payload = {
        "format": "pyWellSFM.WellData",
        "version": "1.0",
        "well": {
            "name": "Well-A",
            "location": {"x": 12.5, "y": -3.0, "z": 0.0},
            "depth": 100.0,
        },
    }

    well = wio.loadWellFromJsonObj(payload)

    assert isinstance(well, Well)
    assert well.name == "Well-A"
    assert well.depth == 100.0
    assert np.allclose(well.wellHeadCoords, np.array([12.5, -3.0, 0.0]))


def test_loadWell_with_markers_path_and_logs(tmp_path: Path) -> None:
    """Test of loadWell with markers, well path and logs.

    :param Path tmp_path: temporary path fixture
    """
    payload = {
        "format": "pyWellSFM.WellData",
        "version": "1.0",
        "well": {
            "name": "Well-B",
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "depth": 100.0,
            "wellPath": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 100.0},
            ],
            "markers": [
                {
                    "name": "M1",
                    "depth": 10.0,
                    "age": 5.0,
                    "stratigraphicType": "Baselap",
                }
            ],
            "striplogs": [
                {
                    "name": "lithology",
                    "intervals": [
                        {"top": 0.0, "base": 15.0, "lithology": "sandstone"},
                        {"top": 15.0, "base": 30.0, "lithology": "shale"},
                    ],
                }
            ],
            "continuousLogs": [
                {
                    "format": "pyWellSFM.CurveData",
                    "version": "1.0",
                    "curve": {
                        "xAxisName": "Depth",
                        "yAxisName": "Gamma",
                        "interpolationMethod": "linear",
                        "data": [
                            {"x": 0.0, "y": 50.0},
                            {"x": 100.0, "y": 80.0},
                        ],
                    },
                }
            ],
        },
    }

    well = wio.loadWellFromJsonObj(payload)

    assert well.name == "Well-B"
    assert well.getDepthLog("lithology") is not None
    assert isinstance(well.getDepthLog("lithology"), Striplog)
    assert "lithology" in well.getDiscreteLogNames()

    gamma = well.getDepthLog("Gamma")
    assert gamma is not None
    assert "Gamma" in well.getContinuousLogNames()

    markers = well.getMarkers()
    assert len(markers) == 1
    marker = next(iter(markers))
    assert marker.name == "M1"
    assert marker.depth == 10.0
    assert marker.age == 5.0
    assert marker.stratigraphicType == StratigraphicSurfaceType.BASELAP


def test_loadWell_rejects_bad_format(tmp_path: Path) -> None:
    """Test of loadWell rejecting bad format.

    :param Path tmp_path: temporary path fixture
    """
    payload = {
        "format": "pyWellSFM.NotAWell",
        "version": "1.0",
        "well": {
            "name": "Well-C",
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "depth": 10.0,
        },
    }
    path = _write_json(tmp_path, payload, "well_bad.json")
    with pytest.raises(ValueError):
        _ = loadWell(path)


def test_loadStriplog_from_csv(tmp_path: Path) -> None:
    """Load a striplog from a CSV file."""
    csv_text = """top,base,lithology
0,15,sandstone
15,30,shale
"""
    csv_path = tmp_path / "striplog.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    striplog = wio._load_striplog_from_csv(csv_path)
    assert isinstance(striplog, Striplog)
    assert len(striplog) == 2
    assert striplog[0].top.middle == 0.0  # type: ignore[union-attr]
    assert striplog[0].base.middle == 15.0  # type: ignore[union-attr]
    assert striplog[1].top.middle == 15.0  # type: ignore[union-attr]
    assert striplog[1].base.middle == 30.0  # type: ignore[union-attr]
    assert striplog[0].components[0]["lithology"] == "sandstone"  # type: ignore[union-attr]
    assert striplog[1].components[0]["lithology"] == "shale"  # type: ignore[union-attr]


def test_loadWellFromLasFile_minimal(tmp_path: Path) -> None:
    """Load a minimal LAS and populate the Well + Curve logs."""
    las_text = """~Version Information
VERS.                  2.0:   CWLS LOG ASCII STANDARD -VERSION 2.0
WRAP.                   NO:   ONE LINE PER DEPTH STEP
~Well Information
WELL.              Well-LAS:   WELL NAME
X.                  123.00:   X COORD
Y.                  456.00:   Y COORD
Z.                   10.00:   ELEVATION
STRT.M                 0.0:   START DEPTH
STOP.M               100.0:   STOP DEPTH
STEP.M                50.0:   STEP
NULL.             -999.25:   NULL VALUE
~Curve Information
DEPT.M                :   Depth
GR.API                :   Gamma Ray
RHOB.KG/M3            :   Bulk density
~A
0    50   2.2
50   60   2.1
100  70   2.0
"""

    las_path = tmp_path / "well.las"
    las_path.write_text(las_text, encoding="utf-8")

    well = wio.loadWellFromLasFile(str(las_path))

    assert isinstance(well, Well)
    assert well.name == "Well-LAS"
    assert np.allclose(well.wellHeadCoords, np.array([123.0, 456.0, 10.0]))
    assert well.depth == 100.0

    # path is vertical with 2 points, z increases with depth.
    assert hasattr(well, "_wellPath")
    assert well._wellPath.shape == (2, 3)
    assert np.allclose(well._wellPath[0], np.array([123.0, 456.0, 10.0]))
    assert np.allclose(well._wellPath[1], np.array([123.0, 456.0, 110.0]))

    gr = well.getDepthLog("GR")
    assert gr is not None
    assert isinstance(gr, Curve)
    assert "GR" in well.getContinuousLogNames()


def test_loadWell() -> None:
    """Test of loadWell with test data."""
    test_data_dir = Path(__file__).parent / "data"
    well_path = test_data_dir / "well.json"

    well = loadWell(str(well_path))

    assert isinstance(well, Well)
    assert well.name == "Well-B"
    assert np.allclose(well.wellHeadCoords, np.array([0.0, 0.0, 0.0]))
    assert well.depth == 100.0

    assert len(well.getMarkers()) == 1

    litho_log = well.getDepthLog("lithology")
    assert litho_log is not None
    assert isinstance(litho_log, Striplog)
    assert "lithology" in well.getDiscreteLogNames()

    gr_log = well.getDepthLog("GR")
    assert gr_log is not None
    assert isinstance(gr_log, Curve)
    assert "GR" in well.getContinuousLogNames()

    density_log = well.getDepthLog("Density")
    assert density_log is not None
    assert isinstance(density_log, Curve)
    assert "Density" in well.getContinuousLogNames()


def test_saveWellToJson_minimal(tmp_path: Path) -> None:
    """Save a minimal Well to JSON and validate against schema."""
    well = Well("Well-S", np.array([1.0, 2.0, 3.0]), 50.0)
    # Ensure a schema-compliant wellPath (2+ points).
    well.setWellPath(
        np.asarray(
            [
                well.wellHeadCoords,
                well.wellHeadCoords + np.asarray([0.0, 0.0, well.depth]),
            ],
            dtype=float,
        )
    )

    out_path = tmp_path / "well_saved.json"
    wio.saveWellToJson(well, str(out_path))

    # Schema validation should pass.
    _ = validateWellJsonFile(str(out_path))

    # Round-trip load
    loaded = loadWell(str(out_path))
    assert isinstance(loaded, Well)
    assert loaded.name == "Well-S"
    assert loaded.depth == 50.0
    assert np.allclose(loaded.wellHeadCoords, np.array([1.0, 2.0, 3.0]))


def test_saveWellToJson_with_markers_logs_and_path(tmp_path: Path) -> None:
    """Save a populated Well to JSON and validate + round-trip."""
    well = Well("Well-T", np.array([0.0, 0.0, 0.0]), 100.0)
    well.setWellPath(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
            dtype=float,
        )
    )

    # Marker
    well.addMarkers(
        Marker(
            name="M1",
            depth=10.0,
            age=5.0,
            stratigraphicType=StratigraphicSurfaceType.BASELAP,
        )
    )

    # Striplog
    striplog = Striplog(
        [
            Interval(
                0.0, 15.0, components=[Component({"lithology": "sandstone"})]
            ),
            Interval(
                15.0, 30.0, components=[Component({"lithology": "shale"})]
            ),
        ]
    )
    print("input striplog", striplog)
    well.addLog("lithology", striplog)

    # Continuous log (Curve)
    gr = Curve(
        "Depth",
        "GR",
        np.asarray([0.0, 100.0], dtype=float),
        np.asarray([50.0, 80.0], dtype=float),
        "linear",
    )
    well.addLog("GR", gr)

    out_path = tmp_path / "well_saved_full.json"
    wio.saveWellToJson(well, str(out_path))

    # Schema validation
    data = validateWellJsonFile(str(out_path))
    assert isinstance(data, dict)
    assert data["format"] == "pyWellSFM.WellData"
    assert data["version"] == "1.0"
    assert isinstance(data.get("well", {}), dict)
    assert isinstance(data["well"].get("continuousLogs"), list)
    assert isinstance(data["well"].get("striplogs"), list)

    # Round-trip load
    loaded = loadWell(str(out_path))
    assert loaded.name == "Well-T"
    assert loaded.getDepthLog("lithology") is not None
    assert "lithology" in loaded.getDiscreteLogNames()
    assert loaded.getDepthLog("GR") is not None
    assert "GR" in loaded.getContinuousLogNames()

    markers = loaded.getMarkers()
    assert len(markers) == 1
    m = next(iter(markers))
    assert m.name == "M1"
    assert m.depth == 10.0
    assert m.age == 5.0
    assert m.stratigraphicType == StratigraphicSurfaceType.BASELAP


def _base_well_payload() -> dict[str, Any]:
    """Helper to create a well JSON payload with required properties."""
    return {
        "format": "pyWellSFM.WellData",
        "version": "1.0",
        "well": {
            "name": "W-1",
            "location": {"x": 1.0, "y": 2.0, "z": 3.0},
            "depth": 100.0,
        },
    }


def _curve_obj(
    *,
    x_axis: str = "Depth",
    y_axis: str = "GR",
    values: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Helper to create a Curve JSON object.

    :param str x_axis: name of the x axis (e.g. "Depth" or "Age")
    :param str y_axis: name of the y axis (e.g. "GR" or "POR")
    :param list[tuple[float, float]] | None values: list of (x, y) values for
    the curve. If None, defaults to [(0.0, 10.0), (100.0, 20.0)].
    """
    if values is None:
        values = [(0.0, 10.0), (100.0, 20.0)]
    return {
        "format": "pyWellSFM.CurveData",
        "version": "1.0",
        "curve": {
            "xAxisName": x_axis,
            "yAxisName": y_axis,
            "interpolationMethod": "linear",
            "data": [{"x": x, "y": y} for x, y in values],
        },
    }


def test_wellToJsonObj_rejects_invalid_head_coords() -> None:
    """Test of wellToJsonObj.

    Test that if we try to save a Well to JSON that has wellHeadCoords that are
    not a 3-element numeric array, we raise a ValueError indicating that the
    wellHeadCoords must be a 3-element numeric array.
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 10.0)
    well.wellHeadCoords = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="3-element"):
        wio.wellToJsonObj(well)


def test_wellToJsonObj_striplog_empty_raises() -> None:
    """Test of wellToJsonObj.

    Test that if we try to save a Well to JSON that has a striplog with no
    intervals, we raise a ValueError indicating that the striplog must contain
    at least 1 interval. This is to prevent accidentally saving a Well with an
    empty striplog, which would result in a JSON file that is technically valid
    but semantically invalid because a striplog with no intervals doesn't make
    sense and would likely cause confusion or errors downstream when we try to
    load the Well and access the striplog and find that it has no intervals.
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 10.0)
    striplog = wio._load_striplog_from_json_obj(
        {
            "name": "lithology",
            "intervals": [
                {"top": 0.0, "base": 1.0, "lithology": "sandstone"},
            ],
        }
    )[1]
    striplog._Striplog__list = []  # type: ignore[attr-defined]
    well._logs["lithology"] = striplog
    with pytest.raises(ValueError, match="must contain at least 1 interval"):
        wio.wellToJsonObj(well)


def test_parse_stratigraphic_type_unknown_variants() -> None:
    """Test of _parse_stratigraphic_type.

    Test that if we provide an unrecognized stratigraphicType value for a
    marker, we set the marker stratigraphicType to
    StratigraphicSurfaceType.UNKNOWN, and that we can still load the Well
    object without error, and access the marker with the expected UNKNOWN
    type. This is to ensure that if users provide a stratigraphicType value
    that is not recognized (e.g. due to a typo, or using a different
    terminology), we handle it gracefully by setting it to UNKNOWN and allowing
    the Well to load, rather than raising an error that would prevent loading
    the Well at all. By handling unrecognized stratigraphicType values in this
    way, we can make the loading process more robust and user-friendly, while
    still providing a clear indication (via the UNKNOWN type) that the provided
    stratigraphicType was not recognized.
    """
    assert (
        wio._parse_stratigraphic_type(123) == StratigraphicSurfaceType.UNKNOWN
    )
    assert (
        wio._parse_stratigraphic_type(" ") == StratigraphicSurfaceType.UNKNOWN
    )
    assert (
        wio._parse_stratigraphic_type("something-else")
        == StratigraphicSurfaceType.UNKNOWN
    )


def test_load_striplog_from_json_obj_happy_path() -> None:
    """Test of _load_striplog_from_json_obj.

    Test that if we provide a valid JSON object representing a striplog, we can
    load it successfully, and that the loaded striplog has the expected name
    and intervals.
    """
    name, striplog = wio._load_striplog_from_json_obj(
        {
            "name": "facies",
            "intervals": [
                {"top": 0.0, "base": 10.0, "lithology": "sandstone"},
            ],
        }
    )
    assert name == "facies"
    assert len(striplog) == 1


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"name": "", "intervals": []}, "non-empty string"),
        ({"name": "facies", "intervals": "bad"}, "must be an array"),
        ({"name": "facies", "intervals": []}, "at least 1 item"),
        ({"name": "facies", "intervals": [1]}, "must be an object"),
        (
            {
                "name": "facies",
                "intervals": [{"top": "x", "base": 1.0, "lithology": "s"}],
            },
            "must be numbers",
        ),
        (
            {
                "name": "facies",
                "intervals": [{"top": 0.0, "base": 1.0, "lithology": ""}],
            },
            "non-empty string",
        ),
        (
            {
                "name": "facies",
                "intervals": [{"top": 2.0, "base": 1.0, "lithology": "s"}],
            },
            "base must be >= top",
        ),
    ],
)
def test_load_striplog_from_json_obj_rejects_invalid(
    payload: dict[str, Any], match: str
) -> None:
    """Test of _load_striplog_from_json_obj.

    Test that if we try to load a striplog from a JSON object that is invalid,
    we raise a ValueError.

    :param dict[str, Any] payload: JSON object representing the striplog
    :param str match: expected error message
    """
    with pytest.raises(ValueError, match=match):
        wio._load_striplog_from_json_obj(payload)


def test_load_striplog_from_csv_rejects_missing_file(tmp_path: Path) -> None:
    """Test of _load_striplog_from_csv.

    Test that if we try to load a striplog from a CSV file that does not exist,
    we raise a FileNotFoundError.

    :param Path tmp_path: temporary path fixture
    """
    with pytest.raises(FileNotFoundError):
        wio._load_striplog_from_csv(tmp_path / "missing.csv")


def test_load_striplog_from_csv_rejects_missing_required_columns(
    tmp_path: Path,
) -> None:
    """Test of _load_striplog_from_csv.

    Test that if we try to load a striplog from a CSV file that is missing
    required columns (e.g. lithology), we raise a ValueError indicating that
    the CSV must contain 'top', 'base', and 'lithology' columns. This is to
    prevent accidentally loading a CSV file that has the wrong format (e.g.
    missing the lithology column), which would result in silently loading
    incorrect data without a lithology log, which would be confusing and lead
    to errors downstream when we try to access the lithology log and it's not
    there. By checking for the required columns upfront and raising a clear
    error, we can help users quickly identify that they are trying to load a
    CSV file with the wrong format, and avoid confusion from a more obscure
    error that would occur if we tried to load the CSV without the required
    columns.

    :param Path tmp_path: temporary path fixture
    """
    p = tmp_path / "bad_striplog.csv"
    p.write_text("top,base\n0,10\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="must contain 'top', 'base', and 'lithology'"
    ):
        wio._load_striplog_from_csv(p)


def test_loadWell_rejects_non_dict_json_payload(tmp_path: Path) -> None:
    """Test of loadWell.

    Test that if we try to load a well from a JSON file that contains a
    non-dict JSON value (e.g. an array), we raise a ValueError indicating that
    the top-level JSON must be an object. This is to prevent confusion from a
    JSONDecodeError or other parsing error that would occur if we tried to
    parse a non-object JSON value as a well, which would be less clear than a
    ValueError indicating that the top-level JSON must be an object.

    :param Path tmp_path: temporary path fixture
    """
    p = tmp_path / "bad.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        loadWell(str(p))


def test_loadWell_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Test of loadWell.

    Test that if we try to load a well from a file with an unsupported
    extension (i.e. not .json or .las), we raise a ValueError. This is to
    prevent accidentally calling loadWell with a .csv or .txt file, which would
    result in a JSONDecodeError or some other parsing error that would be less
    clear than a ValueError indicating that the extension is not supported.
    By checking the extension upfront and raising a clear error, we can help
    users quickly identify that they are trying to load a file with an
    unsupported format, and avoid confusion from a more obscure parsing error
    that would occur if we tried to parse the unsupported file format as JSON
    or LAS. This also allows us to provide a clear error message that indicates
    which extensions are supported, which can help guide users to provide the
    correct file format.

    :param Path tmp_path: temporary path fixture
    """
    p = tmp_path / "well.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=".json or .las"):
        loadWell(str(p))


def test_loadWellFromJsonObj_and_unknown_marker_type() -> None:
    """Test of loadWellFromJsonObj.

    Test that if we provide a marker with an unrecognized stratigraphicType,
    we set the marker stratigraphicType to StratigraphicSurfaceType.UNKNOWN,
    and that we can still load the Well object without error, and access the
    marker with the expected UNKNOWN type.
    """
    payload = _base_well_payload()
    payload["well"]["markers"] = [
        {
            "name": "M1",
            "depth": 10.0,
            "age": 5.0,
            "stratigraphicType": "not-a-known-type",
        }
    ]

    well = wio.loadWellFromJsonObj(payload)
    assert well._wellPath.shape == (2, 3)
    assert well._wellPath[1, 2] == pytest.approx(-97.0)
    marker = next(iter(well.getMarkers()))
    assert marker.stratigraphicType == StratigraphicSurfaceType.UNKNOWN


@pytest.mark.parametrize(
    "bad_wellpath, match",
    [
        ([1, 2], "must be an object"),
        (
            [{"x": "a", "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 1}],
            "must be numeric",
        ),
        ([{"x": 0, "y": 0, "z": 0}], "array of 2+"),
    ],
)
def test_loadWellFromJsonObj_rejects_bad_wellPath(
    bad_wellpath: list[Any], match: str
) -> None:
    """Test of loadWellFromJsonObj.

    Test that if we provide a wellPath that is not an array of objects with
    numeric x,y,z, we raise a ValueError.

    :param list[Any] bad_wellpath: invalid wellPath to test
    :param str match: regex pattern to match in the ValueError message
    """
    payload = _base_well_payload()
    payload["well"]["wellPath"] = bad_wellpath
    with pytest.raises(ValueError, match=match):
        wio.loadWellFromJsonObj(payload)


def test_loadWellFromJsonObj_striplogs_string_path(tmp_path: Path) -> None:
    """Test of loadWellFromJsonObj.

    Test that if we provide a string path for a striplog, we can load the
    striplog from the CSV file at that path, and that the loaded striplog is
    added to the Well object.

    :param Path tmp_path: temporary path fixture
    """
    striplog_csv = tmp_path / "my_striplog.csv"
    striplog_csv.write_text(
        "top,base,lithology\n0,10,sandstone\n10,20,shale\n",
        encoding="utf-8",
    )

    payload = _base_well_payload()
    payload["well"]["striplogs"] = ["my_striplog.csv"]

    well = wio.loadWellFromJsonObj(payload, base_dir=tmp_path)
    assert well.getDepthLog("my_striplog") is not None


def test_loadWellFromJsonObj_striplogs_invalid_item_type() -> None:
    """Test of loadWellFromJsonObj.

    Test that if we try to load striplogs from a list that contains a
    non-object, non-string item, we raise a ValueError.
    """
    payload = _base_well_payload()
    payload["well"]["striplogs"] = [123]
    with pytest.raises(ValueError, match="must be an object or a string path"):
        wio.loadWellFromJsonObj(payload)


def test_loadWellFromJsonObj_continuous_logs_list_rejects_non_object() -> None:
    """Test of loadWellFromJsonObj.

    Test that if we try to load a continuous log from a list that contains a
    non-object item, we raise a ValueError indicating that it must be a Curve
    JSON object.
    """
    payload = _base_well_payload()
    payload["well"]["continuousLogs"] = [123]
    with pytest.raises(ValueError, match="must be a Curve JSON object"):
        wio.loadWellFromJsonObj(payload)


@pytest.mark.parametrize(
    "x_axis, expect_depth, expect_age",
    [
        ("Depth", True, False),
        ("Age", False, True),
        ("Distance", True, False),
    ],
)
def test_loadWellFromJsonObj_continuous_logs_external_file_branches(
    tmp_path: Path,
    x_axis: str,
    expect_depth: bool,
    expect_age: bool,
) -> None:
    """Test of loadWellFromJsonObj.

    Test that if we try to load a continuous log from an external file with
    an invalid reference, we raise a ValueError.

    :param Path tmp_path: temporary path fixture
    :param str x_axis: xAxisName to use in the curve JSON, to test the
        branching logic for determining whether it's a depth log or an age log
    :param bool expect_depth: whether we expect the loaded log to be recognized
        as a depth log (i.e. have "Depth" as xAxisName).
    :param bool expect_age: whether we expect the loaded log to be recognized
        as an age log (i.e. have "Age" as xAxisName).
    """
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(
        json.dumps(_curve_obj(x_axis=x_axis, y_axis="POR")), encoding="utf-8"
    )

    payload = _base_well_payload()
    payload["well"]["continuousLogs"] = {"url": "curve.json"}

    well = wio.loadWellFromJsonObj(payload, base_dir=tmp_path)

    assert (well.getDepthLog("POR") is not None) is expect_depth
    assert (well.getAgeLog("POR") is not None) is expect_age


def test_loadWellFromJsonObj_continuous_log_ext_file_rejects_bad_ref() -> None:
    """Test of loadWellFromJsonObj.

    Test that if we try to load a continuous log from an external file with
    an invalid reference, we raise a ValueError.
    """
    payload = _base_well_payload()
    payload["well"]["continuousLogs"] = {"foo": "bar"}
    with pytest.raises(ValueError, match="only a 'url' property"):
        wio.loadWellFromJsonObj(payload)


def test_loadWellFromLasFile_rejects_missing_file(tmp_path: Path) -> None:
    """Test of loadWellFromLasFile.

    Test that if we try to load a LAS file that does not exist, we raise a
    FileNotFoundError.

    :param Path tmp_path: temporary path fixture
    """
    with pytest.raises(FileNotFoundError):
        wio.loadWellFromLasFile(str(tmp_path / "missing.las"))


def test_loadWellFromLasFile_rejects_no_valid_index(tmp_path: Path) -> None:
    """Test of loadWellFromLasFile.

    Test that if we try to load a LAS file that has no valid depth/index
    column, we raise a ValueError. This is to prevent accidentally loading a
    LAS file that has a non-standard format where the first column is not
    depth, which would result in silently loading incorrect data without a
    depth log.

    :param Path tmp_path: temporary path fixture
    """
    las_text = """~Version Information
VERS.                  2.0:   CWLS LOG ASCII STANDARD -VERSION 2.0
WRAP.                   NO:   ONE LINE PER DEPTH STEP
~Well Information
WELL.              Well-LAS:   WELL NAME
~Curve Information
DEPT.M                :   Depth
GR.API                :   Gamma Ray
~A
NaN  50
NaN  60
"""
    p = tmp_path / "invalid_index.las"
    p.write_text(las_text, encoding="utf-8")

    with pytest.raises(ValueError, match="no valid depth/index"):
        wio.loadWellFromLasFile(str(p))


def test_saveWell_and_saveWellToJson_reject_non_json_extension(
    tmp_path: Path,
) -> None:
    """Test of saveWell.

    Test that if we call saveWell or saveWellToJson with a non-.json extension,
    we raise a ValueError. This is to prevent accidentally calling saveWell
    (which dispatches to saveWellToJson for .json) with a .las extension,
    which would result in silently writing a JSON file with a .las extension,
    which would be confusing and not loadable as a LAS file.

    :param Path tmp_path: temporary path fixture
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 10.0)
    with pytest.raises(ValueError, match=".json extension"):
        wio.saveWellToJson(well, str(tmp_path / "well.txt"))
    with pytest.raises(ValueError, match=".json extension"):
        wio.saveWell(well, str(tmp_path / "well.txt"))


def test_saveWell_json_dispatch_and_roundtrip(tmp_path: Path) -> None:
    """Test of saveWell.

    Test that if we call saveWell with a .json extension, it dispatches to
    saveWellToJson, and that we can round-trip the saved JSON back to a Well
    object with loadWell.

    :param Path tmp_path: temporary path fixture
    """
    well = Well("W", np.array([1.0, 2.0, 3.0]), 20.0)
    well.setWellPath(
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 23.0]], dtype=float)
    )
    well.addLog(
        "GR",
        Curve(
            "Depth",
            "GR",
            np.array([0.0, 20.0], dtype=float),
            np.array([10.0, 20.0], dtype=float),
            "linear",
        ),
    )

    out = tmp_path / "well.json"
    wio.saveWell(well, str(out))

    loaded = loadWell(str(out))
    assert loaded.name == "W"
    assert loaded.getDepthLog("GR") is not None


def test_loadWell_rejects_missing_file(tmp_path: Path) -> None:
    """Test of loadWell.

    Test that if we try to load a well from a JSON file that does not exist,
    we raise a FileNotFoundError.

    :param Path tmp_path: temporary path fixture
    """
    with pytest.raises(FileNotFoundError):
        loadWell(str(tmp_path / "missing.json"))


@pytest.mark.parametrize(
    "well_patch, match",
    [
        ({"well": "bad"}, "Well.well must be an object"),
        (
            {
                "well": {
                    "name": "",
                    "location": {"x": 1, "y": 2, "z": 3},
                    "depth": 1,
                }
            },
            "non-empty string",
        ),
        (
            {"well": {"name": "W", "location": "bad", "depth": 1}},
            "location must be an object",
        ),
        (
            {
                "well": {
                    "name": "W",
                    "location": {"x": "x", "y": 2, "z": 3},
                    "depth": 1,
                }
            },
            "must be numbers",
        ),
        (
            {
                "well": {
                    "name": "W",
                    "location": {"x": 1, "y": 2, "z": 3},
                    "depth": "d",
                }
            },
            "must be a number",
        ),
        (
            {
                "well": {
                    "name": "W",
                    "location": {"x": 1, "y": 2, "z": 3},
                    "depth": 1,
                    "markers": [1],
                }
            },
            "markers\\[0\\] must be an object",
        ),
        (
            {
                "well": {
                    "name": "W",
                    "location": {"x": 1, "y": 2, "z": 3},
                    "depth": 1,
                    "markers": [{"name": "", "depth": 1, "age": 2}],
                }
            },
            "name must be a string",
        ),
        (
            {
                "well": {
                    "name": "W",
                    "location": {"x": 1, "y": 2, "z": 3},
                    "depth": 1,
                    "markers": [{"name": "M", "depth": "d", "age": 2}],
                }
            },
            "depth must be a number",
        ),
        (
            {
                "well": {
                    "name": "W",
                    "location": {"x": 1, "y": 2, "z": 3},
                    "depth": 1,
                    "markers": [{"name": "M", "depth": 1, "age": "a"}],
                }
            },
            "age must be a number",
        ),
    ],
)
def test_loadWellFromJsonObj_rejects_invalid_shapes(
    well_patch: dict[str, Any],
    match: str,
) -> None:
    """Test of loadWellFromJsonObj.

    Test that if the input JSON has invalid shapes/types for the well,
    location, depth, or markers, we raise a ValueError with an appropriate
    message.

    :param dict[str, Any] well_patch: partial JSON payload to merge with a
        valid base payload, to create the test case with the specific invalid
        shape/type
    :param str match: regex pattern to match in the ValueError message
    """
    payload = {"format": "pyWellSFM.WellData", "version": "1.0", **well_patch}
    with pytest.raises(ValueError, match=match):
        wio.loadWellFromJsonObj(payload)


def test_wellToJsonObj_striplog_non_numeric_depth_raises() -> None:
    """Test of wellToJsonObj.

    Test that if the striplog intervals have non-numeric top/base values that
    cannot be approximated as numeric, we raise a ValueError when trying to
    convert to JSON.
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)
    striplog = wio._load_striplog_from_json_obj(
        {
            "name": "litho",
            "intervals": [
                {"top": 0.0, "base": 10.0, "lithology": "sandstone"}
            ],
        }
    )[1]

    class _BadPos:
        def __init__(self, z: object) -> None:
            self.z = z

    class _BadInterval:
        def __init__(self) -> None:
            self.top = _BadPos("bad-top")
            self.base = _BadPos(10.0)
            self.components = [{"lithology": "sandstone"}]

    striplog._Striplog__list = [_BadInterval()]  # type: ignore[assignment]
    well._logs["litho"] = striplog
    with pytest.raises(ValueError, match="is not a number"):
        wio.wellToJsonObj(well)


def test_wellToJsonObj_lithology_fallback_unknown() -> None:
    """Test of wellToJsonObj.

    Test that if the striplog intervals have components without a "lithology"
    key, or with a non-string value, we set the lithology to "Unknown"
    in the output JSON.
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)
    striplog = wio._load_striplog_from_json_obj(
        {
            "name": "litho",
            "intervals": [
                {"top": 0.0, "base": 10.0, "lithology": "sandstone"}
            ],
        }
    )[1]
    striplog[0].components = [{}]  # type: ignore[index]
    well._logs["litho"] = striplog
    obj = wio.wellToJsonObj(well)
    assert (
        obj["well"]["striplogs"][0]["intervals"][0]["lithology"] == "Unknown"
    )


def test_wellToJsonObj_lithology_from_capitalized_key() -> None:
    """Test of wellToJsonObj.

    Test that if the striplog intervals have components with a "Lithology" key
    (capitalized), we use that value in the output JSON.
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)
    striplog = wio._load_striplog_from_json_obj(
        {
            "name": "litho",
            "intervals": [
                {"top": 0.0, "base": 10.0, "lithology": "sandstone"}
            ],
        }
    )[1]
    striplog[0].components = [{"Lithology": "Dolomite"}]  # type: ignore[index]
    well._logs["litho"] = striplog
    obj = wio.wellToJsonObj(well)
    assert (
        obj["well"]["striplogs"][0]["intervals"][0]["lithology"] == "Dolomite"
    )


def test_wellToJsonObj_lithology_from_component_plain_depth_values() -> None:
    """Test of wellToJsonObj.

    Test that if the striplog intervals have components with a "lithology" key,
    we use that value in the output JSON, even if the top/base values are not
    numeric (but can be approximated as numeric).
    """
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)
    striplog = wio._load_striplog_from_json_obj(
        {
            "name": "litho",
            "intervals": [
                {"top": 0.0, "base": 10.0, "lithology": "sandstone"}
            ],
        }
    )[1]

    class _Comp:
        def get(self, key: str) -> str | None:
            if key == "lithology":
                return "Marl"
            return None

    class _Interval:
        def __init__(self) -> None:
            self.top = 1.0
            self.base = 2.0
            self.components = [_Comp()]

    striplog._Striplog__list = [_Interval()]  # type: ignore[assignment]
    well._logs["litho"] = striplog

    obj = wio.wellToJsonObj(well)
    out_itv = obj["well"]["striplogs"][0]["intervals"][0]
    assert out_itv["top"] == pytest.approx(1.0)
    assert out_itv["base"] == pytest.approx(2.0)
    assert out_itv["lithology"] == "Marl"


def test_loadWellFromJsonObj_continuous_logs_list_log_name_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test of loadCurveFromJsonObj.

    Test that if the continuousLogs list has a log with an empty yAxisName,
    we fall back to a default name and still load the log.

    :param pytest.MonkeyPatch monkeypatch: monkeypatch fixture for patching
        lasio.read
    """
    payload = _base_well_payload()
    payload["well"]["continuousLogs"] = [{"curve": {"yAxisName": ""}}]

    fallback_curve = Curve(
        "Depth",
        "FALLBACK",
        np.asarray([0.0, 100.0], dtype=float),
        np.asarray([1.0, 2.0], dtype=float),
        "linear",
    )

    monkeypatch.setattr(wio, "loadCurveFromJsonObj", lambda _: fallback_curve)
    well = wio.loadWellFromJsonObj(payload)
    assert well.getDepthLog("FALLBACK") is not None


def test_loadWellFromLasFile_rejects_too_few_depth_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of loadWellFromLasFile.

    Test that if the LAS file has fewer than 2 depth/index samples,
    we raise a ValueError.

    :param Path tmp_path: temporary path fixture
    :param pytest.MonkeyPatch monkeypatch: monkeypatch fixture for patching
        lasio.read
    """

    class _CurveInfo:
        def __init__(self, mnemonic: str | None) -> None:
            self.mnemonic = mnemonic

    class _Las:
        def __init__(self) -> None:
            self.well: dict[str, object] = {}
            self.params: dict[str, object] = {}
            self.index = np.asarray([0.0], dtype=float)
            self.curves = [_CurveInfo("GR")]
            self.null = None

        def __getitem__(self, _: str) -> np.ndarray:
            return np.asarray([10.0], dtype=float)

    p = tmp_path / "one_sample.las"
    p.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(wio.lasio, "read", lambda *_args, **_kwargs: _Las())

    with pytest.raises(ValueError, match="at least 2 depth/index"):
        wio.loadWellFromLasFile(str(p))


def test_loadWellFromLasFile_handles_curve_filtering_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test of loadWellFromLasFile.

    Test that if the LAS has curves that raise KeyError or have invalid data,
    we skip them but still load the well and any valid curves.

    :param Path tmp_path: temporary path fixture
    :param pytest.MonkeyPatch monkeypatch: monkeypatch fixture for patching
        lasio.read
    """

    class _CurveInfo:
        def __init__(self, mnemonic: str | None) -> None:
            self.mnemonic = mnemonic

    class _Las:
        def __init__(self) -> None:
            self.well: dict[str, object] = {}
            self.params: dict[str, object] = {}
            self.index = np.asarray([0.0, 1.0], dtype=float)
            self.curves = [
                _CurveInfo(None),
                _CurveInfo("DEPT"),
                _CurveInfo("BADGET"),
                _CurveInfo("SIZE"),
                _CurveInfo("NULLS"),
                _CurveInfo("OK"),
            ]
            self.null = -999.25

        def __getitem__(self, key: str) -> np.ndarray:
            if key == "BADGET":
                raise KeyError(key)
            if key == "SIZE":
                return np.asarray([1.0], dtype=float)
            if key == "NULLS":
                return np.asarray([-999.25, np.nan], dtype=float)
            if key == "OK":
                return np.asarray([10.0, 20.0], dtype=float)
            raise KeyError(key)

    p = tmp_path / "branchy.las"
    p.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(wio.lasio, "read", lambda *_args, **_kwargs: _Las())

    well = wio.loadWellFromLasFile(str(p))
    assert well.getDepthLog("OK") is not None


def test_loadWellFromLasFile_drops_duplicate_x_then_skips_curve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test of loadWellFromLasFile.

    Test that if the LAS has duplicate x/index values, we drop them and then
    skip curves that don't have valid data.

    :param Path tmp_path: temporary path fixture
    :param pytest.MonkeyPatch monkeypatch: monkeypatch fixture for patching
        lasio.read
    """

    class _CurveInfo:
        def __init__(self, mnemonic: str) -> None:
            self.mnemonic = mnemonic

    class _Las:
        def __init__(self) -> None:
            self.well: dict[str, object] = {}
            self.params: dict[str, object] = {}
            self.index = np.asarray([0.0, 0.0], dtype=float)
            self.curves = [_CurveInfo("GR")]
            self.null = None

        def __getitem__(self, _: str) -> np.ndarray:
            return np.asarray([10.0, 20.0], dtype=float)

    p = tmp_path / "dup_x.las"
    p.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(wio.lasio, "read", lambda *_args, **_kwargs: _Las())

    well = wio.loadWellFromLasFile(str(p))
    assert well.getDepthLog("GR") is None
