# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
"""Unit tests for Well and Well IO."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from striplog import Component, Interval, Striplog

from pywellsfm.io.json_schema_validation import validateWellJsonFile
from pywellsfm.io.well_io import (
    loadWellFromJsonObj,
    loadWellFromLasFile,
    saveWellToJson,
)
from pywellsfm.model import Curve

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.io import loadWell  # noqa: E402
from pywellsfm.model.Marker import (  # noqa: E402
    Marker,
    StratigraphicSurfaceType,
)
from pywellsfm.model.Well import Well  # noqa: E402


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

    well = loadWellFromJsonObj(payload)

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

    well = loadWellFromJsonObj(payload)

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

    well = loadWellFromLasFile(str(las_path))

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
    saveWellToJson(well, str(out_path))

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
    saveWellToJson(well, str(out_path))

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
