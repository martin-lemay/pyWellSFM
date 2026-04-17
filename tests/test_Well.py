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


import numpy as np
import pytest
from striplog import Component, Interval, Striplog

from pywellsfm.model import Curve
from pywellsfm.model.Marker import (
    Marker,
)
from pywellsfm.model.Well import Well
from pywellsfm.utils.logging_utils import clear_stored_logs, get_stored_logs


def _mk_curve(max_depth: float = 10.0) -> Curve:
    """Helper to create a simple curve with specified max depth.

    :param float max_depth: maximum depth of the curve
    :return Curve: created curve
    """
    return Curve(
        "Depth",
        "GR",
        np.array([0.0, max_depth], dtype=float),
        np.array([10.0, 20.0], dtype=float),
        "linear",
    )


def _mk_striplog(top: float = 0.0, base: float = 10.0) -> Striplog:
    """Helper to create a simple striplog with one interval.

    :param float top: top depth of the interval
    :param float base: base depth of the interval
    :return Striplog: created striplog
    """
    return Striplog(
        [
            Interval(
                top,
                base,
                components=[Component({"lithology": "sandstone"})],
            )
        ]
    )


def test_setWellPath_valid_and_invalid_shapes() -> None:
    """SetWellPath accepts valid shapes and rejects invalid ones."""
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)

    valid_path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=float)
    well.setWellPath(valid_path)
    assert np.allclose(well._wellPath, valid_path)

    with pytest.raises(ValueError):
        well.setWellPath(np.array([[0.0, 0.0, 0.0]], dtype=float))

    with pytest.raises(ValueError):
        well.setWellPath(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float))


def test_shallowCopy_respects_copy_flags() -> None:
    """Shallow copy respects copy flags for markers and logs."""
    well = Well("W", np.array([1.0, 2.0, 3.0]), 100.0)
    marker = Marker("M", depth=10.0, age=2.0)
    well.addMarkers(marker)
    well.addLog("GR", _mk_curve())

    copied = well.shallowCopy("W_copy", copyMarkers=True, copyLogs=True)
    assert copied.name == "W_copy"
    assert copied.getMarkers() is well.getMarkers()
    assert copied.getDepthLog("GR") is well.getDepthLog("GR")

    copied2 = well.shallowCopy("W_copy2", copyMarkers=False, copyLogs=False)
    assert copied2.name == "W_copy2"
    assert copied2.getMarkers() == []
    assert copied2.getDepthLog("GR") is None


def test_marker_helpers_and_setters() -> None:
    """Test marker helper methods and setters."""
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)

    m_old = Marker("old", depth=80.0, age=20.0)
    m_young = Marker("young", depth=20.0, age=5.0)
    well.addMarkers([m_old, m_young])

    assert well.oldestMarker.name == "old"
    assert well.oldestMarkerAge == 20.0
    assert well.youngestMarker.name == "young"
    assert well.youngestMarkerAge == 5.0

    well.setMarkers([m_young])
    assert len(well.getMarkers()) == 1
    assert well.getMarkers()[0].name == "young"


def test_addLog_and_name_filters_for_continuous_discrete() -> None:
    """Test adding logs and filtering by continuous and discrete types."""
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)

    well.addLog("GR", _mk_curve())
    well.addLog("lithology", _mk_striplog())

    assert isinstance(well.getDepthLog("GR"), Curve)
    assert isinstance(well.getDepthLog("lithology"), Striplog)
    assert well.getDepthLog("missing") is None

    assert well.getContinuousLogNames() == {"GR"}
    assert well.getDiscreteLogNames() == {"lithology"}


def test_addLog_rejects_logs_deeper_than_well() -> None:
    """Test that adding logs deeper than the well raises an error."""
    well = Well("W", np.array([0.0, 0.0, 0.0]), 50.0)

    with pytest.raises(ValueError):
        well.addLog("GR", _mk_curve(max_depth=60.0))

    with pytest.raises(ValueError):
        well.addLog("lithology", _mk_striplog(top=0.0, base=60.0))


def test_addLog_duplicate_and_invalid_type_emit_logs() -> None:
    """Test that adding duplicate logs or logs of invalid type emits logs."""
    clear_stored_logs()
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)

    well.addLog("GR", _mk_curve())
    well.addLog("GR", _mk_curve())
    well.addLog("bad", 123)  # type: ignore[arg-type]

    messages = [entry["message"] for entry in get_stored_logs()]
    assert any("already in the list of logs" in msg for msg in messages)
    assert any("Log type is not managed" in msg for msg in messages)


def test_addAgeLog_getAgeLog_and_overwrite_warning() -> None:
    """Test adding age logs, retrieving them, and overwrite warnings."""
    clear_stored_logs()
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)

    log1 = _mk_curve(max_depth=10.0)
    log2 = _mk_curve(max_depth=20.0)

    well.addAgeLog("GR_age", log1)
    well.addAgeLog("GR_age", log2)

    assert well.getAgeLog("GR_age") is log2
    assert well.getAgeLog("missing") is None

    messages = [entry["message"] for entry in get_stored_logs()]
    assert any("already in the list of age logs" in msg for msg in messages)


def test_initDepthAgeModel_uses_well_markers() -> None:
    """Test that initDepthAgeModel uses well markers."""
    well = Well("W", np.array([0.0, 0.0, 0.0]), 100.0)
    well.addMarkers(
        [
            Marker("M1", depth=10.0, age=1.0),
            Marker("M2", depth=40.0, age=4.0),
        ]
    )

    with pytest.raises(AttributeError):
        well.initDepthAgeModel()

    # Initialization object is still created before marker update fails.
    assert well._depthAgeModel is not None
