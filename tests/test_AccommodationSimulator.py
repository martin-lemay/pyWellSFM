# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

from pywellsfm.model import Curve, SubsidenceType
from pywellsfm.simulator.AccommodationSimulator import (
    AccommodationSimulator,
    AccommodationStorage,
)


def _make_curve(values: list[float]) -> Curve:
    """Build a two-point curve for deterministic tests."""
    return Curve(
        "Time",
        "Value",
        np.array([0.0, 10.0], dtype=np.float64),
        np.array(values, dtype=np.float64),
    )


def test_storage_dataclass_fields() -> None:
    """Store and expose accommodation values."""
    out = AccommodationStorage(
        basement=-10.0, seaLevel=2.5, accommodation=12.5
    )
    assert out.basement == -10.0
    assert out.seaLevel == 2.5
    assert out.accommodation == 12.5


def test_init_defaults() -> None:
    """Initialize with expected default state."""
    sim = AccommodationSimulator()
    assert sim.subsidenceCurve is None
    assert sim.eustaticCurve is None
    assert sim.subsidenceType == SubsidenceType.CUMULATIVE
    assert sim.eustatismStart == 0.0
    assert sim.topographyStart == 0.0


def test_set_subsidence_curve_updates_type() -> None:
    """Set curve updates both curve and type."""
    sim = AccommodationSimulator()
    curve = _make_curve([0.0, 40.0])
    sim.setSubsidenceCurve(curve, SubsidenceType.RATE)
    assert sim.subsidenceCurve is curve
    assert sim.getSubsidenceType() == SubsidenceType.RATE


def test_set_subsidence_none_keeps_existing_type() -> None:
    """Setting no curve does not overwrite type."""
    sim = AccommodationSimulator()
    sim.setSubsidenceCurve(_make_curve([0.0, 10.0]), SubsidenceType.RATE)
    sim.setSubsidenceCurve(None, SubsidenceType.CUMULATIVE)
    assert sim.subsidenceCurve is None
    assert sim.getSubsidenceType() == SubsidenceType.RATE


def test_set_eustatic_curve_and_initial_bathymetry() -> None:
    """Set eustasy curve and bathymetry sign convention."""
    sim = AccommodationSimulator()
    curve = _make_curve([100.0, 120.0])
    sim.setEustaticCurve(curve)
    sim.setInitialBathymetry(30.0)
    assert sim.eustaticCurve is curve
    assert sim.topographyStart == -30.0


def test_initial_eustacy_sets_reference_value() -> None:
    """Initialize eustatism reference at input age."""
    sim = AccommodationSimulator()
    sim.setEustaticCurve(_make_curve([100.0, 130.0]))
    sim.initialEustacy(11.0)
    assert sim.eustatismStart == 130.0


def test_prepare_builds_default_curves_when_missing() -> None:
    """Prepare creates flat defaults when curves are missing."""
    sim = AccommodationSimulator()
    sim.prepare()
    assert sim.subsidenceCurve is not None
    assert sim.eustaticCurve is not None
    assert sim.getSubsidenceAt(0.0) == 0.0
    assert sim.getEustatismAt(10.0) == 0.0
    assert sim.eustatismStart == 0.0


def test_prepare_keeps_existing_curves_and_updates_reference() -> None:
    """Prepare keeps custom curves and resets start eustasy."""
    sim = AccommodationSimulator()
    sub = _make_curve([2.0, 5.0])
    eus = _make_curve([7.0, 11.0])
    sim.setSubsidenceCurve(sub, SubsidenceType.CUMULATIVE)
    sim.setEustaticCurve(eus)
    sim.eustatismStart = -1.0

    sim.prepare()

    assert sim.subsidenceCurve is sub
    assert sim.eustaticCurve is eus
    assert sim.eustatismStart == 7.0


def test_get_eustatism_raises_without_curve() -> None:
    """Reject eustatism query when eustatic curve is absent."""
    sim = AccommodationSimulator()
    with pytest.raises(ValueError, match="Eustatic curve is not set"):
        sim.getEustatismAt(1.0)


def test_get_subsidence_raises_without_curve() -> None:
    """Reject subsidence query when subsidence curve is absent."""
    sim = AccommodationSimulator()
    with pytest.raises(ValueError, match="Subsidence curve is not set"):
        sim.getSubsidenceAt(1.0)


def test_get_sea_level_raises_without_curve() -> None:
    """Reject sea-level query when eustatic curve is absent."""
    sim = AccommodationSimulator()
    with pytest.raises(ValueError, match="Eustatic curve is not set"):
        sim.getSeaLevelAt(1.0)


def test_get_sea_level_uses_start_reference() -> None:
    """Compute sea level relative to start eustasy."""
    sim = AccommodationSimulator()
    sim.setEustaticCurve(_make_curve([100.0, 130.0]))
    sim.initialEustacy(0.0)
    assert sim.getSeaLevelAt(11.0) == 30.0


def test_get_basement_elevation_raises_without_curve() -> None:
    """Reject basement query when subsidence curve is absent."""
    sim = AccommodationSimulator()
    with pytest.raises(ValueError, match="Subsidence curve is not set"):
        sim.getBasementElevationAt(1.0)


def test_get_basement_elevation_from_topography_and_subsidence() -> None:
    """Compute basement as topography minus subsidence."""
    sim = AccommodationSimulator()
    sim.setInitialBathymetry(20.0)
    sim.setSubsidenceCurve(_make_curve([2.0, 5.0]), SubsidenceType.CUMULATIVE)
    assert sim.getBasementElevationAt(11.0) == -25.0


def test_get_accommodation_raises_without_subsidence_curve() -> None:
    """Reject accommodation query without subsidence curve."""
    sim = AccommodationSimulator()
    sim.setEustaticCurve(_make_curve([0.0, 10.0]))
    with pytest.raises(ValueError, match="Subsidence curve is not set"):
        sim.getAccommodationAt(10.0)


def test_get_accommodation_raises_without_eustatic_curve() -> None:
    """Reject accommodation query without eustatic curve."""
    sim = AccommodationSimulator()
    sim.setSubsidenceCurve(_make_curve([0.0, 10.0]), SubsidenceType.RATE)
    with pytest.raises(ValueError, match="Eustatic curve is not set"):
        sim.getAccommodationAt(10.0)


def test_get_accommodation_combines_sea_level_and_subsidence() -> None:
    """Compute accommodation as sea level plus subsidence."""
    sim = AccommodationSimulator()
    sim.setSubsidenceCurve(_make_curve([5.0, 8.0]), SubsidenceType.CUMULATIVE)
    sim.setEustaticCurve(_make_curve([100.0, 112.0]))
    sim.initialEustacy(0.0)

    # Sea level at 11 is 12 and subsidence is 8.
    assert sim.getAccommodationAt(11.0) == 20.0
