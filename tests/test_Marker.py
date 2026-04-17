# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

from pywellsfm.model.Marker import Marker, StratigraphicSurfaceType


def test_Marker_init_and_basic_properties() -> None:
    """Marker stores name/depth/age/stratigraphic type as provided."""
    marker = Marker(
        name="M1",
        depth=100.0,
        age=5.0,
        stratigraphicType=StratigraphicSurfaceType.CONFORM,
    )

    assert marker.name == "M1"
    assert marker.depth == 100.0
    assert marker.age == 5.0
    assert marker.stratigraphicType == StratigraphicSurfaceType.CONFORM


def test_Marker_equality_and_hash() -> None:
    """Equality and hash depend on all marker attributes."""
    m1 = Marker("M", 10.0, 2.0, StratigraphicSurfaceType.TOPLAP)
    m2 = Marker("M", 10.0, 2.0, StratigraphicSurfaceType.TOPLAP)
    m3 = Marker("M", 10.0, 3.0, StratigraphicSurfaceType.TOPLAP)

    assert m1 == m2
    assert hash(m1) == hash(m2)
    assert m1 != m3
    assert m1 != "not-a-marker"


def test_Marker_collocation_and_synchrone_checks() -> None:
    """Spatial and temporal relation helpers return expected booleans."""
    m1 = Marker("M1", 10.0, 2.0)
    m2 = Marker("M2", 10.0, 3.0)
    m3 = Marker("M3", 11.0, 2.0)

    assert m1.areCollocated(m2)
    assert not m1.areCollocated(m3)
    assert not m1.areCollocated(1)

    assert m1.areSynchrone(m3)
    assert not m1.areSynchrone(m2)
    assert not m1.areSynchrone(None)


def test_Marker_same_horizon_logic() -> None:
    """Same-horizon requires both matching name and synchronous age."""
    m1 = Marker("TopA", 10.0, 2.0)
    m2 = Marker("TopA", 20.0, 2.0)
    m3 = Marker("TopA", 20.0, 3.0)
    m4 = Marker("TopB", 20.0, 2.0)

    assert m1.areFromSameHorizon(m2)
    assert not m1.areFromSameHorizon(m3)
    assert not m1.areFromSameHorizon(m4)
    assert not m1.areFromSameHorizon("TopA")
