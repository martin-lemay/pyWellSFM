# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import numpy as np
import pytest

from pywellsfm.model.Curve import Curve
from pywellsfm.model.DepthAgeModel import DepthAgeModel
from pywellsfm.model.Marker import Marker


def test_DepthAgeModel_init_defaults() -> None:
    """Model initializes with expected default metadata and empty state."""
    model = DepthAgeModel()
    assert model._xaxisName == "Age"
    assert model._yaxisName == "Depth"
    assert model.depthAgeCurve is None
    assert model.markers == set()
    assert model.interpolationMethod == "linear"


def test_DepthAgeModel_setMarkers_and_getDepth() -> None:
    """Setting markers builds a curve that can be queried in depth domain."""
    model = DepthAgeModel()
    model.setMarkers(
        {
            Marker("M0", depth=100.0, age=0.0),
            Marker("M1", depth=200.0, age=10.0),
        }
    )

    assert model.depthAgeCurve is not None
    assert model.getDepth(0.0) == pytest.approx(100.0)
    assert model.getDepth(10.0) == pytest.approx(200.0)
    assert model.getDepth(5.0) == pytest.approx(150.0)


def test_DepthAgeModel_updateCurve_and_getAge() -> None:
    """Curve update supports exact-match and interpolated age query paths."""
    model = DepthAgeModel()
    age_depths = np.array(
        [[0.0, 100.0], [10.0, 200.0], [20.0, 300.0]],
        dtype=float,
    )
    model.updateCurve(age_depths)

    # Exact depth path in implementation (returns matching ordinate tuple).
    exact = model.getAge(200.0)
    assert exact is not None
    assert exact[0] == pytest.approx(200.0)

    # Interpolation path returns a single age value tuple.
    interp = model.getAge(250.0)
    assert interp is not None
    assert interp[0] == pytest.approx(15.0)


def test_DepthAgeModel_addMarker_with_and_without_existing_curve() -> None:
    """Adding markers works for empty and already initialized models."""
    model = DepthAgeModel()
    model.addMarker(Marker("M0", depth=100.0, age=0.0))
    assert model.depthAgeCurve is not None

    model.addMarker(Marker("M1", depth=200.0, age=10.0))
    assert model.depthAgeCurve is not None
    assert np.array_equal(
        model.depthAgeCurve._abscissa,
        np.array([0.0, 10.0], dtype=float),
    )
    assert np.array_equal(
        model.depthAgeCurve._ordinate,
        np.array([100.0, 200.0], dtype=float),
    )


def test_DepthAgeModel_getDepth_getAge_when_curve_missing() -> None:
    """Queries on empty model return NaN/None sentinel values."""
    model = DepthAgeModel()
    assert np.isnan(model.getDepth(10.0))
    assert model.getAge(100.0) is None


def test_DepthAgeModel_conversion_methods_not_implemented() -> None:
    """Placeholder conversion APIs explicitly raise NotImplementedError."""
    model = DepthAgeModel()
    log = Curve(
        "Age",
        "GR",
        np.array([0.0, 1.0], dtype=float),
        np.array([10.0, 20.0], dtype=float),
        "linear",
    )

    with pytest.raises(NotImplementedError):
        _ = model.convertContinuousLogToDepth(log)
    with pytest.raises(NotImplementedError):
        _ = model.convertContinuousLogToAge(log)
    with pytest.raises(NotImplementedError):
        _ = model.convertDiscreteLogToDepth(log)
    with pytest.raises(NotImplementedError):
        _ = model.convertDiscreteLogToAge(log)
