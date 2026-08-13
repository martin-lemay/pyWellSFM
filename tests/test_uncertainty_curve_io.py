# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

import os
import tempfile

import numpy as np
import pytest

from pywellsfm.io.curve_io import (
    loadUncertaintyCurveFromCsv,
    saveUncertaintyCurveToCsv,
    uncertaintyCurveToBytes,
)
from pywellsfm.model import Curve, UncertaintyCurve


@pytest.fixture
def sample_ucurve() -> UncertaintyCurve:
    """Create a sample UncertaintyCurve for testing."""
    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([5.0, 15.0, 25.0, 35.0])
    curve = Curve("Depth", "WaterDepth", x, y)
    uc = UncertaintyCurve("WaterDepth", curve)
    uc.setMinCurveValues(np.array([3.0, 12.0, 22.0, 32.0]))
    uc.setMaxCurveValues(np.array([7.0, 18.0, 28.0, 38.0]))
    return uc


def test_saveUncertaintyCurveToCsv(
    sample_ucurve: UncertaintyCurve,
) -> None:
    """Test saving UncertaintyCurve to CSV."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        saveUncertaintyCurveToCsv(sample_ucurve, path)
        assert os.path.exists(path)
        with open(path) as fh:
            lines = fh.readlines()
        # Header + 4 data rows
        assert len(lines) == 5
        header = lines[0].strip()
        assert "depth" in header.lower() or "Depth" in header
        assert "min" in header.lower()
        assert "max" in header.lower()
    finally:
        os.unlink(path)


def test_saveUncertaintyCurveToCsv_roundtrip(
    sample_ucurve: UncertaintyCurve,
) -> None:
    """Test CSV round-trip: save then load."""
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        saveUncertaintyCurveToCsv(sample_ucurve, path)
        loaded = loadUncertaintyCurveFromCsv(Path(path))
        np.testing.assert_allclose(
            loaded.getAbscissa(),
            sample_ucurve.getAbscissa(),
        )
        np.testing.assert_allclose(
            loaded.getMinValues(),
            sample_ucurve.getMinValues(),
        )
        np.testing.assert_allclose(
            loaded.getMaxValues(),
            sample_ucurve.getMaxValues(),
        )
    finally:
        os.unlink(path)


def test_saveUncertaintyCurveToCsv_wrong_ext(
    sample_ucurve: UncertaintyCurve,
) -> None:
    """Test that non-.csv extension raises."""
    with pytest.raises(ValueError, match=".csv"):
        saveUncertaintyCurveToCsv(sample_ucurve, "/fake/path.json")


def test_uncertaintyCurveToBytes(
    sample_ucurve: UncertaintyCurve,
) -> None:
    """Test in-memory CSV bytes export."""
    data = uncertaintyCurveToBytes(sample_ucurve)
    assert isinstance(data, bytes)
    text = data.decode("utf-8")
    lines = text.strip().split("\n")
    # Header + 4 data rows
    assert len(lines) == 5
    assert "min" in lines[0].lower()
    assert "max" in lines[0].lower()
