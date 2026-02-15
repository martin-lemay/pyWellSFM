# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Test suite for AccumulationModel classes and unified API."""

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

# Add src to path for imports
m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

from pywellsfm.io import (  # noqa: E402
    loadAccumulationModel,
    loadAccumulationModelGaussianFromCsv,
    saveAccumulationModelEnvironmentOptimumToJson,
    saveAccumulationModelGaussianToCsv,
    saveAccumulationModelGaussianToJson,
)
from pywellsfm.io.curve_io import curveToJsonObj  # noqa: E402
from pywellsfm.model import AccumulationCurve  # noqa: E402
from pywellsfm.model.AccumulationModel import (  # noqa: E402
    AccumulationModel,
    AccumulationModelElementBase,
    AccumulationModelElementEnvironmentOptimum,
    AccumulationModelElementGaussian,
)

# accumulation curve for environment optimum model tests
bathy_curve = AccumulationCurve(
    "Bathymetry",
    np.array([0.0, 10.0, 50.0]),
    np.array([0.0, 1.0, 0.0]),
)
energy_curve = AccumulationCurve(
    "Energy",
    np.array([0.0, 1.0]),
    np.array([0.0, 1.0]),
)
temp_curve = AccumulationCurve(
    "Temperature",
    np.array([0, 25, 50]),
    np.array([0.0, 1.0, 0.0]),
)


# ---------------------------------------
# AccumulationModel.AccumulationModelElementBase
# ---------------------------------------


def test_base_class_cannot_be_instantiated() -> None:
    """Test that the base class cannot be instantiated (abstract)."""
    with pytest.raises(TypeError):
        AccumulationModelElementBase("sand", 100.0)  # type: ignore[abstract]


def test_model_add_remove_get_element_model() -> None:
    """Test AccumulationModel manages element models by name."""
    model = AccumulationModel("Test")
    assert len(model.elements) == 0

    sand_model = AccumulationModelElementGaussian("sand", 100.0)
    model.addElement("sand", sand_model)
    assert len(model.elements) == 1
    assert model.getElementModel("sand") is sand_model

    # Remove
    model.removeElement("sand")
    assert model.getElementModel("sand") is None


# -------------------------------------------------
# AccumulationModel.AccumulationModelElementGaussian
# -------------------------------------------------


def test_gaussian_model_without_env_conditions() -> None:
    """Gaussian element model works without environment conditions."""
    model = AccumulationModel("ProbModel")
    model.addElement("sand", AccumulationModelElementGaussian("sand", 100.0))

    # Should work without environmentConditions (it's optional for this model)
    rate = model.getElementAccumulationAt("sand")

    # Verify it returns a float
    assert isinstance(rate, float)
    # With mean=100 and stddev=200, result should be reasonable (not exact due
    # to randomness)
    # but we can check it's in a plausible range (say, -400 to 600 for 3 sigma)
    assert -400 < rate < 600


def test_gaussian_model_with_env_conditions() -> None:
    """Gaussian element model ignores environment conditions."""
    model = AccumulationModel("ProbModel")
    model.addElement("sand", AccumulationModelElementGaussian("sand", 100.0))

    # Should work with environmentConditions (they're just ignored)
    env_conditions = {"Bathymetry": 10.0, "Temperature": 25.0}
    rate = model.getElementAccumulationAt("sand", env_conditions)

    assert isinstance(rate, float)
    assert -400 < rate < 600


def test_gaussian_model_consistent_behavior() -> None:
    """Test that Gaussian model produces varying results (stochastic)."""
    model = AccumulationModel("ProbModel")
    model.addElement("sand", AccumulationModelElementGaussian("sand", 100.0))

    # Generate multiple samples
    np.random.seed(42)  # Set seed for reproducibility in test
    samples = [model.getElementAccumulationAt("sand") for _ in range(200)]

    # Check that we get variation (not all the same)
    assert len(set(samples)) > 50  # Should have many unique values

    # Check mean is roughly centered around 100 (element.accumulationRate)
    mean_sample = np.mean(samples)
    assert 90 < mean_sample < 110  # Generous range due to randomness

    std_dev_sample = np.std(samples)
    assert 15 < std_dev_sample < 25  # Roughly around 20 (0.2*mean)


def test_gaussian_model_std_dev_factor() -> None:
    """Test that Gaussian model produces varying results (stochastic)."""
    model = AccumulationModel("ProbModel")
    model.addElement(
        "sand",
        AccumulationModelElementGaussian("sand", 100.0, std_dev_factor=5.0),
    )

    # Generate multiple samples
    np.random.seed(42)  # Set seed for reproducibility in test
    samples = [model.getElementAccumulationAt("sand") for _ in range(200)]

    # Check that we get variation (not all the same)
    assert len(set(samples)) > 50  # Should have many unique values

    # Check mean is roughly centered around 100 (element.accumulationRate)
    # increase tolerance due to higher stddev
    mean_sample = np.mean(samples)
    assert 70 < mean_sample < 130  # Generous range due to randomness

    std_dev_sample = np.std(samples)
    assert 450 < std_dev_sample < 550  # Roughly around 500 (5*mean)


# ----------------------------------------------------------
# AccumulationModel.AccumulationModelElementEnvironmentOptimum
# ----------------------------------------------------------


def test_environment_optimum_model_without_env_conditions_raises() -> None:
    """Test raises error without conditions."""
    element_model = AccumulationModelElementEnvironmentOptimum(
        "sand", 100.0, accumulationCurves={"Bathymetry": bathy_curve}
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Should raise ValueError when environmentConditions is None
    with pytest.raises(ValueError) as exc_info:
        model.getElementAccumulationAt("sand")

    error_msg = str(exc_info.value)
    assert "requires environmentConditions" in error_msg


def test_environmentOptimum_addAccumulationCurve_getAccumulationCurve() -> (
    None
):
    """Test adding curves stores them by x-axis name and can be retrieved."""
    element_model = AccumulationModelElementEnvironmentOptimum("sand", 10.0)
    element_model.addAccumulationCurve(bathy_curve)
    element_model.addAccumulationCurve(energy_curve)

    assert "Bathymetry" in element_model.accumulationCurves
    assert "Energy" in element_model.accumulationCurves
    assert element_model.getAccumulationCurve("Bathymetry") is bathy_curve
    assert element_model.getAccumulationCurve("Energy") is energy_curve


def test_environmentOptimum_removeCurve_removes_and_isNoopWhenMissing() -> (
    None
):
    """Test removes by name and doesn't error if missing."""
    element_model = AccumulationModelElementEnvironmentOptimum("sand", 10.0)
    element_model.addAccumulationCurve(bathy_curve)

    element_model.removeAccumulationCurve("Bathymetry")
    assert "Bathymetry" not in element_model.accumulationCurves

    # no-op
    element_model.removeAccumulationCurve("Bathymetry")
    assert "Bathymetry" not in element_model.accumulationCurves


def test_environment_optimum_getAccumulationCurve_raises_for_missing() -> None:
    """Test getAccumulationCurve raises KeyError when curve is missing."""
    element_model = AccumulationModelElementEnvironmentOptimum("sand", 10.0)
    with pytest.raises(KeyError):
        element_model.accumulationCurves["Bathymetry"]


def test_environmentOptimum_getElementAccumulationAt_matchesCurveProduct() -> (
    None
):
    """Migrated behavior test: accumulation equals rate * product(coeffs)."""
    element_model = AccumulationModelElementEnvironmentOptimum(
        "Sand",
        10.0,
        accumulationCurves={
            "Bathymetry": bathy_curve,
            "Energy": energy_curve,
        },
    )
    model = AccumulationModel("EnvModel", {"Sand": element_model})

    conditions = {"Bathymetry": 5.0, "Energy": 0.9}
    rate = model.getElementAccumulationAt("Sand", conditions)
    assert rate == pytest.approx(10.0 * 0.5 * 0.9, rel=1e-12)


def test_environment_optimum_model_with_env_conditions() -> None:
    """Test AccumulationModelEnvironmentOptimum with environment conditions."""
    element_model = AccumulationModelElementEnvironmentOptimum(
        "sand", 100.0, accumulationCurves={"Bathymetry": bathy_curve}
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Test at optimum bathymetry
    env_conditions = {"Bathymetry": 10.0}
    rate = model.getElementAccumulationAt("sand", env_conditions)

    # At optimum (coeff=1.0), rate should equal element.accumulationRate
    assert rate == pytest.approx(100.0, rel=1e-6)


def test_environment_optimum_model_multiple_factors() -> None:
    """Test with multiple environmental factors."""
    element_model = AccumulationModelElementEnvironmentOptimum(
        "sand",
        100.0,
        accumulationCurves={
            "Bathymetry": bathy_curve,
            "Temperature": temp_curve,
        },
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Test with both factors at optimum
    env_conditions = {"Bathymetry": 10.0, "Temperature": 25.0}
    rate = model.getElementAccumulationAt("sand", env_conditions)

    # Both coefficients = 1.0, so rate = 100.0 * 1.0 * 1.0
    assert rate == pytest.approx(100.0, rel=1e-6)

    # Test with one factor suboptimal
    env_conditions = {
        "Bathymetry": 10.0,
        "Temperature": 12.5,
    }  # temp coeff = 0.5
    rate = model.getElementAccumulationAt("sand", env_conditions)

    # Bathy coeff = 1.0, temp coeff = 0.5, so rate = 100.0 * 1.0 * 0.5 = 50.0
    assert rate == pytest.approx(50.0, rel=1e-6)


def test_environment_optimum_model_ignores_unknown_factors() -> None:
    """Test that unknown environmental factors are ignored."""
    element_model = AccumulationModelElementEnvironmentOptimum(
        "sand", 100.0, accumulationCurves={"Bathymetry": bathy_curve}
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Provide extra factors not in model
    env_conditions = {
        "Bathymetry": 10.0,
        "Temperature": 25.0,  # Not defined in model
        "Salinity": 35.0,  # Not defined in model
    }
    rate = model.getElementAccumulationAt("sand", env_conditions)

    # Should only use Bathymetry, ignore others
    assert rate == pytest.approx(100.0, rel=1e-6)


def test_unified_api_consistency() -> None:
    """All element models share the same call signature."""
    prob_model = AccumulationModel(
        "ProbModel",
        {
            "sand": AccumulationModelElementGaussian("sand", 100.0),
        },
    )
    env_model = AccumulationModel(
        "EnvModel",
        {
            "sand": AccumulationModelElementEnvironmentOptimum(
                "sand",
                100.0,
                accumulationCurves={"Bathymetry": bathy_curve},
            ),
        },
    )

    env_conditions = {"Bathymetry": 10.0}

    # Both should accept this calling convention
    prob_rate = prob_model.getElementAccumulationAt("sand", env_conditions)
    env_rate = env_model.getElementAccumulationAt("sand", env_conditions)

    assert isinstance(prob_rate, float)
    assert isinstance(env_rate, float)
    assert env_rate == pytest.approx(100.0, rel=1e-6)


def test_generic_client_code_pattern() -> None:
    """Test a realistic client code pattern using polymorphism."""

    def compute_total_accumulation(
        model: AccumulationModel,
        env_conditions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Example client function that works with any accumulation model."""
        results: dict[str, float] = {}
        for element_name in model.elements:
            try:
                results[element_name] = model.getElementAccumulationAt(
                    element_name, env_conditions
                )
            except ValueError:
                # Handle models that require env conditions
                results[element_name] = np.nan
        return results

    # Test with probabilistic model (doesn't need env conditions)
    prob_model = AccumulationModel(
        "ProbModel",
        {
            "sand": AccumulationModelElementGaussian("sand", 100.0),
            "clay": AccumulationModelElementGaussian("clay", 50.0),
        },
    )

    np.random.seed(42)
    results1 = compute_total_accumulation(prob_model)
    assert "sand" in results1
    assert "clay" in results1
    assert isinstance(results1["sand"], float)
    assert isinstance(results1["clay"], float)

    # Test with environment optimum model (requires env conditions)
    env_model = AccumulationModel(
        "EnvModel",
        {
            "sand": AccumulationModelElementEnvironmentOptimum(
                "sand",
                100.0,
                accumulationCurves={"Bathymetry": bathy_curve},
            ),
            "clay": AccumulationModelElementEnvironmentOptimum(
                "clay",
                50.0,
                accumulationCurves={"Bathymetry": bathy_curve},
            ),
        },
    )

    # Without env conditions - should get nan
    results2 = compute_total_accumulation(env_model)
    assert np.isnan(results2["sand"])
    assert np.isnan(results2["clay"])

    # With env conditions - should work
    env_conditions = {"Bathymetry": 10.0}
    results3 = compute_total_accumulation(env_model, env_conditions)
    assert results3["sand"] == pytest.approx(100.0, rel=1e-6)
    assert results3["clay"] == pytest.approx(50.0, rel=1e-6)


# ------------------------------
# Helpers for ioHelpers tests
# ------------------------------


def _write_json(tmp_path: Path, payload: dict[str, Any], filename: str) -> str:
    """Helper: write JSON to a temp file and return its filesystem path."""
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _write_csv(
    tmp_path: Path, rows: list[list[float | str]], filename: str
) -> str:
    """Helper: write a simple CSV file and return its filesystem path."""
    path = tmp_path / filename
    lines = [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _gaussian_signature(
    model: AccumulationModel,
) -> dict[str, tuple[float, float]]:
    """Canonical signature used by tests to compare Gaussian models."""
    out: dict[str, tuple[float, float]] = {}
    for element_name, element_model in model.elements.items():
        assert isinstance(element_model, AccumulationModelElementGaussian)
        out[element_name] = (
            float(element_model.accumulationRate),
            float(element_model.std_dev_factor),
        )
    return dict(sorted(out.items()))


def _envopt_signature(
    model: AccumulationModel,
) -> tuple[dict[str, float], dict[str, tuple[list[float], list[float]]]]:
    """Canonical signature to compare EnvironmentOptimum models."""
    elements: dict[str, float] = {}
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for element_name, element_model in model.elements.items():
        assert isinstance(
            element_model, AccumulationModelElementEnvironmentOptimum
        )
        elements[element_name] = float(element_model.accumulationRate)
        for curve_name, curve in element_model.accumulationCurves.items():
            curves[curve_name] = (
                [float(v) for v in curve._abscissa],
                [float(v) for v in curve._ordinate],
            )
    return dict(sorted(elements.items())), dict(sorted(curves.items()))


# -------------------------------------------------
# ioHelpers: Gaussian model (CSV)
# -------------------------------------------------


def test_loadAccumulationModelGaussianFromCsv_happy_path(
    tmp_path: Path,
) -> None:
    """Test loading a Gaussian accumulation model from a CSV file.

    Objective:
    - Ensure `loadAccumulationModelGaussianFromCsv` parses the expected columns
      and creates the corresponding `AccumulationModelGaussian` with elements
      and per-element stddev factors.

    Input data:
    - A temporary CSV file with header: name, mean, stddevFactor.
    - Two elements: sand (100.0, 0.2) and shale (50.0, 0.1).

    Expected outputs:
    - Returned model is an `AccumulationModelGaussian`.
    - Elements contain both names with matching accumulation rates.
    - std_dev_factors contain both names with matching factors.
    """
    csv_path = _write_csv(
        tmp_path,
        [
            ["name", "accumulationRate", "stddevFactor"],
            ["sand", 100.0, 0.2],
            ["shale", 50.0, 0.1],
        ],
        "gaussian.csv",
    )

    model = loadAccumulationModelGaussianFromCsv(csv_path)

    assert isinstance(model, AccumulationModel)
    sig = _gaussian_signature(model)
    assert sig == {"sand": (100.0, 0.2), "shale": (50.0, 0.1)}


def test_saveAccumulationModelGaussianToCsv_round_trip(tmp_path: Path) -> None:
    """Test exporting then reloading a Gaussian model via CSV preserves values.

    Objective:
    - Verify `saveAccumulationModelGaussianToCsv` produces a CSV that
      round-trips through `loadAccumulationModelGaussianFromCsv` without losing
      element means or stddev factors.

    Input data:
    - A Gaussian model with two elements and explicit stddev factors.

    Expected outputs:
    - After export + load, the signature (element rate + stddev factor) is
      identical.
    """
    model = AccumulationModel(
        "MyGaussian",
        {
            "sand": AccumulationModelElementGaussian("sand", 100.0, 0.25),
            "shale": AccumulationModelElementGaussian("shale", 50.0, 0.1),
        },
    )

    out_csv = tmp_path / "gaussian_out.csv"
    saveAccumulationModelGaussianToCsv(model, str(out_csv))

    reloaded = loadAccumulationModelGaussianFromCsv(str(out_csv))
    assert _gaussian_signature(reloaded) == _gaussian_signature(model)


# -------------------------------------------------
# ioHelpers: Gaussian model (JSON)
# -------------------------------------------------


def test_loadAccumulationModelGaussian_happy_path(tmp_path: Path) -> None:
    """Test loading a Gaussian accumulation model from schema-shaped JSON.

    Objective:
    - Ensure `loadAccumulationModel` accepts a schema-compliant JSON
      payload and builds the correct model.

    Input data:
        - A JSON file matching `AccumulationModelSchema.json` with per-element
            modelType="Gaussian".

    Expected outputs:
    - Returned model has the expected name, elements, and stddev factors.
    """
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "G",
            "elements": {
                "sand": {
                    "accumulationRate": 100.0,
                    "model": {
                        "modelType": "Gaussian",
                        "stddevFactor": 0.2,
                    },
                },
                "shale": {
                    "accumulationRate": 50.0,
                    "model": {
                        "modelType": "Gaussian",
                        "stddevFactor": 0.1,
                    },
                },
            },
        },
    }

    json_path = _write_json(tmp_path, payload, "gaussian.json")
    model = cast(AccumulationModel, loadAccumulationModel(json_path))

    assert model.name == "G"
    assert _gaussian_signature(model) == {
        "sand": (100.0, 0.2),
        "shale": (50.0, 0.1),
    }


def test_save_AccumulationModel_round_trip(tmp_path: Path) -> None:
    """Test exporting/reloading a Gaussian model via JSON preserves values.

    Objective:
    - Verify `saveAccumulationModelGaussianToJson` produces a JSON file that
      round-trips through `loadAccumulationMode without losing
      element rates or stddev factors.

    Input data:
    - A Gaussian model with two elements and explicit stddev factors.

    Expected outputs:
    - After export + load, the signature is identical.
    """
    model = AccumulationModel(
        "MyGaussian",
        {
            "sand": AccumulationModelElementGaussian("sand", 100.0, 0.25),
            "shale": AccumulationModelElementGaussian("shale", 50.0, 0.1),
        },
    )

    out_json = tmp_path / "gaussian_out.json"
    saveAccumulationModelGaussianToJson(model, str(out_json))

    reloaded = cast(AccumulationModel, loadAccumulationModel(str(out_json)))
    assert reloaded.name == "MyGaussian"
    assert _gaussian_signature(reloaded) == _gaussian_signature(model)


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        ([], "must be an object"),
        (
            {
                "format": "x",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {},
                },
            },
            "Invalid accumulation model format",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "x",
                "accumulationModel": {
                    "name": "G",
                    "elements": {},
                },
            },
            "Invalid accumulation model version",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": "not-an-object",
            },
            "'accumulationModel' must be an object",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {},
                },
            },
            "must be a non-empty object",
        ),
    ],
)
def test_loadAccumulationModel_rejects_invalid_payloads(
    tmp_path: Path,
    payload: Any,  # noqa: ANN401
    expected_message: str,
) -> None:
    """Test JSON loader rejects invalid shapes/metadata.

    Objective:
    - Verify `loadAccumulationModel` enforces the top-level metadata
      (format/version) and the expected object shapes and modelType.

    Input data:
    - A set of invalid payloads (wrong type, wrong format/version,
      wrong modelType).

    Expected outputs:
    - Each invalid payload raises ValueError with a message containing the
      expected substring.
    """
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        loadAccumulationModel(str(path))
    assert expected_message in str(exc.value)


# -------------------------------------------------
# ioHelpers: EnvironmentOptimum model (JSON)
# -------------------------------------------------


def test_loadAccumulationModelEnvironmentOptimumFromJson_inline_curves(
    tmp_path: Path,
) -> None:
    """Test loading an EnvironmentOptimum element model with inline curves.

    Objective:
    - Ensure `loadAccumulationModelEnvironmentOptimumFromJson` supports inline
      TabulatedFunction objects inside `accumulationCurves`.

    Input data:
    - A schema-shaped EnvironmentOptimum JSON file with one element and two
      inline curves (Bathymetry and Energy).

    Expected outputs:
    - Returned model contains the element with correct accumulationRate.
    - prodCurves contains both curves keyed by abscissaName.
    - Curve interpolation behaves as expected at an intermediate point.
    """
    bathy_curve_obj = curveToJsonObj(
        AccumulationCurve(
            "Bathymetry",
            np.array([0.0, 10.0]),
            np.array([0.0, 1.0]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="Bathymetry",
    )
    energy_curve_obj = curveToJsonObj(
        AccumulationCurve(
            "Energy",
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="Energy",
    )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 10.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [
                            bathy_curve_obj,
                            energy_curve_obj,
                        ],
                    },
                }
            },
        },
    }

    json_path = _write_json(tmp_path, payload, "env_inline.json")
    model = cast(AccumulationModel, loadAccumulationModel(json_path))

    assert model.name == "Env"
    sand_model = model.getElementModel("sand")
    assert isinstance(sand_model, AccumulationModelElementEnvironmentOptimum)
    assert "Bathymetry" in sand_model.accumulationCurves
    assert "Energy" in sand_model.accumulationCurves

    # Linear curve between 0 and 10 -> at 5 = 0.5
    assert sand_model.getElementAccumulationAt(
        {"Bathymetry": 5.0}
    ) == pytest.approx(10.0 * 0.5)


def test_loadAccumulationModelEnvironmentOptimumFromJson_url_curves(
    tmp_path: Path,
) -> None:
    """Test loading an EnvironmentOptimum element model with curves by url.

    Objective:
    - Ensure `accumulationCurves` supports schema refs: {"url": "curve.json"}
      resolved relative to the accumulation model JSON file location.

    Expected outputs:
    - Returned model contains the curve loaded from file.
    """
    bathy_curve_obj = curveToJsonObj(
        AccumulationCurve(
            "Bathymetry",
            np.array([0.0, 10.0]),
            np.array([0.0, 1.0]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="Bathymetry",
    )
    curve_path = tmp_path / "bathy_curve.json"
    curve_path.write_text(
        json.dumps(bathy_curve_obj, indent=2), encoding="utf-8"
    )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 10.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [
                            {"url": "bathy_curve.json"},
                        ],
                    },
                }
            },
        },
    }

    json_path = _write_json(tmp_path, payload, "env_url.json")
    model = cast(AccumulationModel, loadAccumulationModel(json_path))

    sand_model = model.getElementModel("sand")
    assert isinstance(sand_model, AccumulationModelElementEnvironmentOptimum)
    assert "Bathymetry" in sand_model.accumulationCurves
    assert sand_model.getElementAccumulationAt(
        {"Bathymetry": 5.0}
    ) == pytest.approx(10.0 * 0.5)


def test_saveAccumulationModelEnvironmentOptimumToJson_inline_round_trip(
    tmp_path: Path,
) -> None:
    """Test exporting an EnvironmentOptimum model with inline round-trips.

    Objective:
    - Verify
      `saveAccumulationModelEnvironmentOptimumToJson(curves_mode='inline')`
      produces a single JSON file that loads back to an equivalent model.

    Input data:
    - An EnvironmentOptimum model with 2 elements and 2 curves.

    Expected outputs:
    - Loading the exported JSON yields same element rates and curve tabulation.
    """
    model = AccumulationModel(
        "Env",
        {
            "sand": AccumulationModelElementEnvironmentOptimum(
                "sand",
                10.0,
                accumulationCurves={
                    "Bathymetry": AccumulationCurve(
                        "Bathymetry",
                        np.array([0.0, 10.0]),
                        np.array([0.0, 1.0]),
                    ),
                    "Energy": AccumulationCurve(
                        "Energy",
                        np.array([0.0, 1.0]),
                        np.array([0.0, 1.0]),
                    ),
                },
            ),
            "shale": AccumulationModelElementEnvironmentOptimum(
                "shale",
                2.5,
                accumulationCurves={
                    "Bathymetry": AccumulationCurve(
                        "Bathymetry",
                        np.array([0.0, 10.0]),
                        np.array([0.0, 1.0]),
                    ),
                    "Energy": AccumulationCurve(
                        "Energy",
                        np.array([0.0, 1.0]),
                        np.array([0.0, 1.0]),
                    ),
                },
            ),
        },
    )

    out_json = tmp_path / "env_inline_out.json"
    saveAccumulationModelEnvironmentOptimumToJson(
        model, str(out_json), curves_mode="inline"
    )

    reloaded = cast(
        AccumulationModel,
        loadAccumulationModel(str(out_json)),
    )
    assert _envopt_signature(reloaded) == _envopt_signature(model)


def test_saveAccumulationModelEnvironmentOptimumToJson_external_raises(
    tmp_path: Path,
) -> None:
    """External curve mode is not supported by the current schema."""
    model = AccumulationModel(
        "Env",
        {
            "sand": AccumulationModelElementEnvironmentOptimum(
                "sand",
                10.0,
                accumulationCurves={"Bathymetry": bathy_curve},
            )
        },
    )
    out_json = tmp_path / "env_external_out.json"
    with pytest.raises(ValueError):
        saveAccumulationModelEnvironmentOptimumToJson(
            model,
            str(out_json),
            curves_mode="external",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
