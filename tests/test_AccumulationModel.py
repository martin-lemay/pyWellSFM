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
from pywellsfm.model import (  # noqa: E402
    AccumulationCurve,
    AccumulationModelBase,
    AccumulationModelEnvironmentOptimum,
    AccumulationModelGaussian,
    Element,
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
# AccumulationModel.AccumulationModelBase
# ---------------------------------------


def test_base_class_cannot_be_instantiated() -> None:
    """Test that the base class cannot be instantiated (abstract)."""
    with pytest.raises(TypeError):
        AccumulationModelBase("TestModel")  # type: ignore[abstract]


def test_base_addElement_adds_element() -> None:
    """Test AccumulationModelBase.addElement adds an element to the set."""
    model = AccumulationModelGaussian("ProbModel")
    assert len(model.elements) == 0

    element = Element("sand", 100.0)
    model.addElement(element)

    assert len(model.elements) == 1
    assert element in model.elements


def test_base_addElement_deduplicates_by_name() -> None:
    """Test addElement uses Element hashing/equality to avoid duplicates."""
    model = AccumulationModelGaussian("ProbModel")
    element1 = Element("sand", 100.0)
    element2 = Element("sand", 999.0)  # Same name => equal to element1

    model.addElement(element1)
    model.addElement(element2)

    assert len(model.elements) == 1
    retrieved = model.getElement("sand")
    assert retrieved is element1


def test_base_getElement_returns_element_or_none() -> None:
    """Test getElement returns the matching element or None when missing."""
    model = AccumulationModelGaussian("ProbModel")
    sand = Element("sand", 100.0)
    model.addElement(sand)

    assert model.getElement("sand") is sand
    assert model.getElement("clay") is None


def test_base_removeElement_removes_existing() -> None:
    """Test removeElement removes by name and leaves other elements intact."""
    model = AccumulationModelGaussian("ProbModel")
    sand = Element("sand", 100.0)
    clay = Element("clay", 50.0)
    model.addElement(sand)
    model.addElement(clay)

    model.removeElement("sand")

    assert model.getElement("sand") is None
    assert model.getElement("clay") is clay
    assert len(model.elements) == 1


def test_base_removeElement_noop_when_missing() -> None:
    """Test removeElement does nothing when the name is not found."""
    model = AccumulationModelGaussian("ProbModel")
    sand = Element("sand", 100.0)
    model.addElement(sand)

    model.removeElement("does-not-exist")

    assert model.getElement("sand") is sand
    assert len(model.elements) == 1


# -------------------------------------------
# AccumulationModel.AccumulationModelGaussian
# -------------------------------------------


def test_gaussian_model_without_env_conditions() -> None:
    """Test AccumulationModelProbabilistic without environment conditions."""
    model = AccumulationModelGaussian("ProbModel")
    element = Element("sand", 100.0)
    model.addElement(element)

    # Should work without environmentConditions (it's optional for this model)
    rate = model.getElementAccumulationAt(element)

    # Verify it returns a float
    assert isinstance(rate, float)
    # With mean=100 and stddev=200, result should be reasonable (not exact due
    # to randomness)
    # but we can check it's in a plausible range (say, -400 to 600 for 3 sigma)
    assert -400 < rate < 600


def test_gaussian_model_with_env_conditions() -> None:
    """Test AccumulationModelProbabilistic with environment conditions."""
    model = AccumulationModelGaussian("ProbModel")
    element = Element("sand", 100.0)
    model.addElement(element)

    # Should work with environmentConditions (they're just ignored)
    env_conditions = {"Bathymetry": 10.0, "Temperature": 25.0}
    rate = model.getElementAccumulationAt(element, env_conditions)

    assert isinstance(rate, float)
    assert -400 < rate < 600


def test_gaussian_model_consistent_behavior() -> None:
    """Test that Gaussian model produces varying results (stochastic)."""
    model = AccumulationModelGaussian("ProbModel")
    element = Element("sand", 100.0)
    model.addElement(element)

    # Generate multiple samples
    np.random.seed(42)  # Set seed for reproducibility in test
    samples = [model.getElementAccumulationAt(element) for _ in range(200)]

    # Check that we get variation (not all the same)
    assert len(set(samples)) > 50  # Should have many unique values

    # Check mean is roughly centered around 100 (element.accumulationRate)
    mean_sample = np.mean(samples)
    assert 90 < mean_sample < 110  # Generous range due to randomness

    std_dev_sample = np.std(samples)
    assert 15 < std_dev_sample < 25  # Roughly around 20 (0.2*mean)


def test_gaussian_model_std_dev_factor() -> None:
    """Test that Gaussian model produces varying results (stochastic)."""
    model = AccumulationModelGaussian("ProbModel")
    element = Element("sand", 100.0)
    model.addElement(element, std_dev_factor=5.0)  # 5x mean

    # Generate multiple samples
    np.random.seed(42)  # Set seed for reproducibility in test
    samples = [model.getElementAccumulationAt(element) for _ in range(200)]

    # Check that we get variation (not all the same)
    assert len(set(samples)) > 50  # Should have many unique values

    # Check mean is roughly centered around 100 (element.accumulationRate)
    # increase tolerance due to higher stddev
    mean_sample = np.mean(samples)
    assert 70 < mean_sample < 130  # Generous range due to randomness

    std_dev_sample = np.std(samples)
    assert 450 < std_dev_sample < 550  # Roughly around 500 (5*mean)


# -----------------------------------------------------
# AccumulationModel.AccumulationModelEnvironmentOptimum
# -----------------------------------------------------


def test_environment_optimum_model_without_env_conditions_raises() -> None:
    """Test raises error without conditions."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    element = Element("sand", 100.0)
    model.addElement(element)

    # Add a production curve
    model.addAccumulationCurve(bathy_curve)

    # Should raise ValueError when environmentConditions is None
    with pytest.raises(ValueError) as exc_info:
        model.getElementAccumulationAt(element)

    error_msg = str(exc_info.value)
    assert "requires environmentConditions" in error_msg
    assert "sand" in error_msg  # Element name should be in error


def test_environment_optimum_addAccumulationCurve_and_getAccumulationCurve(

) -> None:
    """Test adding curves stores them by x-axis name and can be retrieved."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")

    model.addAccumulationCurve(bathy_curve)
    model.addAccumulationCurve(energy_curve)

    assert "Bathymetry" in model.prodCurves
    assert "Energy" in model.prodCurves
    assert model.getAccumulationCurve("Bathymetry") is bathy_curve
    assert model.getAccumulationCurve("Energy") is energy_curve


def test_environment_optimum_removeCurve_removes_and_is_noop_when_missing(

) -> None:
    """Test removes by name and doesn't error if missing."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    model.addAccumulationCurve(bathy_curve)

    model.removeAccumulationCurve("Bathymetry")
    assert "Bathymetry" not in model.prodCurves

    # no-op
    model.removeAccumulationCurve("Bathymetry")
    assert "Bathymetry" not in model.prodCurves


def test_environment_optimum_getAccumulationCurve_raises_for_missing() -> None:
    """Test getAccumulationCurve raises KeyError when curve is missing."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    with pytest.raises(KeyError):
        model.getAccumulationCurve("Bathymetry")


def test_environment_optimum_getElementAccumulationAt_matches_curve_product(

) -> None:
    """Migrated behavior test: accumulation equals rate * product(coeffs)."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    element = Element("Sand", 10.0)
    model.addElement(element)

    model.addAccumulationCurve(bathy_curve)
    model.addAccumulationCurve(energy_curve)

    conditions = {"Bathymetry": 5.0, "Energy": 0.9}
    rate = model.getElementAccumulationAt(element, conditions)
    assert rate == pytest.approx(10.0 * 0.5 * 0.9, rel=1e-12)


def test_environment_optimum_model_with_env_conditions() -> None:
    """Test AccumulationModelEnvironmentOptimum with environment conditions."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    element = Element("sand", 100.0)
    model.addElement(element)
    model.addAccumulationCurve(bathy_curve)

    # Test at optimum bathymetry
    env_conditions = {"Bathymetry": 10.0}
    rate = model.getElementAccumulationAt(element, env_conditions)

    # At optimum (coeff=1.0), rate should equal element.accumulationRate
    assert rate == pytest.approx(100.0, rel=1e-6)


def test_environment_optimum_model_multiple_factors() -> None:
    """Test with multiple environmental factors."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    element = Element("sand", 100.0)
    model.addElement(element)

    # Add multiple production curves
    model.addAccumulationCurve(bathy_curve)
    model.addAccumulationCurve(temp_curve)

    # Test with both factors at optimum
    env_conditions = {"Bathymetry": 10.0, "Temperature": 25.0}
    rate = model.getElementAccumulationAt(element, env_conditions)

    # Both coefficients = 1.0, so rate = 100.0 * 1.0 * 1.0
    assert rate == pytest.approx(100.0, rel=1e-6)

    # Test with one factor suboptimal
    env_conditions = {
        "Bathymetry": 10.0,
        "Temperature": 12.5,
    }  # temp coeff = 0.5
    rate = model.getElementAccumulationAt(element, env_conditions)

    # Bathy coeff = 1.0, temp coeff = 0.5, so rate = 100.0 * 1.0 * 0.5 = 50.0
    assert rate == pytest.approx(50.0, rel=1e-6)


def test_environment_optimum_model_ignores_unknown_factors() -> None:
    """Test that unknown environmental factors are ignored."""
    model = AccumulationModelEnvironmentOptimum("EnvModel")
    element = Element("sand", 100.0)
    model.addElement(element)

    # Add only bathymetry curve
    model.addAccumulationCurve(bathy_curve)

    # Provide extra factors not in model
    env_conditions = {
        "Bathymetry": 10.0,
        "Temperature": 25.0,  # Not defined in model
        "Salinity": 35.0,  # Not defined in model
    }
    rate = model.getElementAccumulationAt(element, env_conditions)

    # Should only use Bathymetry, ignore others
    assert rate == pytest.approx(100.0, rel=1e-6)


def test_unified_api_consistency() -> None:
    """Test that both models can be called with the same unified API."""
    # Create both model types
    prob_model = AccumulationModelGaussian("ProbModel")
    env_model = AccumulationModelEnvironmentOptimum("EnvModel")

    # Setup env model
    env_model.addAccumulationCurve(bathy_curve)

    element = Element("sand", 100.0)
    env_model.addElement(element)
    prob_model.addElement(element)

    env_conditions = {"Bathymetry": 10.0}

    # Both should accept this calling convention
    prob_rate = prob_model.getElementAccumulationAt(element, env_conditions)
    env_rate = env_model.getElementAccumulationAt(element, env_conditions)

    assert isinstance(prob_rate, float)
    assert isinstance(env_rate, float)
    assert env_rate == pytest.approx(100.0, rel=1e-6)


def test_generic_client_code_pattern() -> None:
    """Test a realistic client code pattern using polymorphism."""

    def compute_total_accumulation(
        model: AccumulationModelBase,
        env_conditions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Example client function that works with any accumulation model."""
        results: dict[str, float] = {}
        for element in model.elements:
            try:
                results[element.name] = model.getElementAccumulationAt(
                    element, env_conditions
                )
            except ValueError:
                # Handle models that require env conditions
                results[element.name] = np.nan
        return results

    # Test with probabilistic model (doesn't need env conditions)
    elements = [Element("sand", 100.0), Element("clay", 50.0)]
    prob_model = AccumulationModelGaussian("ProbModel", set(elements))

    np.random.seed(42)
    results1 = compute_total_accumulation(prob_model)
    assert "sand" in results1
    assert "clay" in results1
    assert isinstance(results1["sand"], float)
    assert isinstance(results1["clay"], float)

    # Test with environment optimum model (requires env conditions)
    env_model = AccumulationModelEnvironmentOptimum("EnvModel", set(elements))
    env_model.addAccumulationCurve(bathy_curve)

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
    model: AccumulationModelGaussian,
) -> dict[str, tuple[float, float]]:
    """Canonical signature used by tests to compare Gaussian models."""
    out: dict[str, tuple[float, float]] = {}
    for element in model.elements:
        factor = model.std_dev_factors.get(element.name, model.defaultStdDev)
        out[element.name] = (float(element.accumulationRate), float(factor))
    return dict(sorted(out.items()))


def _envopt_signature(
    model: AccumulationModelEnvironmentOptimum,
) -> tuple[dict[str, float], dict[str, tuple[list[float], list[float]]]]:
    """Canonical signature to compare EnvironmentOptimum models."""
    elements = {e.name: float(e.accumulationRate) for e in model.elements}
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for name, curve in model.prodCurves.items():
        curves[name] = (
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
            ["name", "mean", "stddevFactor"],
            ["sand", 100.0, 0.2],
            ["shale", 50.0, 0.1],
        ],
        "gaussian.csv",
    )

    model = loadAccumulationModelGaussianFromCsv(csv_path)

    assert isinstance(model, AccumulationModelGaussian)
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
    elements = {Element("sand", 100.0), Element("shale", 50.0)}
    model = AccumulationModelGaussian(
        name="MyGaussian",
        elements=elements,
        std_dev_factors={"sand": 0.25, "shale": 0.1},
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
    - A JSON file matching `AccumulationModelSchema.json` with
      modelType="Gaussian".

    Expected outputs:
    - Returned model has the expected name, elements, and stddev factors.
    """
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "G",
            "modelType": "Gaussian",
            "elements": [
                {
                    "name": "sand",
                    "accumulationRate": 100.0,
                    "stddevFactor": 0.2,
                },
                {
                    "name": "shale",
                    "accumulationRate": 50.0,
                    "stddevFactor": 0.1,
                },
            ],
        },
    }

    json_path = _write_json(tmp_path, payload, "gaussian.json")
    model = cast(AccumulationModelGaussian, loadAccumulationModel(json_path))

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
    elements = {Element("sand", 100.0), Element("shale", 50.0)}
    model = AccumulationModelGaussian(
        name="MyGaussian",
        elements=elements,
        std_dev_factors={"sand": 0.25, "shale": 0.1},
    )

    out_json = tmp_path / "gaussian_out.json"
    saveAccumulationModelGaussianToJson(model, str(out_json))

    reloaded = cast(
        AccumulationModelGaussian, loadAccumulationModel(str(out_json))
    )
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
                    "modelType": "Gaussian",
                    "elements": [],
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
                    "modelType": "Gaussian",
                    "elements": [],
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
                    "modelType": "EnvironmentOptimum",
                    "elements": [],
                },
            },
            "must be a non-empty list",
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
    """Test loading an EnvironmentOptimum model with inline TabulatedFunction.

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
    bathy_curve = {
        "format": "pyWellSFM.TabulatedFunctionData",
        "version": "1.0",
        "abscissaName": "Bathymetry",
        "ordinateName": "ReductionCoeff",
        "values": {"xValues": [0.0, 10.0], "yValues": [0.0, 1.0]},
    }
    energy_curve = {
        "format": "pyWellSFM.TabulatedFunctionData",
        "version": "1.0",
        "abscissaName": "Energy",
        "ordinateName": "ReductionCoeff",
        "values": {"xValues": [0.0, 1.0], "yValues": [0.0, 1.0]},
    }

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "modelType": "EnvironmentOptimum",
            "elements": [
                {
                    "name": "sand",
                    "accumulationRate": 10.0,
                    "accumulationCurves": [bathy_curve, energy_curve],
                }
            ],
        },
    }

    json_path = _write_json(tmp_path, payload, "env_inline.json")
    model = model = cast(
        AccumulationModelEnvironmentOptimum,
        loadAccumulationModel(json_path),
    )

    assert model.name == "Env"
    assert model.getElement("sand") is not None
    assert "Bathymetry" in model.prodCurves
    assert "Energy" in model.prodCurves

    # Linear curve between 0 and 10 -> at 5 = 0.5
    assert float(model.prodCurves["Bathymetry"](5.0)) == pytest.approx(0.5)


def test_loadAccuModelEnvironmentOptimumFromJson_external_curve_files(
    tmp_path: Path,
) -> None:
    """Test loading an EnvironmentOptimum model referencing external files.

    Objective:
    - Ensure `loadAccumulationModelEnvironmentOptimumFromJson` supports
      `accumulationCurves` entries that are paths to external JSON and CSV
      files.
    - Ensure relative paths resolve relative to the model JSON directory.

    Input data:
    - Curve1: JSON TabulatedFunction file for Bathymetry.
    - Curve2: CSV numeric x,y file for Energy (name inferred from filename
      stem).
    - Model JSON referencing both with relative paths.

    Expected outputs:
    - Returned model has both curves loaded in prodCurves.
    - Bathymetry curve name comes from the JSON field abscissaName.
    - Energy curve name comes from CSV filename stem.
    """
    bathy_payload = {
        "format": "pyWellSFM.TabulatedFunctionData",
        "version": "1.0",
        "abscissaName": "Bathymetry",
        "ordinateName": "ReductionCoeff",
        "values": {"xValues": [0.0, 10.0], "yValues": [0.0, 1.0]},
    }
    bathy_json = tmp_path / "bathy.json"
    bathy_json.write_text(
        json.dumps(bathy_payload, indent=2), encoding="utf-8"
    )

    energy_csv = tmp_path / "Energy.csv"
    energy_csv.write_text("0,0\n1,1\n", encoding="utf-8")

    model_payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "modelType": "EnvironmentOptimum",
            "elements": [
                {
                    "name": "sand",
                    "accumulationRate": 10.0,
                    "accumulationCurves": ["bathy.json", "Energy.csv"],
                }
            ],
        },
    }

    json_path = _write_json(tmp_path, model_payload, "env_external.json")
    model = cast(
        AccumulationModelEnvironmentOptimum,
        loadAccumulationModel(json_path),
    )

    assert "Bathymetry" in model.prodCurves
    assert "Energy" in model.prodCurves


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
    model = AccumulationModelEnvironmentOptimum("Env")
    model.addElement(Element("sand", 10.0))
    model.addElement(Element("shale", 2.5))

    model.addAccumulationCurve(
        AccumulationCurve(
            "Bathymetry", np.array([0.0, 10.0]), np.array([0.0, 1.0])
        )
    )
    model.addAccumulationCurve(
        AccumulationCurve("Energy", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    )

    out_json = tmp_path / "env_inline_out.json"
    saveAccumulationModelEnvironmentOptimumToJson(
        model, str(out_json), curves_mode="inline"
    )

    reloaded = cast(
        AccumulationModelEnvironmentOptimum,
        loadAccumulationModel(str(out_json)),
    )
    assert _envopt_signature(reloaded) == _envopt_signature(model)


def test_saveAccumulationModelEnvironmentOptimumToJson_external_json_roundTrip(
    tmp_path: Path,
) -> None:
    """Test exporting an EnvironmentOptimum model with external JSON curves.

    Objective:
    - Verify
      `saveAccumulationModelEnvironmentOptimumToJson(curves_mode='external',
      curves_format='json')` produces multiple files (model + curve files) that
      load back to an equivalent model.

    Input data:
    - An EnvironmentOptimum model with 1 element and 2 curves.

    Expected outputs:
    - The curve files exist on disk.
    - Loading the exported model JSON yields the same element/curve signature.
    """
    model = AccumulationModelEnvironmentOptimum("Env")
    model.addElement(Element("sand", 10.0))
    model.addAccumulationCurve(
        AccumulationCurve(
            "Bathymetry", np.array([0.0, 10.0]), np.array([0.0, 1.0])
        )
    )
    model.addAccumulationCurve(
        AccumulationCurve("Energy", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    )

    curves_dir = tmp_path / "curves"
    out_json = tmp_path / "env_external_out.json"
    saveAccumulationModelEnvironmentOptimumToJson(
        model,
        str(out_json),
        curves_mode="external",
        curves_dir=str(curves_dir),
        curves_format="json",
    )

    assert (curves_dir / "Bathymetry.json").exists()
    assert (curves_dir / "Energy.json").exists()

    reloaded = cast(
        AccumulationModelEnvironmentOptimum,
        loadAccumulationModel(str(out_json)),
    )
    assert _envopt_signature(reloaded) == _envopt_signature(model)


def test_saveAccumulationModelEnvironmentOptimumToJson_external_csv_round_trip(
    tmp_path: Path,
) -> None:
    """Test exporting an EnvironmentOptimum model with external CSV curves.

    Objective:
    - Verify
      `saveAccumulationModelEnvironmentOptimumToJson(curves_mode='external',
      curves_format='csv')` produces multiple files (model + curve CSV files)
      that load back to an equivalent model.

    Input data:
    - An EnvironmentOptimum model with 1 element and 2 curves.

    Expected outputs:
    - The curve CSV files exist on disk.
    - Loading the exported model JSON yields the same element/curve signature.
    """
    model = AccumulationModelEnvironmentOptimum("Env")
    model.addElement(Element("sand", 10.0))
    model.addAccumulationCurve(
        AccumulationCurve(
            "Bathymetry", np.array([0.0, 10.0]), np.array([0.0, 1.0])
        )
    )
    model.addAccumulationCurve(
        AccumulationCurve("Energy", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    )

    curves_dir = tmp_path / "curves"
    out_json = tmp_path / "env_external_out.json"
    saveAccumulationModelEnvironmentOptimumToJson(
        model,
        str(out_json),
        curves_mode="external",
        curves_dir=str(curves_dir),
        curves_format="csv",
    )

    assert (curves_dir / "Bathymetry.csv").exists()
    assert (curves_dir / "Energy.csv").exists()

    reloaded = cast(
        AccumulationModelEnvironmentOptimum,
        loadAccumulationModel(str(out_json)),
    )
    assert _envopt_signature(reloaded) == _envopt_signature(model)


def test_loadAccumulationModel_missing_curve_file_raises(
    tmp_path: Path,
) -> None:
    """Test JSON loader errors when a referenced curve file is missing.

    Objective:
    - Ensure `loadAccumulationModel` fails fast when a
      referenced curve file does not exist.

    Input data:
    - A model JSON referencing a non-existent curve file "Missing.json".

    Expected outputs:
    - `FileNotFoundError` is raised.
    """
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "modelType": "EnvironmentOptimum",
            "elements": [
                {
                    "name": "sand",
                    "accumulationRate": 10.0,
                    "accumulationCurves": ["Missing.json"],
                }
            ],
        },
    }

    model_json = _write_json(tmp_path, payload, "env_missing_curve.json")

    with pytest.raises(FileNotFoundError):
        loadAccumulationModel(model_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
