# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Test suite for AccumulationModel classes and unified API."""

import os
import sys

import numpy as np
import pytest

# Add src to path for imports
m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

from pywellsfm.model import AccumulationCurve
from pywellsfm.model.AccumulationModel import (
    AccumulationModel,
    AccumulationModelCombination,
    AccumulationModelElementBase,
    AccumulationModelElementGaussian,
    AccumulationModelElementOptimum,
)

# accumulation curve for environment optimum model tests
bathy_curve = AccumulationCurve(
    "WaterDepth",
    np.array([0.0, 10.0, 50.0]),
    np.array([0.0, 1.0, 0.8]),
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
age_curve = AccumulationCurve(
    "Age",
    np.array([0, 5.0, 10]),
    np.array([0.0, 1.0, 0.8]),
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
    env_conditions = {"WaterDepth": 10.0, "Temperature": 25.0}
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
# AccumulationModel.AccumulationModelElementOptimum
# ----------------------------------------------------------


def test_environment_optimum_model_without_env_conditions() -> None:
    """Test reduction coefficient is 1.0 without conditions."""
    element_model = AccumulationModelElementOptimum(
        "sand", 100.0, accumulationCurves={"WaterDepth": bathy_curve}
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})
    rate = model.getElementAccumulationAt("sand")

    assert rate == 100.0


def test_environmentOptimum_addAccumulationCurve_getAccumulationCurve() -> (
    None
):
    """Test adding curves stores them by x-axis name and can be retrieved."""
    element_model = AccumulationModelElementOptimum("sand", 10.0)
    element_model.addAccumulationCurve(bathy_curve)
    element_model.addAccumulationCurve(energy_curve)

    assert "waterdepth" in element_model.accumulationCurves
    assert "energy" in element_model.accumulationCurves
    assert element_model.getAccumulationCurve("WaterDepth") is bathy_curve
    assert element_model.getAccumulationCurve("Energy") is energy_curve


def test_environmentOptimum_removeCurve_removes_and_isNoopWhenMissing() -> (
    None
):
    """Test removes by name and doesn't error if missing."""
    element_model = AccumulationModelElementOptimum("sand", 10.0)
    element_model.addAccumulationCurve(bathy_curve)

    element_model.removeAccumulationCurve("WaterDepth")
    assert "waterdepth" not in element_model.accumulationCurves

    # no-op
    element_model.removeAccumulationCurve("WaterDepth")
    assert "waterdepth" not in element_model.accumulationCurves


def test_environment_optimum_getAccumulationCurve_raises_for_missing() -> None:
    """Test getAccumulationCurve raises KeyError when curve is missing."""
    element_model = AccumulationModelElementOptimum("sand", 10.0)
    with pytest.raises(KeyError):
        element_model.accumulationCurves["WaterDepth"]


def test_environmentOptimum_getElementAccumulationAt_matchesCurveProduct() -> (
    None
):
    """Migrated behavior test: accumulation equals rate * product(coeffs)."""
    element_model = AccumulationModelElementOptimum(
        "Sand",
        10.0,
        accumulationCurves={
            "WaterDepth": bathy_curve,
            "Energy": energy_curve,
        },
    )
    model = AccumulationModel("EnvModel", {"Sand": element_model})

    conditions = {"WaterDepth": 5.0, "Energy": 0.9}
    rate = model.getElementAccumulationAt("Sand", conditions)
    assert rate == pytest.approx(10.0 * 0.5 * 0.9, rel=1e-12)


def test_environment_optimum_model_with_env_conditions() -> None:
    """Test AccumulationModelEnvironmentOptimum with environment conditions."""
    element_model = AccumulationModelElementOptimum(
        "sand", 100.0, accumulationCurves={"WaterDepth": bathy_curve}
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Test at optimum waterDepth
    env_conditions = {"WaterDepth": 10.0}
    rate = model.getElementAccumulationAt("sand", env_conditions)

    # At optimum (coeff=1.0), rate should equal element.accumulationRate
    assert rate == pytest.approx(100.0, rel=1e-6)


def test_environment_optimum_model_multiple_factors() -> None:
    """Test with multiple environmental factors."""
    element_model = AccumulationModelElementOptimum(
        "sand",
        100.0,
        accumulationCurves={
            "WaterDepth": bathy_curve,
            "Temperature": temp_curve,
            "Age": age_curve,
        },
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Test with both factors at optimum
    age: float = 5.0
    env_conditions = {"WaterDepth": 10.0, "Temperature": 25.0}
    rate = model.getElementAccumulationAt("sand", env_conditions, age)

    # Both coefficients = 1.0, so rate = 100.0 * 1.0 * 1.0
    assert rate == pytest.approx(100.0, rel=1e-6)

    # Test with one factor suboptimal
    env_conditions = {
        "WaterDepth": 10.0,
        "Temperature": 12.5,
    }  # temp coeff = 0.5
    rate = model.getElementAccumulationAt("sand", env_conditions, age)

    # Bathy coeff = 1.0, temp coeff = 0.5, age coeff = 1.0,
    # so rate = 100.0 * 1.0 * 0.5 * 1.0 = 50.0
    assert rate == pytest.approx(50.0, rel=1e-6)

    # Test with age factor suboptimal
    env_conditions = {
        "WaterDepth": 10.0,
        "Temperature": 25.0,
    }  # temp coeff = 1.0
    rate = model.getElementAccumulationAt("sand", env_conditions, age=10.0)

    # Bathy coeff = 1.0, temp coeff = 1.0, age coeff = 0.8,
    # so rate = 100.0 * 1.0 * 1.0 * 0.8 = 80.0
    assert rate == pytest.approx(80.0, rel=1e-6)


def test_environment_optimum_model_ignores_unknown_factors() -> None:
    """Test that unknown environmental factors are ignored."""
    element_model = AccumulationModelElementOptimum(
        "sand", 100.0, accumulationCurves={"WaterDepth": bathy_curve}
    )
    model = AccumulationModel("EnvModel", {"sand": element_model})

    # Provide extra factors not in model
    env_conditions = {
        "WaterDepth": 10.0,
        "Temperature": 25.0,  # Not defined in model
        "Salinity": 35.0,  # Not defined in model
    }
    rate = model.getElementAccumulationAt("sand", env_conditions)

    # Should only use WaterDepth, ignore others
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
            "sand": AccumulationModelElementOptimum(
                "sand",
                100.0,
                accumulationCurves={"WaterDepth": bathy_curve},
            ),
        },
    )
    combination_model = AccumulationModel(
        "CombinationModel",
        {
            "sand": AccumulationModelCombination(
                [
                    AccumulationModelElementOptimum(
                        "sand",
                        100.0,
                        accumulationCurves={"WaterDepth": bathy_curve},
                    ),
                    AccumulationModelElementGaussian("sand", 100.0),
                ]
            )
        },
    )

    env_conditions = {"WaterDepth": 10.0}

    # Both should accept this calling convention
    prob_rate = prob_model.getElementAccumulationAt("sand", env_conditions)
    env_rate = env_model.getElementAccumulationAt("sand", env_conditions)
    comb_rate = combination_model.getElementAccumulationAt(
        "sand", env_conditions
    )
    assert isinstance(prob_rate, float)
    assert isinstance(env_rate, float)
    assert isinstance(comb_rate, float)
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

    # Test with environment optimum model
    env_model = AccumulationModel(
        "EnvModel",
        {
            "sand": AccumulationModelElementOptimum(
                "sand",
                100.0,
                accumulationCurves={"WaterDepth": bathy_curve},
            ),
            "clay": AccumulationModelElementOptimum(
                "clay",
                50.0,
                accumulationCurves={"WaterDepth": bathy_curve},
            ),
        },
    )

    # Without env conditions - should get max accumulation
    results2 = compute_total_accumulation(env_model)
    assert results2["sand"] == pytest.approx(100.0, rel=1e-6)
    assert results2["clay"] == pytest.approx(50.0, rel=1e-6)

    # With env conditions - should work
    env_conditions = {"WaterDepth": 50.0}
    results3 = compute_total_accumulation(env_model, env_conditions)
    assert results3["sand"] == pytest.approx(0.8 * 100.0, rel=1e-6)
    assert results3["clay"] == pytest.approx(0.8 * 50.0, rel=1e-6)

    # Test with environment optimum model
    combination_model = AccumulationModel(
        "CombinationModel",
        {
            "sand": AccumulationModelCombination(
                [
                    AccumulationModelElementOptimum(
                        "sand",
                        100.0,
                        accumulationCurves={"WaterDepth": bathy_curve},
                    ),
                    AccumulationModelElementGaussian("sand", 100.0),
                ]
            ),
            "clay": AccumulationModelCombination(
                [
                    AccumulationModelElementOptimum(
                        "clay",
                        50.0,
                        accumulationCurves={"WaterDepth": bathy_curve},
                    ),
                    AccumulationModelElementGaussian("clay", 50.0),
                ]
            ),
        },
    )

    results4 = compute_total_accumulation(combination_model)
    assert "sand" in results4
    assert "clay" in results4
    assert isinstance(results4["sand"], float)
    assert isinstance(results4["clay"], float)


class _BasePassThrough(AccumulationModelElementBase):
    """Concrete class calling base abstract implementation."""

    def getAccumulationCoefficientAt(
        self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        return super().getAccumulationCoefficientAt(  # type: ignore[safe-super]
            environmentConditions,
            age,
        )


class _ConstantModel(AccumulationModelElementBase):
    """Simple deterministic element model for tests."""

    def __init__(
        self,
        elementName: str,
        accumulationRate: float,
        coeff: float,
    ) -> None:
        """Create test model with fixed coefficient."""
        super().__init__(elementName, accumulationRate)
        self.coeff = coeff

    def getAccumulationCoefficientAt(
        self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Return fixed coefficient."""
        return self.coeff


def test_base_pass_implementation_returns_none() -> None:
    """Base abstract body is executed via super call."""
    model = _BasePassThrough("sand", 10.0)
    assert model.getAccumulationCoefficientAt() is None


def test_combination_empty_models_returns_zero() -> None:
    """Empty combinations are inconsistent and return zero."""
    combination = AccumulationModelCombination([])

    assert combination.checkModelsConsistency() is False
    assert combination.getAccumulationCoefficientAt() == 0.0


def test_combination_sets_defaults_when_names_and_rates_differ() -> None:
    """Inconsistent combinations still set element and reference rate."""
    m1 = _ConstantModel("sand", 10.0, 0.8)
    m2 = _ConstantModel("clay", 20.0, 0.6)

    combination = AccumulationModelCombination([m1, m2])

    assert combination.elementName == "sand"
    assert combination.accumulationRate == pytest.approx(15.0)


def test_combination_returns_zero_when_all_coeffs_near_zero() -> None:
    """Near-zero coefficients force a zero result."""
    m1 = _ConstantModel("sand", 10.0, 1e-7)
    m2 = _ConstantModel("sand", 10.0, 1e-8)
    combination = AccumulationModelCombination([m1, m2])

    assert combination.getAccumulationCoefficientAt() == 0.0


def test_get_element_accumulation_unknown_element_returns_zero() -> None:
    """Unknown elements return zero accumulation."""
    model = AccumulationModel("Test")

    assert model.getElementAccumulationAt("missing") == 0.0


def test_get_total_accumulation_sums_all_elements() -> None:
    """Total accumulation is the sum of each element accumulation."""
    sand = _ConstantModel("sand", 10.0, 0.5)
    clay = _ConstantModel("clay", 20.0, 0.25)
    model = AccumulationModel("Test", {"sand": sand, "clay": clay})

    assert model.getTotalAccumulationAt() == pytest.approx(10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
