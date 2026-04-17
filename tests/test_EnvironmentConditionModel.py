# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import numpy as np
import pytest

from pywellsfm.model.Curve import Curve
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelBase,
    EnvironmentConditionModelCombination,
    EnvironmentConditionModelConstant,
    EnvironmentConditionModelCurve,
    EnvironmentConditionModelGaussian,
    EnvironmentConditionModelStats,
    EnvironmentConditionModelTriangular,
    EnvironmentConditionModelUniform,
    EnvironmentConditionsModel,
)


class _ConcreteBase(EnvironmentConditionModelBase):
    """Concrete helper for abstract-base behavior tests."""

    def getEnvironmentConditionAt(
        self,
        relatedCondition: float | None = None,
    ) -> float:
        return 1.0


class _DeterministicStats(EnvironmentConditionModelStats):
    """Stats helper with deterministic reference and sampled values."""

    def __init__(
        self,
        environmentConditionName: str,
        value: float,
        reference: float,
        minValue: float,
        maxValue: float,
    ) -> None:
        super().__init__(environmentConditionName, minValue, maxValue)
        self._value = value
        self._reference = reference

    def getReferenceValue(self) -> float:
        return self._reference

    def getEnvironmentConditionAt(
        self,
        relatedCondition: float | None = None,
    ) -> float:
        return self._value


def _linear_curve(
    x_name: str,
    y_name: str,
    x_values: list[float],
    y_values: list[float],
) -> Curve:
    """Build a simple linear curve used by model tests."""
    return Curve(
        x_name,
        y_name,
        np.array(x_values, dtype=float),
        np.array(y_values, dtype=float),
        "linear",
    )


def test_abstract_methods_can_be_called_explicitly() -> None:
    """Call abstract methods explicitly to cover base placeholders."""
    base = _ConcreteBase("energy")
    assert (
        EnvironmentConditionModelBase.getEnvironmentConditionAt(base) is None
    )

    stat_model = EnvironmentConditionModelConstant("temperature", 2.0)
    assert EnvironmentConditionModelStats.getReferenceValue(stat_model) is None
    assert (
        EnvironmentConditionModelStats.getEnvironmentConditionAt(stat_model)
        is None
    )


def test_stats_range_property_and_zero_reference_coefficient() -> None:
    """Range helpers and zero-reference coefficient path are supported."""
    model = _DeterministicStats("energy", 5.0, 0.0, 1.0, 9.0)

    assert model.range == (1.0, 9.0)
    assert model.getEnvironmentConditionCoefficientAt() == 5.0


def test_curve_model_requires_related_condition() -> None:
    """Curve models reject missing related condition values."""
    curve = _linear_curve("waterDepth", "oxygen", [0.0, 10.0], [1.0, 0.0])
    model = EnvironmentConditionModelCurve("oxygen", curve)

    with pytest.raises(ValueError, match="relatedCondition"):
        model.getEnvironmentConditionAt()


def test_combination_handles_empty_models_list() -> None:
    """Combination returns false consistency and zero condition if empty."""
    combo = EnvironmentConditionModelCombination([])

    assert combo.checkModelsConsistency() is False
    assert combo.getEnvironmentConditionAt() == 0.0


def test_combination_accepts_mixed_condition_names() -> None:
    """Combination still initializes when condition names differ."""
    m1 = EnvironmentConditionModelConstant("energy", 1.0)
    m2 = EnvironmentConditionModelConstant("salinity", 1.0)

    combo = EnvironmentConditionModelCombination([m1, m2])

    assert combo.envConditionName in {"energy", "salinity"}


def test_combination_without_stats_uses_other_models_mean() -> None:
    """Combination can use mean of non-stat models as reference."""
    c1 = _linear_curve("waterDepth", "energy", [0.0, 10.0], [0.0, 10.0])
    c2 = _linear_curve("waterDepth", "energy", [0.0, 10.0], [10.0, 20.0])
    model1 = EnvironmentConditionModelCurve("energy", c1)
    model2 = EnvironmentConditionModelCurve("energy", c2)

    combo = EnvironmentConditionModelCombination([model1, model2])
    value = combo.getEnvironmentConditionAt(relatedCondition=4.0)

    assert value == pytest.approx(9.0)


def test_combination_without_other_models_uses_stat_reference() -> None:
    """Combination uses stat-model coefficients with stat reference only."""
    s1 = _DeterministicStats("energy", 20.0, 10.0, 0.0, 100.0)
    s2 = _DeterministicStats("energy", 5.0, 10.0, 0.0, 100.0)

    combo = EnvironmentConditionModelCombination([s1, s2])

    assert combo.getEnvironmentConditionAt() == pytest.approx(10.0)


def test_environment_conditions_add_overwrite_and_remove() -> None:
    """Overwrite keeps model present and remove deletes condition."""
    model = EnvironmentConditionsModel()
    model.addEnvironmentConditionModel(
        "energy",
        EnvironmentConditionModelConstant("energy", 1.0),
    )

    model.addEnvironmentConditionModel(
        "energy",
        EnvironmentConditionModelConstant("energy", 2.0),
    )
    assert model.isEnvironmentConditionModelPresent("energy")
    assert model.environmentConditionNames == ["energy"]

    model.removeEnvironmentConditionModel("energy")
    assert not model.isEnvironmentConditionModelPresent("energy")


def test_uniform_model_reference_and_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uniform model returns midpoint reference and sampled value."""
    model = EnvironmentConditionModelUniform("energy", 2.0, 8.0)

    monkeypatch.setattr(np.random, "uniform", lambda a, b: 5.5)

    assert model.getReferenceValue() == pytest.approx(5.0)
    assert model.getEnvironmentConditionAt() == pytest.approx(5.5)


def test_triangular_model_reference_and_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triangular model returns mode reference and sampled value."""
    model = EnvironmentConditionModelTriangular("energy", 4.0, 1.0, 10.0)

    monkeypatch.setattr(np.random, "triangular", lambda a, m, b: 4.5)

    assert model.getReferenceValue() == pytest.approx(4.0)
    assert model.getEnvironmentConditionAt() == pytest.approx(4.5)


def test_gaussian_model_default_std_and_clamping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gaussian model applies default std dev and clamps to bounds."""
    model = EnvironmentConditionModelGaussian(
        "energy",
        meanValue=10.0,
        stdDev=None,
        minValue=0.0,
        maxValue=12.0,
    )

    assert model.stdDev == pytest.approx(2.0)
    assert model.getReferenceValue() == pytest.approx(10.0)

    monkeypatch.setattr(np.random, "normal", lambda mean, std: 15.0)
    assert model.getEnvironmentConditionAt() == pytest.approx(12.0)

    monkeypatch.setattr(np.random, "normal", lambda mean, std: -3.0)
    assert model.getEnvironmentConditionAt() == pytest.approx(0.0)


def test_get_environment_conditions_handles_waterdepth_dependency() -> None:
    """Water-depth dependent conditions are resolved directly."""
    depth_to_oxygen = EnvironmentConditionModelCurve(
        "oxygen",
        _linear_curve("waterDepth", "oxygen", [0.0, 10.0], [1.0, 0.0]),
    )
    model = EnvironmentConditionsModel([depth_to_oxygen])

    values = model.getEnvironmentConditionsAt(waterDepth=2.0)

    assert values["oxygen"] == pytest.approx(0.8)


def test_combination_curve_dependencies_can_be_ambiguous() -> None:
    """Combination with mixed curve sources raises ambiguity error."""
    c_depth = EnvironmentConditionModelCurve(
        "energy",
        _linear_curve("waterDepth", "energy", [0.0, 1.0], [0.0, 1.0]),
    )
    c_age = EnvironmentConditionModelCurve(
        "energy",
        _linear_curve("age", "energy", [0.0, 1.0], [0.0, 1.0]),
    )
    combo = EnvironmentConditionModelCombination([c_depth, c_age])
    model = EnvironmentConditionsModel([combo])

    with pytest.raises(ValueError, match="ambiguous"):
        model.getCurveModelDependencies()


def test_get_environment_conditions_resolves_age_then_dependencies() -> None:
    """Age-dependent condition is resolved before dependent conditions."""
    age_to_salinity = EnvironmentConditionModelCurve(
        "salinity",
        _linear_curve("age", "salinity", [0.0, 10.0], [30.0, 40.0]),
    )
    salinity_to_energy = EnvironmentConditionModelCurve(
        "energy",
        _linear_curve("salinity", "energy", [30.0, 40.0], [1.0, 2.0]),
    )
    temperature = EnvironmentConditionModelConstant("temperature", 22.0)

    model = EnvironmentConditionsModel(
        [temperature, age_to_salinity, salinity_to_energy]
    )
    values = model.getEnvironmentConditionsAt(waterDepth=0.0, age=5.0)

    assert values["temperature"] == pytest.approx(22.0)
    assert values["salinity"] == pytest.approx(35.0)
    assert values["energy"] == pytest.approx(1.5)


def test_get_environment_conditions_raises_on_unresolved_dependency() -> None:
    """Missing source condition raises unresolved dependency error."""
    salinity_to_energy = EnvironmentConditionModelCurve(
        "energy",
        _linear_curve("salinity", "energy", [30.0, 40.0], [1.0, 2.0]),
    )
    model = EnvironmentConditionsModel([salinity_to_energy])

    with pytest.raises(ValueError, match="Unresolved dependencies"):
        model.getEnvironmentConditionsAt(waterDepth=0.0)


def test_curve_dependencies_reject_multiple_sources_for_same_condition() -> (
    None
):
    """Duplicate condition with different sources is rejected."""

    class _DuplicateItemsDict(dict[str, EnvironmentConditionModelBase]):
        def items(self):  # noqa: ANN202
            first = EnvironmentConditionModelCurve(
                "energy",
                _linear_curve("waterDepth", "energy", [0.0, 1.0], [0.0, 1.0]),
            )
            second = EnvironmentConditionModelCurve(
                "energy",
                _linear_curve("age", "energy", [0.0, 1.0], [0.0, 1.0]),
            )
            return [("energy", first), ("energy", second)]

    model = EnvironmentConditionsModel()
    model.envConditionModels = _DuplicateItemsDict()

    with pytest.raises(ValueError, match="multiple source"):
        model.getCurveModelDependencies()
