# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from typing import cast

import numpy as np
import pytest

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

from pywellsfm.model.AccumulationModel import AccumulationModel
from pywellsfm.simulator.AccumulationSimulator import AccumulationSimulator


class _DummyAccumulationModel:
    def __init__(self) -> None:
        self.elements: list[str] = []

    def getElementAccumulationAt(
        self,
        elementName: str,
        environmentConditions: dict[str, float],
        age: float | None,
    ) -> float:
        raise NotImplementedError


def test_init_defaults_to_no_model() -> None:
    """Initialize simulator with no accumulation model."""
    sim = AccumulationSimulator()
    assert sim.accumulationModel is None


def test_set_accumulation_model_assigns_reference() -> None:
    """Attach a model instance to the simulator."""
    sim = AccumulationSimulator()
    model = _DummyAccumulationModel()
    sim.setAccumulationModel(cast(AccumulationModel, model))
    assert sim.accumulationModel is model


def test_prepare_raises_when_model_is_missing() -> None:
    """Raise when prepare is called before setting a model."""
    sim = AccumulationSimulator()
    with pytest.raises(ValueError, match="Accumulation model is not set"):
        sim.prepare()


def test_prepare_succeeds_when_model_is_set() -> None:
    """Allow prepare when a model is already configured."""
    sim = AccumulationSimulator()
    sim.setAccumulationModel(
        cast(AccumulationModel, _DummyAccumulationModel())
    )
    sim.prepare()


def test_compute_all_raises_without_model() -> None:
    """Raise on all-element rate query when model is missing."""
    sim = AccumulationSimulator()
    with pytest.raises(
        ValueError, match="Accumulation model is not set in the simulator"
    ):
        sim.computeAccumulationRatesForAllElements({"depth": 10.0})


def test_compute_all_delegates_to_model_for_each_element() -> None:
    """Delegate all-element rates to model with expected arguments."""

    class _Model(_DummyAccumulationModel):
        def __init__(self) -> None:
            self.elements = ["sand", "shale"]

        def getElementAccumulationAt(
            self,
            elementName: str,
            environmentConditions: dict[str, float],
            age: float | None,
        ) -> float:
            assert environmentConditions == {"depth": 120.0}
            assert age == 8.5
            return {"sand": 2.0, "shale": 3.5}[elementName]

    sim = AccumulationSimulator()
    sim.setAccumulationModel(cast(AccumulationModel, _Model()))

    rates = sim.computeAccumulationRatesForAllElements(
        {"depth": 120.0}, age=8.5
    )

    assert rates == {"sand": 2.0, "shale": 3.5}


def test_compute_all_sets_nan_when_model_raises_value_error() -> None:
    """Set NaN for elements whose model computation raises ValueError."""

    class _Model(_DummyAccumulationModel):
        def __init__(self) -> None:
            self.elements = ["sand", "shale"]

        def getElementAccumulationAt(
            self,
            elementName: str,
            environmentConditions: dict[str, float],
            age: float | None,
        ) -> float:
            if elementName == "sand":
                return 1.0
            raise ValueError("missing environment condition")

    sim = AccumulationSimulator()
    sim.setAccumulationModel(cast(AccumulationModel, _Model()))

    rates = sim.computeAccumulationRatesForAllElements({"depth": 40.0})

    assert rates["sand"] == 1.0
    assert np.isnan(rates["shale"])


def test_compute_element_rate_raises_without_model() -> None:
    """Raise on single-element rate query when model is missing."""
    sim = AccumulationSimulator()
    with pytest.raises(
        ValueError, match="Accumulation model is not set in the simulator"
    ):
        sim.computeElementAccumulationRate("sand", {"depth": 12.0})


def test_compute_element_rate_delegates_to_model() -> None:
    """Delegate single-element rate computation to model API."""

    class _Model(_DummyAccumulationModel):
        def __init__(self) -> None:
            self.elements = ["sand"]

        def getElementAccumulationAt(
            self,
            elementName: str,
            environmentConditions: dict[str, float],
            age: float | None,
        ) -> float:
            assert elementName == "sand"
            assert environmentConditions == {"depth": 15.0}
            assert age == 4.0
            return 2.25

    sim = AccumulationSimulator()
    sim.setAccumulationModel(cast(AccumulationModel, _Model()))

    rate = sim.computeElementAccumulationRate("sand", {"depth": 15.0}, age=4.0)

    assert rate == 2.25


def test_compute_total_rate_is_nan_safe_sum_of_all_elements() -> None:
    """Sum all-element rates while ignoring NaN values."""

    class _Model(_DummyAccumulationModel):
        def __init__(self) -> None:
            self.elements = ["sand", "shale", "lime"]

        def getElementAccumulationAt(
            self,
            elementName: str,
            environmentConditions: dict[str, float],
            age: float | None,
        ) -> float:
            if elementName == "sand":
                return 1.5
            if elementName == "shale":
                raise ValueError("missing condition")
            return 2.0

    sim = AccumulationSimulator()
    sim.setAccumulationModel(cast(AccumulationModel, _Model()))

    total = sim.computeTotalAccumulationRate({"depth": 100.0}, age=2.0)

    assert total == 3.5
