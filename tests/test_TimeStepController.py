# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Callable

import numpy as np
import pytest

from pywellsfm.model.FSSimulationParameters import FSSimulatorParameters
from pywellsfm.simulator.TimeStepController import TimeStepController


def _constant_change(
    value: float,
) -> Callable[[float, float, np.ndarray], float]:
    """Return a compute_max_change callback that always returns *value*."""

    def _fn(
        t: float,
        dt: float,
        rates: np.ndarray,
    ) -> float:
        return value

    return _fn


def _sequential_changes(
    *values: float,
) -> Callable[[float, float, np.ndarray], float]:
    """Return a callback that yields *values* in order."""
    it = iter(values)

    def _fn(
        t: float,
        dt: float,
        rates: np.ndarray,
    ) -> float:
        return next(it)

    return _fn


class TestAdaptTimeStep:
    """Tests for TimeStepController.adapt()."""

    def test_uses_dt_hi_when_constraint_is_safe(self) -> None:
        """Adapt picks dt_hi branch and applies safety."""
        params = FSSimulatorParameters(
            max_waterDepth_change_per_step=1.0,
            dt_min=0.1,
            dt_max=1.0,
            safety=0.9,
        )
        ctrl = TimeStepController(params, _constant_change(0.5))

        dt = ctrl.adapt(
            t=30.0,
            curWaterDepths=np.array([10.0]),
            rates=np.array([1.0]),
            remaining=0.8,
        )

        assert dt == pytest.approx(0.72)

    def test_clamps_to_next_exact_age(self) -> None:
        """Adapt clamps dt to hit the next required exact age."""
        params = FSSimulatorParameters(
            max_waterDepth_change_per_step=1.0,
            dt_min=0.1,
            dt_max=1.0,
            safety=1.0,
        )
        ctrl = TimeStepController(params, _constant_change(0.2))
        ctrl.set_exact_ages({29.7})

        dt = ctrl.adapt(
            t=30.0,
            curWaterDepths=np.array([10.0]),
            rates=np.array([1.0]),
            remaining=1.0,
        )

        assert dt == pytest.approx(0.3)

    def test_raises_if_dt_min_cannot_satisfy(self) -> None:
        """Adapt raises with diagnostics when dt_min violates constraints."""
        params = FSSimulatorParameters(
            max_waterDepth_change_per_step=0.1,
            dt_min=0.2,
            dt_max=1.0,
        )
        ctrl = TimeStepController(params, _constant_change(9.0))

        with pytest.raises(
            RuntimeError, match="Cannot satisfy waterDepth-change"
        ):
            ctrl.adapt(
                t=30.0,
                curWaterDepths=np.array([1.0]),
                rates=np.array([1.0]),
                remaining=1.0,
            )

    def test_raises_if_selected_step_still_violates(self) -> None:
        """Adapt raises when final selected dt still breaks constraints."""
        params = FSSimulatorParameters(
            max_waterDepth_change_per_step=1.0,
            dt_min=0.1,
            dt_max=1.0,
            safety=1.0,
        )
        ctrl = TimeStepController(params, _sequential_changes(0.2, 0.2, 1.2))

        with pytest.raises(
            RuntimeError, match="Chosen timestep still violates"
        ):
            ctrl.adapt(
                t=30.0,
                curWaterDepths=np.array([1.0]),
                rates=np.array([1.0]),
                remaining=1.0,
            )

    def test_uses_binary_search_branch(self) -> None:
        """Adapt uses binary search when dt_max violates the constraint."""
        params = FSSimulatorParameters(
            max_waterDepth_change_per_step=1.0,
            dt_min=0.1,
            dt_max=1.0,
            safety=1.0,
        )
        # dt_min check passes (0.1), dt_max check fails (2.0),
        # binary search always returns 0.5 -> final check passes (0.5)
        ctrl = TimeStepController(
            params, _sequential_changes(0.1, 2.0, *([0.5] * 42))
        )

        dt = ctrl.adapt(
            t=30.0,
            curWaterDepths=np.array([1.0]),
            rates=np.array([1.0]),
            remaining=1.0,
        )

        assert 0.1 <= dt <= 1.0


class TestBinarySearch:
    """Tests for TimeStepController._binary_search()."""

    def test_returns_interval_value(self) -> None:
        """Binary search returns a bounded dt."""
        params = FSSimulatorParameters(max_waterDepth_change_per_step=1.0)
        ctrl = TimeStepController(params, _constant_change(0.5))

        dt = ctrl._binary_search(
            t=30.0,
            curWaterDepths=np.array([10.0, 12.0]),
            rates=np.array([1.0, 1.0]),
            dt_lo=0.1,
            dt_hi=0.9,
        )

        assert 0.1 <= dt <= 0.9

    def test_covers_hi_update_branch(self) -> None:
        """Binary search updates upper bound when condition is false."""
        params = FSSimulatorParameters(max_waterDepth_change_per_step=1.0)
        ctrl = TimeStepController(params, _constant_change(2.0))

        dt = ctrl._binary_search(
            t=30.0,
            curWaterDepths=np.array([10.0]),
            rates=np.array([1.0]),
            dt_lo=0.1,
            dt_hi=0.9,
        )

        assert dt == pytest.approx(0.1)


class TestSetExactAges:
    """Tests for TimeStepController.set_exact_ages()."""

    def test_stores_sorted_descending(self) -> None:
        """Exact ages are stored sorted in descending order."""
        params = FSSimulatorParameters()
        ctrl = TimeStepController(params, _constant_change(0.0))

        ctrl.set_exact_ages({10.0, 30.0, 20.0, 5.0})

        assert ctrl._sorted_exact_ages == [30.0, 20.0, 10.0, 5.0]

    def test_resets_index(self) -> None:
        """Index is reset to 0 on each call."""
        params = FSSimulatorParameters()
        ctrl = TimeStepController(params, _constant_change(0.0))

        ctrl.set_exact_ages({10.0})
        ctrl._exact_age_idx = 5
        ctrl.set_exact_ages({20.0, 10.0})

        assert ctrl._exact_age_idx == 0
        assert ctrl._sorted_exact_ages == [20.0, 10.0]

    def test_empty_set(self) -> None:
        """Empty set produces empty list."""
        params = FSSimulatorParameters()
        ctrl = TimeStepController(params, _constant_change(0.0))

        ctrl.set_exact_ages(set())

        assert ctrl._sorted_exact_ages == []
        assert ctrl._exact_age_idx == 0
