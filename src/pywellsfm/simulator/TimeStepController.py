# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from collections.abc import Callable
from typing import Optional, Self

import numpy as np
import numpy.typing as npt

from pywellsfm.model.FSSimulationParameters import FSSimulatorParameters
from pywellsfm.utils import get_logger

logger = get_logger(__name__)


class TimeStepController:
    """Domain-agnostic adaptive time-step controller.

    Chooses an optimal time step duration such that a user-defined
    constraint (maximum change per step) is not exceeded. The actual
    evaluation of the constraint is delegated to a callback so that
    the controller remains independent of the geological domain.
    """

    def __init__(
        self: Self,
        params: FSSimulatorParameters,
        compute_max_change: Callable[
            [float, float, npt.NDArray[np.float64]], float
        ],
    ) -> None:
        """Initialize the TimeStepController.

        :param FSSimulatorParameters params: stepping parameters.
        :param compute_max_change: callback ``(t, dt, rates)`` that
            returns the maximum state change for a candidate
            time step ``dt`` at time ``t``.
        """
        self.params = params
        self._compute_max_change = compute_max_change
        self._sorted_exact_ages: list[float] = []
        self._exact_age_idx: int = 0

    def set_exact_ages(self: Self, exactAges: set[float]) -> None:
        """Pre-sort exact ages in descending order for efficient lookup.

        Must be called before the first call to :meth:`adapt`.

        :param set[float] exactAges: exact ages to include.
        """
        self._sorted_exact_ages = sorted(exactAges, reverse=True)
        self._exact_age_idx = 0

    def adapt(
        self: Self,
        t: float,
        curWaterDepths: npt.NDArray[np.float64],
        rates: npt.NDArray[np.float64],
        remaining: float,
    ) -> float:
        """Choose an adaptive time step based on the change constraint.

        :param float t: current time.
        :param npt.NDArray[np.float64] curWaterDepths: current water
            depth for each realization.
        :param npt.NDArray[np.float64] rates: deposition rates for
            each realization.
        :param float remaining: remaining time to stop.
        :return float: chosen time step.
        """
        # Advance index past ages >= t (already visited)
        ages = self._sorted_exact_ages
        while (
            self._exact_age_idx < len(ages) and ages[self._exact_age_idx] >= t
        ):
            self._exact_age_idx += 1
        nextExactAge: Optional[float] = (
            ages[self._exact_age_idx]
            if self._exact_age_idx < len(ages)
            else None
        )

        # set dt max as the minimum of user-defined dt_max and the dt
        # that would cause max deposition (use max deposition rate as
        # proxy)
        # TODO: to improve, better evaluate max deposition rate (need
        # to cumulate over waterDepth range)
        dt_max = min(
            self.params.dt_max,
            self.params.max_waterDepth_change_per_step
            / float(np.nanmax(rates)),
        )
        dt_hi = float(min(dt_max, remaining))
        dt_lo = float(min(self.params.dt_min, dt_hi))
        max_change = self.params.max_waterDepth_change_per_step

        # Check if constraint can be satisfied at dt_min
        max_change_at_dt_min = self._compute_max_change(t, dt_lo, rates)
        if max_change_at_dt_min > max_change:
            raise RuntimeError(
                "Cannot satisfy waterDepth-change constraint"
                " at dt_min.\n"
                f"  - time: {t:.6f} Myr,"
                f" dt_min: {dt_lo:.6g} Myr\n"
                f"  - allowed max change: {max_change:.6g} m\n"
                "  - computed max change at dt_min: "
                f"{max_change_at_dt_min:.6g} m\n"
                "Possible causes:\n"
                "  1. Discontinuous forcing curve (e.g.,"
                " step/lower-bound "
                "interpolation causing jumps).\n"
                "  2. Forcing or accumulation rates too"
                " strong for current "
                "constraints.\n"
                "  3. Units mismatch between rates"
                " (m/Myr), curves, and "
                "timestep (Myr).\n"
                "Try: decreasing dt_min, smoothing input"
                " curves "
                "(e.g., linear interpolation), increasing "
                "max_waterDepth_change_per_step, or"
                " reducing "
                "forcing/rates."
            )

        # Check if dt_max satisfies constraint
        max_change_at_dt_max = self._compute_max_change(t, dt_hi, rates)
        if max_change_at_dt_max <= max_change:
            dt = dt_hi
        else:
            # Binary search for optimal dt between dt_lo and dt_hi
            dt = self._binary_search(t, curWaterDepths, rates, dt_lo, dt_hi)

        # Apply safety factor
        dt = float(
            max(
                self.params.dt_min,
                min(dt * self.params.safety, dt_hi),
            )
        )

        # clamp dt to not overshoot next exact age
        if nextExactAge is not None and dt > t - nextExactAge:
            dt = t - nextExactAge

        # Final safety check to ensure chosen dt still satisfies
        # constraint.
        max_change_at_selected_dt = self._compute_max_change(t, dt, rates)
        if max_change_at_selected_dt > max_change:
            raise RuntimeError(
                "Chosen timestep still violates"
                " waterDepth-change "
                "constraint.\n"
                f"  - time: {t:.6f} Myr\n"
                f"  - dt_min: {dt_lo:.6g} Myr,"
                f" dt_max: {dt_hi:.6g} Myr\n"
                f"  - selected dt: {dt:.6g} Myr\n"
                f"  - allowed max change: {max_change:.6g} m\n"
                f"  - change at dt_max:"
                f" {max_change_at_dt_max:.6g} m\n"
                "  - change at selected dt: "
                f"{max_change_at_selected_dt:.6g} m\n"
                "Possible causes:\n"
                "  1. Discontinuous forcing curve (e.g.,"
                " step/lower-bound "
                "interpolation causing jumps).\n"
                "  2. Constraint too strict for current"
                " forcing/rates.\n"
                "  3. Numerical edge case near dt bounds"
                " or remaining time.\n"
                "Try: smoothing input curves (e.g., linear"
                " interpolation), "
                "increasing"
                " max_waterDepth_change_per_step, reducing "
                "forcing/rates, or lowering dt_max."
            )
        return dt

    def _binary_search(
        self: Self,
        t: float,
        curWaterDepths: npt.NDArray[np.float64],
        rates: npt.NDArray[np.float64],
        dt_lo: float,
        dt_hi: float,
    ) -> float:
        """Binary search for optimal time step.

        :param float t: current time.
        :param npt.NDArray[np.float64] curWaterDepths: current water
            depth for each realization.
        :param npt.NDArray[np.float64] rates: deposition rates at
            time t.
        :param float dt_lo: lower bound for time step.
        :param float dt_hi: upper bound for time step.
        :return float: optimal time step.
        """
        max_change = float(self.params.max_waterDepth_change_per_step)
        lo = dt_lo
        hi = dt_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            # check no change from offshore to onshore or vice versa
            # (i.e., no sign change in water depth)
            deltaWD = self._compute_max_change(t, mid, rates)
            newWds = curWaterDepths + deltaWD
            changeWaterDepthSign = np.all(
                np.sign(newWds) == np.sign(curWaterDepths)
            )
            if deltaWD <= max_change and changeWaterDepthSign:
                lo = mid
            else:
                hi = mid
        return lo
