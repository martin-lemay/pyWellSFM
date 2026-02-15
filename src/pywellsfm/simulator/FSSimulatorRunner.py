# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Callable, Mapping, Optional, Self, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr
from attr import dataclass

from pywellsfm.model.Marker import Marker
from pywellsfm.model.SimulationParameters import RealizationData, Scenario
from pywellsfm.simulator.AccommodationSimulator import AccommodationSimulator
from pywellsfm.simulator.FSSimulator import FSSimulator


@dataclass
class FSSimulatorRunnerData:
    #: scenario to simulate
    scenario: Scenario
    #: list of realization data, one per realization
    realizationDataList: list[RealizationData]
    #: maximum bathymetry change and accumulated thickness
    #: per step (in meters). Default is 0.5 m.
    max_bathymetry_change_per_step: float = 0.5
    #: minimum time step (in Myr). Default is 1e-3 Myr.
    dt_min: float = 1e-3
    #: maximum time step (in Myr). Default is 0.1 Myr.
    dt_max: float = 0.1
    #: safety factor for time step adjustment. Must be in (0, 1].
    #: Default is 0.9.
    safety: float = 0.9
    #: maximum number of steps. Default is 1e9.
    max_steps: int = int(1e9)


class FSSimulatorRunner:
    def __init__(
        self: Self,
        scenario: Scenario,
        realizationDataList: list[RealizationData],
        max_bathymetry_change_per_step: float = 0.5,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        safety: float = 0.9,
        max_steps: int = int(1e9),
    ) -> None:
        """Defines a Forward Stratigraphic Simulator runner.

        The FS simulator runner is used to run one or multiple realizations of
        a scenario over time using adaptive time steps. Time step is adjusted
        such as accumulated thickness and bathymetry change do not exceed the
        given threshold (by default 0.5m).

        The FS simulator runner also manages uncertainties over multiple
        realizations.

        :param Scenario scenario: scenario to simulate.
        :param list[RealizationData] realizationDataList: list of realization
            data.
        :param float max_bathymetry_change_per_step: maximum bathymetry change
            and  accumulated thickness per step. Unit is meters.
            Default is 0.5 m.
        :param float dt_min: minimum time step. Unit is Myr.
            Default is 1e-3 Myr.
        :param float dt_max: maximum time step. Unit is Myr.
            Default is 0.1 Myr.
        :param float safety: safety factor for time step adjustment.
            Must be in (0, 1]. Default is 0.9.
        :param int max_steps: maximum number of steps. Default is 1e9.
        """
        self.scenario: Scenario = scenario
        self.realizationDataList: list[RealizationData] = realizationDataList
        self.fsSimulators: list[FSSimulator] = []

        # sea level is the same for all realizations, so we use the first
        # FS simulator's accommodation simulator
        self.seaLevelSimulator: AccommodationSimulator

        # Adaptive step configuration
        self.n_real: int = len(
            self.realizationDataList
        )  # number of realizations
        self.max_bathymetry_change_per_step: float = (
            max_bathymetry_change_per_step
        )
        self.dt_min: float = dt_min
        self.dt_max: float = dt_max
        self.safety: float = safety
        self.max_steps: int = max_steps

        # State tracking for ensemble results
        self.times: list[float] = []
        self.sea_levels: list[npt.NDArray[np.float64]] = []
        self.subsidences: list[npt.NDArray[np.float64]] = []
        self.basements: list[npt.NDArray[np.float64]] = []
        self.accommodations: list[npt.NDArray[np.float64]] = []
        self.depo_rate_totals: list[npt.NDArray[np.float64]] = []
        self.thickness_steps: list[npt.NDArray[np.float64]] = []
        self.thickness_cumul: list[npt.NDArray[np.float64]] = []
        self.bathymetries: list[npt.NDArray[np.float64]] = []
        self.delta_bathymetries: list[npt.NDArray[np.float64]] = []
        self.dts: list[float] = []
        self.initial_bathymetries: Optional[npt.NDArray[np.float64]] = None

        self.outputs: Optional[xr.Dataset] = None

        # Optional environment function
        self.env_fn: Optional[
            Callable[
                [Mapping[str, Any]],
                dict[str, float] | Sequence[dict[str, float]],
            ]
        ] = None

    def setEnvironmentFunction(
        self: Self,
        env_fn: Callable[
            [Mapping[str, Any]], dict[str, float] | Sequence[dict[str, float]]
        ],
    ) -> None:
        """Set the environment function for computing environmental conditions.

        :param env_fn: callback to compute environment conditions from state.
        """
        self.env_fn = env_fn

    def prepare(self: Self) -> None:
        """Prepare the FS simulator for running."""
        # Validate configuration
        if self.max_bathymetry_change_per_step <= 0:
            raise ValueError("max_bathymetry_change_per_step must be > 0")
        if self.dt_min <= 0 or self.dt_max <= 0:
            raise ValueError("dt_min and dt_max must be > 0")
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if not (0.0 < self.safety <= 1.0):
            raise ValueError("safety must be in (0, 1]")

        # Create FSSimulator instances for each realization
        self.fsSimulators = [
            FSSimulator(self.scenario, realizationData)
            for realizationData in self.realizationDataList
        ]

        # Prepare all simulators
        for fsSimulator in self.fsSimulators:
            fsSimulator.prepare()

        # Extract initial bathymetries
        # Initial bathymetry = -topographyStart (since topography=-bathymetry)
        self.initial_bathymetries = np.array(
            [
                realizationData.initialBathymetry
                for realizationData in self.realizationDataList
            ],
            dtype=np.float64,
        )

        # Use the first simulator's accommodation simulator for sea level
        self.seaLevelSimulator = self.fsSimulators[0].accommodationSimulator

        # Reset state tracking
        self.times = []
        self.sea_levels = []
        self.subsidences = []
        self.basements = []
        self.accommodations = []
        self.depo_rate_totals = []
        self.thickness_steps = []
        self.thickness_cumul = []
        self.bathymetries = []
        self.delta_bathymetries = []
        self.dts = []

    def getStartAge(self: Self, markerStart: Optional[Marker] = None) -> float:
        """Get the start age of the simulation.

        :return float: start age.
        """
        return min(
            fsSimulator.getFirstMarkerAge()
            for fsSimulator in self.fsSimulators
        )

    def getAgeEnd(self: Self, markerEnd: Optional[Marker]) -> float:
        """Get the end age of the simulation.

        :return float: end age.
        """
        if markerEnd is None:
            # Get the maximum age across all realizations
            return max(
                fsSimulator.getLastMarkerAge()
                for fsSimulator in self.fsSimulators
            )
        return markerEnd.age

    def run(self: Self, markerEnd: Optional[Marker] = None) -> None:
        """Run the FS simulator until a given marker or to the top.

        Time decreases from start to stop (e.g., from 100 Myr to 0 Myr).

        :param Optional[Marker] markerEnd: marker until which to run the
            simulation. If None, the simulation runs to the top of wells.
        """
        if not self.fsSimulators:
            raise RuntimeError("Must call prepare() before run()")

        # Determine start and stop times
        # absolute geological age (in Myr) at start of simulation
        start = self.getStartAge()
        # absolute geological age (in Myr) at end of simulation
        stop: float = self.getAgeEnd(markerEnd)
        if stop >= start:
            raise ValueError("stop must be < start")

        # Initialize bathymetry state
        if self.initial_bathymetries is None:
            raise RuntimeError("initial_bathymetries not set")
        bathy_t = self.initial_bathymetries.copy()

        # Set initial time
        self.times.append(start)
        sea_level_t = 0.0
        subs_t = np.zeros(self.n_real, dtype=np.float64)

        t = start
        for _ in range(self.max_steps):
            if t <= stop:
                break

            print(f"Running time step at age {t:.4f} over {stop:.4f} Myr...")

            # Build state for environment function
            # may add energy, temperature, or other conditions in the future
            state: dict[str, Any] = {
                "time": float(t),
                "bathymetry": bathy_t,
            }

            # Get environment conditions
            env_raw = self.env_fn(state) if self.env_fn is not None else {}
            env_list = self._as_env_per_realization(env_raw, self.n_real)

            # Compute deposition rates for each realization
            accumulationRates = np.array(
                [
                    self.fsSimulators[i].getTotalAccumulationRate(env_list[i])
                    for i in range(self.n_real)
                ],
                dtype=np.float64,
            )

            # Choose adaptive time step
            remaining = t - stop

            dt = self._adaptTimeStep(t, accumulationRates, remaining=remaining)

            t2 = t - dt
            # Get sea level variation between t1 and t2
            delta_sea_level_t = self._getDeltaSeaLevel(t, t2)
            sea_level_t += delta_sea_level_t

            # Get subsidence variation between t1 and t2 for all realizations
            delta_subs_t = self._getDeltaSubsidence(t, t2)
            subs_t += delta_subs_t

            # positive subsidence means sinking
            basement_t = (-self.initial_bathymetries) - subs_t
            acco_t = subs_t + sea_level_t

            # Compute thickness and bathymetry change
            thickness_step = self._getAccumulatedThickness(
                accumulationRates, dt
            )
            delta_bathy = self._getBathymetryVariation(
                delta_sea_level_t, delta_subs_t, thickness_step
            )
            delta_bathy_array: npt.NDArray[np.float64]
            if isinstance(delta_bathy, np.ndarray):
                delta_bathy_array = delta_bathy.copy()
            else:
                delta_bathy_array = np.full(
                    (self.n_real,), delta_bathy, dtype=np.float64
                )

            # Record state at time t (step-start)
            self.bathymetries.append(bathy_t.copy())
            self.thickness_steps.append(thickness_step)
            self.thickness_cumul.append(
                thickness_step
                if not self.thickness_cumul
                else self.thickness_cumul[-1] + thickness_step
            )
            self.delta_bathymetries.append(delta_bathy_array)
            self.sea_levels.append(
                np.full((self.n_real,), sea_level_t, dtype=np.float64)
            )
            self.subsidences.append(subs_t.copy())
            self.basements.append(basement_t)
            self.accommodations.append(acco_t)
            self.depo_rate_totals.append(accumulationRates)
            self.dts.append(dt)

            # Update bathymetry for next step
            bathy_t = bathy_t + delta_bathy_array

            # Advance time
            t -= dt
            self.times.append(t)
        else:
            raise RuntimeError("Reached max_steps without reaching stop")

        print(
            f"Finished simulation at age {t:.4f} Myr after "
            + f"{len(self.times) - 1} steps."
        )

    def _adaptTimeStep(
        self: Self, t: float, rates: npt.NDArray[np.float64], remaining: float
    ) -> float:
        """Choose an adaptive time step based on bathymetry change constraint.

        Max time step duration is limited by the maximum accumulated thickness
        and bathymetry change allowed.

        :param float t: current time.
        :param npt.NDArray[np.float64] rates: deposition rates for each
            realization.
        :param float remaining: remaining time to stop.
        :return float: chosen time step.
        """
        dt_max = min(
            self.dt_max,
            self.max_bathymetry_change_per_step / float(np.nanmax(rates)),
        )
        dt_hi = float(min(dt_max, remaining))
        dt_lo = float(min(self.dt_min, dt_hi))
        max_change = self.max_bathymetry_change_per_step

        # Check if constraint can be satisfied at dt_min
        if self._computeMaxBathymetryChange(t, rates, dt_lo) > max_change:
            raise RuntimeError(
                "Cannot satisfy bathymetry-change constraint at dt_min. "
                "Increase dt_min, increase max_bathymetry_change_per_step, "
                "or reduce forcing/rates."
            )

        # Check if dt_max satisfies constraint
        if self._computeMaxBathymetryChange(t, rates, dt_hi) <= max_change:
            dt = dt_hi
        else:
            # Binary search for optimal dt between dt_lo and dt_hi
            dt = self._binarySearchForOptimalDt(t, rates, dt_lo, dt_hi)

        # Apply safety factor
        dt = float(max(self.dt_min, min(dt * self.safety, dt_hi)))
        return dt

    def _getDeltaSeaLevel(self: Self, t1: float, t2: float) -> float:
        """Get the change in sea level between t1 and t2.

        :param float t1: start time.
        :param float t2: end time.
        :return float: change in sea level (sea level at t2 - sea level at t1).
        """
        sea_level_t1 = self.seaLevelSimulator.getSeaLevelAt(t1)
        sea_level_t2 = self.seaLevelSimulator.getSeaLevelAt(t2)
        return sea_level_t2 - sea_level_t1

    def _getDeltaSubsidence(
        self: Self, t1: float, t2: float
    ) -> npt.NDArray[np.float64]:
        """Get the change in subsidence between t1 and t2 for all realizations.

        :param float t1: start time.
        :param float t2: end time.
        :return npt.NDArray[np.float64]: change in subsidence (subsidence at
            t2 - subsidence at t1) for all realizations.
        """
        delta_subs = np.zeros((self.n_real,), dtype=np.float64)
        for i, fsSimulator in enumerate(self.fsSimulators):
            subs_t1 = fsSimulator.getSubsidenceAtAge(t1)
            subsType = fsSimulator.getSubsidenceType()
            if subsType == "cumulative":
                subs_t2 = fsSimulator.getSubsidenceAtAge(t2)
                delta_subs[i] = subs_t2 - subs_t1
            elif subsType == "rate":
                delta_subs[i] = subs_t1 * (t2 - t1)
            else:
                raise ValueError(f"Unknown subsidence type: {subsType}")
        return delta_subs

    def _computeMaxBathymetryChange(
        self: Self, t1: float, rates: npt.NDArray[np.float64], dt: float
    ) -> float:
        """Compute maximum bathymetry change across realizations for a dt.

        :param float t1: start time of the step.
        :param npt.NDArray[np.float64] rates: deposition rates at t1.
        :param float dt: candidate time step.
        :return float: maximum absolute bathymetry change.
        """
        t2 = float(t1 + dt)

        # Get sea level variation between t1 and t2 (same for all realizations)
        delta_sea_level = self._getDeltaSeaLevel(t1, t2)

        # Get subsidence variation between t1 and t2 for all realizations
        delta_subsidences = self._getDeltaSubsidence(t1, t2)

        # Compute accumulated thickness between t1 and t2 for all realizations
        thicknesses_step = self._getAccumulatedThickness(rates, dt)

        # Compute bathymetry change
        d_bathy = self._getBathymetryVariation(
            delta_sea_level,
            delta_subsidences,
            thicknesses_step,
        )

        # Return max absolute change
        finite = d_bathy[np.isfinite(d_bathy)]  # type: ignore
        if finite.size == 0:
            return float("inf")
        return float(np.max(np.abs(finite)))

    def _binarySearchForOptimalDt(
        self: Self,
        t: float,
        rates: npt.NDArray[np.float64],
        dt_lo: float,
        dt_hi: float,
    ) -> float:
        """Binary search for optimal time step between dt_lo and dt_hi.

        :param float t: current time.
        :param npt.NDArray[np.float64] rates: deposition rates at time t.
        :param float dt_lo: lower bound for time step.
        :param float dt_hi: upper bound for time step.
        :return float: optimal time step.
        """
        max_change = float(self.max_bathymetry_change_per_step)
        lo = dt_lo
        hi = dt_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._computeMaxBathymetryChange(t, rates, mid) <= max_change:
                lo = mid
            else:
                hi = mid
        return lo

    @staticmethod
    def _as_env_per_realization(
        env: dict[str, float] | Sequence[dict[str, float]],
        n_real: int,
    ) -> list[dict[str, float]]:
        """Normalize environment conditions to a per-realization list.

        :param env: environment conditions (single dict or sequence).
        :param int n_real: number of realizations.
        :return list[dict[str, float]]: list of environment dicts.
        """
        if isinstance(env, Mapping):
            return [dict(env) for _ in range(int(n_real))]
        if len(env) != n_real:
            raise ValueError(
                "env sequence length must match number of realizations"
            )
        return [dict(e) for e in env]

    def finalize(self: Self) -> None:
        """Finalize the FS simulator after running."""
        # create output xarray Dataset
        self.outputs = self._buildEnsembleDataset()

        # create simulated wells
        for fsSimulator in self.fsSimulators:
            fsSimulator.finalize()

    def _buildEnsembleDataset(self: Self) -> xr.Dataset:
        """Build the ensemble dataset after running all realizations.

        :return xr.Dataset: xarray.Dataset containing ensemble results.
        """
        if not self.times:
            raise RuntimeError(
                "Must call run() before _buildEnsembleDataset()"
            )

        n_real = len(self.fsSimulators)

        # Convert lists to arrays
        # Note: we stored state at each step-start; last `times` entry has
        # no state.
        time_axis = np.array(self.times[:-1], dtype=np.float64)
        dt_axis = np.array(self.dts, dtype=np.float64)

        sea_level_arr = np.stack(self.sea_levels, axis=1)  # (real, time)
        subs_arr = np.stack(self.subsidences, axis=1)
        basement_arr = np.stack(self.basements, axis=1)
        acco_arr = np.stack(self.accommodations, axis=1)
        depo_arr = np.stack(self.depo_rate_totals, axis=1)
        thickness_step_arr = np.stack(self.thickness_steps, axis=1)
        thickness_cumul_arr = np.stack(self.thickness_cumul, axis=1)
        bathy_arr = np.stack(self.bathymetries, axis=1)
        delta_bathy_arr = np.stack(self.delta_bathymetries, axis=1)

        # Build realization IDs
        realization_ids = np.arange(n_real, dtype=np.int64)

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                "sea_level": (("time",), sea_level_arr[0, :]),
                "dt": (("time",), dt_axis),
                "initial_bathymetry": (
                    ("realization",),
                    self.initial_bathymetries,
                ),
                "subsidence": (("realization", "time"), subs_arr),
                "basement": (("realization", "time"), basement_arr),
                "accommodation": (("realization", "time"), acco_arr),
                "depo_rate_total": (("realization", "time"), depo_arr),
                "thickness_step": (
                    ("realization", "time"),
                    thickness_step_arr,
                ),
                "thickness_cumul": (
                    ("realization", "time"),
                    thickness_cumul_arr,
                ),
                "bathymetry": (("realization", "time"), bathy_arr),
                "delta_bathymetry": (("realization", "time"), delta_bathy_arr),
            },
            coords={
                "time": (("time",), time_axis),
                "realization": (("realization",), realization_ids),
            },
            attrs={
                "scenario_name": self.scenario.name,
                "start": float(self.times[0]),
                "stop": float(self.times[-1]),
                "max_bathymetry_change_per_step": float(
                    self.max_bathymetry_change_per_step
                ),
            },
        )

        return ds

    @staticmethod
    def _getAccumulatedThickness(
        accumulationRate: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        """Get the deposited thickness at a step given the accumulation rate.

        :param npt.NDArray[np.float64] accumulationRate: accumulation rate
            (e.g., in m/Myr).
        :param float dt: time step duration (e.g., in Myr).
        :return npt.NDArray[np.float64]: deposited thickness over the
            time step.
        """
        return accumulationRate * dt

    @staticmethod
    def _getBathymetryVariation(
        delta_seaLevel: float,
        delta_subs: npt.NDArray[np.float64],
        thickness_step: npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        """Compute bathymetry variation over a time step.

        :param float delta_seaLevel: change in sea level over the time step.
        :param npt.NDArray[np.float64] delta_subs: change in subsidence of each
            realization over the time step.
        :param npt.NDArray[np.float64] thickness_step: deposited thickness of
            each realization over the time step.
        :return float | npt.NDArray[np.float64]: bathymetry variation over the
            time step.
        """
        return delta_seaLevel + delta_subs - thickness_step
