# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import xarray as xr
from attr import dataclass

from pywellsfm.model.DepositionalEnvironment import DepositionalEnvironment
from pywellsfm.model.FSSimulationParameters import RealizationData, Scenario
from pywellsfm.model.Marker import Marker

from .AccommodationSimulator import AccommodationSimulator
from .DepositionalEnvironmentSimulator import (
    DepositionalEnvironmentSimulator,
    DESimulatorParameters,
)
from .Realization import Realization


@dataclass
class FSSimulatorParameters:
    """Parameters for the Forward Stratigraphic Simulator (FSSimulator)."""

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


class FSSimulator:
    def __init__(
        self: Self,
        scenario: Scenario,
        realizationDataList: list[RealizationData],
        use_depositional_environment_simulator: bool = False,
        deSimulator_weights: dict[str, float] | None = None,
        deSimulator_params: DESimulatorParameters | None = None,
        fsSimulator_params: FSSimulatorParameters = FSSimulatorParameters(),
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
        :param bool use_depositional_environment_simulator: whether to use the
            depositional environment simulator. Default is False.
        :param dict[str, float] | None deSimulator_weights: weights for the
            depositional environment simulator. Keys are depositional
            environment names, values are weights. If None, all environments
            have equal weight. Default is None.
        :param DESimulatorParameters | None deSimulator_params: parameters for
            the depositional environment simulator. If None, default parameters
            are used. Default is None.
        :param FSSimulatorParameters fsSimulator_params: parameters for the FS
            simulator.
        """
        self.scenario: Scenario = scenario
        self.realizationDataList: list[RealizationData] = realizationDataList
        self.fsSimulators: list[Realization] = []

        # sea level is the same for all realizations, so we use the first
        # FS simulator's accommodation simulator
        self.seaLevelSimulator: AccommodationSimulator

        # depositional environment simulator is set when a depositional
        # environment model is provided in the scenario
        self.depositionalEnvironmentSimulator: Optional[
            DepositionalEnvironmentSimulator
        ] = None
        self.use_deSimulator = use_depositional_environment_simulator
        self.deSimulator_weights = deSimulator_weights
        self.deSimulator_params = deSimulator_params

        # Adaptative step configuration
        self.n_real: int = len(
            self.realizationDataList
        )  # number of realizations
        self.params: FSSimulatorParameters = fsSimulator_params

        # State tracking for ensemble results
        self.times: list[float] = []
        self.sea_levels: list[npt.NDArray[np.float64]] = []
        self.subsidences: list[npt.NDArray[np.float64]] = []
        self.basements: list[npt.NDArray[np.float64]] = []
        self.accommodations: list[npt.NDArray[np.float64]] = []
        self.depo_rate_totals: list[npt.NDArray[np.float64]] = []
        self.depo_rate_elements: list[list[dict[str, float]]] = []
        self.thickness_steps: list[npt.NDArray[np.float64]] = []
        self.thickness_cumul: list[npt.NDArray[np.float64]] = []
        self.bathymetries: list[npt.NDArray[np.float64]] = []
        self.delta_bathymetries: list[npt.NDArray[np.float64]] = []
        self.environments: list[npt.NDArray[np.str_]] = []
        self.dts: list[float] = []
        self.initial_bathymetries: Optional[npt.NDArray[np.float64]] = None

        self.outputs: Optional[xr.Dataset] = None

    def prepare(self: Self) -> None:
        """Prepare the FS simulator for running."""
        # Validate configuration
        if self.params.max_bathymetry_change_per_step <= 0:
            raise ValueError("max_bathymetry_change_per_step must be > 0")
        if self.params.dt_min <= 0 or self.params.dt_max <= 0:
            raise ValueError("dt_min and dt_max must be > 0")
        if self.params.dt_min > self.params.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if not (0.0 < self.params.safety <= 1.0):
            raise ValueError("safety must be in (0, 1]")

        # Create FSSimulator instances for each realization
        self.fsSimulators = [
            Realization(self.scenario, realizationData)
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

        # Initialize depositional environment simulator if needed
        deModel = self.scenario.depositionalEnvironmentModel
        if self.use_deSimulator and (deModel is not None):
            self.depositionalEnvironmentSimulator = (
                DepositionalEnvironmentSimulator(
                    deModel,
                    params=self.deSimulator_params,
                    weights=self.deSimulator_weights,
                )
            )
            self.depositionalEnvironmentSimulator.prepare()

        # Reset state tracking
        self.times = []
        self.sea_levels = []
        self.subsidences = []
        self.basements = []
        self.accommodations = []
        self.depo_rate_totals = []
        self.depo_rate_elements = []
        self.thickness_steps = []
        self.thickness_cumul = []
        self.bathymetries = []
        self.delta_bathymetries = []
        self.environments = []
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

        # Initialize depositional environment state if needed
        depEnv_t: Optional[list[DepositionalEnvironment | None]] = None
        if self.depositionalEnvironmentSimulator is not None:
            depEnv_t = [
                real.initialEnvironment for real in self.realizationDataList
            ]

        # set initial sea level and subsidence values
        sea_level_t = 0.0
        cumul_subs_t = np.zeros(self.n_real, dtype=np.float64)

        # Set initial time
        t = start
        for _ in range(self.params.max_steps):
            if t <= stop:
                break

            print(f"Running time step at age {t:.4f} over {stop:.4f} Myr...")

            # Get environment conditions from depositional environments
            env_list = self._computeEnvironmentalConditions(bathy_t, depEnv_t)

            # Compute deposition rates for each realization
            accumulationRates = np.array(
                [
                    self.fsSimulators[i].getTotalAccumulationRate(env_list[i])
                    for i in range(self.n_real)
                ],
                dtype=np.float64,
            )
            print(f"  Accumulation rates: {accumulationRates}")

            # Choose adaptive time step
            remaining = t - stop

            dt = self._adaptTimeStep(t, accumulationRates, remaining=remaining)

            t2 = t - dt

            # Record state at time t (step-start)
            self.times.append(t)
            # Record time step
            self.dts.append(dt)

            # Record bathymetry for this step
            self.bathymetries.append(bathy_t.copy())

            # Record depositional environment for this step if needed
            if depEnv_t is not None:
                self.environments.append(
                    np.array(
                        [
                            env.name if env is not None else "none"
                            for env in depEnv_t
                        ],
                        dtype=str,
                    )
                )

            # Record deposition rates for this step
            self.depo_rate_totals.append(accumulationRates)

            # element accumulation rates for this step
            accu_rates_step: list[dict[str, float]] = [
                {} for _ in range(self.n_real)
            ]
            for i in range(self.n_real):
                for name in self.scenario.accumulationModel.elements:
                    accuRate = self.fsSimulators[i].getElementAccumulationRate(
                        env_list[i], name
                    )
                    accu_rates_step[i][name] = accuRate
            self.depo_rate_elements.append(accu_rates_step)

            # Compute and record deposited thickness for this step
            thickness_step = self._getAccumulatedThickness(
                accumulationRates, dt
            )
            self.thickness_steps.append(thickness_step)
            self.thickness_cumul.append(
                thickness_step
                if not self.thickness_cumul
                else self.thickness_cumul[-1] + thickness_step
            )

            # Get sea level variation between t and t2
            delta_sea_level_t = self._getDeltaSeaLevel(t, t2)
            sea_level_t += delta_sea_level_t
            self.sea_levels.append(
                np.full((self.n_real,), sea_level_t, dtype=np.float64)
            )

            # Get subsidence variation between t and t2 for all realizations
            delta_subs_t = self._getDeltaSubsidence(t, t2)
            cumul_subs_t += delta_subs_t
            self.subsidences.append(cumul_subs_t.copy())

            # Compute bathymetry change for this step
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
            self.delta_bathymetries.append(delta_bathy_array)

            # compute and record basement elevation
            # (positive subsidence means sinking)
            basement_t = (-self.initial_bathymetries) - cumul_subs_t
            self.basements.append(basement_t)

            # compute and record accommodation
            acco_t = cumul_subs_t + sea_level_t
            self.accommodations.append(acco_t)

            # update state for next step
            # Update bathymetry for next step
            bathy_t = bathy_t + delta_bathy_array

            # update depositional environment for next step if needed
            if self.depositionalEnvironmentSimulator is not None:
                window: int = min(
                    self.depositionalEnvironmentSimulator.params.trend_window,
                    len(self.environments),
                )
                prev_env: list[list[str] | None] = [
                    [] for _ in range(self.n_real)
                ]
                if window > 0:
                    # get the last `window` environments for this realization,
                    # in the correct order (oldest to most recent)
                    for i in range(self.n_real):
                        prev_env[i] = [
                            self.environments[-k][i]
                            for k in range(1, window + 1)
                        ][::-1]
                depEnv_t = self._computeDepositionalEnvironment(
                    bathy_t, prev_env
                )

            # Advance time
            t -= dt
        else:
            raise RuntimeError("Reached max_steps without reaching stop")

        # Record final state at time t (step-end)
        self.times.append(t)
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
        # set dt max as the minimum of user-defined dt_max and the dt that
        # would cause max deposition (use max deposition rate as proxy)
        # TODO: to improve, better evaluate max deposition rate (need to
        # cumulate over bathymetry range)
        dt_max = min(
            self.params.dt_max,
            self.params.max_bathymetry_change_per_step
            / float(np.nanmax(rates)),
        )
        dt_hi = float(min(dt_max, remaining))
        dt_lo = float(min(self.params.dt_min, dt_hi))
        max_change = self.params.max_bathymetry_change_per_step

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
        dt = float(
            max(self.params.dt_min, min(dt * self.params.safety, dt_hi))
        )
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
                delta_subs[i] = subs_t1 * (t1 - t2)
            else:
                raise ValueError(f"Unknown subsidence type: {subsType}")
        return delta_subs

    def _computeMaxBathymetryChange(
        self: Self, t1: float, rates: npt.NDArray[np.float64], dt: float
    ) -> float:
        """Compute maximum bathymetry change across realizations for a dt.

        :param float t1: start time of the step.
        :param npt.NDArray[np.float64] rates: deposition rates at t1 for each
            realization.
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
        max_change = float(self.params.max_bathymetry_change_per_step)
        lo = dt_lo
        hi = dt_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._computeMaxBathymetryChange(t, rates, mid) <= max_change:
                lo = mid
            else:
                hi = mid
        return lo

    def _computeDepositionalEnvironment(
        self: Self,
        bathy: npt.NDArray[np.float64],
        prev_env: list[list[str] | None],
    ) -> Optional[list[DepositionalEnvironment | None]]:
        """Compute depositional environment for each realization.

        :param npt.NDArray[np.float64] bathy: bathymetry value for each
            realization.
        :return list[list[str] | None]: list of previous depositional
            environment names for each realization.
        """
        if self.depositionalEnvironmentSimulator is None:
            return None

        assert len(prev_env) == self.n_real, (
            "prev_env must have the same length as number of realizations"
        )
        assert len(bathy) == self.n_real, (
            "bathy must have the same length as number of realizations"
        )

        env: list[DepositionalEnvironment | None] = []
        for i in range(self.n_real):
            _, env_i = self.depositionalEnvironmentSimulator.run(
                bathymetry_value=bathy[i],
                previous_environments=prev_env[i],
            )
            env.append(env_i)
        return env

    def _computeEnvironmentalConditions(
        self: Self,
        bathy: npt.NDArray[np.float64],
        env: Optional[list[DepositionalEnvironment | None]] = None,
    ) -> list[dict[str, float]]:
        """Compute environmental conditions based on DepostionalEnvironment.

        :param npt.NDArray[np.float64] bathy: bathymetry value for each
            realization.
        :param list[DepositionalEnvironment|None], optional env: list of
            depositional environments for each realization. Defaults to None.

        :return list[dict[str, float]]: list of dictionaries containing
            environmental conditions for each realization.
        """
        env_conds: list[dict[str, float]] = []
        for i, bathy_i in enumerate(bathy):
            env_conds_i: dict[str, float] = {"bathymetry": float(bathy_i)}
            if (env is not None) and (env[i] is not None):
                env_conds_i.update(
                    self._getAllConditionsForEnvironment(env[i], bathy_i)  # type: ignore[assignment]
                )
            env_conds.append(env_conds_i)
        return env_conds

    def _getAllConditionsForEnvironment(
        self: Self, env: DepositionalEnvironment | None, bathy: float
    ) -> dict[str, float]:
        """Get all environment condition values from known bathymetry.

        Resolution plan
        ---------------
        The objective is to compute every property value for ``env`` from a
        known bathymetry ``bathy`` while respecting curve dependencies.

        1. Build the property universe

          - Start from ``env.other_property_ranges`` keys.
          - Keep ``"bathymetry"`` as an already known
            base variable with value ``bathy``.

        2. Analyze dependency graph from curves

          - For each curve in ``env.property_curves``, read the pair
            ``x_axis -> y_axis`` from the curve axis names.
          - Interpret this as: ``y_axis`` depends on ``x_axis``.
          - Record, for every target property, its required
            source property.

        3. Initialize independent properties

          - Independent property = no curve where this
            property is a y-axis.
          - Assign these values directly using their range midpoint via
            ``env.getPropertyMid(property_name)``.

        4. Resolve properties directly dependent on bathymetry

          - For every unresolved property with dependency
            ``x_axis == "bathymetry"``, compute value with:
            ``env.getValueFromCurveAt(
            "bathymetry", property_name, bathy)``.

        5. Resolve remaining properties iteratively

          - Repeatedly scan unresolved properties.
          - If a property depends on ``x_axis`` that is already resolved,
            compute it with
            ``env.getValueFromCurveAt(
            x_axis, property_name, value_of_x_axis)``.
          - Continue until no new property can be resolved.

        6. Validate and fail fast on invalid definitions

          - If unresolved properties remain, raise ``ValueError``
            describing missing prerequisites, missing curves,
            or cyclic dependencies.
          - If a curve references unknown properties,
            raise ``ValueError``.

        7. Return merged conditions

          - Return a dict containing all resolved properties (excluding
            ``"bathymetry"``).

        Notes:
        - This strategy is equivalent to a topological dependency resolution.
        - It supports mixed configurations: some properties fixed by midpoint,
          others constrained by curves.
        - Deterministic ordering is recommended (e.g., sorted property names)
          for reproducibility when multiple properties become
          solvable at once.

        :return dict[str, float]: environment condition dict.
        """
        if env is None:
            return {}

        properties: set[str] = set(env.other_property_ranges.keys())
        dependencies: dict[str, str] = env.getCurveDependencies()

        env_conds: dict[str, float] = {}
        # resolve independent properties (no curve where this property is a
        # y-axis)
        for property_name in sorted(properties):
            if property_name not in dependencies:
                env_conds[property_name] = env.getPropertyMid(property_name)

        # resolve properties with curve dependencies iteratively
        unresolved: set[str] = set(properties) - set(env_conds.keys())
        progress = (
            True  # True as long as a curve is resolved at each iteration
        )
        while unresolved and progress:
            progress = False
            for property_name in sorted(unresolved):
                x_axis = dependencies.get(property_name)
                if x_axis is None:
                    continue
                # resolve properties directly dependent on bathymetry
                if x_axis == "bathymetry":
                    env_conds[property_name] = env.getValueFromCurveAt(
                        "bathymetry", property_name, bathy
                    )
                    progress = True
                    continue
                # resolve properties dependent on other properties
                if x_axis in env_conds:
                    env_conds[property_name] = env.getValueFromCurveAt(
                        x_axis, property_name, env_conds[x_axis]
                    )
                    progress = True
            unresolved = set(properties) - set(env_conds.keys())

        if unresolved:
            unresolved_with_sources = {
                property_name: dependencies.get(property_name)
                for property_name in sorted(unresolved)
            }
            raise ValueError(
                "Could not resolve all environment properties for "
                f"{env.name}. Unresolved dependencies: "
                f"{unresolved_with_sources}."
            )

        return env_conds

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
        data_vars = {
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
        }
        for elementName in self.scenario.accumulationModel.elements:
            element_arr = np.array(
                [
                    [
                        self.depo_rate_elements[t][r][elementName]
                        for t in range(len(self.times) - 1)
                    ]
                    for r in range(n_real)
                ],
                dtype=np.float64,
            )
            data_vars[f"depo_rate_{elementName}"] = (
                ("realization", "time"),
                element_arr,
            )

        if self.depositionalEnvironmentSimulator is not None:
            env_arr = np.stack(self.environments, axis=1)
            data_vars["environment"] = (("realization", "time"), env_arr)  # type: ignore[assignment]

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": (("time",), time_axis),
                "realization": (("realization",), realization_ids),
            },
            attrs={
                "scenario_name": self.scenario.name,
                "start": float(self.times[0]),
                "stop": float(self.times[-1]),
                "max_bathymetry_change_per_step": float(
                    self.params.max_bathymetry_change_per_step
                ),
            },
        )

        return ds

    @staticmethod
    def _getAccumulatedThickness(
        accumulationRate: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        """Get the deposited thickness at a step given the accumulation rate.

        :param npt.NDArray[np.float64] accumulationRate: total accumulation
            rate for each realization (e.g., in m/Myr).
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
