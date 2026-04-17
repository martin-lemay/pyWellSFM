# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import xarray as xr

from pywellsfm.model.DepositionalEnvironment import DepositionalEnvironment
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelUniform,
)
from pywellsfm.model.FSSimulationParameters import (
    FSSimulatorParameters,
    RealizationData,
    Scenario,
)
from pywellsfm.model.Marker import Marker
from pywellsfm.utils import get_logger

from .AccommodationSimulator import AccommodationSimulator
from .AccumulationSimulator import AccumulationSimulator
from .DepositionalEnvironmentSimulator import (
    DepositionalEnvironmentModel,
    DepositionalEnvironmentSimulator,
    DESimulatorParameters,
)
from .EnvironmentConditionSimulator import EnvironmentConditionSimulator
from .TimeStepController import TimeStepController

logger = get_logger(__name__)


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
        such as accumulated thickness and waterDepth change do not exceed the
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
        # store scenario and realization data
        self.scenario: Scenario = scenario
        self.realizationDataList: list[RealizationData] = realizationDataList

        # realization simulators
        # depositional environment simulator is shared by all realizations
        # It is set when the depositional environment model contains multiple
        # environments, otherwise we consider a single environment
        self.depositionalEnvironmentSimulator: Optional[
            DepositionalEnvironmentSimulator
        ] = None
        # environment condition simulator is shared by all realizations
        self.environmentConditionSimulator = EnvironmentConditionSimulator()
        # accumulation simulator is shared by all realizations
        self.accumulationSimulator = AccumulationSimulator()

        # accommodation simulators, one for each realization
        self.accommodationSimulators: list[AccommodationSimulator] = [
            AccommodationSimulator() for _ in self.realizationDataList
        ]
        # sea level is shared by all realizations, will use the first
        # element of accommodationSimulators list
        self.seaLevelSimulator: AccommodationSimulator

        # depositional environment simulator configuration
        self.use_deSimulator = use_depositional_environment_simulator
        self.deSimulator_weights = deSimulator_weights
        self.deSimulator_params = deSimulator_params

        # Adaptative step configuration
        self.n_real: int = len(
            self.realizationDataList
        )  # number of realizations
        self.params: FSSimulatorParameters = fsSimulator_params

        # Initialize adaptive time-step controller
        self._time_step_controller = TimeStepController(
            self.params, self._computeMaxWaterDepthChange
        )

        # marker ages
        self.markerAges: set[float] = set()

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
        self.waterDepths: list[npt.NDArray[np.float64]] = []
        self.delta_waterDepths: list[npt.NDArray[np.float64]] = []
        self.environments: list[npt.NDArray[np.str_]] = []
        self.dts: list[float] = []
        self.initial_waterDepths: Optional[npt.NDArray[np.float64]] = None

        self.outputs: Optional[xr.Dataset] = None

        self._ready: bool = False

    def prepare(self: Self) -> None:
        """Prepare the FS simulator for running."""
        # Validate configuration
        if self.params.max_waterDepth_change_per_step <= 0:
            raise ValueError("max_waterDepth_change_per_step must be > 0")
        if self.params.dt_min <= 0 or self.params.dt_max <= 0:
            raise ValueError("dt_min and dt_max must be > 0")
        if self.params.dt_min > self.params.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if not (0.0 < self.params.safety <= 1.0):
            raise ValueError("safety must be in (0, 1]")

        # Set accumulation simulator data
        self.accumulationSimulator.setAccumulationModel(
            self.scenario.accumulationModel
        )
        self.accumulationSimulator.prepare()

        # Set environment condition simulator data
        deModel = self.scenario.depositionalEnvironmentModel
        if deModel is None:
            # create a model where water depth ranges from 0 to 10000m
            deModel = DepositionalEnvironmentModel(
                name="Default",
                environments=[
                    DepositionalEnvironment(
                        name="Default",
                        waterDepthModel=EnvironmentConditionModelUniform(
                            "waterDepth", -np.inf, np.inf
                        ),
                    )
                ],
            )
        self.environmentConditionSimulator.setEnvironmentModel(deModel)
        self.environmentConditionSimulator.prepare()

        # set accommodation simulator data for each realization
        for i in range(self.n_real):
            self.accommodationSimulators[i].setSubsidenceCurve(
                self.realizationDataList[i].subsidenceCurve,
                self.realizationDataList[i].subsidenceType,
            )
            self.accommodationSimulators[i].setInitialBathymetry(
                self.realizationDataList[i].initialBathymetry
            )
            if self.scenario.eustaticCurve is not None:
                # set the eustatic curve if defined, otherwise considers
                # no eustacy variations
                self.accommodationSimulators[i].setEustaticCurve(
                    self.scenario.eustaticCurve
                )
            self.accommodationSimulators[i].prepare()
            self.accommodationSimulators[i].initialEustacy(self.getStartAge())

        # Use the first accommodation simulator for sea level calculation
        self.seaLevelSimulator = self.accommodationSimulators[0]

        # Initialize depositional environment simulator if needed
        # no need if a single environment
        if self.use_deSimulator and deModel.getEnvironmentCount() > 1:
            self.depositionalEnvironmentSimulator = (
                DepositionalEnvironmentSimulator(
                    deModel,
                    params=self.deSimulator_params,
                    weights=self.deSimulator_weights,
                )
            )
            self.depositionalEnvironmentSimulator.prepare()

        # Extract initial waterDepths
        self.initial_waterDepths = np.array(
            [
                realizationData.initialBathymetry
                for realizationData in self.realizationDataList
            ],
            dtype=np.float64,
        )

        # extract exact time step from markers
        self.markerAges = {
            marker.age
            for rd in self.realizationDataList
            for marker in rd.well._markers
        }

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
        self.waterDepths = []
        self.delta_waterDepths = []
        self.environments = []
        self.dts = []

        # mark as ready for running
        self._ready = True

    def getStartAge(self: Self, markerStart: Optional[Marker] = None) -> float:
        """Get the start age of the simulation.

        :return float: start age.
        """
        if markerStart is None:
            return min(
                rd.well.oldestMarkerAge for rd in self.realizationDataList
            )
        return markerStart.age

    def getAgeEnd(self: Self, markerEnd: Optional[Marker] = None) -> float:
        """Get the end age of the simulation.

        :return float: end age.
        """
        if markerEnd is None:
            # Get the maximum age across all realizations
            return max(
                rd.well.youngestMarkerAge for rd in self.realizationDataList
            )
        return markerEnd.age

    def run(
        self: Self,
        markerEnd: Optional[Marker] = None,
        exactAges: Optional[set[float]] = None,
    ) -> None:
        """Run the FS simulator until a given marker or to the top.

        Time decreases from start to stop (e.g., from 100 Myr to 0 Myr).

        :param Optional[Marker] markerEnd: marker until which to run the
            simulation. If None, the simulation runs to the top of wells.
        :param Optional[set[float]] exactAges: set of exact ages to include in
            the simulation. These ages will be included in the set of ages at
            which the simulator state is recorded. If None, only marker ages
            are included. Default is None.
        """
        if not self._ready:
            raise RuntimeError("Must call prepare() before run()")

        # combine exact ages from markers and user input
        if exactAges is None:
            exactAges = self.markerAges
        else:
            exactAges = exactAges.union(self.markerAges)

        # Pre-sort exact ages for efficient lookup during stepping
        self._time_step_controller.set_exact_ages(exactAges)

        # Determine start and stop times
        # absolute geological age (in Myr) at start of simulation
        start = self.getStartAge()
        # absolute geological age (in Myr) at end of simulation
        stop: float = self.getAgeEnd(markerEnd)
        if stop >= start:
            raise ValueError("stop must be < start")

        # Initialize water depth state
        if self.initial_waterDepths is None:
            raise RuntimeError("initial_waterDepths not set")
        waterDepth_t = self.initial_waterDepths.copy()

        # Initialize depositional environment state
        depEnv_t = self._initializeDepositionalEnvironments(waterDepth_t)

        # set initial sea level and subsidence values
        sea_level_t = 0.0
        cumul_subs_t = np.zeros(self.n_real, dtype=np.float64)

        # Set initial time
        t = start
        for _ in range(self.params.max_steps):
            if t <= stop:
                break

            logger.info(
                "Running time step at age %.4f over %.4f Myr...", t, stop
            )

            # Record state at time t (step-start)
            self.times.append(t)

            # **** 1. CONDITIONS AND ACCUMULATION RATES AT STEP t ****

            # Record water depth for this step
            self.waterDepths.append(waterDepth_t.copy())

            # Get environment conditions from depositional environments
            env_list = self._computeEnvironmentalConditions(
                waterDepth_t, depEnv_t, t
            )

            # Record depositional environment for this step if needed
            self.environments.append(
                np.array(
                    [
                        env.name if env is not None else "none"
                        for env in depEnv_t
                    ],
                    dtype=str,
                )
            )

            # Compute and record deposition rates for this step
            accumulationRates = self._computeAccumulationRates(env_list, t)
            self.depo_rate_totals.append(accumulationRates)

            # element accumulation rates for this step
            accu_rates_step: list[dict[str, float]] = [
                {} for _ in range(self.n_real)
            ]
            for name in self.scenario.accumulationModel.elements:
                accuRate = self._computeElementAccumulationRates(
                    name, env_list, t
                )
                for i in range(self.n_real):
                    accu_rates_step[i][name] = accuRate[i]
            self.depo_rate_elements.append(accu_rates_step)

            # **** 2. DURATION OF STEP t ****
            # Choose adaptive time step (step duration)
            remaining = t - stop
            dt = self._time_step_controller.adapt(
                t, waterDepth_t, accumulationRates, remaining
            )
            t2 = t - dt

            # Record time step duration for this step
            self.dts.append(dt)

            # **** 3. DEPOSITED THICKNESS AND ACCOMODATION AT STEP t ****
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

            # Compute waterDepth change for this step
            delta_waterDepth = self._getWaterDepthVariation(
                delta_sea_level_t, delta_subs_t, thickness_step
            )
            delta_waterDepth_array: npt.NDArray[np.float64]
            if isinstance(delta_waterDepth, np.ndarray):
                delta_waterDepth_array = delta_waterDepth.copy()
            else:
                delta_waterDepth_array = np.full(
                    (self.n_real,), delta_waterDepth, dtype=np.float64
                )
            self.delta_waterDepths.append(delta_waterDepth_array)

            # compute and record basement elevation
            # (positive subsidence means sinking)
            basement_t = (-self.initial_waterDepths) - cumul_subs_t
            self.basements.append(basement_t)

            # compute and record accommodation
            acco_t = cumul_subs_t + sea_level_t
            self.accommodations.append(acco_t)

            # **** 4. UPDATE WATER DEPTH AND DEP ENVIRONMENT FOR NEXT STEP ****
            # Update waterDepthmetry for next step
            waterDepth_t = waterDepth_t + delta_waterDepth_array

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
                    waterDepth_t, prev_env
                )

            # Advance time
            t = t2
        else:
            raise RuntimeError("Reached max_steps without reaching stop")

        # Record final state at time t (step-end)
        self.times.append(t)
        logger.info(
            "Finished simulation at age %.4f Myr after %d steps.",
            t,
            len(self.times) - 1,
        )

        # mark as not ready for running again until prepare is called
        self._ready = False

    def _initializeDepositionalEnvironments(
        self: Self, waterDepth: npt.NDArray[np.float64]
    ) -> list[Optional[DepositionalEnvironment]]:
        depEnv: list[Optional[DepositionalEnvironment]] = [
            None for _ in self.realizationDataList
        ]
        deModel = self.environmentConditionSimulator.environmentModel
        if deModel is not None:
            for i in range(self.n_real):
                envName = self.realizationDataList[i].initialEnvironmentName
                env = None
                if envName is not None:
                    env = deModel.getEnvironmentByName(envName)

                if (
                    env is None
                    and self.depositionalEnvironmentSimulator is not None
                ):
                    # simulate environment from the simulator
                    _, env_i = self.depositionalEnvironmentSimulator.run(
                        waterDepth_value=waterDepth[i],
                        previous_environments=None,
                    )
                    depEnv[i] = env_i
        return depEnv

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
        for i, accommodationSimulator in enumerate(
            self.accommodationSimulators
        ):
            subs_t1 = accommodationSimulator.getSubsidenceAt(t1)
            subsType = accommodationSimulator.getSubsidenceType()
            if subsType == "cumulative":
                subs_t2 = accommodationSimulator.getSubsidenceAt(t2)
                delta_subs[i] = subs_t2 - subs_t1
            elif subsType == "rate":
                delta_subs[i] = subs_t1 * (t1 - t2)
            else:
                raise ValueError(f"Unknown subsidence type: {subsType}")
        return delta_subs

    def _computeMaxWaterDepthChange(
        self: Self, t1: float, dt: float, rates: npt.NDArray[np.float64]
    ) -> float:
        """Compute maximum water depth change across realizations for a dt.

        :param float t1: start time of the step.
        :param npt.NDArray[np.float64] rates: deposition rates at t1 for each
            realization.
        :param float dt: candidate time step.
        :return float: maximum absolute water depth change.
        """
        t2 = t1 - dt

        # Get sea level variation between t1 and t2 (same for all realizations)
        delta_sea_level = self._getDeltaSeaLevel(t1, t2)

        # Get subsidence variation between t1 and t2 for all realizations
        delta_subsidences = self._getDeltaSubsidence(t1, t2)

        # Compute accumulated thickness between t1 and t2 for all realizations
        thicknesses_step = self._getAccumulatedThickness(rates, dt)

        # Compute water depth change
        d_water_depth = self._getWaterDepthVariation(
            delta_sea_level,
            delta_subsidences,
            thicknesses_step,
        )

        # Return max absolute change
        finite = d_water_depth[np.isfinite(d_water_depth)]  # type: ignore
        if finite.size == 0:
            return float("inf")
        return float(np.max(np.abs(finite)))

    def _computeAccumulationRates(
        self: Self,
        env_list: list[dict[str, float]],
        age: float,
    ) -> npt.NDArray[np.float64]:
        """Compute accumulation rates for all realizations from the model.

        :param list[dict[str, float]] env_list: list of environment conditions
            for each realization.
        :param float age: age at the location (only needed if some conditions
            depend on age).
        :return npt.NDArray[np.float64]: accumulation rates for all
            realizations.
        """
        accumulationRates = np.array(
            [
                self.accumulationSimulator.computeTotalAccumulationRate(
                    env_list[i], age
                )
                for i in range(self.n_real)
            ],
            dtype=np.float64,
        )
        return accumulationRates

    def _computeElementAccumulationRates(
        self: Self,
        elementName: str,
        env_list: list[dict[str, float]],
        age: float,
    ) -> npt.NDArray[np.float64]:
        """Compute accumulation rates for all realizations from the model.

        :param str elementName: name of the accumulation element to compute.
        :param list[dict[str, float]] env_list: list of environment conditions
            for each realization.
        :param float age: age at the location (only needed if some conditions
            depend on age).
        :return npt.NDArray[np.float64]: accumulation rates for all
            realizations.
        """
        accumulationRates = np.array(
            [
                self.accumulationSimulator.computeElementAccumulationRate(
                    elementName, env_list[i], age
                )
                for i in range(self.n_real)
            ],
            dtype=np.float64,
        )
        return accumulationRates

    def _computeDepositionalEnvironment(
        self: Self,
        waterDepth: npt.NDArray[np.float64],
        prev_env: list[list[str] | None],
    ) -> list[DepositionalEnvironment | None]:
        """Compute depositional environment for each realization.

        :param npt.NDArray[np.float64] waterDepth: waterDepth value for each
            realization.
        :return list[list[str] | None]: list of previous depositional
            environment names for each realization.
        """
        if self.depositionalEnvironmentSimulator is None:
            return [None] * self.n_real

        if len(prev_env) != self.n_real:
            raise ValueError(
                "prev_env must have the same length as number of realizations"
            )
        if len(waterDepth) != self.n_real:
            raise ValueError(
                "waterDepth must have the same length as number of "
                + "realizations"
            )

        env: list[DepositionalEnvironment | None] = []
        for i in range(self.n_real):
            _, env_i = self.depositionalEnvironmentSimulator.run(
                waterDepth_value=waterDepth[i],
                previous_environments=prev_env[i],
            )
            env.append(env_i)
        return env

    def _computeEnvironmentalConditions(
        self: Self,
        waterDepth: npt.NDArray[np.float64],
        envs: list[DepositionalEnvironment | None],
        age: float,
    ) -> list[dict[str, float]]:
        """Compute environmental conditions based on DepositionalEnvironment.

        :param npt.NDArray[np.float64] waterDepth: waterDepth value for each
            realization.
        :param list[DepositionalEnvironment|None] envs: list of depositional
            environments for each realization.
        :param float age: age at the location (only needed if some conditions
            depend on age).

        :return list[dict[str, float]]: list of dictionaries containing
            environmental conditions for each realization.
        """
        env_conds: list[dict[str, float]] = []
        for i, waterDepth_i in enumerate(waterDepth):
            env_conds_i: dict[str, float] = {"waterDepth": float(waterDepth_i)}
            env_i = envs[i]
            if env_i is not None:
                env_conds_i.update(
                    self.environmentConditionSimulator.computeEnvironmentalConditions(
                        env_i.name,
                        waterDepth_i,
                        age,
                    )
                )
            env_conds.append(env_conds_i)
        return env_conds

    def finalize(self: Self) -> None:
        """Finalize the FS simulator after running."""
        # create output xarray Dataset
        self.outputs = self._buildEnsembleDataset()

        # TODO: create simulated wells

    def _buildEnsembleDataset(self: Self) -> xr.Dataset:
        """Build the ensemble dataset after running all realizations.

        :return xr.Dataset: xarray.Dataset containing ensemble results.
        """
        if not self.times:
            raise RuntimeError(
                "Must call run() before _buildEnsembleDataset()"
            )

        n_real = len(self.realizationDataList)

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
        waterDepth_arr = np.stack(self.waterDepths, axis=1)
        delta_waterDepth_arr = np.stack(self.delta_waterDepths, axis=1)

        # Build realization IDs
        realization_ids = np.arange(n_real, dtype=np.int64)

        # Create xarray Dataset
        data_vars = {
            "sea_level": (("time",), sea_level_arr[0, :]),
            "dt": (("time",), dt_axis),
            "initial_waterDepth": (
                ("realization",),
                self.initial_waterDepths,
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
            "waterDepth": (("realization", "time"), waterDepth_arr),
            "delta_waterDepth": (
                ("realization", "time"),
                delta_waterDepth_arr,
            ),
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
                "max_waterDepth_change_per_step": float(
                    self.params.max_waterDepth_change_per_step
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
    def _getWaterDepthVariation(
        delta_seaLevel: float,
        delta_subs: npt.NDArray[np.float64],
        thickness_step: npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        """Compute water depth variation over a time step.

        :param float delta_seaLevel: change in sea level over the time step.
        :param npt.NDArray[np.float64] delta_subs: change in subsidence of each
            realization over the time step.
        :param npt.NDArray[np.float64] thickness_step: deposited thickness of
            each realization over the time step.
        :return float | npt.NDArray[np.float64]: water depth variation over the
            time step.
        """
        return delta_seaLevel + delta_subs - thickness_step
