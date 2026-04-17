# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import pytest

from pywellsfm.io.fssimulation_io import loadFSSimulation
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelUniform,
)
from pywellsfm.model.Marker import Marker
from pywellsfm.simulator.FSSimulator import FSSimulator, FSSimulatorParameters


class _DummyAccommodationSimulator:
    def __init__(self: Self, subs_type: str, t1: float, t2: float) -> None:
        self._subs_type = subs_type
        self._values = {10.0: t1, 9.0: t2}

    def getSubsidenceAt(self: Self, t: float) -> float:
        return self._values[t]

    def getSubsidenceType(self: Self) -> str:
        return self._subs_type


class _DummyDESimulator:
    def __init__(self: Self, env: DepositionalEnvironment) -> None:
        self._env = env
        self.params = SimpleNamespace(trend_window=2)

    def run(
        self: Self,
        waterDepth_value: float,
        previous_environments: list[str] | None,
    ) -> tuple[float, DepositionalEnvironment]:
        _ = waterDepth_value
        _ = previous_environments
        return 1.0, self._env


@pytest.fixture()
def simulation_data_path() -> Path:
    """Provide the default simulation fixture path."""
    return Path(__file__).parent / "data" / "simulation.json"


@pytest.fixture()
def fs_sim(simulation_data_path: Path) -> FSSimulator:
    """Load a fresh simulator from JSON fixture data."""
    return loadFSSimulation(str(simulation_data_path))


def test_prepare_sets_initial_state(fs_sim: FSSimulator) -> None:
    """Prepare initializes simulation state and defaults."""
    fs_sim.prepare()

    assert fs_sim._ready is True
    assert fs_sim.seaLevelSimulator is fs_sim.accommodationSimulators[0]
    assert fs_sim.initial_waterDepths is not None
    assert np.allclose(fs_sim.initial_waterDepths, np.array([15.0, 20.0]))
    assert fs_sim.markerAges == {30.0, 10.0}
    assert fs_sim.times == []
    assert fs_sim.dts == []


@pytest.mark.parametrize(
    ("params", "msg"),
    [
        (
            FSSimulatorParameters(max_waterDepth_change_per_step=0.0),
            "max_waterDepth_change_per_step must be > 0",
        ),
        (
            FSSimulatorParameters(dt_min=0.0),
            "dt_min and dt_max must be > 0",
        ),
        (
            FSSimulatorParameters(dt_min=0.2, dt_max=0.1),
            "dt_min must be <= dt_max",
        ),
        (
            FSSimulatorParameters(safety=0.0),
            r"safety must be in \(0, 1\]",
        ),
    ],
)
def test_prepare_rejects_invalid_parameters(
    fs_sim: FSSimulator,
    params: FSSimulatorParameters,
    msg: str,
) -> None:
    """Prepare rejects invalid FS parameter ranges."""
    fs_sim.params = params

    with pytest.raises(ValueError, match=msg):
        fs_sim.prepare()


def test_get_age_helpers_support_marker_override(fs_sim: FSSimulator) -> None:
    """Age helpers use markers and explicit overrides."""
    fs_sim.prepare()

    start_override = Marker("Top", depth=1.0, age=25.0)
    end_override = Marker("Base", depth=2.0, age=12.0)

    assert fs_sim.getStartAge() == 30.0
    assert fs_sim.getStartAge(start_override) == 25.0
    assert fs_sim.getAgeEnd() == 10.0
    assert fs_sim.getAgeEnd(end_override) == 12.0


def test_run_requires_prepare(fs_sim: FSSimulator) -> None:
    """Run fails when called before prepare."""
    with pytest.raises(RuntimeError, match=r"Must call prepare\(\) before"):
        fs_sim.run()


def test_run_rejects_stop_greater_or_equal_to_start(
    fs_sim: FSSimulator,
) -> None:
    """Run rejects a stop age that is not younger than start."""
    fs_sim.prepare()

    with pytest.raises(ValueError, match="stop must be < start"):
        fs_sim.run(markerEnd=Marker("TooOld", depth=1.0, age=31.0))


def test_run_rejects_missing_initial_water_depths(fs_sim: FSSimulator) -> None:
    """Run fails when initial water depths are not prepared."""
    fs_sim.prepare()
    fs_sim.initial_waterDepths = None

    with pytest.raises(RuntimeError, match="initial_waterDepths not set"):
        fs_sim.run()


def test_run_raises_when_max_steps_reached(fs_sim: FSSimulator) -> None:
    """Run raises when max_steps does not allow progress."""
    fs_sim.params = FSSimulatorParameters(max_steps=0)
    fs_sim.prepare()

    with pytest.raises(RuntimeError, match="Reached max_steps"):
        fs_sim.run()


def test_run_finalize_builds_dataset(
    fs_sim: FSSimulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run and finalize produce expected ensemble outputs."""
    fs_sim.params = FSSimulatorParameters(dt_min=1.0, dt_max=2.0)
    fs_sim.prepare()
    monkeypatch.setattr(
        fs_sim._time_step_controller,
        "adapt",
        lambda _t, _wd, _rates, remaining: min(2.0, remaining),
    )
    fs_sim.run()
    fs_sim.finalize()

    assert fs_sim.outputs is not None
    ds = fs_sim.outputs

    assert "time" in ds.coords
    assert "realization" in ds.coords
    assert "depo_rate_total" in ds.data_vars
    assert "depo_rate_CarbonateShallow" in ds.data_vars
    assert "depo_rate_CarbonateIntermediate" in ds.data_vars
    assert "depo_rate_CarbonateDeep" in ds.data_vars
    assert ds.attrs["scenario_name"] == "Scenario1"
    assert ds.attrs["start"] == 30.0
    assert ds.attrs["stop"] == 10.0


def test_build_ensemble_dataset_requires_time(fs_sim: FSSimulator) -> None:
    """Dataset build fails when no run has been performed."""
    fs_sim.prepare()

    with pytest.raises(RuntimeError, match=r"Must call run\(\) before"):
        fs_sim._buildEnsembleDataset()


def test_get_delta_subsidence_supports_modes_and_errors(
    fs_sim: FSSimulator,
) -> None:
    """Delta subsidence handles cumulative, rate, and invalid types."""
    fs_sim.n_real = 2
    fs_sim.accommodationSimulators = [
        cast(
            Any,
            _DummyAccommodationSimulator("cumulative", t1=10.0, t2=12.0),
        ),
        cast(Any, _DummyAccommodationSimulator("rate", t1=3.0, t2=0.0)),
    ]

    out = fs_sim._getDeltaSubsidence(10.0, 9.0)
    assert np.allclose(out, np.array([2.0, 3.0]))

    fs_sim.accommodationSimulators = [
        cast(Any, _DummyAccommodationSimulator("unknown", t1=1.0, t2=2.0))
    ]
    fs_sim.n_real = 1
    with pytest.raises(ValueError, match="Unknown subsidence type"):
        fs_sim._getDeltaSubsidence(10.0, 9.0)


def test_compute_max_water_depth_change_handles_all_nan(
    fs_sim: FSSimulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max water-depth change returns inf for all-NaN variation."""
    monkeypatch.setattr(fs_sim, "_getDeltaSeaLevel", lambda *_: 0.0)
    monkeypatch.setattr(
        fs_sim,
        "_getDeltaSubsidence",
        lambda *_: np.array([0.0, 0.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        fs_sim,
        "_getWaterDepthVariation",
        lambda *_: np.array([np.nan, np.nan], dtype=np.float64),
    )

    rates = np.array([1.0, 2.0], dtype=np.float64)
    assert fs_sim._computeMaxWaterDepthChange(30.0, 1.0, rates) == float("inf")


def test_compute_accumulation_rates_calls_accumulation_simulator(
    fs_sim: FSSimulator,
) -> None:
    """Rate helpers query total and element accumulation methods."""

    class DummyAccumulation:
        def computeTotalAccumulationRate(
            self: Self,
            env_cond: dict[str, float],
            age: float,
        ) -> float:
            return env_cond["waterDepth"] + age

        def computeElementAccumulationRate(
            self: Self,
            element: str,
            env_cond: dict[str, float],
            age: float,
        ) -> float:
            return len(element) + env_cond["waterDepth"] + age

    fs_sim.accumulationSimulator = DummyAccumulation()  # type: ignore[assignment]
    fs_sim.n_real = 2
    envs = [{"waterDepth": 1.0}, {"waterDepth": 2.0}]

    totals = fs_sim._computeAccumulationRates(envs, age=3.0)
    elems = fs_sim._computeElementAccumulationRates("Ca", envs, age=3.0)

    assert np.allclose(totals, np.array([4.0, 5.0]))
    assert np.allclose(elems, np.array([6.0, 7.0]))


def test_compute_depositional_environment_variants(
    fs_sim: FSSimulator,
) -> None:
    """Depositional environment helper handles all code paths."""
    fs_sim.n_real = 2
    water_depth = np.array([5.0, 7.0], dtype=np.float64)

    out = fs_sim._computeDepositionalEnvironment(water_depth, [[], []])
    assert out == [None, None]

    env = DepositionalEnvironment(
        name="Default",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", -np.inf, np.inf
        ),
    )
    fs_sim.depositionalEnvironmentSimulator = _DummyDESimulator(env)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="prev_env"):
        fs_sim._computeDepositionalEnvironment(water_depth, [[]])

    with pytest.raises(ValueError, match="waterDepth"):
        fs_sim._computeDepositionalEnvironment(
            np.array([3.0], dtype=np.float64),
            [[], []],
        )

    out = fs_sim._computeDepositionalEnvironment(water_depth, [[], []])
    assert out == [env, env]


def test_initialize_and_compute_environmental_conditions(
    fs_sim: FSSimulator,
) -> None:
    """Environment helpers initialize and merge env conditions."""
    env = DepositionalEnvironment(
        name="E1",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", -np.inf, np.inf
        ),
    )
    model = DepositionalEnvironmentModel(name="M", environments=[env])

    fs_sim.n_real = 1
    rd0 = fs_sim.realizationDataList[0].__class__(
        well=fs_sim.realizationDataList[0].well,
        initialBathymetry=fs_sim.realizationDataList[0].initialBathymetry,
        initialEnvironmentName=None,
        subsidenceCurve=fs_sim.realizationDataList[0].subsidenceCurve,
        subsidenceType=fs_sim.realizationDataList[0].subsidenceType,
    )
    fs_sim.realizationDataList = [rd0]
    fs_sim.environmentConditionSimulator.setEnvironmentModel(model)
    fs_sim.depositionalEnvironmentSimulator = cast(Any, _DummyDESimulator(env))

    dep_env = fs_sim._initializeDepositionalEnvironments(np.array([2.0]))
    assert dep_env == [env]

    conds = fs_sim._computeEnvironmentalConditions(
        waterDepth=np.array([2.0]),
        envs=[env],
        age=1.0,
    )
    assert conds[0]["waterDepth"] == 2.0


def test_prepare_initializes_de_simulator_for_multiple_envs(
    fs_sim: FSSimulator,
) -> None:
    """Prepare creates DE simulator when enabled with many environments."""
    env1 = DepositionalEnvironment(
        name="E1",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", -np.inf, 0.0
        ),
    )
    env2 = DepositionalEnvironment(
        name="E2",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", 0.0, np.inf
        ),
    )
    de_model = DepositionalEnvironmentModel("M", [env1, env2])
    fs_sim.scenario = replace(
        fs_sim.scenario,
        eustaticCurve=None,
        depositionalEnvironmentModel=de_model,
    )
    fs_sim.use_deSimulator = True

    fs_sim.prepare()

    assert fs_sim.depositionalEnvironmentSimulator is not None


def test_run_covers_exact_age_union_scalar_and_env_window(
    fs_sim: FSSimulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run covers exact-age merge and DE trend-window logic."""
    fs_sim.n_real = 1
    fs_sim.realizationDataList = [fs_sim.realizationDataList[0]]
    fs_sim.initial_waterDepths = np.array([10.0], dtype=np.float64)
    fs_sim.markerAges = {29.0}
    fs_sim._ready = True
    fs_sim.params = FSSimulatorParameters(max_steps=10)

    class DummyDESim:
        params = SimpleNamespace(trend_window=2)

    prev_env_seen: list[list[list[str] | None]] = []

    def fake_compute_de_env(
        water_depth: np.ndarray,
        prev_env: list[list[str] | None],
    ) -> list[None]:
        _ = water_depth
        prev_env_seen.append(prev_env)
        return [None]

    monkeypatch.setattr(fs_sim, "getStartAge", lambda *_: 30.0)
    monkeypatch.setattr(fs_sim, "getAgeEnd", lambda *_: 28.0)
    monkeypatch.setattr(
        fs_sim,
        "_initializeDepositionalEnvironments",
        lambda *_: [None],
    )
    monkeypatch.setattr(
        fs_sim,
        "_computeEnvironmentalConditions",
        lambda *_: [{"waterDepth": 10.0}],
    )
    monkeypatch.setattr(
        fs_sim,
        "_computeAccumulationRates",
        lambda *_: np.array([1.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        fs_sim,
        "_computeElementAccumulationRates",
        lambda *_: np.array([0.5], dtype=np.float64),
    )
    monkeypatch.setattr(
        fs_sim._time_step_controller, "adapt", lambda *a, **k: 1.0
    )
    monkeypatch.setattr(fs_sim, "_getDeltaSeaLevel", lambda *_: 0.0)
    monkeypatch.setattr(
        fs_sim,
        "_getDeltaSubsidence",
        lambda *_: np.array([0.0], dtype=np.float64),
    )
    monkeypatch.setattr(fs_sim, "_getWaterDepthVariation", lambda *_: 0.0)
    monkeypatch.setattr(
        fs_sim,
        "_computeDepositionalEnvironment",
        fake_compute_de_env,
    )
    fs_sim.depositionalEnvironmentSimulator = DummyDESim()  # type: ignore[assignment]

    fs_sim.run(exactAges={28.5})

    assert fs_sim.times == [30.0, 29.0, 28.0]
    assert prev_env_seen[0] == [["none"]]


def test_initialize_depositional_environment_reads_named_env(
    fs_sim: FSSimulator,
) -> None:
    """Initialize path queries named environments from the DE model."""
    env = DepositionalEnvironment(
        name="Named",
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth", -np.inf, np.inf
        ),
    )
    model = DepositionalEnvironmentModel(name="M", environments=[env])
    rd0 = fs_sim.realizationDataList[0].__class__(
        well=fs_sim.realizationDataList[0].well,
        initialBathymetry=fs_sim.realizationDataList[0].initialBathymetry,
        initialEnvironmentName="Named",
        subsidenceCurve=fs_sim.realizationDataList[0].subsidenceCurve,
        subsidenceType=fs_sim.realizationDataList[0].subsidenceType,
    )
    fs_sim.n_real = 1
    fs_sim.realizationDataList = [rd0]
    fs_sim.environmentConditionSimulator.setEnvironmentModel(model)

    dep_env = fs_sim._initializeDepositionalEnvironments(np.array([1.0]))

    assert dep_env == [None]


def test_compute_max_water_depth_change_returns_finite_value(
    fs_sim: FSSimulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max water-depth helper returns finite absolute maxima."""
    monkeypatch.setattr(fs_sim, "_getDeltaSeaLevel", lambda *_: 0.0)
    monkeypatch.setattr(
        fs_sim,
        "_getDeltaSubsidence",
        lambda *_: np.array([1.0, -2.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        fs_sim,
        "_getWaterDepthVariation",
        lambda *_: np.array([1.0, -2.0], dtype=np.float64),
    )

    out = fs_sim._computeMaxWaterDepthChange(
        t1=30.0,
        rates=np.array([1.0, 1.0], dtype=np.float64),
        dt=1.0,
    )
    assert out == 2.0


def test_build_dataset_includes_environment_when_simulator_is_set(
    fs_sim: FSSimulator,
) -> None:
    """Dataset includes environment variable with DE simulator enabled."""
    fs_sim.initial_waterDepths = np.array([10.0], dtype=np.float64)
    fs_sim.times = [30.0, 29.0]
    fs_sim.dts = [1.0]
    fs_sim.sea_levels = [np.array([0.0], dtype=np.float64)]
    fs_sim.subsidences = [np.array([0.0], dtype=np.float64)]
    fs_sim.basements = [np.array([-10.0], dtype=np.float64)]
    fs_sim.accommodations = [np.array([0.0], dtype=np.float64)]
    fs_sim.depo_rate_totals = [np.array([1.0], dtype=np.float64)]
    fs_sim.thickness_steps = [np.array([1.0], dtype=np.float64)]
    fs_sim.thickness_cumul = [np.array([1.0], dtype=np.float64)]
    fs_sim.waterDepths = [np.array([10.0], dtype=np.float64)]
    fs_sim.delta_waterDepths = [np.array([0.0], dtype=np.float64)]
    fs_sim.environments = [np.array(["none"], dtype=str)]
    fs_sim.depo_rate_elements = [
        [
            {
                "CarbonateShallow": 0.1,
                "CarbonateIntermediate": 0.2,
                "CarbonateDeep": 0.3,
            }
        ]
    ]
    fs_sim.depositionalEnvironmentSimulator = SimpleNamespace(  # type: ignore[assignment]
        params=SimpleNamespace(trend_window=1)
    )
    fs_sim.n_real = 1
    fs_sim.realizationDataList = [fs_sim.realizationDataList[0]]

    ds = fs_sim._buildEnsembleDataset()

    assert "environment" in ds.data_vars
