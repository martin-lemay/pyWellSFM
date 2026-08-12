# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import pytest
from striplog import Striplog

from pywellsfm.io.fssimulation_io import loadFSSimulation
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelUniform,
)
from pywellsfm.model.Facies import (
    Facies,
    FaciesCriteria,
    FaciesCriteriaType,
    FaciesModel,
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


def test_classify_main_element_picks_dominant(
    fs_sim: FSSimulator,
) -> None:
    """MainElement classification picks element with highest rate."""
    result = FSSimulator._classifyMainElement(
        {"Sand": 0.5, "Shale": 0.3, "Carbonate": 0.2}
    )
    assert result == "Sand"


def test_classify_main_element_all_zero(
    fs_sim: FSSimulator,
) -> None:
    """MainElement returns Unclassified when all rates are zero."""
    result = FSSimulator._classifyMainElement({"A": 0.0, "B": 0.0})
    assert result == "Unclassified"


def test_classify_main_element_empty(
    fs_sim: FSSimulator,
) -> None:
    """MainElement returns Unclassified for empty dict."""
    assert FSSimulator._classifyMainElement({}) == "Unclassified"


def test_classify_facies_single_match() -> None:
    """Facies classification returns the matching facies name."""
    f1 = Facies(
        "Limestone",
        FaciesCriteria(
            "Carbonate", 0.5, 1.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        ),
    )
    f2 = Facies(
        "Marl",
        FaciesCriteria(
            "Carbonate", 0.0, 0.5, FaciesCriteriaType.SEDIMENTOLOGICAL
        ),
    )
    model = FaciesModel({f1, f2})
    result = FSSimulator._classifyFacies(model, {"Carbonate": 0.7})
    assert result == "Limestone"


def test_classify_facies_no_match() -> None:
    """Facies classification returns Unclassified when nothing matches."""
    f1 = Facies(
        "Limestone",
        FaciesCriteria(
            "Carbonate", 0.8, 1.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        ),
    )
    model = FaciesModel({f1})
    result = FSSimulator._classifyFacies(model, {"Carbonate": 0.3})
    assert result == "Unclassified"


def test_classify_facies_multiple_match_picks_most_specific() -> None:
    """When multiple facies match, pick the most specific (narrowest)."""
    broad = Facies(
        "Carbonate",
        FaciesCriteria(
            "Carbonate", 0.0, 1.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        ),
    )
    narrow = Facies(
        "PureLimestone",
        FaciesCriteria(
            "Carbonate", 0.7, 1.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        ),
    )
    model = FaciesModel({broad, narrow})
    result = FSSimulator._classifyFacies(model, {"Carbonate": 0.8})
    assert result == "PureLimestone"


def test_classify_facies_multi_criteria() -> None:
    """Facies with multiple criteria must all be satisfied."""
    f = Facies(
        "ShallowCarbonate",
        {
            FaciesCriteria(
                "Carbonate", 0.5, 1.0, FaciesCriteriaType.SEDIMENTOLOGICAL
            ),
            FaciesCriteria(
                "waterDepth", 0.0, 20.0, FaciesCriteriaType.ENVIRONMENTAL
            ),
        },
    )
    model = FaciesModel({f})
    # Both criteria satisfied
    assert (
        FSSimulator._classifyFacies(
            model, {"Carbonate": 0.8, "waterDepth": 10.0}
        )
        == "ShallowCarbonate"
    )
    # waterDepth out of range
    assert (
        FSSimulator._classifyFacies(
            model, {"Carbonate": 0.8, "waterDepth": 30.0}
        )
        == "Unclassified"
    )


def test_classify_facies_missing_property() -> None:
    """Facies does not match if a required property is missing."""
    f = Facies(
        "Limestone",
        FaciesCriteria(
            "Carbonate", 0.5, 1.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        ),
    )
    model = FaciesModel({f})
    result = FSSimulator._classifyFacies(model, {"waterDepth": 10.0})
    assert result == "Unclassified"


def test_build_striplog_merges_adjacent_same_label() -> None:
    """Adjacent intervals with same label are merged."""
    intervals = [
        (0.0, 10.0, "Sand"),
        (10.0, 20.0, "Sand"),
        (20.0, 30.0, "Shale"),
    ]
    result = FSSimulator._buildStriplog(intervals)
    assert isinstance(result, Striplog)
    assert len(result) == 2
    assert result[0].primary["lithology"] == "Sand"
    assert float(result[0].top.z) == pytest.approx(0.0)
    assert float(result[0].base.z) == pytest.approx(20.0)
    assert result[1].primary["lithology"] == "Shale"


def test_build_striplog_no_merge_different_labels() -> None:
    """Intervals with different labels are kept separate."""
    intervals = [
        (0.0, 10.0, "Sand"),
        (10.0, 20.0, "Shale"),
        (20.0, 30.0, "Sand"),
    ]
    result = FSSimulator._buildStriplog(intervals)
    assert len(result) == 3


def test_build_striplog_single_interval() -> None:
    """Single interval produces a single-entry Striplog."""
    result = FSSimulator._buildStriplog([(5.0, 15.0, "Limestone")])
    assert len(result) == 1
    assert result[0].primary["lithology"] == "Limestone"
    assert float(result[0].top.z) == pytest.approx(5.0)
    assert float(result[0].base.z) == pytest.approx(15.0)


def test_build_striplog_all_same_label() -> None:
    """All intervals with same label merge into one."""
    intervals = [
        (0.0, 5.0, "A"),
        (5.0, 10.0, "A"),
        (10.0, 15.0, "A"),
    ]
    result = FSSimulator._buildStriplog(intervals)
    assert len(result) == 1
    assert float(result[0].top.z) == pytest.approx(0.0)
    assert float(result[0].base.z) == pytest.approx(15.0)


def test_build_simulated_well_main_element_only(
    fs_sim: FSSimulator,
) -> None:
    """Build simulated well with MainElement logs when no FaciesModel."""
    # Set up minimal simulation state: 2 steps, 2 realizations
    fs_sim.times = [30.0, 20.0, 10.0]  # 2 steps
    fs_sim.depo_rate_elements = [
        # step 0 (age 30->20): [real0, real1]
        [
            {"CarbonateShallow": 0.5, "CarbonateDeep": 0.3},
            {"CarbonateShallow": 0.2, "CarbonateDeep": 0.6},
        ],
        # step 1 (age 20->10): [real0, real1]
        [
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.8},
            {"CarbonateShallow": 0.7, "CarbonateDeep": 0.1},
        ],
    ]
    fs_sim.thickness_steps = [
        np.array([5.0, 8.0], dtype=np.float64),  # step 0
        np.array([9.0, 3.0], dtype=np.float64),  # step 1
    ]
    fs_sim.waterDepths = [
        np.array([15.0, 20.0], dtype=np.float64),  # step 0
        np.array([10.0, 12.0], dtype=np.float64),  # step 1
    ]
    fs_sim.environments = [
        np.array(["Shallow", "Deep"], dtype=str),  # step 0
        np.array(["Deep", "Shallow"], dtype=str),  # step 1
    ]

    well = fs_sim._buildSimulatedWell(0)

    assert well.name == "Well1_sim_0"
    # Should have MainElement and Environment in both domains
    age_log = well.getAgeLog("MainElement")
    depth_log = well.getDepthLog("MainElement")
    assert age_log is not None
    assert depth_log is not None
    assert well.getAgeLog("Environment") is not None
    assert well.getDepthLog("Environment") is not None
    # No Facies log (no faciesModel in scenario)
    assert well.getAgeLog("Facies") is None
    assert well.getDepthLog("Facies") is None

    # Check age-domain: step0 dominant=CarbonateShallow (age 30->20),
    # step1 dominant=CarbonateDeep (age 20->10) -> 2 intervals.
    # Striplog orders ascending top, so index 0 = youngest (top=10.0).
    assert isinstance(age_log, Striplog)
    assert len(age_log) == 2

    # Check depth-domain: stacked from oldest marker (depth=90.0)
    # upward. Step0: 5.0m thick (base=90, top=85), step1: 9.0m thick
    # (base=85, top=76). Striplog orders ascending top: [0]=top=76, [1]=top=85.
    assert isinstance(depth_log, Striplog)
    assert len(depth_log) == 2
    assert float(depth_log[1].base.z) == pytest.approx(90.0)
    assert float(depth_log[1].top.z) == pytest.approx(85.0)
    assert float(depth_log[0].base.z) == pytest.approx(85.0)
    assert float(depth_log[0].top.z) == pytest.approx(76.0)


def test_build_simulated_well_with_facies_model(
    fs_sim: FSSimulator,
) -> None:
    """Build simulated well includes Facies logs when FaciesModel set."""
    # Add a facies model to scenario
    f_shallow = Facies(
        "ShallowCarbonate",
        FaciesCriteria(
            "CarbonateShallow",
            0.5,
            1.0,
            FaciesCriteriaType.SEDIMENTOLOGICAL,
        ),
    )
    f_deep = Facies(
        "DeepCarbonate",
        FaciesCriteria(
            "CarbonateDeep",
            0.5,
            1.0,
            FaciesCriteriaType.SEDIMENTOLOGICAL,
        ),
    )
    fm = FaciesModel({f_shallow, f_deep})
    fs_sim.scenario = replace(fs_sim.scenario, faciesModel=fm)

    # Set up 2 steps, use realization 0
    fs_sim.times = [30.0, 20.0, 10.0]
    fs_sim.depo_rate_elements = [
        [
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
        [
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
        ],
    ]
    fs_sim.thickness_steps = [
        np.array([4.0, 6.0], dtype=np.float64),
        np.array([6.0, 4.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [
        np.array([15.0, 20.0], dtype=np.float64),
        np.array([10.0, 12.0], dtype=np.float64),
    ]
    fs_sim.environments = [
        np.array(["Shallow", "Deep"], dtype=str),
        np.array(["Deep", "Shallow"], dtype=str),
    ]

    well = fs_sim._buildSimulatedWell(0)

    age_facies = well.getAgeLog("Facies")
    depth_facies = well.getDepthLog("Facies")
    assert age_facies is not None
    assert depth_facies is not None
    assert isinstance(age_facies, Striplog)
    # Step0 (30->20 Ma): CarbonateShallow=0.8 -> ShallowCarbonate
    # Step1 (20->10 Ma): CarbonateDeep=0.9 -> DeepCarbonate
    # Striplog orders ascending top: [0]=top=10.0 (DeepCarbonate, younger),
    # [1]=top=20.0 (ShallowCarbonate, older).
    assert len(age_facies) == 2
    assert age_facies[0].primary["lithology"] == "DeepCarbonate"
    assert age_facies[1].primary["lithology"] == "ShallowCarbonate"


def test_build_simulated_well_merges_same_label_steps(
    fs_sim: FSSimulator,
) -> None:
    """Adjacent steps with same dominant element are merged."""
    fs_sim.times = [30.0, 25.0, 20.0, 10.0]  # 3 steps
    fs_sim.depo_rate_elements = [
        # step 0+1: same dominant (CarbonateShallow), step 2: different
        [
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
        [
            {"CarbonateShallow": 0.7, "CarbonateDeep": 0.3},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
        [
            {"CarbonateShallow": 0.2, "CarbonateDeep": 0.8},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
    ]
    fs_sim.thickness_steps = [
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([5.0, 3.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [
        np.array([15.0, 20.0], dtype=np.float64),
        np.array([12.0, 16.0], dtype=np.float64),
        np.array([10.0, 13.0], dtype=np.float64),
    ]
    fs_sim.environments = [
        np.array(["Shallow", "Deep"], dtype=str),
        np.array(["Shallow", "Deep"], dtype=str),
        np.array(["Deep", "Shallow"], dtype=str),
    ]

    well = fs_sim._buildSimulatedWell(0)
    age_log = well.getAgeLog("MainElement")
    assert isinstance(age_log, Striplog)
    # Steps 0+1 merge (both CarbonateShallow), step 2 is CarbonateDeep
    assert len(age_log) == 2

    depth_log = well.getDepthLog("MainElement")
    assert isinstance(depth_log, Striplog)
    # Merged: CarbonateShallow (steps 0+1: 3+2=5m, base=90→top=85),
    # CarbonateDeep (step 2: 5m, base=85→top=80).
    # Striplog orders ascending top: [0]=top=80 (CarbonateDeep), [1]=top=85.
    assert len(depth_log) == 2
    assert float(depth_log[1].base.z) == pytest.approx(90.0)
    assert float(depth_log[1].top.z) == pytest.approx(85.0)
    assert float(depth_log[0].base.z) == pytest.approx(85.0)
    assert float(depth_log[0].top.z) == pytest.approx(80.0)


def test_build_simulated_well_environment_log(
    fs_sim: FSSimulator,
) -> None:
    """Build simulated well includes Environment logs with merging."""
    fs_sim.times = [30.0, 25.0, 20.0, 10.0]  # 3 steps
    fs_sim.depo_rate_elements = [
        [
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
        [
            {"CarbonateShallow": 0.7, "CarbonateDeep": 0.3},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
        [
            {"CarbonateShallow": 0.2, "CarbonateDeep": 0.8},
            {"CarbonateShallow": 0.1, "CarbonateDeep": 0.9},
        ],
    ]
    fs_sim.thickness_steps = [
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([5.0, 3.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [
        np.array([15.0, 20.0], dtype=np.float64),
        np.array([12.0, 16.0], dtype=np.float64),
        np.array([10.0, 13.0], dtype=np.float64),
    ]
    # Steps 0+1 same environment ("Shallow"), step 2 different ("Deep")
    fs_sim.environments = [
        np.array(["Shallow", "Deep"], dtype=str),
        np.array(["Shallow", "Deep"], dtype=str),
        np.array(["Deep", "Shallow"], dtype=str),
    ]

    well = fs_sim._buildSimulatedWell(0)

    age_env = well.getAgeLog("Environment")
    depth_env = well.getDepthLog("Environment")
    assert age_env is not None
    assert depth_env is not None
    assert isinstance(age_env, Striplog)
    assert isinstance(depth_env, Striplog)

    # Steps 0+1 merge (both "Shallow"), step 2 is "Deep" -> 2 intervals
    assert len(age_env) == 2
    # Ascending top: [0]=Deep, [1]=Shallow
    assert age_env[0].primary["lithology"] == "Deep"
    assert age_env[1].primary["lithology"] == "Shallow"

    # Depth: merged Shallow (steps 0+1: 3+2=5m, base=90→top=85),
    # Deep (step 2: 5m, base=85→top=80).
    assert len(depth_env) == 2
    assert float(depth_env[1].base.z) == pytest.approx(90.0)
    assert float(depth_env[1].top.z) == pytest.approx(85.0)
    assert depth_env[1].primary["lithology"] == "Shallow"
    assert float(depth_env[0].base.z) == pytest.approx(85.0)
    assert float(depth_env[0].top.z) == pytest.approx(80.0)
    assert depth_env[0].primary["lithology"] == "Deep"


def test_build_simulated_well_creates_erosive_marker_on_hiatus(
    fs_sim: FSSimulator,
) -> None:
    """Hiatus (rate<=0) creates an Erosive marker at onset age/depth."""
    from pywellsfm.model.Marker import StratigraphicSurfaceType

    # 3 steps: deposit, hiatus, deposit
    fs_sim.times = [30.0, 25.0, 20.0, 10.0]
    fs_sim.depo_rate_totals = [
        np.array([1.0, 1.0], dtype=np.float64),  # step 0: depositing
        np.array([0.0, 0.0], dtype=np.float64),  # step 1: hiatus
        np.array([1.0, 1.0], dtype=np.float64),  # step 2: depositing
    ]
    fs_sim.depo_rate_elements = [
        [
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
        ],
        [
            {"CarbonateShallow": 0.0, "CarbonateDeep": 0.0},
            {"CarbonateShallow": 0.0, "CarbonateDeep": 0.0},
        ],
        [
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
            {"CarbonateShallow": 0.8, "CarbonateDeep": 0.2},
        ],
    ]
    fs_sim.thickness_steps = [
        np.array([5.0, 5.0], dtype=np.float64),  # step 0
        np.array([0.0, 0.0], dtype=np.float64),  # step 1: no thickness
        np.array([5.0, 5.0], dtype=np.float64),  # step 2
    ]
    fs_sim.waterDepths = [
        np.array([15.0, 20.0], dtype=np.float64),
        np.array([15.0, 20.0], dtype=np.float64),
        np.array([15.0, 20.0], dtype=np.float64),
    ]
    fs_sim.environments = [
        np.array(["Shallow", "Shallow"], dtype=str),
        np.array(["Shallow", "Shallow"], dtype=str),
        np.array(["Shallow", "Shallow"], dtype=str),
    ]

    well = fs_sim._buildSimulatedWell(0)

    # Original well has 2 markers (at ages 30 and 10)
    # Simulated well should have those 2 + 1 erosive marker
    markers = well.getMarkers()
    erosive_markers = [
        m
        for m in markers
        if m.stratigraphicType == StratigraphicSurfaceType.EROSIVE
    ]
    assert len(erosive_markers) == 1
    m = erosive_markers[0]
    # Onset at age 25.0 (step 1), duration = 25.0 - 20.0 = 5.0
    assert m.name == "Hiatus_5"
    assert m.age == 25.0
    # Depth = oldest marker depth (90.0) - step0 thickness (5.0) = 85.0
    assert m.depth == pytest.approx(85.0)


def test_build_simulated_well_multiple_hiatus_periods(
    fs_sim: FSSimulator,
) -> None:
    """Multiple non-adjacent hiatus periods create separate markers."""
    from pywellsfm.model.Marker import StratigraphicSurfaceType

    # 5 steps: deposit, hiatus, deposit, hiatus, deposit
    fs_sim.times = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
    fs_sim.depo_rate_totals = [
        np.array([1.0, 1.0], dtype=np.float64),  # step 0: deposit
        np.array([0.0, 0.0], dtype=np.float64),  # step 1: hiatus
        np.array([1.0, 1.0], dtype=np.float64),  # step 2: deposit
        np.array([0.0, 0.0], dtype=np.float64),  # step 3: hiatus
        np.array([1.0, 1.0], dtype=np.float64),  # step 4: deposit
    ]
    fs_sim.depo_rate_elements = [
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
        [{"CarbonateShallow": 0.0, "CarbonateDeep": 0.0}] * 2,
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
        [{"CarbonateShallow": 0.0, "CarbonateDeep": 0.0}] * 2,
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
    ]
    fs_sim.thickness_steps = [
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [np.array([15.0, 20.0], dtype=np.float64)] * 5
    fs_sim.environments = [np.array(["Shallow", "Shallow"], dtype=str)] * 5

    well = fs_sim._buildSimulatedWell(0)

    markers = well.getMarkers()
    erosive_markers = [
        m
        for m in markers
        if m.stratigraphicType == StratigraphicSurfaceType.EROSIVE
    ]
    assert len(erosive_markers) == 2
    # Sort by age descending (oldest first)
    erosive_markers.sort(key=lambda m: m.age, reverse=True)
    # First hiatus: onset at 40.0, ends at 30.0, duration=10
    assert erosive_markers[0].name == "Hiatus_10"
    assert erosive_markers[0].age == 40.0
    # Second hiatus: onset at 20.0, ends at 10.0, duration=10
    assert erosive_markers[1].name == "Hiatus_10"
    assert erosive_markers[1].age == 20.0


def test_build_simulated_well_hiatus_at_end(
    fs_sim: FSSimulator,
) -> None:
    """Hiatus extending to end of simulation still creates a marker."""
    from pywellsfm.model.Marker import StratigraphicSurfaceType

    # 2 steps: deposit, then hiatus until end
    fs_sim.times = [30.0, 20.0, 10.0]
    fs_sim.depo_rate_totals = [
        np.array([1.0, 1.0], dtype=np.float64),  # step 0: deposit
        np.array([0.0, 0.0], dtype=np.float64),  # step 1: hiatus
    ]
    fs_sim.depo_rate_elements = [
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
        [{"CarbonateShallow": 0.0, "CarbonateDeep": 0.0}] * 2,
    ]
    fs_sim.thickness_steps = [
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [np.array([15.0, 20.0], dtype=np.float64)] * 2
    fs_sim.environments = [np.array(["Shallow", "Shallow"], dtype=str)] * 2

    well = fs_sim._buildSimulatedWell(0)

    markers = well.getMarkers()
    erosive_markers = [
        m
        for m in markers
        if m.stratigraphicType == StratigraphicSurfaceType.EROSIVE
    ]
    assert len(erosive_markers) == 1
    m = erosive_markers[0]
    # Onset at 20.0, simulation ends at 10.0, duration=10
    assert m.name == "Hiatus_10"
    assert m.age == 20.0
    assert m.depth == pytest.approx(85.0)


def test_build_simulated_well_hiatus_at_start(
    fs_sim: FSSimulator,
) -> None:
    """Hiatus at the very start of simulation creates a marker."""
    from pywellsfm.model.Marker import StratigraphicSurfaceType

    # 2 steps: hiatus, then deposit
    fs_sim.times = [30.0, 20.0, 10.0]
    fs_sim.depo_rate_totals = [
        np.array([0.0, 0.0], dtype=np.float64),  # step 0: hiatus
        np.array([1.0, 1.0], dtype=np.float64),  # step 1: deposit
    ]
    fs_sim.depo_rate_elements = [
        [{"CarbonateShallow": 0.0, "CarbonateDeep": 0.0}] * 2,
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
    ]
    fs_sim.thickness_steps = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [np.array([15.0, 20.0], dtype=np.float64)] * 2
    fs_sim.environments = [np.array(["Shallow", "Shallow"], dtype=str)] * 2

    well = fs_sim._buildSimulatedWell(0)

    markers = well.getMarkers()
    erosive_markers = [
        m
        for m in markers
        if m.stratigraphicType == StratigraphicSurfaceType.EROSIVE
    ]
    assert len(erosive_markers) == 1
    m = erosive_markers[0]
    # Onset at age 30.0, ends at 20.0, duration=10
    assert m.name == "Hiatus_10"
    assert m.age == 30.0
    # Depth = oldest marker depth (90.0) — no deposition before hiatus
    assert m.depth == pytest.approx(90.0)


def test_build_simulated_well_consecutive_hiatus_steps_one_marker(
    fs_sim: FSSimulator,
) -> None:
    """Multiple consecutive hiatus steps produce a single marker."""
    from pywellsfm.model.Marker import StratigraphicSurfaceType

    # 4 steps: deposit, hiatus, hiatus, deposit
    fs_sim.times = [40.0, 30.0, 20.0, 10.0, 0.0]
    fs_sim.depo_rate_totals = [
        np.array([1.0, 1.0], dtype=np.float64),  # step 0: deposit
        np.array([0.0, 0.0], dtype=np.float64),  # step 1: hiatus
        np.array([0.0, 0.0], dtype=np.float64),  # step 2: hiatus
        np.array([1.0, 1.0], dtype=np.float64),  # step 3: deposit
    ]
    fs_sim.depo_rate_elements = [
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
        [{"CarbonateShallow": 0.0, "CarbonateDeep": 0.0}] * 2,
        [{"CarbonateShallow": 0.0, "CarbonateDeep": 0.0}] * 2,
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
    ]
    fs_sim.thickness_steps = [
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [np.array([15.0, 20.0], dtype=np.float64)] * 4
    fs_sim.environments = [np.array(["Shallow", "Shallow"], dtype=str)] * 4

    well = fs_sim._buildSimulatedWell(0)

    markers = well.getMarkers()
    erosive_markers = [
        m
        for m in markers
        if m.stratigraphicType == StratigraphicSurfaceType.EROSIVE
    ]
    assert len(erosive_markers) == 1
    m = erosive_markers[0]
    # Onset at 30.0, resumes at 10.0, duration=20
    assert m.name == "Hiatus_20"
    assert m.age == 30.0
    assert m.depth == pytest.approx(85.0)


def test_build_simulated_well_no_hiatus_no_erosive_markers(
    fs_sim: FSSimulator,
) -> None:
    """Continuous deposition produces no Erosive markers."""
    from pywellsfm.model.Marker import StratigraphicSurfaceType

    fs_sim.times = [30.0, 20.0, 10.0]
    fs_sim.depo_rate_totals = [
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    ]
    fs_sim.depo_rate_elements = [
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
        [{"CarbonateShallow": 0.8, "CarbonateDeep": 0.2}] * 2,
    ]
    fs_sim.thickness_steps = [
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
    ]
    fs_sim.waterDepths = [np.array([15.0, 20.0], dtype=np.float64)] * 2
    fs_sim.environments = [np.array(["Shallow", "Shallow"], dtype=str)] * 2

    well = fs_sim._buildSimulatedWell(0)

    markers = well.getMarkers()
    erosive_markers = [
        m
        for m in markers
        if m.stratigraphicType == StratigraphicSurfaceType.EROSIVE
    ]
    assert len(erosive_markers) == 0


def test_finalize_creates_simulated_wells(
    fs_sim: FSSimulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize creates one simulated well per realization."""
    fs_sim.params = FSSimulatorParameters(dt_min=1.0, dt_max=2.0)
    fs_sim.prepare()
    monkeypatch.setattr(
        fs_sim._time_step_controller,
        "adapt",
        lambda _t, _wd, _rates, remaining: min(2.0, remaining),
    )
    fs_sim.run()
    fs_sim.finalize()

    assert len(fs_sim.simulatedWells) == 2
    assert fs_sim.simulatedWells[0].name == "Well1_sim_0"
    assert fs_sim.simulatedWells[1].name == "Well2_sim_1"

    # Each well should have MainElement and Environment in both domains
    for well in fs_sim.simulatedWells:
        assert well.getAgeLog("MainElement") is not None
        assert well.getDepthLog("MainElement") is not None
        assert well.getAgeLog("Environment") is not None
        assert well.getDepthLog("Environment") is not None
        # No Facies (no faciesModel in simulation.json)
        assert well.getAgeLog("Facies") is None
        assert well.getDepthLog("Facies") is None
