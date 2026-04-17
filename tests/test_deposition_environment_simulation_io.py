# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from pywellsfm.io.depositional_environment_simulation_io import (
    depositionalEnvironmentSimulationToJsonObj,
    loadDepositionalEnvironmentSimulation,
    loadDepositionalEnvironmentSimulationFromJsonObj,
    loadSimulatorParametersFromJsonObj,
    loadSimulatorWeights,
    saveDepositionalEnvironmentSimulation,
    simulatorParametersToJsonObj,
)
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.model.EnvironmentConditionModel import (
    EnvironmentConditionModelUniform,
)
from pywellsfm.simulator.DepositionalEnvironmentSimulator import (
    DepositionalEnvironmentSimulator,
    DESimulatorParameters,
)
from pywellsfm.utils import IntervalDistanceMethod


def _make_environment(
    name: str,
    min_depth: float,
    max_depth: float,
) -> DepositionalEnvironment:
    return DepositionalEnvironment(
        name=name,
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth",
            min_depth,
            max_depth,
        ),
    )


@pytest.fixture()
def simple_envs() -> DepositionalEnvironmentModel:
    """Provide a minimal three-environment model for I/O tests."""
    envs = [
        _make_environment("shallow", 0.0, 10.0),
        _make_environment("mid", 10.0, 50.0),
        _make_environment("deep", 50.0, 200.0),
    ]
    return DepositionalEnvironmentModel("simple3", envs)


@pytest.fixture()
def simple_sim(
    simple_envs: DepositionalEnvironmentModel,
) -> DepositionalEnvironmentSimulator:
    """Provide a simulator fixture built from the simple model."""
    return DepositionalEnvironmentSimulator(simple_envs)


class TestSimulationIO:
    def test_json_obj_roundtrip(
        self,
        simple_envs: DepositionalEnvironmentModel,
    ) -> None:
        """Test export/load round-trip from in-memory JSON object."""
        sim = DepositionalEnvironmentSimulator(
            simple_envs,
            weights={"shallow": 3.0, "mid": 2.0, "deep": 1.0},
            params=DESimulatorParameters(
                waterDepth_sigma=7.0,
                transition_sigma=9.0,
                trend_sigma=0.5,
                trend_window=4,
                interval_distance_method=IntervalDistanceMethod.CENTER,
            ),
        )

        payload = depositionalEnvironmentSimulationToJsonObj(sim)
        loaded = loadDepositionalEnvironmentSimulationFromJsonObj(payload)

        assert loaded.environment_names == sim.environment_names
        assert loaded.params.waterDepth_sigma == 7.0
        assert loaded.params.transition_sigma == 9.0
        assert loaded.params.trend_sigma == 0.5
        assert loaded.params.trend_window == 4
        assert (
            loaded.params.interval_distance_method
            == IntervalDistanceMethod.CENTER
        )
        prior = loaded.compute_prior()
        assert math.isclose(prior["shallow"], 3.0 / 6.0)
        assert math.isclose(prior["mid"], 2.0 / 6.0)
        assert math.isclose(prior["deep"], 1.0 / 6.0)

    def test_save_and_load_file(
        self,
        tmp_path: Path,
        simple_envs: DepositionalEnvironmentModel,
    ) -> None:
        """Test save/load functions with a JSON file path."""
        sim = DepositionalEnvironmentSimulator(simple_envs)
        sim.prepare()
        out_path = Path(tmp_path) / "de_simulation.json"

        saveDepositionalEnvironmentSimulation(sim, str(out_path))
        loaded = loadDepositionalEnvironmentSimulation(str(out_path))

        assert loaded.environment_names == sim.environment_names
        assert loaded.params == sim.params

    def test_missing_weights_key_raises(
        self,
        simple_envs: DepositionalEnvironmentModel,
    ) -> None:
        """Test load raises when weights are missing env names."""
        sim = DepositionalEnvironmentSimulator(simple_envs)
        payload = depositionalEnvironmentSimulationToJsonObj(sim)
        payload["weights"] = {"shallow": 1.0, "mid": 1.0}

        with pytest.raises(ValueError, match="missing keys"):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)

    def test_unknown_weights_key_raises(
        self,
        simple_envs: DepositionalEnvironmentModel,
    ) -> None:
        """Test load raises when weights contain unknown env names."""
        sim = DepositionalEnvironmentSimulator(simple_envs)
        payload = depositionalEnvironmentSimulationToJsonObj(sim)
        payload["weights"] = {
            "shallow": 1.0,
            "mid": 1.0,
            "deep": 1.0,
            "unknown": 1.0,
        }

        with pytest.raises(ValueError, match="unknown environments"):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)


class TestSimulatorParameterParsing:
    def test_load_simulator_parameters_success(self) -> None:
        """Parse a valid simulator params object."""
        obj: dict[str, Any] = {
            "waterDepth_sigma": 7,
            "waterDepth_weight": 0.3,
            "transition_sigma": 9.0,
            "transition_weight": 0.6,
            "trend_sigma": 0.5,
            "trend_window": 4,
            "trend_weight": 0.1,
            "interval_distance_method": "center",
        }

        params = loadSimulatorParametersFromJsonObj(obj)

        assert params.waterDepth_sigma == 7.0
        assert params.transition_sigma == 9.0
        assert params.trend_sigma == 0.5
        assert params.trend_window == 4
        assert params.waterDepth_weight == 0.3
        assert params.transition_weight == 0.6
        assert params.trend_weight == 0.1
        assert params.interval_distance_method == IntervalDistanceMethod.CENTER

    @pytest.mark.parametrize(
        "obj, message",
        [
            ({"waterDepth_sigma": "bad"}, "waterDepth_sigma must be a number"),
            ({"transition_sigma": "bad"}, "transition_sigma must be a number"),
            ({"trend_sigma": "bad"}, "trend_sigma must be a number"),
            (
                {"waterDepth_weight": "bad"},
                "waterDepth_weight must be a number",
            ),
            (
                {"transition_weight": "bad"},
                "transition_weight must be a number",
            ),
            ({"trend_weight": "bad"}, "trend_weight must be a number"),
            ({"trend_window": 3.2}, "trend_window must be an integer"),
            (
                {"interval_distance_method": 1},
                "interval_distance_method must be a string",
            ),
            (
                {"interval_distance_method": "not-a-method"},
                "interval_distance_method must be one of",
            ),
            (
                {"unknown": 1.0},
                "DESimulation.params contains unsupported properties",
            ),
        ],
    )
    def test_load_simulator_parameters_rejects_invalid_values(
        self,
        obj: dict[str, Any],
        message: str,
    ) -> None:
        """Reject invalid simulator params fields and values."""
        with pytest.raises(ValueError, match=message):
            loadSimulatorParametersFromJsonObj(obj)

    def test_simulator_parameters_to_json_obj(self) -> None:
        """Serialize params into schema-compatible JSON fields."""
        params = DESimulatorParameters(
            waterDepth_sigma=7.0,
            waterDepth_weight=0.5,
            transition_sigma=8.0,
            transition_weight=0.8,
            trend_sigma=0.7,
            trend_window=6,
            trend_weight=0.4,
            interval_distance_method=IntervalDistanceMethod.GAP,
        )

        payload = simulatorParametersToJsonObj(params)

        assert payload == {
            "waterDepth_sigma": 7.0,
            "waterDepth_weight": 0.5,
            "transition_sigma": 8.0,
            "transition_weight": 0.8,
            "trend_sigma": 0.7,
            "trend_window": 6,
            "trend_weight": 0.4,
            "interval_distance_method": "gap",
        }


class TestSimulatorWeightsParsing:
    def test_load_simulator_weights_success(self) -> None:
        """Parse valid environment weights as floats."""
        parsed = loadSimulatorWeights({"A": 2, "B": 1.5})
        assert parsed == {"A": 2.0, "B": 1.5}

    @pytest.mark.parametrize(
        "obj, message",
        [
            ({"": 1.0}, "keys must be non-empty strings"),
            ({"A": "bad"}, "must be a number"),
            ({"A": 0}, "must be > 0"),
            ({"A": -1}, "must be > 0"),
        ],
    )
    def test_load_simulator_weights_rejects_invalid_values(
        self,
        obj: dict[str, Any],
        message: str,
    ) -> None:
        """Reject invalid weights entries."""
        with pytest.raises(ValueError, match=message):
            loadSimulatorWeights(obj)


class TestSimulationLoadValidation:
    def test_load_rejects_invalid_format(
        self,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Reject payloads with an unexpected format string."""
        payload = depositionalEnvironmentSimulationToJsonObj(simple_sim)
        payload["format"] = "invalid.format"

        with pytest.raises(
            ValueError,
            match="Invalid deposition environment simulation format",
        ):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)

    def test_load_rejects_invalid_version(
        self,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Reject payloads with an unsupported version."""
        payload = depositionalEnvironmentSimulationToJsonObj(simple_sim)
        payload["version"] = "2.0"

        with pytest.raises(
            ValueError,
            match="Invalid deposition environment simulation version",
        ):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)

    def test_load_rejects_extra_top_level_key(
        self,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Reject unexpected top-level DESimulation keys."""
        payload = depositionalEnvironmentSimulationToJsonObj(simple_sim)
        payload["unexpected"] = 1

        with pytest.raises(
            ValueError,
            match="DESimulation contains unsupported properties",
        ):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)

    def test_load_rejects_non_object_deposition_model(
        self,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Reject non-object depositionalEnvironmentModel values."""
        payload = depositionalEnvironmentSimulationToJsonObj(simple_sim)
        payload["depositionalEnvironmentModel"] = "bad"

        with pytest.raises(
            ValueError,
            match="depositionalEnvironmentModel must be an object",
        ):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)

    def test_load_rejects_non_object_params(
        self,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Reject params when not represented as an object."""
        payload = depositionalEnvironmentSimulationToJsonObj(simple_sim)
        payload["params"] = 1

        with pytest.raises(
            ValueError,
            match="DESimulation.params must be an object",
        ):
            loadDepositionalEnvironmentSimulationFromJsonObj(payload)

    def test_to_json_omits_default_params(
        self,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Skip params block when simulator uses defaults."""
        payload = depositionalEnvironmentSimulationToJsonObj(simple_sim)
        assert "params" not in payload


class TestSimulationFileValidation:
    def test_load_rejects_non_json_extension(self, tmp_path: Path) -> None:
        """Reject loading simulation data from non-JSON files."""
        path = tmp_path / "sim.txt"
        path.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Expected '.json', got '.txt'"):
            loadDepositionalEnvironmentSimulation(str(path))

    def test_save_rejects_non_json_extension(
        self,
        tmp_path: Path,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Reject saving simulation data to non-JSON files."""
        with pytest.raises(ValueError, match="must have a .json extension"):
            saveDepositionalEnvironmentSimulation(
                simple_sim,
                str(tmp_path / "sim.txt"),
            )

    def test_save_creates_parent_directory_and_roundtrips(
        self,
        tmp_path: Path,
        simple_sim: DepositionalEnvironmentSimulator,
    ) -> None:
        """Create parent folders during save and verify round-trip load."""
        output_path = tmp_path / "nested" / "dir" / "sim.json"

        saveDepositionalEnvironmentSimulation(simple_sim, str(output_path))
        loaded = loadDepositionalEnvironmentSimulation(str(output_path))

        assert output_path.exists()
        assert loaded.environment_names == simple_sim.environment_names
