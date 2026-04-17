# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable module-import position and import-order checks

from __future__ import annotations

import os
import sys

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

import math
from typing import Self

import numpy as np
import pytest

from pywellsfm.model.DepositionalEnvironment import (
    CarbonateProtectedRampDepositionalEnvironmentModel,
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


# ======================================================================
# Fixtures
# ======================================================================
def _make_environment(
    name: str,
    min_depth: float,
    max_depth: float,
    distality: float | None = None,
) -> DepositionalEnvironment:
    return DepositionalEnvironment(
        name=name,
        waterDepthModel=EnvironmentConditionModelUniform(
            "waterDepth",
            min_depth,
            max_depth,
        ),
        distality=distality,
    )


@pytest.fixture()
def simple_envs() -> DepositionalEnvironmentModel:
    """Three non-overlapping environments for basic tests."""
    envs = [
        _make_environment("shallow", 0.0, 10.0),
        _make_environment("mid", 10.0, 50.0),
        _make_environment("deep", 50.0, 200.0),
    ]
    return DepositionalEnvironmentModel("simple3", envs)


@pytest.fixture()
def simple_envs2() -> DepositionalEnvironmentModel:
    """Five non-overlapping environments for more detailed tests."""
    envs = [
        _make_environment("shallow1", 0.0, 5.0),
        _make_environment("shallow2", 5.0, 10.0),
        _make_environment("mid", 10.0, 50.0),
        _make_environment("deep1", 50.0, 200.0),
        _make_environment("deep2", 200.0, 1000.0),
    ]
    return DepositionalEnvironmentModel("simple5", envs)


@pytest.fixture()
def simple_sim(
    simple_envs: DepositionalEnvironmentModel,
) -> DepositionalEnvironmentSimulator:
    """Simulator with default parameters and simple environments."""
    return DepositionalEnvironmentSimulator(simple_envs)


@pytest.fixture()
def carbonate_envs() -> DepositionalEnvironmentModel:
    """Carbonate-platform environments from :meth:`from_breakpoints`."""
    model = CarbonateProtectedRampDepositionalEnvironmentModel(
        tidal_range=2.0,
        lagoon_max_waterDepth=5.0,
        fairweather_wave_base_waterDepth=20.0,
        storm_wave_base_waterDepth=50.0,
        shelf_break_waterDepth=200.0,
        slope_toe_max_waterDepth=1000.0,
    )
    return model


# ======================================================================
# Constructor validation
# ======================================================================


class TestConstructor:
    def test_invalid_waterDepth_sigma(self: Self) -> None:
        """Test invalid waterDepth sigma raises an error."""
        with pytest.raises(ValueError, match="waterDepth_sigma"):
            DESimulatorParameters(waterDepth_sigma=0.0)

    def test_invalid_transition_sigma(self: Self) -> None:
        """Test invalid transition sigma raises an error."""
        with pytest.raises(ValueError, match="transition_sigma"):
            DESimulatorParameters(transition_sigma=0.0)

    def test_invalid_trend_sigma(self: Self) -> None:
        """Test invalid trend sigma raises an error."""
        with pytest.raises(ValueError, match="trend_sigma"):
            DESimulatorParameters(trend_sigma=0.0)

    def test_invalid_trend_window(self: Self) -> None:
        """Test invalid trend window raises an error."""
        with pytest.raises(ValueError, match="trend_window"):
            DESimulatorParameters(trend_window=1)

    def test_environment_names(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test environment names are exposed in order.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        assert simple_sim.environment_names == ["shallow", "mid", "deep"]

    def test_environments_copy(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test environments property returns a defensive copy.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        envs = simple_sim.environments
        assert envs is not simple_sim.environments  # returns a copy

    def test_params_property(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test params property exposes simulator parameters."""
        assert isinstance(simple_sim.params, DESimulatorParameters)

    def test_missing_model_raises(self: Self) -> None:
        """Test constructor rejects a missing model."""
        with pytest.raises(ValueError, match="must"):
            DepositionalEnvironmentSimulator(None)  # type: ignore[arg-type]


class TestStaticHelpers:
    def test_interval_distance_all_methods(self: Self) -> None:
        """Test interval distance supports all declared methods."""
        methods = [
            IntervalDistanceMethod.GAP,
            IntervalDistanceMethod.CENTER,
            IntervalDistanceMethod.HAUSDORFF,
            IntervalDistanceMethod.WASSERSTEIN2,
            IntervalDistanceMethod.GAP_TIMES_CENTER,
            IntervalDistanceMethod.GAP_OVERLAPPING_WIDTH,
        ]
        for method in methods:
            val = DepositionalEnvironmentSimulator._interval_distance(
                0.0,
                1.0,
                2.0,
                3.0,
                method,
            )
            assert val >= 0.0

    def test_interval_distance_invalid_method(self: Self) -> None:
        """Test unsupported interval distance method raises an error."""
        with pytest.raises(ValueError, match="Unsupported"):
            DepositionalEnvironmentSimulator._interval_distance(
                0.0,
                1.0,
                2.0,
                3.0,
                method="invalid",  # type: ignore[arg-type]
            )

    def test_estimate_distality_slope_empty(self: Self) -> None:
        """Test empty series has zero estimated slope."""
        slope = DepositionalEnvironmentSimulator._estimate_distality_slope([])
        assert slope == 0.0

    def test_normalize_weights_negative_total(self: Self) -> None:
        """Test negative total weights raise an error."""
        with pytest.raises(ValueError, match="non-negative"):
            DepositionalEnvironmentSimulator._normalize_weights(
                -1.0,
                -1.0,
                -1.0,
            )

    def test_normalize_weights_zero_total(self: Self) -> None:
        """Test zero total weights trigger division by zero."""
        with pytest.raises(ZeroDivisionError):
            DepositionalEnvironmentSimulator._normalize_weights(
                0.0,
                0.0,
                0.0,
            )

    def test_weighted_likelihood_requires_names(self: Self) -> None:
        """Test names are required when likelihood is None."""
        with pytest.raises(ValueError, match="names"):
            DepositionalEnvironmentSimulator._weightedLikelihood(
                None,
                1.0,
                None,
            )


# ======================================================================
# Prior
# ======================================================================


class TestPrior:
    def test_sums_to_one(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test prior probabilities sum to one.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        prior = simple_sim.compute_prior()
        assert math.isclose(sum(prior.values()), 1.0)

    def test_uniform_weights(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test equal weights produce a uniform prior.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        prior = simple_sim.compute_prior()
        expected = 1.0 / 3.0
        for p in prior.values():
            assert math.isclose(p, expected)

    def test_non_uniform_weights(self: Self) -> None:
        """Test non-uniform weights produce weighted prior."""
        envs = [
            _make_environment("a", 0.0, 10.0),
            _make_environment("b", 10.0, 50.0),
        ]
        weights = {"a": 3.0, "b": 1.0}
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("ab", envs), weights=weights
        )
        prior = sim.compute_prior()
        assert math.isclose(prior["a"], 0.75)
        assert math.isclose(prior["b"], 0.25)


# ======================================================================
# WaterDepth likelihood
# ======================================================================


class TestWaterDepthLikelihood:
    def test_unconstrained(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test unconstrained waterDepth likelihood is uniform.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_waterDepth_likelihood()
        for v in lik.values():
            assert v == 1.0

    def test_both_supplied_raises(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test passing value and range together raises an error.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        with pytest.raises(ValueError, match="not both"):
            simple_sim.compute_waterDepth_likelihood(
                waterDepth_value=5.0, waterDepth_range=(0.0, 10.0)
            )

    def test_value_inside_environment(self: Self) -> None:
        """Test that nearest environment has higher likelihood."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 100.0, 200.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs)
        )
        # Value 5 is close to A [0,10], far from B [100,200]
        lik = sim.compute_waterDepth_likelihood(waterDepth_value=5.0)
        assert lik["A"] > lik["B"]

    def test_value_matching_env_range(self: Self) -> None:
        """Test expected likelihood for a known endpoint distance."""
        envs = [_make_environment("X", 0.5, 10.0)]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("X", envs),
            params=DESimulatorParameters(waterDepth_sigma=100.0),
        )
        lik = sim.compute_waterDepth_likelihood(waterDepth_value=0.0)
        # delta = 1.5 (1 + gap of 0.5), so likelihood = exp(-0.5 * (1.5/100)^2)
        expected = math.exp(-0.5 * (1.5 / 100.0) ** 2)
        assert math.isclose(lik["X"], expected)

    def test_range_constraint(self: Self) -> None:
        """Test range-based waterDepth likelihood computation."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 0.0, 100.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs),
            params=DESimulatorParameters(waterDepth_sigma=50.0),
        )
        lik = sim.compute_waterDepth_likelihood(waterDepth_range=(0.0, 10.0))
        # A: delta=0, B: delta=|0-0|+|10-100|=90
        assert lik["A"] == 1.0  # exact match
        assert lik["A"] > lik["B"]

    def test_all_positive(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test Gaussian likelihoods are non-negative.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_waterDepth_likelihood(waterDepth_value=5000.0)
        for v in lik.values():
            assert v >= 0.0  # may underflow to 0 for extreme values

    def test_negative_obs_excludes_nonnegative_env(self: Self) -> None:
        """Test negative values exclude non-negative-only environments."""
        envs = [
            _make_environment("shore", 0.0, 10.0),
            _make_environment("subaerial", -10.0, -1.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("shore_subaerial", envs)
        )
        lik = sim.compute_waterDepth_likelihood(waterDepth_value=-2.0)
        assert lik["shore"] == 0.0
        assert lik["subaerial"] > 0.0


# ======================================================================
# Transition likelihood
# ======================================================================


class TestTransitionLikelihood:
    def test_none_previous(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test no previous environment gives neutral likelihood.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_transition_likelihood(None)
        for v in lik.values():
            assert v == 1.0

    def test_same_env_highest(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test staying in same environment has highest transition score.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_transition_likelihood("shallow")
        assert lik["shallow"] > lik["mid"]
        assert lik["shallow"] > lik["deep"]

    def test_adjacency_ordering(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test adjacent environments are favored over distant ones.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_transition_likelihood("shallow")
        assert lik["mid"] > lik["deep"]

    def test_unknown_env_raises(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test unknown previous environment raises an error.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        with pytest.raises(ValueError, match="Unknown"):
            simple_sim.compute_transition_likelihood("nonexistent")

    def test_self_likelihood_is_one(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test self-transition likelihood equals one.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_transition_likelihood("mid")
        assert lik["mid"] == 1.0


# ======================================================================
# Distality trend likelihood
# ======================================================================


class TestDistalityTrendLikelihood:
    def test_no_history_is_unconstrained(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test missing history yields neutral trend likelihood.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_distality_trend_likelihood(None)
        for value in lik.values():
            assert value == 1.0

    def test_short_history_is_unconstrained(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test short history yields neutral trend likelihood.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_distality_trend_likelihood(["mid"])
        for value in lik.values():
            assert value == 1.0

    def test_unknown_history_environment_raises(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test unknown environment in history raises an error.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        with pytest.raises(ValueError, match="Unknown environment"):
            simple_sim.compute_distality_trend_likelihood(["mid", "unknown"])

    def test_positive_trend_favours_more_distal(
        self: Self, simple_envs2: DepositionalEnvironmentModel
    ) -> None:
        """Test positive trend favors more distal environments.

        Args:
            simple_envs2 (DepositionalEnvironmentModel): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DESimulatorParameters(
                transition_weight=0.0,  # disable transition influence
                trend_sigma=0.2,
                trend_window=5,
            ),
        )
        sim.prepare()
        lik = sim.compute_distality_trend_likelihood(
            ["shallow1", "shallow2", "mid"]
        )
        assert lik["shallow1"] < lik["deep2"] < lik["deep1"]

    def test_negative_trend_favours_more_proximal(
        self: Self, simple_envs2: DepositionalEnvironmentModel
    ) -> None:
        """Test negative trend favors more proximal environments.

        Args:
            simple_envs2 (DepositionalEnvironmentModel): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DESimulatorParameters(
                transition_weight=0.0,  # disable transition influence
                trend_sigma=0.2,
                trend_window=2,
            ),
        )
        sim.prepare()
        lik = sim.compute_distality_trend_likelihood(
            (["deep2", "deep1", "mid"])
        )
        assert lik["shallow2"] > lik["shallow1"] > lik["deep2"]

    def test_custom_distality_mapping(
        self: Self, simple_envs2: DepositionalEnvironmentModel
    ) -> None:
        """Test custom distality mapping affects trend likelihood.

        Args:
            simple_envs2 (DepositionalEnvironmentModel): Environments fixture.
        """
        envs_copy = [
            _make_environment(e.name, e.waterDepth_min, e.waterDepth_max)
            for e in simple_envs2.environments
        ]
        envs_copy[0].distality = 4.0
        envs_copy[1].distality = 3.0
        envs_copy[2].distality = 2.0
        envs_copy[3].distality = 1.0
        envs_copy[4].distality = 0.0
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("simple5_copy", envs_copy),
            params=DESimulatorParameters(
                transition_weight=0.0,  # disable transition influence
                trend_sigma=0.2,
                trend_window=2,
            ),
        )
        sim.prepare()
        lik = sim.compute_distality_trend_likelihood(
            ["shallow1", "shallow2", "mid"]
        )
        # In this custom mapping, trend is toward smaller values.
        assert lik["shallow1"] < lik["deep2"] < lik["deep1"]

    def test_empty_distality_map_is_unconstrained(
        self: Self, simple_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test empty distality map yields neutral trend likelihood."""
        sim = DepositionalEnvironmentSimulator(simple_envs)
        lik = sim.compute_distality_trend_likelihood(["shallow", "mid"])
        for value in lik.values():
            assert value == 1.0

    def test_flat_slope_is_unconstrained(
        self: Self, simple_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test near-flat trend slope yields neutral trend likelihood."""
        sim = DepositionalEnvironmentSimulator(simple_envs)
        sim.prepare()
        sim._trend_threshold = 10.0
        lik = sim.compute_distality_trend_likelihood(["shallow", "mid"])
        for value in lik.values():
            assert value == 1.0

    def test_two_step_trend_sign_branches(
        self: Self, simple_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test two-step trend uses sign-based mismatch branch."""
        sim = DepositionalEnvironmentSimulator(simple_envs)
        sim.prepare()
        lik = sim.compute_distality_trend_likelihood(["shallow", "mid"])
        assert lik["mid"] == 1.0
        assert lik["deep"] == 1.0
        assert lik["shallow"] < 1.0


# ======================================================================
# Posterior
# ======================================================================


class TestPosterior:
    def test_sums_to_one(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test posterior probabilities sum to one.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        post = simple_sim.compute_posterior(waterDepth_value=5.0)
        assert math.isclose(sum(post.values()), 1.0)

    def test_unconstrained_equals_prior(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test unconstrained posterior equals prior.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        prior = simple_sim.compute_prior()
        post = simple_sim.compute_posterior()
        for name in simple_sim.environment_names:
            assert math.isclose(post[name], prior[name])

    def test_waterDepth_constraint_shifts_posterior(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test waterDepth observation shifts posterior probabilities.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        post = simple_sim.compute_posterior(waterDepth_value=5.0)
        # shallow [0,10] should dominate for waterDepth=5
        assert post["shallow"] > post["deep"]

    def test_combined_constraints(self: Self) -> None:
        """Test waterDepth and transition jointly affect posterior."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 30.0),
            _make_environment("C", 30.0, 100.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("ABC", envs),
            params=DESimulatorParameters(
                waterDepth_sigma=15.0,
                transition_sigma=15.0,
            ),
        )
        sim.prepare()
        post = sim.compute_posterior(
            waterDepth_value=15.0, previous_environments=["A"]
        )
        assert math.isclose(sum(post.values()), 1.0)
        # B [10,30] is best for waterDepth=15, and adjacent to A
        assert post["B"] > post["C"]

    def test_fallback_returns_valid_posterior(self: Self) -> None:
        """Test underflow fallback still returns a normalized posterior."""
        envs = [
            _make_environment("A", 0.0, 1.0),
            _make_environment("B", 1000.0, 2000.0),
        ]
        # Very tight sigma → posteriors may underflow
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs),
            params=DESimulatorParameters(
                waterDepth_sigma=0.001,
                transition_sigma=0.001,
            ),
        )
        sim.prepare()
        post = sim.compute_posterior(
            waterDepth_value=500.0, previous_environments=["A"]
        )
        assert math.isclose(sum(post.values()), 1.0)

    def test_trend_influences_posterior(
        self: Self, simple_envs2: DepositionalEnvironmentModel
    ) -> None:
        """Test trend likelihood influences posterior ranking.

        Args:
            simple_envs2 (DepositionalEnvironmentModel): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DESimulatorParameters(
                transition_weight=0.0,  # disable transition influence
                trend_sigma=0.5,
                trend_window=5,
            ),
        )
        sim.prepare()
        post = sim.compute_posterior(
            previous_environments=["shallow1", "shallow2", "mid"],
        )
        assert post["deep2"] > post["shallow2"] > post["shallow1"]

    def test_fallback_1a_returns_relaxed_trend(self: Self) -> None:
        """Test fallback 1a returns once relaxed trend is non-zero."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 20.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs)
        )
        outputs = [
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 1.0, "B": 0.0}, 1.0),
        ]

        def fake_compute(
            *_args: object, **_kwargs: object
        ) -> tuple[dict[str, float], float]:
            return outputs.pop(0)

        sim._compute_posterior = fake_compute  # type: ignore[method-assign]
        post = sim.compute_posterior(
            waterDepth_value=5.0,
            previous_environments=["A", "B"],
        )
        assert post == {"A": 1.0, "B": 0.0}

    def test_fallback_1b_returns_without_trend(self: Self) -> None:
        """Test fallback 1b returns after dropping trend likelihood."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 20.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs)
        )
        outputs = [
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 1.0}, 1.0),
        ]

        def fake_compute(
            *_args: object, **_kwargs: object
        ) -> tuple[dict[str, float], float]:
            return outputs.pop(0)

        sim._compute_posterior = fake_compute  # type: ignore[method-assign]
        post = sim.compute_posterior(
            waterDepth_value=5.0,
            previous_environments=["A", "B"],
        )
        assert post == {"A": 0.0, "B": 1.0}

    def test_fallback_2a_returns_relaxed_transition(self: Self) -> None:
        """Test fallback 2a returns once transition is relaxed."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 20.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs)
        )
        outputs = [
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 1.0, "B": 0.0}, 1.0),
        ]

        def fake_compute(
            *_args: object, **_kwargs: object
        ) -> tuple[dict[str, float], float]:
            return outputs.pop(0)

        sim._compute_posterior = fake_compute  # type: ignore[method-assign]
        post = sim.compute_posterior(
            waterDepth_value=5.0,
            previous_environments=["A", "B"],
        )
        assert post == {"A": 1.0, "B": 0.0}

    def test_fallback_2b_returns_without_transition(self: Self) -> None:
        """Test fallback 2b returns after dropping transition."""
        envs = [
            _make_environment("A", 0.0, 10.0),
            _make_environment("B", 10.0, 20.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs)
        )
        outputs = [
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 0.0}, 0.0),
            ({"A": 0.0, "B": 1.0}, 1.0),
        ]

        def fake_compute(
            *_args: object, **_kwargs: object
        ) -> tuple[dict[str, float], float]:
            return outputs.pop(0)

        sim._compute_posterior = fake_compute  # type: ignore[method-assign]
        post = sim.compute_posterior(
            waterDepth_value=5.0,
            previous_environments=["A", "B"],
        )
        assert post == {"A": 0.0, "B": 1.0}


# ======================================================================
# Prepare
# ======================================================================


class TestPrepare:
    def test_prepare_keeps_results_identical(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test cache preparation preserves computed probabilities.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        prior_before = simple_sim.compute_prior()
        trans_before = simple_sim.compute_transition_likelihood("mid")
        post_before = simple_sim.compute_posterior(
            waterDepth_value=12.0,
            previous_environments=["mid"],
        )

        simple_sim.prepare()
        prior_after = simple_sim.compute_prior()
        trans_after = simple_sim.compute_transition_likelihood("mid")
        post_after = simple_sim.compute_posterior(
            waterDepth_value=12.0,
            previous_environments=["mid"],
        )

        for name in simple_sim.environment_names:
            assert math.isclose(prior_before[name], prior_after[name])
            assert math.isclose(trans_before[name], trans_after[name])
            assert math.isclose(post_before[name], post_after[name])

    def test_prepare_caches_none_transition(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test prepared cache returns neutral transition for `None`.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        lik = simple_sim.compute_transition_likelihood(None)
        for value in lik.values():
            assert value == 1.0


# ======================================================================
# Sampling
# ======================================================================


class TestSampling:
    def test_deterministic_seed(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test sampling is deterministic for a fixed seed.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        post = simple_sim.compute_posterior(waterDepth_value=5.0)
        result1 = simple_sim.sample_environment(post, seed=42)
        result2 = simple_sim.sample_environment(post, seed=42)
        assert result1 == result2

    def test_result_in_environments(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test sampled environment belongs to known environments.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        post = simple_sim.compute_posterior(waterDepth_value=5.0)
        result = simple_sim.sample_environment(post, seed=0)
        assert result in simple_sim.environment_names

    def test_rng_parameter(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test sampler accepts a NumPy random generator.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        post = simple_sim.compute_posterior(waterDepth_value=5.0)
        rng = np.random.default_rng(123)
        simple_sim.prepare()
        result = simple_sim.sample_environment(post, rng=rng)
        assert result in simple_sim.environment_names

    def test_all_zero_raises(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test zero-probability posterior raises an error.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        zero_post = dict.fromkeys(simple_sim.environment_names, 0.0)
        with pytest.raises(ValueError, match="zero"):
            simple_sim.prepare()
            simple_sim.sample_environment(zero_post, seed=0)

    def test_degenerate_posterior(self: Self) -> None:
        """Test deterministic outcome for degenerate posterior."""
        envs = [
            _make_environment("only", 0.0, 10.0),
            _make_environment("other", 10.0, 20.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            DepositionalEnvironmentModel("AB", envs)
        )
        post = {"only": 1.0, "other": 0.0}
        for seed in range(10):
            sim.prepare()
            assert sim.sample_environment(post, seed=seed) == "only"


# ======================================================================
# Run helper
# ======================================================================


class TestRun:
    def test_run_matches_manual_workflow(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test helper `run` matches manual posterior+sample workflow.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        posterior, result = simple_sim.run(
            waterDepth_value=5.0,
            previous_environments=["shallow"],
            seed=123,
        )

        manual_posterior = simple_sim.compute_posterior(
            waterDepth_value=5.0,
            previous_environments=["shallow"],
        )
        manual_result = simple_sim.sample_environment(
            manual_posterior, seed=123
        )

        for name in simple_sim.environment_names:
            assert math.isclose(posterior[name], manual_posterior[name]), (
                f"Posterior mismatch for environment '{name}'"
            )
        assert result is not None
        assert result.name == manual_result

    def test_run_is_deterministic_with_seed(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test helper `run` is deterministic for fixed seed.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        simple_sim.prepare()
        _, result1 = simple_sim.run(waterDepth_value=5.0, seed=42)
        _, result2 = simple_sim.run(waterDepth_value=5.0, seed=42)
        assert result1 == result2

    def test_run_with_previous_environments(
        self: Self, simple_envs2: DepositionalEnvironmentModel
    ) -> None:
        """Test helper `run` with previous environments history.

        Args:
            simple_envs2 (DepositionalEnvironmentModel): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DESimulatorParameters(
                transition_weight=0.0,  # disable transition influence
                trend_sigma=0.2,
                trend_window=2,
            ),
        )
        sim.prepare()
        posterior, result = sim.run(
            previous_environments=["shallow1", "shallow2", "mid"],
            seed=123,
        )
        assert math.isclose(sum(posterior.values()), 1.0)
        assert result is not None
        assert result.name in sim.environment_names

    def test_run_out_of_model_range_returns_none(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test out-of-range waterDepth interval returns None result."""
        simple_sim.prepare()
        posterior, result = simple_sim.run(waterDepth_range=(-5.0, 500.0))
        assert result is None
        assert set(posterior.values()) == {0.0}


# ======================================================================
# Integration: carbonate platform preset
# ======================================================================


class TestCarbonatePlatformIntegration:
    def test_full_workflow(
        self: Self, carbonate_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test full carbonate workflow from posterior to sampling.

        Args:
            carbonate_envs (DepositionalEnvironmentModel): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)

        post = sim.compute_posterior(
            waterDepth_value=3.0, previous_environments=["ReefCrest"]
        )
        assert math.isclose(sum(post.values()), 1.0)

        result = sim.sample_environment(post, seed=42)
        assert result in sim.environment_names

    def test_deep_waterDepth_favours_deep_envs(
        self: Self, carbonate_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test deep waterDepth favors deep carbonate environments.

        Args:
            carbonate_envs (DepositionalEnvironmentModel): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            carbonate_envs,
            params=DESimulatorParameters(
                waterDepth_sigma=50.0,
            ),
        )
        post = sim.compute_posterior(waterDepth_value=400.0)
        # Basin [200, 1000] should dominate
        shallow_total = sum(
            post[n]
            for n in ["SupraTidal", "Shore", "Lagoon", "BackReef", "ReefCrest"]
        )
        assert sum((post["ShelfSlope"], post["Basin"])) > shallow_total

    def test_run_with_carbonate_platform(
        self: Self, carbonate_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test helper `run` with carbonate preset environments.

        Args:
            carbonate_envs (DepositionalEnvironmentModel): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)
        posterior, result = sim.run(
            waterDepth_value=3.0,
            previous_environments=["ReefCrest"],
            seed=42,
        )
        assert math.isclose(sum(posterior.values()), 1.0)
        assert result is not None
        assert result.name in sim.environment_names

    def test_main_env_in_carbonate_platform(
        self: Self, carbonate_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test dominant sampled environments for carbonate scenario.

        Args:
            carbonate_envs (DepositionalEnvironmentModel): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)
        results_all = []
        for i in range(100):
            posterior, result = sim.run(
                waterDepth_value=10.0,
                previous_environments=["ReefCrest"],
                seed=i,
            )
            assert result is not None
            results_all.append(result.name)
        counts = np.unique_counts(results_all)

        order = np.argsort(counts.counts)[::-1]
        sorted_values = counts.values[order]
        print(sorted_values)
        assert sorted_values.tolist() == ["ForeReef", "Lagoon", "Buildup"]

    def test_trend_in_carbonate_platform(
        self: Self, carbonate_envs: DepositionalEnvironmentModel
    ) -> None:
        """Test trend-influenced sampled environments in carbonate case.

        Args:
            carbonate_envs (DepositionalEnvironmentModel): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)
        results_all = []
        for i in range(100):
            sim.prepare()
            _, result = sim.run(
                waterDepth_value=3.0,
                previous_environments=[
                    "ShelfSlope",
                    "OuterRamp",
                    "ForeReef",
                    "ReefCrest",
                ],
                seed=i,
            )
            assert result is not None
            results_all.append(result.name)
        counts = np.unique_counts(results_all)

        order = np.argsort(counts.counts)[::-1]
        sorted_values = counts.values[order]
        print(sorted_values)
        assert sorted_values.tolist() == [
            "ReefCrest",
            "ForeReef",
            "BackReef",
            "Buildup",
        ]
