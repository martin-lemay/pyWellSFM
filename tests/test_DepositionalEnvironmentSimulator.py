# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

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

from pywellsfm.simulator.DepositionalEnvironmentSimulator import (
    DepositionalEnvironmentSimulator,
    DepositionalEnvironmentSimulatorParameters,
    EnvironmentDefinition,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def simple_envs() -> list[EnvironmentDefinition]:
    """Three non-overlapping environments for basic tests."""
    return [
        EnvironmentDefinition("shallow", 0.0, 10.0),
        EnvironmentDefinition("mid", 10.0, 50.0),
        EnvironmentDefinition("deep", 50.0, 200.0),
    ]


@pytest.fixture()
def simple_envs2() -> list[EnvironmentDefinition]:
    """Five non-overlapping environments for more detailed tests."""
    return [
        EnvironmentDefinition("shallow1", 0.0, 5.0),
        EnvironmentDefinition("shallow2", 5.0, 10.0),
        EnvironmentDefinition("mid", 10.0, 50.0),
        EnvironmentDefinition("deep1", 50.0, 200.0),
        EnvironmentDefinition("deep2", 200.0, 1000.0),
    ]


@pytest.fixture()
def simple_sim(
    simple_envs: list[EnvironmentDefinition],
) -> DepositionalEnvironmentSimulator:
    """Simulator with default parameters and simple environments."""
    return DepositionalEnvironmentSimulator(simple_envs)


@pytest.fixture()
def carbonate_envs() -> list[EnvironmentDefinition]:
    """Carbonate-platform environments from :meth:`from_breakpoints`."""
    return DepositionalEnvironmentSimulator.from_breakpoints(
        lagoon_max_bathymetry=5.0,
        fairweather_wave_breaking_bathymetry=2.0,
        fairweather_wave_base_bathymetry=20.0,
        storm_wave_base_bathymetry=50.0,
        shelf_break_bathymetry=200.0,
        basin_max_bathymetry=1000.0,
    )


# ======================================================================
# EnvironmentDefinition validation
# ======================================================================


class TestEnvironmentDefinition:
    def test_valid(self: Self) -> None:
        """Test valid environment creation."""
        env = EnvironmentDefinition("a", 0.0, 10.0, weight=2.0)
        assert env.name == "a"
        assert env.bathymetry_min == 0.0
        assert env.bathymetry_max == 10.0
        assert env.weight == 2.0

    def test_mid(self: Self) -> None:
        """Test computation of bathymetry midpoint."""
        env = EnvironmentDefinition("a", 0.0, 10.0)
        assert env.bathymetry_mid == 5.0

    def test_range_width(self: Self) -> None:
        """Test computation of bathymetry range width."""
        env = EnvironmentDefinition("a", 5.0, 20.0)
        assert env.bathymetry_range_width == 15.0

    def test_invalid_range(self: Self) -> None:
        """Test invalid decreasing range raises an error."""
        with pytest.raises(ValueError, match="bathymetry_min"):
            EnvironmentDefinition("bad", 10.0, 5.0)

    def test_equal_range(self: Self) -> None:
        """Test equal min and max range raises an error."""
        with pytest.raises(ValueError, match="bathymetry_min"):
            EnvironmentDefinition("bad", 5.0, 5.0)

    def test_zero_weight(self: Self) -> None:
        """Test zero environment weight raises an error."""
        with pytest.raises(ValueError, match="weight"):
            EnvironmentDefinition("bad", 0.0, 10.0, weight=0.0)

    def test_negative_weight(self: Self) -> None:
        """Test negative environment weight raises an error."""
        with pytest.raises(ValueError, match="weight"):
            EnvironmentDefinition("bad", 0.0, 10.0, weight=-1.0)


# ======================================================================
# Constructor validation
# ======================================================================


class TestConstructor:
    def test_empty_environments(self: Self) -> None:
        """Test constructor rejects empty environment lists."""
        with pytest.raises(ValueError, match="At least one"):
            DepositionalEnvironmentSimulator([])

    def test_duplicate_names(self: Self) -> None:
        """Test constructor rejects duplicate environment names."""
        envs = [
            EnvironmentDefinition("a", 0.0, 10.0),
            EnvironmentDefinition("a", 10.0, 20.0),
        ]
        with pytest.raises(ValueError, match="unique"):
            DepositionalEnvironmentSimulator(envs)

    def test_default_params(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test simulator default parameter values.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.

        Returns:
            None: This test does not return a value.
        """
        assert simple_sim.params.bathymetry_sigma == 5.0
        assert simple_sim.params.transition_sigma == 5.0
        assert simple_sim.params.trend_sigma == 2.0
        assert simple_sim.params.trend_window == 5
        assert simple_sim.params.transition_mode == "adjacency"

    def test_invalid_trend_sigma(self: Self) -> None:
        """Test invalid trend sigma raises an error."""
        with pytest.raises(ValueError, match="trend_sigma"):
            DepositionalEnvironmentSimulatorParameters(trend_sigma=0.0)

    def test_invalid_trend_window(self: Self) -> None:
        """Test invalid trend window raises an error."""
        with pytest.raises(ValueError, match="trend_window"):
            DepositionalEnvironmentSimulatorParameters(trend_window=1)

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


# ======================================================================
# from_breakpoints
# ======================================================================


class TestFromBreakpoints:
    def test_default_breakpoints(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test default breakpoint preset content.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        names = [e.name for e in carbonate_envs]
        assert "shore" in names
        assert "lagoon" in names
        assert "reef_crest" in names
        assert "basin" in names
        assert len(carbonate_envs) == 8

    def test_all_ranges_valid(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test all preset ranges are strictly increasing.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        for env in carbonate_envs:
            assert env.bathymetry_min < env.bathymetry_max

    def test_custom_breakpoints(self: Self) -> None:
        """Test custom breakpoints override default values."""
        envs = DepositionalEnvironmentSimulator.from_breakpoints(
            lagoon_max_bathymetry=10.0,
            storm_wave_base_bathymetry=30.0,
        )
        lagoon = next(e for e in envs if e.name == "lagoon")
        assert lagoon.bathymetry_max == 10.0


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
            EnvironmentDefinition("a", 0.0, 10.0, weight=3.0),
            EnvironmentDefinition("b", 10.0, 50.0, weight=1.0),
        ]
        sim = DepositionalEnvironmentSimulator(envs)
        prior = sim.compute_prior()
        assert math.isclose(prior["a"], 0.75)
        assert math.isclose(prior["b"], 0.25)


# ======================================================================
# Bathymetry likelihood
# ======================================================================


class TestBathymetryLikelihood:
    def test_unconstrained(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test unconstrained bathymetry likelihood is uniform.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        lik = simple_sim.compute_bathymetry_likelihood()
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
            simple_sim.compute_bathymetry_likelihood(
                bathymetry_value=5.0, bathymetry_range=(0.0, 10.0)
            )

    def test_value_inside_environment(self: Self) -> None:
        """Test that nearest environment has higher likelihood."""
        envs = [
            EnvironmentDefinition("A", 0.0, 10.0),
            EnvironmentDefinition("B", 100.0, 200.0),
        ]
        sim = DepositionalEnvironmentSimulator(envs)
        # Value 5 is close to A [0,10], far from B [100,200]
        lik = sim.compute_bathymetry_likelihood(bathymetry_value=5.0)
        assert lik["A"] > lik["B"]

    def test_value_matching_env_range(self: Self) -> None:
        """Test expected likelihood for a known endpoint distance."""
        envs = [EnvironmentDefinition("X", 0.5, 10.0)]
        sim = DepositionalEnvironmentSimulator(
            envs,
            params=DepositionalEnvironmentSimulatorParameters(
                bathymetry_sigma=100.0
            ),
        )
        lik = sim.compute_bathymetry_likelihood(bathymetry_value=0.0)
        # delta = 1.5 (1 + gap of 0.5), so likelihood = exp(-0.5 * (1.5/100)^2)
        expected = math.exp(-0.5 * (1.5 / 100.0) ** 2)
        assert math.isclose(lik["X"], expected)

    def test_range_constraint(self: Self) -> None:
        """Test range-based bathymetry likelihood computation."""
        envs = [
            EnvironmentDefinition("A", 0.0, 10.0),
            EnvironmentDefinition("B", 0.0, 100.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            envs,
            params=DepositionalEnvironmentSimulatorParameters(
                bathymetry_sigma=50.0
            ),
        )
        lik = sim.compute_bathymetry_likelihood(bathymetry_range=(0.0, 10.0))
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
        lik = simple_sim.compute_bathymetry_likelihood(bathymetry_value=5000.0)
        for v in lik.values():
            assert v >= 0.0  # may underflow to 0 for extreme values


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

    def test_mode_none(
        self: Self, simple_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test transition mode `none` disables transition weighting.

        Args:
            simple_envs (list[EnvironmentDefinition]): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs,
            params=DepositionalEnvironmentSimulatorParameters(
                transition_mode="none"
            ),
        )
        lik = sim.compute_transition_likelihood("shallow")
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
        self: Self, simple_envs2: list[EnvironmentDefinition]
    ) -> None:
        """Test positive trend favors more distal environments.

        Args:
            simple_envs2 (list[EnvironmentDefinition]): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DepositionalEnvironmentSimulatorParameters(
                transition_mode="none",
                trend_sigma=0.2,
                trend_window=5,
            ),
        )
        lik = sim.compute_distality_trend_likelihood(
            ["shallow1", "shallow2", "mid"]
        )
        assert lik["shallow1"] < lik["deep2"] < lik["deep1"]

    def test_negative_trend_favours_more_proximal(
        self: Self, simple_envs2: list[EnvironmentDefinition]
    ) -> None:
        """Test negative trend favors more proximal environments.

        Args:
            simple_envs2 (list[EnvironmentDefinition]): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DepositionalEnvironmentSimulatorParameters(
                transition_mode="none",
                trend_sigma=0.2,
                trend_window=2,
            ),
        )
        lik = sim.compute_distality_trend_likelihood(
            (["deep2", "deep1", "mid"])
        )
        assert lik["shallow2"] > lik["shallow1"] > lik["deep2"]

    def test_custom_distality_mapping(
        self: Self, simple_envs2: list[EnvironmentDefinition]
    ) -> None:
        """Test custom distality mapping affects trend likelihood.

        Args:
            simple_envs2 (list[EnvironmentDefinition]): Environments fixture.
        """
        custom_distality = {
            "shallow1": 4.0,
            "shallow2": 3.0,
            "mid": 2.0,
            "deep1": 1.0,
            "deep2": 0.0,
        }
        sim = DepositionalEnvironmentSimulator(
            environments=simple_envs2,
            distality_by_environment=custom_distality,
            params=DepositionalEnvironmentSimulatorParameters(
                transition_mode="none",
                trend_sigma=0.2,
                trend_window=2,
            ),
        )
        lik = sim.compute_distality_trend_likelihood(
            ["shallow1", "shallow2", "mid"]
        )
        # In this custom mapping, trend is toward smaller values.
        assert lik["shallow1"] < lik["deep2"] < lik["deep1"]


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
        post = simple_sim.compute_posterior(bathymetry_value=5.0)
        assert math.isclose(sum(post.values()), 1.0)

    def test_unconstrained_equals_prior(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test unconstrained posterior equals prior.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        prior = simple_sim.compute_prior()
        post = simple_sim.compute_posterior()
        for name in simple_sim.environment_names:
            assert math.isclose(post[name], prior[name])

    def test_bathymetry_constraint_shifts_posterior(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test bathymetry observation shifts posterior probabilities.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        post = simple_sim.compute_posterior(bathymetry_value=5.0)
        # shallow [0,10] should dominate for bathy=5
        assert post["shallow"] > post["deep"]

    def test_combined_constraints(self: Self) -> None:
        """Test bathymetry and transition jointly affect posterior."""
        envs = [
            EnvironmentDefinition("A", 0.0, 10.0),
            EnvironmentDefinition("B", 10.0, 30.0),
            EnvironmentDefinition("C", 30.0, 100.0),
        ]
        sim = DepositionalEnvironmentSimulator(
            envs,
            params=DepositionalEnvironmentSimulatorParameters(
                bathymetry_sigma=15.0,
                transition_sigma=15.0,
            ),
        )
        post = sim.compute_posterior(
            bathymetry_value=15.0, previous_environments=["A"]
        )
        assert math.isclose(sum(post.values()), 1.0)
        # B [10,30] is best for bathy=15, and adjacent to A
        assert post["B"] > post["C"]

    def test_fallback_returns_valid_posterior(self: Self) -> None:
        """Test underflow fallback still returns a normalized posterior."""
        envs = [
            EnvironmentDefinition("A", 0.0, 1.0),
            EnvironmentDefinition("B", 1000.0, 2000.0),
        ]
        # Very tight sigma → posteriors may underflow
        sim = DepositionalEnvironmentSimulator(
            envs,
            params=DepositionalEnvironmentSimulatorParameters(
                bathymetry_sigma=0.001,
                transition_sigma=0.001,
            ),
        )
        post = sim.compute_posterior(
            bathymetry_value=500.0, previous_environments=["A"]
        )
        assert math.isclose(sum(post.values()), 1.0)

    def test_trend_influences_posterior(
        self: Self, simple_envs2: list[EnvironmentDefinition]
    ) -> None:
        """Test trend likelihood influences posterior ranking.

        Args:
            simple_envs2 (list[EnvironmentDefinition]): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DepositionalEnvironmentSimulatorParameters(
                transition_mode="none",
                trend_sigma=0.5,
                trend_window=5,
            ),
        )
        post = sim.compute_posterior(
            previous_environments=["shallow1", "shallow2", "mid"],
        )
        assert post["deep2"] > post["shallow2"] > post["shallow1"]


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
        prior_before = simple_sim.compute_prior()
        trans_before = simple_sim.compute_transition_likelihood("mid")
        post_before = simple_sim.compute_posterior(
            bathymetry_value=12.0,
            previous_environments=["mid"],
        )

        simple_sim.prepare()

        prior_after = simple_sim.compute_prior()
        trans_after = simple_sim.compute_transition_likelihood("mid")
        post_after = simple_sim.compute_posterior(
            bathymetry_value=12.0,
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
        post = simple_sim.compute_posterior(bathymetry_value=5.0)
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
        post = simple_sim.compute_posterior(bathymetry_value=5.0)
        result = simple_sim.sample_environment(post, seed=0)
        assert result in simple_sim.environment_names

    def test_rng_parameter(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test sampler accepts a NumPy random generator.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        post = simple_sim.compute_posterior(bathymetry_value=5.0)
        rng = np.random.default_rng(123)
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
            simple_sim.sample_environment(zero_post, seed=0)

    def test_degenerate_posterior(self: Self) -> None:
        """Test deterministic outcome for degenerate posterior."""
        envs = [
            EnvironmentDefinition("only", 0.0, 10.0),
            EnvironmentDefinition("other", 10.0, 20.0),
        ]
        sim = DepositionalEnvironmentSimulator(envs)
        post = {"only": 1.0, "other": 0.0}
        for seed in range(10):
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
        posterior, result = simple_sim.run(
            bathymetry_value=5.0,
            previous_environments=["shallow"],
            seed=123,
        )

        manual_posterior = simple_sim.compute_posterior(
            bathymetry_value=5.0,
            previous_environments=["shallow"],
        )
        manual_result = simple_sim.sample_environment(
            manual_posterior, seed=123
        )

        for name in simple_sim.environment_names:
            assert math.isclose(posterior[name], manual_posterior[name])
        assert result == manual_result

    def test_run_is_deterministic_with_seed(
        self: Self, simple_sim: DepositionalEnvironmentSimulator
    ) -> None:
        """Test helper `run` is deterministic for fixed seed.

        Args:
            simple_sim (DepositionalEnvironmentSimulator): Simulator fixture.
        """
        _, result1 = simple_sim.run(bathymetry_value=5.0, seed=42)
        _, result2 = simple_sim.run(bathymetry_value=5.0, seed=42)
        assert result1 == result2

    def test_run_with_previous_environments(
        self: Self, simple_envs2: list[EnvironmentDefinition]
    ) -> None:
        """Test helper `run` with previous environments history.

        Args:
            simple_envs2 (list[EnvironmentDefinition]): Environments fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            simple_envs2,
            params=DepositionalEnvironmentSimulatorParameters(
                transition_mode="none",
                trend_sigma=0.2,
                trend_window=2,
            ),
        )
        posterior, result = sim.run(
            previous_environments=["shallow1", "shallow2", "mid"],
            seed=123,
        )
        assert math.isclose(sum(posterior.values()), 1.0)
        assert result in sim.environment_names


# ======================================================================
# Integration: carbonate platform preset
# ======================================================================


class TestCarbonatePlatformIntegration:
    def test_full_workflow(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test full carbonate workflow from posterior to sampling.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)

        post = sim.compute_posterior(
            bathymetry_value=3.0, previous_environments=["reef_crest"]
        )
        assert math.isclose(sum(post.values()), 1.0)

        result = sim.sample_environment(post, seed=42)
        assert result in sim.environment_names

    def test_deep_bathymetry_favours_deep_envs(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test deep bathymetry favors deep carbonate environments.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(
            carbonate_envs,
            params=DepositionalEnvironmentSimulatorParameters(
                bathymetry_sigma=50.0,
            ),
        )
        post = sim.compute_posterior(bathymetry_value=400.0)
        # basin [200, 1000] should dominate
        shallow_total = sum(
            post[n] for n in ["shore", "lagoon", "back_reef", "reef_crest"]
        )
        assert post["basin"] > shallow_total

    def test_run_with_carbonate_platform(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test helper `run` with carbonate preset environments.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)
        posterior, result = sim.run(
            bathymetry_value=3.0,
            previous_environments=["reef_crest"],
            seed=42,
        )
        assert math.isclose(sum(posterior.values()), 1.0)
        assert result in sim.environment_names

    def test_main_env_in_carbonate_platform(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test dominant sampled environments for carbonate scenario.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)
        results_all = []
        for i in range(100):
            posterior, result = sim.run(
                bathymetry_value=10.0,
                previous_environments=["reef_crest"],
                seed=i,
            )
            results_all.append(result)
        counts = np.unique_counts(results_all)

        order = np.argsort(counts.counts)[::-1]
        sorted_values = counts.values[order]
        assert sorted_values.tolist() == [
            "fore_reef",
            "back_reef",
            "lagoon",
            "reef_crest",
            "shore",
        ]

    def test_trend_in_carbonate_platform(
        self: Self, carbonate_envs: list[EnvironmentDefinition]
    ) -> None:
        """Test trend-influenced sampled environments in carbonate case.

        Args:
            carbonate_envs (list[EnvironmentDefinition]): Preset fixture.
        """
        sim = DepositionalEnvironmentSimulator(carbonate_envs)
        results_all = []
        for i in range(100):
            posterior, result = sim.run(
                bathymetry_value=10.0,
                previous_environments=[
                    "basin",
                    "outer_platform",
                    "fore_reef",
                    "reef_crest",
                ],
                seed=i,
            )
            results_all.append(result)
        counts = np.unique_counts(results_all)

        order = np.argsort(counts.counts)[::-1]
        sorted_values = counts.values[order]
        assert sorted_values.tolist() == [
            "back_reef",
            "fore_reef",
            "lagoon",
            "reef_crest",
            "shore",
        ]
