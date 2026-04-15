# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Self

import numpy as np

from pywellsfm.model import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from pywellsfm.utils import (
    IntervalDistanceMethod,
    center_distance,
    gap_distance,
    gap_overlapping_width_distance,
    gap_times_center_distance,
    hausdorff_distance,
    wasserstein2_distance,
)

__doc__ = """
This module contains the DepositionalEnvironmentSimulator class, which
simulates the depositional environment at a given point based on its location,
waterDepth, and energy conditions.

Let's assume a list of possible depositional environments. Each environment is
defined by its location on a 2D map, and by waterDepth and energy ranges.
For instance, for a protected carbonate platform, the list includes from
proximal to distal:

- shore: between the subaerial beach and the lagoon, waterDepth <5m, low energy
  zone.
- lagoon: between shore and reef environments, waterDepth <15m, very low energy
  zone.
- back reef: located between laggon and reef crest, waterDepth <5m,
  intermediate energy.
- reef crest: very high energy zone (wave swash zone), waterDepth <2m
- fore reef: in the front of the reef, high energy zone (wave surf zone),
  waterDepth 2-20m
- outer platform: more distal than fore reef, before shelf, low energy zone
  (intermediate if storms), waterDepth 20-50m
- shelf: more distal than outer platform, waterDepth 50-100s m, very low energy
- basin: deep than shelf, waterDepth >100s m, very low energy.

Exact waterDepth ranges depends on user inputs including lagoon maximum depth,
fairweather wave breaking depth, fairweather wave base depth, storm wave base
depth, and shelf break depth.

The Depositional environment simulator stochastically simulate in which zone a
given point is. The unconstrained case uses waterDepth ranges to compute
probabilities to be in a given zone, then pick one of the zones according to
these probabilities.

A constrained simulation by the waterDepth at the given point, recompute the
probabilities such as the given waterDepth at the point preferentially fall
within the range of the selected zone.

A forward constrained simulation computes the probabilities by assuming
a progressive transition from one zone to another one. So at a given point,
probabilities are computed according to:

- the selected environment at the previous step. The probability to be in given
  environment increases if this environment is the same (1.0) or adjacent to
  the previous one. Adjacency is computed by combining water depth range and
  distality (if defined) differences.
- the distality trend from previously selected environments. The probability to
  be in a given environment increases if the trend (transgressive or
  regressive) is respected. In pratice, the probability increases as the
  variance of the derivative of the distality increases.

All constraints can be combined together.

The conditioning algorithm is based on bayesian approach, where the prior
probabilities are computed based on the waterDepth ranges defined for each
environment, and the likelihood is computed based on the waterDepth constraint
and the previous environment constraints. The posterior probabilities are then
computed by multiplying the prior and the likelihood, and normalizing the
result. Finally, the environment is selected by sampling from the posterior
probabilities.

Likelihoods can be weighted to control the influence of each constraint
(i.e., water depth, transition, distality trend). When a constraint weight
differs from 1, the initial likelihood values for the given constraint are
weighted against 1 (unconstrained), so that the more the weight, the more the
likelihood values are close to the initial ones, and the less the weight,
the more the likelihood values are close to 1 for all environments
(unconstrained). By default, all weights are equal (1.0), meaning that all
constraints are equally important.


This module contains the :class:`DepositionalEnvironmentSimulator` class,
which selects a depositional environment (a discrete state) at a *given
time* and *given location/order* using Bayesian conditioning.

The algorithm combines:

- **Prior** probabilities computed from environment weights.
- **waterDepth likelihood** from a measured or uncertain waterDepth value /
  range.
- **Transition likelihood** from an optional *previous-step* environment
  (Walther's law / spatial continuity).
- **Distality trend likelihood** from an optional list of *previous-step*
  environments (progressive transitions).
- **Weights** for each likelihood to control their influence.

Posterior:

.. math::

    P(e \\mid D, S) \\propto P(e)\\,P(D \\mid e)\\,P(S \\mid e)\\,P(T \\mid e)

where:

- :math:`P(e)` is the prior probability of environment :math:`e`.
- :math:`P(D \\mid e)` is the water depth likelihood of environment :math:`e`
    given waters depth evidence :math:`D`.
    :math:`P(D \\mid e) = wdWeight P(D \\mid e)_0 + (1 - wdWeight)`,
    where :math:`P(D \\mid e)_0` is the initial likelihood computed from the
    water depth ranges, and :math:`wdWeight` is the weight for the waterDepth
    likelihood.
- :math:`P(S \\mid e)` is the transition likelihood of environment :math:`e`
    given previous environment evidence :math:`S`.
    :math:`P(S \\mid e) = transWeight P(S \\mid e)_0 + (1 - transWeight)`,
    where :math:`P(S \\mid e)_0` is the initial likelihood computed from the
    previous environment, and :math:`transWeight` is the weight for the
    transition likelihood.
- :math:`P(T \\mid e)` is the distality trend likelihood of environment
    :math:`e` given previous environment history evidence :math:`T`.
    :math:`P(T \\mid e) = trendWeight P(T \\mid e)_0 + (1 - trendWeight)`,
    where :math:`P(T \\mid e)_0` is the initial likelihood computed from the
    previous environment history, and :math:`trendWeight` is the weight for the
    distality trend likelihood.


"""


# ======================================================================
# Data classes
# ======================================================================


@dataclass(frozen=True)
class DESimulatorParameters:
    """Configuration parameters for :class:`DepositionalEnvironmentSimulator`.

    :param float waterDepth_sigma: standard-deviation (metres) for the
        Gaussian waterDepth likelihood kernel.  Controls tolerance to
        waterDepth mismatches.
    :param float waterDepth_weight: weight for the waterDepth likelihood in
        the posterior computation.
    :param float transition_sigma: standard-deviation (metres) for the
        Gaussian transition (adjacency) likelihood kernel.
    :param float transition_weight: weight for the transition likelihood in
        the posterior computation.
    :param float trend_sigma: standard-deviation for the Gaussian distality
        trend likelihood kernel. Controls tolerance to distality trend
        mismatches.
    :param int trend_window: number of previous environments to consider for
        distality trend likelihood.
    :param float trend_weight: weight for the distality trend likelihood in
        the posterior computation.
    :param IntervalDistanceMethod interval_distance_method: method to compute
        the distance between waterDepth intervals for likelihood
        computation.
    """

    waterDepth_sigma: float = 2.0
    waterDepth_weight: float = 1.0
    transition_sigma: float = 1.0
    transition_weight: float = 1.0
    trend_sigma: float = 2.0
    trend_window: int = 5
    trend_weight: float = 1.0
    interval_distance_method: IntervalDistanceMethod = (
        IntervalDistanceMethod.GAP_OVERLAPPING_WIDTH
    )

    def __post_init__(self: Self) -> None:
        """Validate parameter constraints."""
        if self.waterDepth_sigma <= 0:
            raise ValueError("waterDepth_sigma must be > 0.")
        if self.transition_sigma <= 0:
            raise ValueError("transition_sigma must be > 0.")
        if self.trend_sigma <= 0:
            raise ValueError("trend_sigma must be > 0.")
        if self.trend_window < 2:
            raise ValueError("trend_window must be >= 2.")


# ======================================================================
# Simulator
# ======================================================================


class DepositionalEnvironmentSimulator:
    """Bayesian simulator that selects a depositional environment.

    At a *single* time-slice and location the simulator:

    1. Computes **prior** probabilities from environment weights.
    2. Computes **waterDepth likelihood** from a measured (or uncertain)
       waterDepth value / range.
    3. Computes **transition likelihood** from an optional
       *previous-step* environment (spatial continuity / Walther's law).
    4. Multiplies prior × likelihoods and normalises to obtain the
       **posterior**.
    5. **Samples** one environment from the posterior.

    :param list[DepositionalEnvironment] environments: non-empty list of
        environment definitions with unique names. Consider ordering
        environments from proximal to distal if
        *DepositionalEnvironment.distality* is not provided.
    :param dict[str, float] | None weights: optional explicit
        mapping from environment name to weights.
        If ``None``, weights are equal to 1.0 for all environments.
    :param DESimulatorParameters | None params:
        simulator tuning knobs; when ``None`` the defaults are used.

    :raises ValueError: if *environments* is empty or contains
        duplicate names.
    """

    def __init__(
        self: Self,
        depositionalEnvironmentModel: DepositionalEnvironmentModel,
        weights: dict[str, float] | None = None,
        params: DESimulatorParameters | None = None,
    ) -> None:
        """Initialise the simulator with environments and parameters."""
        if not depositionalEnvironmentModel:
            raise ValueError(
                "A depositional environment model must " + " be provided."
            )
        #: depositional environment model with environment definitions
        self.depositionalEnvironmentModel = depositionalEnvironmentModel

        #: list of environment names. The order is important for
        # distality trend likelihoods if no distality was defined.
        self._names = [
            e.name for e in depositionalEnvironmentModel.environments
        ]
        #: mapping from environment name to definition
        self._environments: dict[str, DepositionalEnvironment] = {
            e.name: e for e in depositionalEnvironmentModel.environments
        }
        self._weights: dict[str, float] = {}
        if weights is not None:
            self._weights = dict(weights)
        else:
            # use equal weights if none provided
            self._weights = dict.fromkeys(self._names, 1.0)

        # populated by prepare()
        #: mapping of environment name to distality value.
        self._distality_by_environment: dict[str, float] = {}

        #: simulator parameters
        self._params: DESimulatorParameters = params or DESimulatorParameters()

        # cached prior probabilities and likelihoods for efficiency;
        # populated by prepare()
        self._cached_prior: dict[str, float] | None = None
        self._cached_transition_likelihood: dict[
            str | None, dict[str, float]
        ] = {}
        # threshold for considering a distality trend as significant
        self._trend_threshold: float = 0.01

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def environments(self: Self) -> list[DepositionalEnvironment]:
        """Return a copy of the environments."""
        return list(self.depositionalEnvironmentModel.environments)

    @property
    def environment_names(self: Self) -> list[str]:
        """Return a copy of the environment name list."""
        return list(self._names)

    @property
    def params(self: Self) -> DESimulatorParameters:
        """Return the simulator parameters."""
        return self._params

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------
    # Gaussian kernel for likelihood computation
    @staticmethod
    def _gaussian_kernel(delta: float, sigma: float) -> float:
        r"""Unnormalised Gaussian kernel.

        .. math::

            k = \exp\!\left(-\frac{\delta^2}{2\,\sigma^2}\right)
        """
        return math.exp(-0.5 * (delta / sigma) ** 2)

    # Helper for distance between intervals (used for water depth ranges)
    @staticmethod
    def _interval_distance(
        min1: float,
        max1: float,
        min2: float,
        max2: float,
        method: IntervalDistanceMethod = IntervalDistanceMethod.WASSERSTEIN2,
    ) -> float:
        """Compute the interval distance between two intervals.

        Returns 0 when the two intervals are identical.
        """
        match method:
            case IntervalDistanceMethod.GAP:
                return gap_distance(min1, max1, min2, max2)
            case IntervalDistanceMethod.CENTER:
                return center_distance(min1, max1, min2, max2)
            case IntervalDistanceMethod.HAUSDORFF:
                return hausdorff_distance(min1, max1, min2, max2)
            case IntervalDistanceMethod.WASSERSTEIN2:
                return wasserstein2_distance(min1, max1, min2, max2)
            case IntervalDistanceMethod.GAP_TIMES_CENTER:
                return gap_times_center_distance(min1, max1, min2, max2)
            case IntervalDistanceMethod.GAP_OVERLAPPING_WIDTH:
                return gap_overlapping_width_distance(min1, max1, min2, max2)
            case _:
                raise ValueError(
                    f"Unsupported interval distance method: {method}"
                )

    # helper to estimate distality trend from a series of distality values
    @staticmethod
    def _estimate_distality_slope(distality_series: list[float]) -> float:
        """Estimate the trend slope from a distality history series."""
        # TODO: remove. Use derivative, the closer the better.
        if len(distality_series) < 2:
            # no trend defined
            return 0.0

        if len(distality_series) == 2:
            return distality_series[1] - distality_series[0]

        x = np.arange(len(distality_series), dtype=float)
        y = np.asarray(distality_series, dtype=float)
        return float(np.polyfit(x, y, deg=1)[0])

    # helper to normalize likelihoods
    @staticmethod
    def _normalize(
        values: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Normalise a dict of non-negative values to sum to 1.

        Returns ``(normalised_dict, total_before_normalisation)``.
        """
        total = sum(values.values())
        if total > 0:
            return {k: v / total for k, v in values.items()}, total
        return values, 0.0

    # helper to normalize constraint weights
    @staticmethod
    def _normalize_weights(
        waterDepth_weight: float, transition_weight: float, trend_weight: float
    ) -> tuple[float, float, float]:
        """Normalize a dict of non-negative weights to sum to 1."""
        total = waterDepth_weight + transition_weight + trend_weight
        if total < 0:
            raise ValueError("Weights must be non-negative.")

        if total == 0:
            total = 1.0

        waterDepth_weightNorm = waterDepth_weight / total
        transition_weightNorm = transition_weight / total
        trend_weightNorm = trend_weight / total
        maxi = max(
            waterDepth_weightNorm, transition_weightNorm, trend_weightNorm
        )
        return (
            waterDepth_weightNorm / maxi,
            transition_weightNorm / maxi,
            trend_weightNorm / maxi,
        )

    # helper to apply weighting to likelihoods,
    # with fallback to 1.0 if likelihood is None
    @staticmethod
    def _weightedLikelihood(
        likelihood: dict[str, float] | None,
        weight: float,
        names: list[str] | None = None,
    ) -> dict[str, float]:
        if likelihood is None:
            if names is None:
                raise ValueError(
                    "If likelihood is None, names must be provided."
                )
            return dict.fromkeys(names, 1.0)
        return {k: v * weight + (1 - weight) for k, v in likelihood.items()}

    # helper to apply weighting to all likelihoods at once,
    # with fallback to 1.0 if likelihood is None
    @staticmethod
    def _weightedLikelihoods(
        L_bathy: dict[str, float] | None,
        L_trans: dict[str, float] | None,
        L_trend: dict[str, float] | None,
        waterDepth_weight: float,
        transition_weight: float,
        trend_weight: float,
        names: list[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        weights = DepositionalEnvironmentSimulator._normalize_weights(
            waterDepth_weight if L_bathy is not None else 0.0,
            transition_weight if L_trans is not None else 0.0,
            trend_weight if L_trend is not None else 0.0,
        )
        return (
            DepositionalEnvironmentSimulator._weightedLikelihood(
                L_bathy, weights[0], names
            ),
            DepositionalEnvironmentSimulator._weightedLikelihood(
                L_trans, weights[1], names
            ),
            DepositionalEnvironmentSimulator._weightedLikelihood(
                L_trend, weights[2], names
            ),
        )

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------

    def compute_prior(self: Self) -> dict[str, float]:
        r"""Compute prior probabilities from environment weights.

        .. math::

            P(e) = \frac{w_e}{\sum_{e'} w_{e'}}

        :returns: normalised prior probability for each environment.
        """
        if self._cached_prior is not None:
            return dict(self._cached_prior)

        total = sum(
            self._weights[env.name] for env in self._environments.values()
        )
        return {
            e.name: self._weights[e.name] / total
            for e in self._environments.values()
        }

    # ------------------------------------------------------------------
    # waterDepth likelihood
    # ------------------------------------------------------------------

    def compute_waterDepth_likelihood(
        self: Self,
        *,
        waterDepth_value: float | None = None,
        waterDepth_range: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        r"""Compute likelihood of each environment given waterDepth evidence.

        The likelihood uses a Gaussian kernel on the *interval distance*
        between the observation and each environment's waterDepth range:

        .. math::

            P(D \mid e) = \exp\!\left(
                -\frac{\delta(D,\, \text{range}(e))^2}
                       {2\,\sigma_{\text{waterDepth}}^2}
            \right)

        If neither *waterDepth_value* nor *waterDepth_range* is given,
        all likelihoods are 1 (unconstrained).

        :param float | None waterDepth_value: exact waterDepth
            measurement.
        :param tuple[float, float] | None waterDepth_range: uncertain
            waterDepth interval ``(min, max)``.
        :returns: likelihood value for each environment (unnormalised).
        :raises ValueError: if both parameters are specified
            simultaneously.
        """
        if waterDepth_value is not None and waterDepth_range is not None:
            raise ValueError(
                "Provide either waterDepth_value or waterDepth_range, "
                "not both."
            )

        # Unconstrained
        if waterDepth_value is None and waterDepth_range is None:
            return dict.fromkeys(self._names, 1.0)

        # Compute likelihoods based on interval distance to each environment's
        # waterDepth range.
        sigma = self._params.waterDepth_sigma

        obs_min: float = 0.0
        obs_max: float = 0.0
        if waterDepth_value is not None:
            obs_min = obs_max = waterDepth_value
        else:
            obs_min, obs_max = waterDepth_range  # type: ignore[assignment]

        result: dict[str, float] = {}
        for env in self._environments.values():
            delta = self._interval_distance(
                obs_min,
                obs_max,
                env.waterDepth_min,
                env.waterDepth_max,
                method=self._params.interval_distance_method,
            )
            # split purely onshore from purely offshore facies
            if obs_min > 0 and env.waterDepth_max <= 0:
                # set delta to inf for environments with max waterDepth <= 0
                delta = float("inf")
            elif obs_max < 0 and env.waterDepth_min >= 0:
                # set delta to inf for environments with min waterDepth >= 0
                delta = float("inf")
            res = self._gaussian_kernel(delta, sigma)
            result[env.name] = res
        return result

    # ------------------------------------------------------------------
    # Transition likelihood
    # ------------------------------------------------------------------

    def compute_transition_likelihood(
        self: Self,
        previous_environment: str | None = None,
        *,
        _sigma_override: float | None = None,
    ) -> dict[str, float]:
        """Compute transition (adjacency) likelihood.

        When *previous_environment* is ``None`` all likelihoods
        are 1 (unconstrained).

        The likelihood is a Gaussian kernel on the combined distance of
        waterDepth range interval distance and the distality difference
        between the previous environment's and each candidate's range.

        :param str | None previous_environment: name of the environment
            selected at the previous step, or ``None``.
        :param float | None _sigma_override: optional override for the
            transition sigma (for testing and fallback purposes).
        :returns: likelihood value for each environment (unnormalised).
        :raises ValueError: if *previous_environment* is not a known
            name.
        """
        if (previous_environment is not None) and (
            previous_environment not in self._names
        ):
            raise ValueError(
                f"Unknown previous environment '{previous_environment}'. "
                f"Known: {self._names}"
            )

        if _sigma_override is None:
            cached = self._cached_transition_likelihood.get(
                previous_environment
            )
            if cached is not None:
                return dict(cached)

        if previous_environment is None:
            return dict.fromkeys(self._names, 1.0)

        prev = self._environments[previous_environment]
        sigma = (
            _sigma_override
            if _sigma_override is not None
            else self._params.transition_sigma
        )

        result: dict[str, float] = {}
        for env in self._environments.values():
            # water depth distance term
            wdDelta = self._interval_distance(
                prev.waterDepth_min,
                prev.waterDepth_max,
                env.waterDepth_min,
                env.waterDepth_max,
                method=self._params.interval_distance_method,
            )

            # add distality term
            distalityDelta = 0.0
            if (
                self._distality_by_environment is not None
                and prev.name in self._distality_by_environment
                and env.name in self._distality_by_environment
            ):
                distalityDelta = abs(
                    self._distality_by_environment[prev.name]
                    - self._distality_by_environment[env.name]
                )

            delta = wdDelta + distalityDelta
            result[env.name] = self._gaussian_kernel(delta, sigma)
        return result

    # ------------------------------------------------------------------
    # Distality trend likelihood
    # -----------------------------------------------------------------
    def compute_distality_trend_likelihood(
        self: Self,
        previous_environments: list[str] | None = None,
        *,
        _sigma_override: float | None = None,
    ) -> dict[str, float]:
        """Compute distality trend likelihood from previous environments.

        Compute the slope of the derivative of the distality series, and
        compare to the implied increment from each candidate environment.
        The more the mismatch, the lower the likelihood.

        :param list[str] | None previous_environments: list of previous
            environment names, or ``None``.
        :param float | None _sigma_override: optional override for the
            trend sigma (for testing and fallback purposes).
        :returns: likelihood value for each environment (unnormalised).
        :raises ValueError: if any name in *previous_environments* is not a
            known environment.
        """
        for name in previous_environments or []:
            if (name not in self._environments) and (name.lower() != "none"):
                raise ValueError(
                    f"Unknown environment '{name}' in previous_environments. "
                    f"Known: {self._names}"
                )

        # no use of distality trend or no trend defined yet
        if (previous_environments is None) or (len(previous_environments) < 2):
            return dict.fromkeys(self._names, 1.0)

        # _distality_by_environment is None
        if len(self._distality_by_environment) == 0:
            print(
                "Warning: distality distances were not computed, "
                + "distality influence is disregarded."
            )
            return dict.fromkeys(self._names, 1.0)

        # compute the slope of the derivative of the distality series, and
        # compare to the implied increment from each candidate environment.
        # The more the mismatch, the lower the likelihood.
        trend_window = self._params.trend_window
        history_tail = [
            name
            for name in previous_environments[-trend_window:]
            if name.lower() != "none"
        ]
        distality_series = [
            self._distality_by_environment[name] for name in history_tail
        ]
        trend_slope = self._estimate_distality_slope(distality_series)

        if abs(trend_slope) < self._trend_threshold:
            # if the trend is very flat, consider there is no trend
            return dict.fromkeys(self._names, 1.0)

        sigma = (
            _sigma_override
            if _sigma_override is not None
            else self._params.trend_sigma
        )
        result: dict[str, float] = {}
        for env in self._environments.values():
            implied_increment = (
                self._distality_by_environment[env.name]
                - self._distality_by_environment[previous_environments[-1]]
            )
            mismatch: float = 0.0
            if len(previous_environments) == 2:
                # keep the sign of the derivative, but ignore the magnitude
                if (abs(implied_increment) < self._trend_threshold) or (
                    abs(trend_slope) < self._trend_threshold
                ):
                    # if either is very flat, consider there is no trend
                    mismatch = 0.0
                else:
                    implied_increment_norm = implied_increment / abs(
                        implied_increment
                    )
                    trend_slope_norm = trend_slope / abs(trend_slope)
                    mismatch = implied_increment_norm - trend_slope_norm
            else:
                # consider sign and magnitude
                mismatch = implied_increment - trend_slope
            result[env.name] = self._gaussian_kernel(mismatch, sigma)
        return result

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------
    # helper to compute the posterior with given likelihoods and weights.
    def _compute_posterior(
        self: Self,
        prior: dict[str, float],
        L_bathy: dict[str, float] | None = None,
        L_trans: dict[str, float] | None = None,
        L_trend: dict[str, float] | None = None,
        waterDepth_weight: float = 1.0,
        transition_weight: float = 1.0,
        trend_weight: float = 1.0,
    ) -> tuple[dict[str, float], float]:
        wL_bathy, wL_trans, wL_trend = (
            DepositionalEnvironmentSimulator._weightedLikelihoods(
                L_bathy,
                L_trans,
                L_trend,
                waterDepth_weight,
                transition_weight,
                trend_weight,
                self._names,
            )
        )
        posterior_unnorm = {
            name: prior[name]
            * (wL_bathy[name] * wL_trans[name] * wL_trend[name])
            for name in self._names
        }
        return self._normalize(posterior_unnorm)

    def compute_posterior(
        self: Self,
        *,
        waterDepth_value: float | None = None,
        waterDepth_range: tuple[float, float] | None = None,
        previous_environments: list[str] | None = None,
    ) -> dict[str, float]:
        r"""Compute posterior probabilities via Bayesian conditioning.

        .. math::

            P(e \mid D, S, R) \propto
                P(e)\,P(D \mid e)\,P(S \mid e)\,P(R \mid e)

        where:

        - :math:`P(e)` is the prior probability of environment :math:`e`.
        - :math:`P(D \\mid e)` is the water depth likelihood of environment
          :math:`e` given waters depth evidence :math:`D`.
          :math:`P(D \\mid e) = wdWeight P(D \\mid e)_0 + (1 - wdWeight)`,
          where :math:`P(D \\mid e)_0` is the initial likelihood computed from
          the water depth ranges, and :math:`wdWeight` is the weight for the
          waterDepth likelihood.
        - :math:`P(S \\mid e)` is the transition likelihood of environment
          :math:`e` given previous environment evidence :math:`S`.
          :math:`P(S \\mid e) = transWeight P(S \\mid e)_0 + (1-transWeight)`,
          where :math:`P(S \\mid e)_0` is the initial likelihood computed from
          the previous environment, and :math:`transWeight` is the weight for
          the transition likelihood.
        - :math:`P(T \\mid e)` is the distality trend likelihood of environment
          :math:`e` given previous environment history evidence :math:`T`.
          :math:`P(T \\mid e) = trendWeight P(T \\mid e)_0 + (1-trendWeight)`,
          where :math:`P(T \\mid e)_0` is the initial likelihood computed from
          the previous environment history, and :math:`trendWeight` is the
          weight for the distality trend likelihood.

        If the posterior is numerically zero everywhere, the method applies
        a five-level fallback strategy:

          1. **Relax** the transition constraint (σ_transition × 10)
             while keeping waterDepth and trend constraints.
          2. **Drop** the transition constraint entirely; keep waterDepth
             and trend constraints.
          3. **Relax** trend constraint (σ_trend × 10); while keeping
             waterDepth constraint and dropping transition constraint.
          4. **Drop** trend constraint entirely (and trend constraint); while
             keeping only waterDepth constraint.
          5. **Return** the prior (no likelihoods at all).

        :param float | None waterDepth_value: exact waterDepth
            measurement (mutually exclusive with *waterDepth_range*).
        :param tuple[float, float] | None waterDepth_range: uncertain
            waterDepth interval ``(min, max)``.
        :param list[str] | None previous_environments: full ordered
            history of previous environments.
        :returns: normalised posterior probabilities.
        """
        previous_environment: str | None = None
        if (previous_environments is not None) and (
            len(previous_environments) > 0
        ):
            previous_environment = previous_environments[-1]

        prior = self.compute_prior()
        L_bathy = self.compute_waterDepth_likelihood(
            waterDepth_value=waterDepth_value,
            waterDepth_range=waterDepth_range,
        )
        L_trans = self._cached_transition_likelihood.get(
            previous_environment, dict.fromkeys(self._names, 1.0)
        )
        L_trend = self.compute_distality_trend_likelihood(
            previous_environments,
        )

        # initial posterior computation
        posterior, total = self._compute_posterior(
            prior,
            L_bathy,
            L_trans,
            L_trend,
            self._params.waterDepth_weight,
            self._params.transition_weight,
            self._params.trend_weight,
        )
        if total > 0:
            return posterior

        # ---- Fallback 1: relax distality trend σ × 10 or drop ------------
        print("Posterior all-zero; fallback 1a - relaxing distality trend.")
        L_trend_relaxed = self.compute_distality_trend_likelihood(
            previous_environments,
            _sigma_override=self._params.trend_sigma * 10.0,
        )
        posterior, total = self._compute_posterior(
            prior,
            L_bathy,
            L_trans,
            L_trend_relaxed,
            self._params.waterDepth_weight,
            self._params.transition_weight,
            self._params.trend_weight,
        )
        if total > 0:
            return posterior

        # drop distality trend constraint entirely
        print("Posterior still zero; fallback 1b - ignoring distality trend.")
        posterior, total = self._compute_posterior(
            prior,
            L_bathy,
            L_trans,
            None,
            self._params.waterDepth_weight,
            self._params.transition_weight,
        )
        if total > 0:
            return posterior

        # ---- Fallback 2: relax transition σ × 10 or drop ---------------
        print(
            "Posterior all-zero; fallback 2a - ignoring distality trend "
            + "and relaxing transition sigma."
        )
        L_trans_relaxed = self.compute_transition_likelihood(
            previous_environment,
            _sigma_override=self._params.transition_sigma * 10.0,
        )
        posterior, total = self._compute_posterior(
            prior,
            L_bathy,
            L_trans_relaxed,
            None,
            self._params.waterDepth_weight,
            self._params.transition_weight,
        )
        if total > 0:
            return posterior

        # drop transition constraint entirely
        print(
            "Posterior still zero; fallback 2b - ignoring distality trend "
            + "and transition."
        )
        posterior, total = self._compute_posterior(prior, L_bathy, None, None)
        if total > 0:
            return posterior

        # ---- Fallback 3: prior only ----------------------------------
        print(
            "Posterior still zero; fallback 3 - returning prior. "
            + "Likelihoods values were:\n"
            + f"  waterDepth likelihood: {L_bathy}\n"
            + f"  transition likelihood: {L_trans}\n"
            + f"  distality trend likelihood: {L_trend}\n"
        )
        return prior

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_environment(
        self: Self,
        posterior: dict[str, float],
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> str:
        """Sample one environment from *posterior* probabilities.

        :param dict[str, float] posterior: mapping
            ``{name: probability}``.
        :param numpy.random.Generator | None rng: random number
            generator.  If ``None``, a new generator is created from
            *seed*.
        :param int | None seed: seed for a new generator when *rng* is
            ``None``.
        :returns: name of the sampled environment.
        :raises ValueError: if all posterior probabilities are zero.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        names = list(posterior.keys())
        probs = np.array([posterior[n] for n in names], dtype=float)

        total = probs.sum()
        if total == 0:
            raise ValueError(
                "Cannot sample: all posterior probabilities are zero."
            )
        probs /= total  # ensure exact normalisation for numpy

        idx: int = int(rng.choice(len(names), p=probs))
        return names[idx]

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------
    def prepare(self: Self) -> None:
        """Precompute and cache invariant probabilities.

        Caches:

        - prior probabilities (environment weights),
        - transition likelihood dictionaries for ``None`` and each
          environment name at default transition sigma.
        - distality trend likelihood dictionaries for all combinations of
          previous environment histories up to trend_window length at default
          trend sigma.

        The cache is safe because environments and parameters are fixed
        after initialisation.
        """
        self._cached_prior = self.compute_prior()

        # cached distality distances by environment
        # compute distality threshold as the minimum non-zero increment between
        # environments, to avoid considering very flat trends as significant.
        # This is a simple heuristic that can be refined.
        trend_threshold = min(
            (
                x
                for x in list(self._distality_by_environment.values())
                if x > 0
            ),
            default=None,
        )
        if (trend_threshold is not None) and (trend_threshold > 0):
            self._trend_threshold = trend_threshold * 0.01

        #: mapping from environment name to distality
        #: (the higher, the more distal)
        distality_by_environment_tmp: dict[str, float | None] = {
            e.name: e.distality for e in self._environments.values()
        }
        if any(d is None for d in distality_by_environment_tmp.values()):
            # if at least one distality is not defined, use the ordering in
            # the environment list as distality
            self._distality_by_environment = {
                e.name: float(i)
                for i, e in enumerate(self._environments.values())
            }
        else:
            # use ordering based on distality values
            sorted_envs = sorted(
                self._environments.values(),
                key=lambda e: e.distality,  # type: ignore
            )
            self._distality_by_environment = {
                e.name: float(i) for i, e in enumerate(sorted_envs)
            }

        # cached transition likelihood - computed after distality since
        # distality is used for transition
        transition_cache: dict[str | None, dict[str, float]] = {
            None: dict.fromkeys(self._names, 1.0)
        }

        for name in self._names:
            transition_cache[name] = self.compute_transition_likelihood(
                name,
                _sigma_override=self._params.transition_sigma,
            )

        self._cached_transition_likelihood = transition_cache

    def run(
        self: Self,
        waterDepth_value: float | None = None,
        waterDepth_range: tuple[float, float] | None = None,
        previous_environments: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[dict[str, float], Optional[DepositionalEnvironment]]:
        """Compute posterior and sample one environment.

        If water depth is lower than the minimum of all environments, the
        simulator will select the more proximal environment if water depth
        range is inside, None otherwise.

        :param float | None waterDepth_value: exact waterDepth value.
        :param tuple[float, float] | None waterDepth_range: uncertain
            waterDepth range.
        :param list[str] | None previous_environments: ordered history
            of previous-step environments.
        :param dict[str, float] | None distality_by_environment:
            optional explicit distality mapping.
        :param int | None seed: optional seed for deterministic
            sampling.
        :returns: ``(posterior, sampled_environment)``.
        """
        # water depth not in the range of any environment, return None for
        # constrained case
        if waterDepth_value is not None or waterDepth_range is not None:
            obs_min: float = 0.0
            obs_max: float = 0.0
            if waterDepth_value is not None:
                obs_min = obs_max = waterDepth_value
            else:
                obs_min, obs_max = waterDepth_range  # type: ignore[assignment]
            if obs_min < min(
                env.waterDepth_min for env in self._environments.values()
            ) or obs_max > max(
                env.waterDepth_max for env in self._environments.values()
            ):
                return dict.fromkeys(self._names, 0.0), None

        posterior = self.compute_posterior(
            waterDepth_value=waterDepth_value,
            waterDepth_range=waterDepth_range,
            previous_environments=previous_environments,
        )
        sampled_environment = self.sample_environment(posterior, seed=seed)
        return posterior, self._environments[sampled_environment]
