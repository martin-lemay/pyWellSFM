# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

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
bathymetry, and energy conditions.

Let's assume a list of possible depositional environments. Each environment is
defined by its location on a 2D map, and by bathymetry and energy ranges.
For instance, for a protected carbonate platform, the list includes from
proximal to distal:

- shore: between the subaerial beach and the lagoon, bathymetry <5m, low energy
  zone.
- lagoon: between shore and reef environments, bathymetry <15m, very low energy
  zone.
- back reef: located between laggon and reef crest, bathymetry <5m,
  intermediate energy.
- reef crest: very high energy zone (wave swash zone), bathymetry <2m
- fore reef: in the front of the reef, high energy zone (wave surf zone),
  bathymetry 2-20m
- outer platform: more distal than fore reef, before shelf, low energy zone
  (intermediate if storms), bathymetry 20-50m
- shelf: more distal than outer platform, bathymetry 50-100s m, very low energy
- basin: deep than shelf, bathymetry >100s m, very low energy.

Exact bathymetry ranges depends on user inputs including lagoon maximum depth,
fairweather wave breaking depth, fairweather wave base depth, storm wave base
depth, and shelf break depth.

The Depositional environment simulator stochastically simulate in which zone a
given point is. The unconstrained case uses bathymetry ranges to compute
probabilities to be in a given zone, then pick one of the zones according to
these probabilities.

A constrained simulation by the bathymetry at the given point, recompute the
probabilities such as the given bathymetry at the point preferentially fall
within the range of the selected zone.

A forward constrained simulation computes the probabilities by assuming
a progressive transition from one zone to another one. So at a given point,
probabilities are computed according to:
- the selected environment at the previous step. The probability to be in given
  environment increases if this environment is the same or adjacent to the
  previous one.
- the distality trend from previously selected environments. The probability to
  be in a given environment increases if the trend (transgressive or
  regressive) is respected. In pratice, the probability increases as the
  variance of the derivative of the distality increases.

All constraints can be combined together.

The conditioning algorithm is based on bayesian approach, where the prior
probabilities are computed based on the bathymetry ranges defined for each
environment, and the likelihood is computed based on the bathymetry constraint
and the previous environment constraint. The posterior probabilities are then
computed by multiplying the prior and the likelihood, and normalizing the
result. Finally, the environment is selected by sampling from the posterior
probabilities.


This module contains the :class:`DepositionalEnvironmentSimulator` class,
which selects a depositional environment (a discrete state) at a *given
time* and *given location/order* using Bayesian conditioning.

The algorithm combines:

- **Prior** probabilities computed from environment weights.
- **Bathymetry likelihood** from a measured or uncertain bathymetry value /
  range.
- **Transition likelihood** from an optional *previous-step* environment
  (Walther's law / spatial continuity).
- **Distality trend likelihood** from an optional list of *previous-step*
  environments (progressive transitions).

Posterior:

.. math::

    P(e \\mid D, S) \\propto P(e)\\,P(D \\mid e)\\,P(S \\mid e)\\,P(T \\mid e)

The simulator does **not** perform forward simulation in time.
The "previous steps" refers to the previous *samples* in an ordered
traversal at the *same* time (e.g., along a profile or grid scanning
order).
"""


# ======================================================================
# Data classes
# ======================================================================


@dataclass(frozen=True)
class EnvironmentDefinition:
    """Definition of a single depositional environment.

    :param str name: unique label for the environment.
    :param float bathymetry_min: minimum bathymetry in metres (positive
        downward).  Must be strictly less than *bathymetry_max*.
    :param float bathymetry_max: maximum bathymetry in metres (positive
        downward).
    :param float weight: prior weight (must be > 0, default 1.0).
    """

    name: str
    bathymetry_min: float
    bathymetry_max: float
    weight: float = 1.0

    def __post_init__(self: Self) -> None:
        """Validate environment definition constraints."""
        if self.bathymetry_min >= self.bathymetry_max:
            raise ValueError(
                f"bathymetry_min ({self.bathymetry_min}) must be strictly "
                f"less than bathymetry_max ({self.bathymetry_max}) for "
                f"environment '{self.name}'."
            )
        if self.weight <= 0:
            raise ValueError(
                f"weight must be > 0, got {self.weight} for environment "
                f"'{self.name}'."
            )

    @property
    def bathymetry_mid(self: Self) -> float:
        """Mid-point of the bathymetry range."""
        return (self.bathymetry_min + self.bathymetry_max) / 2.0

    @property
    def bathymetry_range_width(self: Self) -> float:
        """Width of the bathymetry range."""
        return self.bathymetry_max - self.bathymetry_min


@dataclass(frozen=True)
class DepositionalEnvironmentSimulatorParameters:
    """Configuration parameters for :class:`DepositionalEnvironmentSimulator`.

    :param float bathymetry_sigma: standard-deviation (metres) for the
        Gaussian bathymetry likelihood kernel.  Controls tolerance to
        bathymetry mismatches.
    :param float transition_sigma: standard-deviation (metres) for the
        Gaussian transition (adjacency) likelihood kernel.
    :param str transition_mode: ``"none"`` disables transition
        likelihood; ``"adjacency"`` computes it from bathymetry-range
        distances.
    :param IntervalDistanceMethod interval_distance_method: method to compute
        the distance between bathymetry intervals for likelihood
        computation.
    :param float trend_sigma: standard-deviation for the Gaussian distality
        trend likelihood kernel. Controls tolerance to distality trend
        mismatches.
    :param int trend_window: number of previous environments to consider for
        distality trend likelihood.
    """

    bathymetry_sigma: float = 5.0
    transition_sigma: float = 5.0
    trend_sigma: float = 2.0
    trend_window: int = 5
    transition_mode: Literal["none", "adjacency"] = "adjacency"
    interval_distance_method: IntervalDistanceMethod = (
        IntervalDistanceMethod.GAP_OVERLAPPING_WIDTH
    )

    def __post_init__(self: Self) -> None:
        """Validate parameter constraints."""
        if self.bathymetry_sigma <= 0:
            raise ValueError("bathymetry_sigma must be > 0.")
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
    2. Computes **bathymetry likelihood** from a measured (or uncertain)
       bathymetry value / range.
    3. Computes **transition likelihood** from an optional
       *previous-step* environment (spatial continuity / Walther's law).
    4. Multiplies prior × likelihoods and normalises to obtain the
       **posterior**.
    5. **Samples** one environment from the posterior.

    :param list[EnvironmentDefinition] environments: non-empty list of
        environment definitions with unique names. Consider ordering
        environments from proximal to distal if *distality_by_environment* is
        not provided.
    :param dict[str, float] | None distality_by_environment: optional explicit
        mapping from environment name to distality (e.g., distance to shoreline
         or index such as 0 for shore, 1 for lagoon, etc.).
         If ``None``, distality is inferred from the ordering of
        environments.
    :param DepositionalEnvironmentSimulatorParameters | None params:
        simulator tuning knobs; when ``None`` the defaults are used.

    :raises ValueError: if *environments* is empty or contains
        duplicate names.
    """

    def __init__(
        self: Self,
        environments: list[EnvironmentDefinition],
        distality_by_environment: dict[str, float] | None = None,
        params: DepositionalEnvironmentSimulatorParameters | None = None,
    ) -> None:
        """Initialise the simulator with environments and parameters."""
        if not environments:
            raise ValueError("At least one environment must be provided.")

        #: environment names
        self._names: list[str] = [e.name for e in environments]
        #: mapping from environment name to definition
        self._environments: dict[str, EnvironmentDefinition] = {
            e.name: e for e in environments
        }
        #: mapping from environment name to distality
        #: (the higher, the more distal)
        self._distality_by_environment: dict[str, float] = {}
        if distality_by_environment is not None:
            self._distality_by_environment = dict(distality_by_environment)
        else:
            # use ordering in the environment list as distality
            self._distality_by_environment = {
                e.name: i for i, e in enumerate(environments)
            }

        #: simulator parameters
        self._params: DepositionalEnvironmentSimulatorParameters = (
            params or DepositionalEnvironmentSimulatorParameters()
        )

        # cached prior probabilities and likelihoods for efficiency;
        # populated by prepare()
        self._cached_prior: dict[str, float] | None = None
        self._cached_transition_likelihood: dict[
            str | None, dict[str, float]
        ] = {}
        # threshold for considering a distality trend as significant
        self._trend_threshold: float = 0.01

        self.__post_init__()

    def __post_init__(self: Self) -> None:
        """Check for input parameters coherency."""
        # TODO: check all environments have distinct names, and if a distality
        # list is set, that it is defined for all environments.
        if len(self._names) != len(set(self._names)):
            raise ValueError(
                "Environment names must be unique. "
                f"Got duplicates in: {self._names}"
            )
        if set(self._names) != set(self._environments.keys()):
            raise ValueError(
                "Inconsistent environment names between list and dict. "
                f"List: {self._names}, "
                + f"dict keys: {list(self._environments.keys())}"
            )

        if self._distality_by_environment is not None:
            unknown_envs = set(
                self._distality_by_environment.keys()
            ).difference(self._names)
            if unknown_envs:
                raise ValueError(
                    "Unknown environment names in distality_by_environment: "
                    f"{sorted(unknown_envs)}. "
                    + f"Known environments: {self._names}"
                )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_breakpoints(
        *,
        lagoon_max_bathymetry: float = 15.0,
        fairweather_wave_breaking_bathymetry: float = 5.0,
        fairweather_wave_base_bathymetry: float = 20.0,
        storm_wave_base_bathymetry: float = 50.0,
        shelf_break_bathymetry: float = 200.0,
        basin_max_bathymetry: float = 1000.0,
    ) -> list[EnvironmentDefinition]:
        """Build a carbonate-platform environment list from breakpoints.

        The returned list follows the proximal → distal ordering of a
        protected carbonate platform.

        :param float lagoon_max_bathymetry: maximum depth of the lagoon
            (default 15 m).
        :param float fairweather_wave_breaking_bathymetry: fairweather
            wave-breaking depth (default 5 m).
        :param float fairweather_wave_base_bathymetry: fairweather
            wave-base depth (default 20 m).
        :param float storm_wave_base_bathymetry: storm wave-base depth
            (default 50 m).
        :param float shelf_break_bathymetry: shelf-break depth
            (default 200 m).
        :param float basin_max_bathymetry: practical upper limit for
            basin depth (default 1000 m).
        :returns: list of :class:`EnvironmentDefinition` suitable for
            :meth:`__init__`.
        """
        shore_max = min(5.0, lagoon_max_bathymetry)
        back_reef_max = min(5.0, lagoon_max_bathymetry)

        return [
            EnvironmentDefinition("shore", 0.0, shore_max),
            EnvironmentDefinition("lagoon", 0.0, lagoon_max_bathymetry),
            EnvironmentDefinition("back_reef", 0.0, back_reef_max),
            EnvironmentDefinition("reef_crest", 0.0, 2.0),
            EnvironmentDefinition(
                "fore_reef", 2.0, fairweather_wave_base_bathymetry
            ),
            EnvironmentDefinition(
                "outer_platform",
                fairweather_wave_base_bathymetry,
                storm_wave_base_bathymetry,
            ),
            EnvironmentDefinition(
                "shelf",
                storm_wave_base_bathymetry,
                shelf_break_bathymetry,
            ),
            EnvironmentDefinition(
                "basin",
                shelf_break_bathymetry,
                basin_max_bathymetry,
            ),
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def environments(self: Self) -> list[EnvironmentDefinition]:
        """Return a copy of the environment definitions."""
        return list(self._environments.values())

    @property
    def environment_names(self: Self) -> list[str]:
        """Return a copy of the environment name list."""
        return list(self._names)

    @property
    def params(self: Self) -> DepositionalEnvironmentSimulatorParameters:
        """Return the simulator parameters."""
        return self._params

    @staticmethod
    def _gaussian_kernel(delta: float, sigma: float) -> float:
        r"""Unnormalised Gaussian kernel.

        .. math::

            k = \exp\!\left(-\frac{\delta^2}{2\,\sigma^2}\right)
        """
        return math.exp(-0.5 * (delta / sigma) ** 2)

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

        total = sum(e.weight for e in self._environments.values())
        return {e.name: e.weight / total for e in self._environments.values()}

    # ------------------------------------------------------------------
    # Bathymetry likelihood
    # ------------------------------------------------------------------

    def compute_bathymetry_likelihood(
        self: Self,
        *,
        bathymetry_value: float | None = None,
        bathymetry_range: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        r"""Compute likelihood of each environment given bathymetry evidence.

        The likelihood uses a Gaussian kernel on the *interval distance*
        between the observation and each environment's bathymetry range:

        .. math::

            P(D \mid e) = \exp\!\left(
                -\frac{\delta(D,\, \text{range}(e))^2}
                       {2\,\sigma_{\text{bathy}}^2}
            \right)

        If neither *bathymetry_value* nor *bathymetry_range* is given,
        all likelihoods are 1 (unconstrained).

        :param float | None bathymetry_value: exact bathymetry
            measurement.
        :param tuple[float, float] | None bathymetry_range: uncertain
            bathymetry interval ``(min, max)``.
        :returns: likelihood value for each environment (unnormalised).
        :raises ValueError: if both parameters are specified
            simultaneously.
        """
        if bathymetry_value is not None and bathymetry_range is not None:
            raise ValueError(
                "Provide either bathymetry_value or bathymetry_range, "
                "not both."
            )

        # Unconstrained
        if bathymetry_value is None and bathymetry_range is None:
            return dict.fromkeys(self._names, 1.0)

        # Compute likelihoods based on interval distance to each environment's
        # bathymetry range.
        sigma = self._params.bathymetry_sigma

        if bathymetry_value is not None:
            obs_min = obs_max = bathymetry_value
        else:
            assert bathymetry_range is not None
            obs_min, obs_max = bathymetry_range

        result: dict[str, float] = {}
        for env in self._environments.values():
            delta = self._interval_distance(
                obs_min,
                obs_max,
                env.bathymetry_min,
                env.bathymetry_max,
                method=self._params.interval_distance_method,
            )
            result[env.name] = self._gaussian_kernel(delta, sigma)
        return result

    # ------------------------------------------------------------------
    # Helper for distance between intervals
    # ------------------------------------------------------------------
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

        When *previous_environment* is ``None`` or
        :attr:`params.transition_mode` is ``"none"``, all likelihoods
        are 1 (unconstrained).

        In ``"adjacency"`` mode the likelihood is a Gaussian kernel on
        the interval distance between the previous environment's
        bathymetry range and each candidate's range.

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

        if (
            previous_environment is None
            or self._params.transition_mode == "none"
        ):
            return dict.fromkeys(self._names, 1.0)

        prev = self._environments[previous_environment]
        sigma = (
            _sigma_override
            if _sigma_override is not None
            else self._params.transition_sigma
        )

        result: dict[str, float] = {}
        for env in self._environments.values():
            delta = self._interval_distance(
                prev.bathymetry_min,
                prev.bathymetry_max,
                env.bathymetry_min,
                env.bathymetry_max,
                method=self._params.interval_distance_method,
            )
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
            if name not in self._environments:
                raise ValueError(
                    f"Unknown environment '{name}' in previous_environments. "
                    f"Known: {self._names}"
                )

        # no use of distality trend or no trend defined yet
        if (previous_environments is None) or (len(previous_environments) < 2):
            return dict.fromkeys(self._names, 1.0)

        # compute the slope of the derivative of the distality series, and
        # compare to the implied increment from each candidate environment.
        # The more the mismatch, the lower the likelihood.
        trend_window = self._params.trend_window
        history_tail = previous_environments[-trend_window:]
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

        transition_cache: dict[str | None, dict[str, float]] = {
            None: dict.fromkeys(self._names, 1.0)
        }

        for name in self._names:
            transition_cache[name] = self.compute_transition_likelihood(
                name,
                _sigma_override=self._params.transition_sigma,
            )

        self._cached_transition_likelihood = transition_cache

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

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

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

    def compute_posterior(
        self: Self,
        *,
        bathymetry_value: float | None = None,
        bathymetry_range: tuple[float, float] | None = None,
        previous_environments: list[str] | None = None,
    ) -> dict[str, float]:
        r"""Compute posterior probabilities via Bayesian conditioning.

        .. math::

            P(e \mid D, S, R) \propto
                P(e)\,P(D \mid e)\,P(S \mid e)\,P(R \mid e)

          If the posterior is numerically zero everywhere, the method
          applies a four-level fallback strategy:

        1. **Relax** the transition constraint (σ_transition × 10)
              while keeping bathymetry and trend constraints.
          2. **Drop** the transition constraint entirely; keep bathymetry
              and trend constraints.
          3. **Drop** trend as well; keep only bathymetry.
          4. **Return** the prior (no likelihoods at all).

        :param float | None bathymetry_value: exact bathymetry
            measurement (mutually exclusive with *bathymetry_range*).
        :param tuple[float, float] | None bathymetry_range: uncertain
            bathymetry interval ``(min, max)``.
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
        L_bathy = self.compute_bathymetry_likelihood(
            bathymetry_value=bathymetry_value,
            bathymetry_range=bathymetry_range,
        )
        L_trans = self.compute_transition_likelihood(previous_environment)
        L_trend = self.compute_distality_trend_likelihood(
            previous_environments,
        )

        posterior_unnorm = {
            name: prior[name] * L_bathy[name] * L_trans[name] * L_trend[name]
            for name in self._names
        }
        posterior, total = self._normalize(posterior_unnorm)
        if total > 0:
            return posterior

        # ---- Fallback 1: relax distality trend σ × 10 or drop ------------
        print(
            "Posterior all-zero; fallback 1a - relaxing distality trend sigma."
        )
        L_trend_relaxed = self.compute_distality_trend_likelihood(
            previous_environments,
            _sigma_override=self._params.transition_sigma * 10.0,
        )
        posterior_unnorm = {
            name: (
                prior[name]
                * L_bathy[name]
                * L_trans[name]
                * L_trend_relaxed[name]
            )
            for name in self._names
        }
        posterior, total = self._normalize(posterior_unnorm)
        if total > 0:
            return posterior

        # drop trend constraint entirely
        print("Posterior still zero; fallback 1b - ignoring distality trend.")
        posterior_unnorm = {
            name: prior[name] * L_bathy[name] * L_trans[name]
            for name in self._names
        }
        posterior, total = self._normalize(posterior_unnorm)
        if total > 0:
            return posterior

        # ---- Fallback 2: relax transition σ × 10 or drop ---------------
        print("Posterior all-zero; fallback 2a - relaxing transition sigma.")
        L_trans_relaxed = self.compute_transition_likelihood(
            previous_environment,
            _sigma_override=self._params.transition_sigma * 10.0,
        )
        posterior_unnorm = {
            name: prior[name]
            * L_bathy[name]
            * L_trans_relaxed[name]
            * L_trend[name]
            for name in self._names
        }
        posterior, total = self._normalize(posterior_unnorm)
        if total > 0:
            return posterior

        # drop transition constraint entirely
        print("Posterior still zero; fallback 2b - ignoring transition.")
        posterior_unnorm = {
            name: prior[name] * L_bathy[name] for name in self._names
        }
        posterior, total = self._normalize(posterior_unnorm)
        if total > 0:
            return posterior

        # ---- Fallback 4: prior only ----------------------------------
        print("Posterior still zero; fallback 4 - returning prior.")
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

    def run(
        self: Self,
        bathymetry_value: float | None = None,
        bathymetry_range: tuple[float, float] | None = None,
        previous_environments: list[str] | None = None,
        seed: int | None = None,
    ) -> tuple[dict[str, float], str]:
        """Compute posterior and sample one environment.

        :param float | None bathymetry_value: exact bathymetry value.
        :param tuple[float, float] | None bathymetry_range: uncertain
            bathymetry range.
        :param list[str] | None previous_environments: ordered history
            of previous-step environments.
        :param dict[str, float] | None distality_by_environment:
            optional explicit distality mapping.
        :param int | None seed: optional seed for deterministic
            sampling.
        :returns: ``(posterior, sampled_environment)``.
        """
        posterior = self.compute_posterior(
            bathymetry_value=bathymetry_value,
            bathymetry_range=bathymetry_range,
            previous_environments=previous_environments,
        )
        sampled_environment = self.sample_environment(posterior, seed=seed)
        return posterior, sampled_environment
