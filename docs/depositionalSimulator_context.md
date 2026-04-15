Context / Spec: DepositionalEnvironmentSimulator (single time-slice)

Goal

Define a generic, parameterized simulator that selects a depositional
environment (a discrete state) at a *given time* and *given location/order*
using Bayesian conditioning:

- Prior probabilities: computed from environment definitions only (no
  constraints).
- Likelihood(s): computed from user constraints:
  - a water depth constraint (exact water depth, or water depth range)
  - an optional “previous-step” environment constraint (to encourage spatial
    continuity along a sampling order).
  - an optional “long-term distality trend” constraint (to encourage
    coherence with the distal/proximal trend inferred from the previously
    simulated environments).
- Posterior probabilities: proportional to prior × likelihood(s).
- Selection: sample one environment according to the posterior.

Non-goals

- No forward simulation in time.
- No dynamics/transition in time; “previous step” refers to previous *sample*
  in an ordered traversal at the *same* time (e.g., along a profile or grid
  scanning order).

Key concepts

1) Environment definition

An environment is a discrete label with parameterized validity/affinity rules.
Keep it generic so new environment sets can be plugged in later.

Minimum required attributes:

- `name: str`
- `waterDepth_range: (min_waterDepth, max_waterDepth)` in meters (positive downward)
- `weight: float` (optional prior weight; default 1.0)

Optional (future extensibility):

- `energy_range` or categorical energy classes
- `map_mask` / spatial rule (2D location constraint)
- custom feature likelihoods

2) User-defined parameters (parameterization)

The simulator must be fully configurable from user inputs. Two levels are
useful:

A) Predefined environment lists with “Physical breakpoints” inputs (example: protected carbonate platform)

- `lagoon_max_waterDepth`
- `fairweather_wave_breaking_waterDepth`
- `fairweather_wave_base_waterDepth`
- `storm_wave_base_waterDepth`
- `shelf_break_waterDepth`

These breakpoints can be converted into environment waterDepth ranges.

B) Direct environment-list inputs (generic mode)

User provides a list of environments with explicit `waterDepth_range` (and
optionally `weight`). This is the most generic representation and should be
the simulator’s internal canonical form.

3) Constraints

Water depth constraint can be expressed as:

- `waterDepth_value: float` (exact water depth at the point), OR
- `waterDepth_range: (min_waterDepth, max_waterDepth)` (uncertain waterDepth / interval), OR
- no waterDepth constraint (unconstrained run).

Previous-steps environments (previous-step constraint and Long-term distality trend constraint):

- `previous_environments: list[str]` (or any ordered sequence), representing
  the previously simulated environments along the same sampling order.
- `distality`: a scalar measure of distance to shoreline, defined per
  environment either as:
  - absolute distances (e.g., metres), OR
  - ordinal integers (0 for most proximal; increasing as environments become
    more distal).

This constraint is meant to express the *trend* in distality rather than a
single-step adjacency. It uses the derivative (slope) of distality computed
from the recent history, and biases the next sampled environment so its
distality increment is consistent with that trend.

Notes:

- If there is no history (empty/length-1 list), this constraint is
  unconstrained (all-ones likelihood).
- The history refers to previous *samples* at the same time-slice (not time
  steps).

Example environment set (protected carbonate platform)

From proximal to distal (illustrative; actual ranges depend on breakpoints):

- shore: waterDepth < 5 m, low energy
- lagoon: waterDepth < 15 m, very low energy
- back reef: waterDepth < 5 m, intermediate energy
- reef crest: waterDepth < 2 m, very high energy
- fore reef: 2–20 m, high energy
- outer platform: 20–50 m, low energy (intermediate during storms)
- shelf: 50–100s m, very low energy
- basin: > 100s m, very low energy

Note: because several environments may overlap in waterDepth (e.g., shore/back reef
both “shallow”), the prior must handle overlaps gracefully.

Algorithm (Bayesian conditioning)

Notation

- Environments: $e \in E$
- Prior: $P(e)$
- Water depth evidence: $D$ (either a value or a range)
- Previous environment evidence: $S$ (previous state)
- Long-term trend evidence: $R$ (distality trend inferred from history)
- Likelihoods: $P(D\mid e)$, $P(S\mid e)$, and $P(R\mid e)$
- Posterior: $P(e\mid D,S,R)$

Posterior

$$
P(e\mid D,S,R) \propto P(e)\,P(D\mid e)\,P(S\mid e)\,P(R\mid e)
$$

Normalize across $E$.

Required methods (API-level)

The implementation should expose these core methods (names flexible, but keep
responsibilities separated):

1) Parameterization / configuration

- Build a canonical list of environments parameterized from user parameters.
  - `from_breakpoints(...) -> list[EnvironmentDefinition]` (optional helper)
  - `from_definitions(defs) -> DepositionalEnvironmentSimulator`

2) Prior probabilities (no constraints)

- `compute_prior() -> dict[name, prob]`

For a strictly “no constraint” prior, return a
normalized vector based on environment `weight` and optionally a default
uninformative prior.

However, if you want priors informed by the *environment definitions*
themselves, a common approach is to assign prior mass
proportional to the environment’s “support” (e.g., range water depth) times weight
`base_mass(e) = weight(e) * range_waterDepth(e)`.
This remains unconstrained by a specific point measurement.

Default (generic):

- `base_mass(e) = weight(e)`
- `P(e) = base_mass(e) / sum(base_mass)`

3) Likelihood: Water depth constraint

- `compute_waterDepth_likelihood(waterDepth_value=None, waterDepth_range=None) -> dict[name, L]`

Use a likelihood proportional to overlap fraction:
- If $D$ is a value: $P(D\mid e)=(D - range(e).min) + (D - range(e).max)$
- If $D$ is a range: $P(D\mid e)=(range(D).min - range(e).min) + (range(D).max - range(e).max)$

Normalization by range(e)?

Default recommendation: implement *both*; keep hard as the baseline and allow
soft via a flag to avoid brittle all-zero posteriors.

4) Likelihood: previous-step environment constraint

- `compute_transition_likelihood(previous_environments[-1]: str | None) -> dict[name, L]`

This is not a time simulation; it’s a spatial/ordering continuity constraint
based on geological Walter's law.

Generic ways to encode it:

- Adjacency list: adjacency distance from waterDepth ranges.
- Transition matrix: user-configurable $T[prev, next]$.

Default using adjacency distance in list: same distance function as waterDepth
constraint, taken environment waterDepth ranges.
Ex: $likelihood=(range(e_prev).min - range(e).min) + (range(e_prev).max - range(e).max)$

Then normalize or leave unnormalized (it is a likelihood term).

Important: if `previous_environments is empty`, return all-ones likelihood.

5) Likelihood: long-term distality trend constraint

- `compute_distality_trend_likelihood(previous_environments: list[str], distality_by_environment: dict[str, float] | None = None) -> dict[name, L]`

Purpose: encourage the next environment to follow the *recent distality
trend* rather than only the immediate previous environment.

Inputs:

- `previous_environments`: ordered history of sampled environments (most
  recent last).
- `distality_by_environment` (optional): mapping from environment name to a
  scalar distality value. If not provided, an ordinal distality can be derived
  from the canonical environment ordering (0..N-1).

Trend extraction (generic):

- Convert history to a numeric series $d_t$ using `distality(e_t)`.
- Estimate the recent derivative $\hat{g}$ from the last $k$ points (window
  length configurable):
  - simplest: $\hat{g} = d_{t} - d_{t-1}$,
  - or robust: slope of a linear fit to the last $k$ samples.

Candidate scoring:

- For each candidate environment $e$, compute its implied increment relative
  to the last state: $\Delta(e) = distality(e) - distality(e_{t})$.
- Likelihood can be a smooth kernel around the mismatch between $\Delta(e)$
  and $\hat{g}$, e.g. a Gaussian:

$$
P(R\mid e) = \exp\left(-\frac{(\Delta(e) - \hat{g})^2}{2\,\sigma_{trend}^2}\right)
$$

Recommended defaults / edge handling:

- If `previous_environments` has fewer than 2 entries, return all ones.
- If `distality_by_environment` is missing a key, either raise a validation
  error or fallback to ordinal distality for that environment.
- Keep this likelihood *soft* (non-zero everywhere) to avoid brittle
  all-zero posteriors; let σ control strength.

6) Posterior probabilities

- `compute_posterior(waterDepth_value=None, waterDepth_range=None, previous_environments=None, distality_by_environment=None) -> dict[name, prob]`

Compute:

- `prior = compute_prior()`
- `L_waterDepth = compute_waterDepth_likelihood(...)`
- `L_trans = compute_transition_likelihood(previous_environments[-1])`
- `L_trend = compute_distality_trend_likelihood(previous_environments, distality_by_environment)`
- `posterior_unnorm[e] = prior[e] * L_waterDepth[e] * L_trans[e] * L_trend[e]`
- normalize to sum to 1

Edge case: if all posterior_unnorm are 0 (e.g., hard constraint impossible),
fallback strategy must be defined:

Fall back applied to following order:
- 1. relax environment constraint, to take the next environment
  according to the trend that honor the waterDepth constraint
- 2. ignore environment constraint, use only waterDepth constraint to compute the likelyhood
- 3. use prior probabilities without likelyhood, with debug info

If still error, error message with debug info.

7) Selection / sampling

- `sample_environment(posterior: dict[name, prob], rng=None) -> str`

Implementation detail: use `numpy.random.Generator.choice` (or Python
`random.choices`) for categorical sampling. Ensure deterministic runs by
accepting an RNG seed or Generator.

API sketch (Python, indicative)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EnvironmentDefinition:
  name: str
  waterDepth_min: float
  waterDepth_max: float
  weight: float = 1.0


@dataclass(frozen=True)
class SimulatorParameters:
  waterDepth_sigma: float = 1.0
  ordered_environments: tuple[str, ...] | None = None
  trend_window: int = 5
  trend_sigma: float = 1.0


class DepositionalEnvironmentSimulator:
  def __init__(self, envs: list[EnvironmentDefinition], params: SimulatorParameters):
    ...

  def compute_prior(self) -> dict[str, float]:
    ...

  def compute_waterDepth_likelihood(self, *, waterDepth_value: float | None = None,
                 waterDepth_range: tuple[float, float] | None = None) -> dict[str, float]:
    ...

  def compute_transition_likelihood(self, previous_environment: str | None) -> dict[str, float]:
    ...

  def compute_distality_trend_likelihood(
      self,
      previous_environments: list[str] | None,
      *,
      distality_by_environment: dict[str, float] | None = None,
  ) -> dict[str, float]:
    ...

  def compute_posterior(self, *, waterDepth_value: float | None = None,
              waterDepth_range: tuple[float, float] | None = None,
              previous_environment: str | None = None,
              previous_environments: list[str] | None = None,
              distality_by_environment: dict[str, float] | None = None) -> dict[str, float]:
    ...

  def sample_environment(self, posterior: dict[str, float], *, seed: int | None = None) -> str:
    ...

  def prepare(self) -> None:
    ...

  def run(self, waterDepth_value: float,
        waterDepth_range: tuple[float, float],
        previous_environments list[str] | None = None,
        seed: int | None = None) -> tuple[dict[str, float], str]:
    ...
  
```

Data model

- `EnvironmentDefinition`
  - `name: str`
  - `waterDepth_min: float`
  - `waterDepth_max: float`
  - `weight: float = 1.0`

- `DESimulatorParameters`
  - `waterDepth_sigma: float` (controls strength/tolerance of the waterDepth likelihood)
  - `transition_sigma: float` (controls strength/tolerance of the transition likelihood)
  - `trend_window: int` (history window length used to estimate distality slope)
  - `trend_sigma: float` (controls strength/tolerance of the trend likelihood)
  - `distality_by_environment: dict[str, float] | None` (optional mapping; if missing, derive ordinal distality from ordering)
  - `interval_distance_method: IntervalDistanceMethod` (distance function to compare waterDepth ranges)

Implementation plan (Python)

1) Define minimal domain objects

- Create a new module (suggested): `src/pywellsfm/simulator/DepositionalEnvironmentSimulator.py`
- Add:
  - `@dataclass EnvironmentDefinition`
  - `@dataclass DESimulatorParameters`
  - `class DepositionalEnvironmentSimulator`

2) Implement configuration / parameterization

- Accept either:
  - `environment_definitions: list[EnvironmentDefinition]` (generic)
  - or user breakpoints + a preset builder that returns definitions
- Validate inputs:
  - waterDepth_min < waterDepth_max
  - unique names
  - weights > 0

3) Implement `compute_prior()`

- Start with the simplest generic prior:
  - equals to `weight`
- Normalize to probabilities.

4) Implement waterDepth likelihood

- define a “distance to interval” function $\delta$ (0 if intervals middle are equal, 
  distance increases as ranges are different. ex: sum of distance between mins and max or ranges)
- likelihood $\exp(-(\delta/\sigma)^2/2)$

5) Implement transition likelihood

- Implement adjacency-based likelihood using `environments`:
  - compute distance from environments (using waterDepth ranges)
- Add optional full matrix support later if needed.

6) Implement posterior

- Multiply components and normalize.
- Implement the all-zero fallback strategy (configurable).

7) Implement sampling

- `sample_environment()` with injected RNG.

8) Tests

- Add unit tests under `tests/` (new file suggested):
  - `test_DepositionalEnvironmentSimulator.py`
- Cover:
  - prior normalization
  - hard waterDepth constraint selects only compatible environments
  - transition likelihood biases toward same/adjacent
  - posterior sums to 1
  - deterministic sampling with seed

9) Documentation

- Add a short doc page under `docs/` describing:
  - parameterization options
  - the Bayesian formulation
  - examples for a carbonate platform preset


Concrete example: carbonate platform preset mapping (breakpoints → ranges)

This section is intentionally a *template*; tune to domain needs.

Inputs (meters, positive downward):

- `lagoon_max_waterDepth` (e.g., 15)
- `fairweather_wave_breaking_waterDepth` (e.g., 5)
- `fairweather_wave_base_waterDepth` (e.g., 20)
- `storm_wave_base_waterDepth` (e.g., 50)
- `shelf_break_waterDepth` (e.g., 200)

Example derived ranges (simple, monotonic; can overlap where appropriate):

- `reef_crest`: [0, 2]
- `fore_reef`: [2, fairweather_wave_breaking_waterDepth]
- `outer_platform`: [fairweather_wave_breaking_waterDepth, storm_wave_base_waterDepth]
- `shelf`: [storm_wave_base_waterDepth, shelf_break_waterDepth]
- `basin`: [shelf_break_waterDepth, +inf] (implement as a large max or a special case)
- `lagoon`: [0, lagoon_max_waterDepth] (overlaps with reef_crest/shore/back_reef)
- `shore`: [0, min(5, lagoon_max_waterDepth)] (example)
- `back_reef`: [0, min(5, lagoon_max_waterDepth)] (example)

Overlaps are acceptable; the Bayesian conditioning + weights control which
state wins under constraints.

Minimal usage example (single evaluation)

Given a water depth measurement at a point and an optional previous environment:

1) compute posterior
2) sample an environment

No time-stepping is performed; the caller can iterate over points if desired.


Related literature (HMM focus)
-------------------------------

The simulator design in this document (categorical latent state + observation
likelihood from waterDepth + adjacency/continuity term from the previous
state) is closely aligned with a Hidden Markov Model (HMM) formulation.

Foundational HMM references (highly cited)

- Rabiner, L. R. (1989). “A tutorial on Hidden Markov Models and selected
  applications in speech recognition.” Proceedings of the IEEE.
- Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). “A maximization
  technique occurring in the statistical analysis of probabilistic functions
  of Markov chains.” The Annals of Mathematical Statistics.
- Viterbi, A. J. (1967). “Error bounds for convolutional codes and an
  asymptotically optimum decoding algorithm.” IEEE Transactions on
  Information Theory. (Viterbi decoding; often used for MAP state sequences.)

Books / modern treatments

- Cappé, O., Moulines, E., & Rydén, T. (2005). Inference in Hidden Markov
  Models. Springer.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
  (Sequential/latent-variable models; HMM as a standard case.)
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective.
  (HMMs, forward–backward, Viterbi, parameter estimation.)

Pointers for depositional/facies applications

HMMs are commonly applied to “segmentation” / classification problems where
the hidden state is lithofacies or depositional environment and observations
are well logs or other attributes. To locate domain-specific papers, useful
search queries include:

- “hidden Markov model” well log facies classification
- HMM lithofacies classification gamma ray
- HMM stratigraphic sequence segmentation
- Markov chain facies transition probabilities Walther’s law

Notes on mapping to this simulator

- Hidden state $e_t$: depositional environment at sample index $t$ (not time;
  it can be a spatial traversal order).
- Transition model $P(e_t\mid e_{t-1})$: adjacency/continuity constraint.
- Emission model $P(D_t\mid e_t)$: waterDepth-value or waterDepth-interval
  likelihood.
