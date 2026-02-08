"""Ensemble 1D forward-simulation skeleton (xarray-friendly).

This module is intentionally "thin": it reuses existing pyWellSFM model/simulator
objects (Scenario / Curve / AccommodationSimulator / AccumulationSimulator) but
stores *many* 1D realizations in a single labeled container.

Recommended stack:
- xarray: labeled N-D arrays for ensemble + time dims
- arviz: posterior storage for inverse modeling (optional)
- emcee or pymc: sampling (optional)

The design goal is to keep your forward model pure and to treat uncertainty as an
extra dimension ("realization" or "draw") rather than as overloaded numeric types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from pywellsfm.model.Curve import Curve, UncertaintyCurve
from pywellsfm.model.SimulationParameters import Scenario
from pywellsfm.simulator.AccommodationSimulator import AccommodationSimulator
from pywellsfm.simulator.AccumulationSimulator import AccumulationSimulator


def _require_xarray() -> Any:
    """Import xarray with a helpful error message."""
    try:
        import xarray as xr  # type: ignore

        return xr
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "xarray is required for this skeleton. Install with: pip install xarray"
        ) from exc


@dataclass(frozen=True)
class EnsembleAxes:
    """Axes definition for an ensemble simulation."""

    time: npt.NDArray[np.float64]
    realization: npt.NDArray[np.int64]


@dataclass(frozen=True)
class Observation:
    """A simple Gaussian observation model for inverse modeling."""

    name: str
    time: npt.NDArray[np.float64]
    value: npt.NDArray[np.float64]
    sigma: float | npt.NDArray[np.float64]


@dataclass(frozen=True)
class AdaptiveStepConfig:
    """Configuration for adaptive time stepping.

    The core constraint is:
        abs(delta_bathymetry_per_step) <= max_bathymetry_change_per_step

    with the bathymetry variation defined (per your formulation) as:
        delta_bathy = delta_sea_level - delta_subsidence + thickness_step

    where deposited thickness (for the step) is approximated as:
        thickness_step = deposition_rate * dt

    Notes:
        - Units must be consistent. In your codebase, accumulation rates are
          documented as (m/My). If `time` is in My, then dt is in My and the
          resulting thickness is in meters.
        - This configuration is intentionally generic; you can later refine the
          thickness estimate (e.g., element-wise, compaction, bypass, erosion).
    """

    max_bathymetry_change_per_step: float = 1.0
    dt_min: float = 1e-6
    dt_max: float = 1.0
    safety: float = 0.9
    max_steps: int = 1_000_000


def make_time_axis(
    start: float,
    stop: float,
    nb_steps: int,
) -> npt.NDArray[np.float64]:
    """Create a monotonic time/age axis.

    Notes:
        Your code currently uses `age` increasing from 0 to 1; this helper is more
        explicit and supports any range.
    """
    if nb_steps < 2:
        raise ValueError("nb_steps must be >= 2")
    return np.linspace(float(start), float(stop), int(nb_steps), dtype=np.float64)


def _get_eustatic_curve(scenario: Scenario) -> Curve:
    """Return a usable eustatic curve.

    In the current codebase `Scenario.eustaticCurve` is optional.
    """
    if scenario.eustaticCurve is not None:
        return scenario.eustaticCurve
    return Curve("Time", "Eustacy", np.array([0.0, 1.0]), np.array([0.0, 0.0]))


def _as_env_per_realization(
    env: dict[str, float] | Sequence[dict[str, float]],
    n_real: int,
) -> list[dict[str, float]]:
    """Normalize environment conditions to a per-realization list."""
    if isinstance(env, Mapping):
        return [dict(env) for _ in range(int(n_real))]
    if len(env) != n_real:
        raise ValueError("env sequence length must match number of realizations")
    return [dict(e) for e in env]


def _total_deposition_rate_from_env(
    prod_sim: AccumulationSimulator,
    env: dict[str, float],
) -> float:
    """Compute total deposition rate (sum of element production rates)."""
    element_rates = prod_sim.computeAccumulationRatesForAllElements(env)
    return float(np.nansum(list(element_rates.values())))


def _max_abs_delta_bathymetry(
    *,
    acco_sims: list[AccommodationSimulator],
    deposition_rates_t1: npt.NDArray[np.float64],
    t1: float,
    dt: float,
) -> float:
    """Return max |Δbathy| across realizations for a candidate dt.

    Bathymetry variation is computed as:
        Δbathy = Δsea_level + Δsubsidence - thickness_step

    Thickness is estimated explicitly from start-of-step rates:
        thickness_step = deposition_rates_1 * dt

    Args:
        acco_sims: list of accommodation simulators (one per realization)
        deposition_rates_t1: array of deposition rates at t1 (one per realization)
        t1: start time of the step
        dt: candidate time step
    Returns:
        Maximum absolute bathymetry change across all realizations.
    """
    t2 = float(t1 + dt)
    # all simulators share the same eustatic curve
    sea_level_t1 = acco_sims[0].getSeaLevelAt(t1)
    subsidence_t1 = np.asarray(
        [acco_sim.getSubsidenceAt(t1) for acco_sim in acco_sims], dtype=np.float64
    )

    sea_level_t2 = acco_sims[0].getSeaLevelAt(t2)
    subsidence_t2 = np.asarray(
        [acco_sim.getSubsidenceAt(t2) for acco_sim in acco_sims], dtype=np.float64
    )

    d_sea = sea_level_t2 - float(sea_level_t1)
    d_sub = subsidence_t2 - subsidence_t1
    thickness_step = deposition_rates_t1 * float(dt)
    d_bathy = d_sea + d_sub - thickness_step

    finite = d_bathy[np.isfinite(d_bathy)]
    if finite.size == 0:
        return float("inf")
    return float(np.max(np.abs(finite)))


def eval_curve(curve: Curve, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vector-evaluate a `Curve` on a numpy array."""
    return np.asarray([float(curve(float(xi))) for xi in x], dtype=np.float64)


def eval_curves(
    curves: Iterable[Curve],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate many curves on a common x-axis.

    Returns:
        array of shape (n_curves, n_time)
    """
    out: list[npt.NDArray[np.float64]] = [eval_curve(c, x) for c in curves]
    return np.stack(out, axis=0)


def sample_scalar_from_uncertainty(
    u: UncertaintyCurve,
    at: float,
    size: int,
    rng: np.random.Generator | None = None,
    method: str = "uniform",
) -> npt.NDArray[np.float64]:
    """Sample a scalar from an `UncertaintyCurve` at a given abscissa.

    This is a convenience sampler for parameters like initial bathymetry. It treats
    the uncertainty range (min/median/max) as bounds.

    Args:
        u: uncertainty curve
        at: abscissa where to read the range
        size: number of samples
        rng: numpy RNG
        method: "uniform" or "triangular" (mode at median)
    """
    rng0 = rng if rng is not None else np.random.default_rng()
    ymin, ymed, ymax = (float(v) for v in u.getRangeAt(float(at)))

    if not (np.isfinite(ymin) and np.isfinite(ymed) and np.isfinite(ymax)):
        raise ValueError("UncertaintyCurve range contains non-finite values")
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    if method == "uniform":
        return rng0.uniform(ymin, ymax, size=int(size)).astype(np.float64)
    if method == "triangular":
        return rng0.triangular(ymin, ymed, ymax, size=int(size)).astype(np.float64)

    raise ValueError(f"Unknown sampling method: {method}")


def build_ensemble_dataset(
    scenario: Scenario,
    time: npt.NDArray[np.float64],
    initial_bathymetry: npt.NDArray[np.float64],
    subsidence_curves: list[Curve],
    realization_ids: npt.NDArray[np.int64] | None = None,
) -> Any:
    """Build an `xarray.Dataset` that holds inputs for an ensemble run.

    Inputs are split into:
    - shared forcings: e.g. `sea_level(time)` (from scenario.eustaticCurve)
    - realization-specific inputs: `initial_bathymetry(realization)`,
      `subsidence(realization,time)`
    """
    xr = _require_xarray()

    if realization_ids is None:
        realization_ids = np.arange(initial_bathymetry.size, dtype=np.int64)

    if len(subsidence_curves) != initial_bathymetry.size:
        raise ValueError(
            "subsidence_curves length must match initial_bathymetry length"
        )

    time0 = np.asarray(time, dtype=np.float64)
    bathy0 = np.asarray(initial_bathymetry, dtype=np.float64)

    eustatic = _get_eustatic_curve(scenario)

    # Shared sea-level relative to start datum.
    e0 = float(eustatic(float(time0[0])))
    sea_level = eval_curve(eustatic, time0) - e0

    subsidence = eval_curves(subsidence_curves, time0)

    ds = xr.Dataset(
        data_vars={
            "sea_level": ("time", sea_level),
            "initial_bathymetry": ("realization", bathy0),
            "subsidence": (("realization", "time"), subsidence),
        },
        coords={
            "time": ("time", time0),
            "realization": ("realization", np.asarray(realization_ids, np.int64)),
        },
        attrs={
            "scenario_name": scenario.name,
        },
    )

    return ds


def run_forward_ensemble_adaptive(
    scenario: Scenario,
    start: float,
    stop: float,
    initial_bathymetry: npt.NDArray[np.float64],
    subsidence_curves: list[Curve],
    step_cfg: AdaptiveStepConfig | None = None,
    realization_ids: npt.NDArray[np.int64] | None = None,
    env_fn: Callable[[Mapping[str, Any]], dict[str, float] | Sequence[dict[str, float]]]
    | None = None,
) -> Any:
    """Run an ensemble forward simulation with adaptive time steps.

    This variant is meant for your case where you do NOT know the number of steps
    upfront. It chooses dt each step so that the (estimated) deposited thickness
    stays below `step_cfg.max_thickness_per_step`.

    What this returns:
        An xarray.Dataset similar to `run_forward_ensemble`, but with a dynamically
        built `time` coordinate.

    Important limitation (by design for a skeleton):
        The thickness-per-step constraint is enforced using a simple estimate
        thickness ~= total_deposition_rate * dt.
        If your production model depends on realization-specific environmental
        conditions (e.g., water depth), provide an `env_fn` that returns a list of
        env dicts (one per realization).
    """
    # 1. prepare run
    xr = _require_xarray()

    step_cfg0 = step_cfg if step_cfg is not None else AdaptiveStepConfig()
    if step_cfg0.max_bathymetry_change_per_step <= 0:
        raise ValueError("max_bathymetry_change_per_step must be > 0")
    if step_cfg0.dt_min <= 0 or step_cfg0.dt_max <= 0:
        raise ValueError("dt_min and dt_max must be > 0")
    if step_cfg0.dt_min > step_cfg0.dt_max:
        raise ValueError("dt_min must be <= dt_max")
    if not (0.0 < step_cfg0.safety <= 1.0):
        raise ValueError("safety must be in (0, 1]")

    bathy0 = np.asarray(initial_bathymetry, dtype=np.float64)
    n_real = int(bathy0.size)
    if len(subsidence_curves) != n_real:
        raise ValueError("subsidence_curves length must match initial_bathymetry")

    if realization_ids is None:
        realization_ids = np.arange(n_real, dtype=np.int64)

    # Prepare simulators.
    acco_sims = []
    for real in range(n_real):
        acco_sim = AccommodationSimulator()
        acco_sim.setEustaticCurve(_get_eustatic_curve(scenario))
        acco_sim.setSubsidenceCurve(subsidence_curves[real])
        acco_sim.setInitialBathymetry(bathy0[real])
        acco_sim.prepare()
        acco_sims.append(acco_sim)

    prod_sim = AccumulationSimulator()
    prod_sim.setAccumulationModel(scenario.accumulationModel)
    prod_sim.prepare()

    start0 = float(start)
    stop0 = float(stop)
    if stop0 <= start0:
        raise ValueError("stop must be > start")

    # datum for sea-level (relative to start)
    e0 = acco_sims[0].getEustatismAt(start0)

    times: list[float] = [start0]
    sea_levels: list[npt.NDArray[np.float64]] = []
    subsidences: list[npt.NDArray[np.float64]] = []
    topographies: list[npt.NDArray[np.float64]] = []
    accommodations: list[npt.NDArray[np.float64]] = []
    depo_rate_totals: list[npt.NDArray[np.float64]] = []
    thickness_steps: list[npt.NDArray[np.float64]] = []
    bathymetries: list[npt.NDArray[np.float64]] = []
    delta_bathymetries: list[npt.NDArray[np.float64]] = []
    dts: list[float] = []

    bathy_t = bathy0.copy()

    t = start0
    for _step in range(step_cfg0.max_steps):
        if t >= stop0:
            break

        sea_level_t = float(acco_sims[0].getEustatismAt(t) - e0)
        subs_t = np.asarray([float(c(t)) for c in subsidence_curves], dtype=np.float64)
        topo_t = (-bathy0) + subs_t
        acco_t = subs_t + sea_level_t

        state: dict[str, Any] = {
            "time": float(t),
            "sea_level": float(sea_level_t),
            "subsidence": subs_t,
            "topography": topo_t,
            "accommodation": acco_t,
            "bathymetry": bathy_t,
        }

        env_raw = env_fn(state) if env_fn is not None else {}
        env_list = _as_env_per_realization(env_raw, n_real)

        rates = np.asarray(
            [_total_deposition_rate_from_env(prod_sim, e) for e in env_list],
            dtype=np.float64,
        )

        # Choose dt to satisfy |Δbathy| <= max_bathymetry_change_per_step.
        remaining = stop0 - t
        dt_hi = float(min(step_cfg0.dt_max, remaining))
        dt_lo = float(min(step_cfg0.dt_min, dt_hi))

        max_change = float(step_cfg0.max_bathymetry_change_per_step)

        if (
            _max_abs_delta_bathymetry(
                acco_sims=acco_sims,
                deposition_rates_t1=rates,
                t1=t,
                dt=dt_lo,
            )
            > max_change
        ):
            raise RuntimeError(
                "Cannot satisfy bathymetry-change constraint at dt_min. "
                "Increase dt_min, increase max_bathymetry_change_per_step, "
                "or reduce forcing/rates."
            )

        if (
            _max_abs_delta_bathymetry(
                acco_sims=acco_sims,
                deposition_rates_t1=rates,
                t1=t,
                dt=dt_hi,
            )
            <= max_change
        ):
            dt = dt_hi
        else:
            lo = dt_lo
            hi = dt_hi
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if (
                    _max_abs_delta_bathymetry(
                        acco_sims=acco_sims,
                        deposition_rates_t1=rates,
                        t1=t,
                        dt=mid,
                    )
                    <= max_change
                ):
                    lo = mid
                else:
                    hi = mid
            dt = lo

        dt = float(max(step_cfg0.dt_min, min(dt * step_cfg0.safety, dt_hi)))

        # Compute and apply step update for bathymetry.
        t2 = t + dt
        sea_level_t2 = float(acco_sims[0].getEustatismAt(t2) - e0)
        subs_t2 = np.asarray(
            [acco_sim.getSubsidenceAt(t2) for acco_sim in acco_sims],
            dtype=np.float64,
        )
        thickness_step = rates * dt
        delta_bathy = (sea_level_t2 - sea_level_t) + (subs_t2 - subs_t) - thickness_step

        # Record state at time t (step-start)
        bathymetries.append(bathy_t.copy())
        thickness_steps.append(thickness_step)
        delta_bathymetries.append(delta_bathy)

        # record step state at time t
        sea_levels.append(np.full((n_real,), sea_level_t, dtype=np.float64))
        subsidences.append(subs_t)
        topographies.append(topo_t)
        accommodations.append(acco_t)
        depo_rate_totals.append(rates)
        dts.append(dt)

        bathy_t = bathy_t + delta_bathy

        t = t + dt
        times.append(t)

    else:
        raise RuntimeError("Reached max_steps without reaching stop")

    # Convert to arrays on (realization, time).
    # Note: we stored state at each step-start; last `times` entry has no state.
    time_axis = np.asarray(times[:-1], dtype=np.float64)
    dt_axis = np.asarray(dts, dtype=np.float64)

    sea_level_arr = np.stack(sea_levels, axis=1)  # (real, time)
    subs_arr = np.stack(subsidences, axis=1)
    topo_arr = np.stack(topographies, axis=1)
    acco_arr = np.stack(accommodations, axis=1)
    depo_arr = np.stack(depo_rate_totals, axis=1)
    thickness_arr = np.stack(thickness_steps, axis=1)
    bathy_arr = np.stack(bathymetries, axis=1)
    delta_bathy_arr = np.stack(delta_bathymetries, axis=1)

    out = xr.Dataset(
        data_vars={
            "sea_level": (("time",), sea_level_arr[0, :]),
            "dt": (("time",), dt_axis),
            "initial_bathymetry": (("realization",), bathy0),
            "subsidence": (("realization", "time"), subs_arr),
            "topography": (("realization", "time"), topo_arr),
            "accommodation": (("realization", "time"), acco_arr),
            "depo_rate_total": (("realization", "time"), depo_arr),
            "thickness_step": (("realization", "time"), thickness_arr),
            "bathymetry": (("realization", "time"), bathy_arr),
            "delta_bathymetry": (("realization", "time"), delta_bathy_arr),
        },
        coords={
            "time": (("time",), time_axis),
            "realization": (("realization",), np.asarray(realization_ids, np.int64)),
        },
        attrs={
            "scenario_name": scenario.name,
            "start": start0,
            "stop": stop0,
            "max_bathymetry_change_per_step": float(
                step_cfg0.max_bathymetry_change_per_step
            ),
        },
    )

    return out


def run_forward_ensemble(
    scenario: Scenario,
    ds: Any,
    env_fn: Callable[[Mapping[str, Any]], dict[str, float]] | None = None,
) -> Any:
    """Run a simple forward simulation for all realizations.

    This is a *skeleton* that computes accommodation components and calls your
    `AccumulationSimulator` for production rates. It is intentionally minimal:
    - It does not yet build stratigraphic layers or depth-age models.
    - It provides a clean place (`env_fn`) to derive environmental conditions
      (e.g., water depth) from current state.

    Args:
        scenario: shared scenario (production model, eustatic curve)
        ds: dataset produced by `build_ensemble_dataset`
        env_fn: callback to compute environment conditions dict from a state mapping.
            If None, uses an empty dict.

    Returns:
        xarray.Dataset with new variables:
          - topography(realization,time)
          - accommodation(realization,time)
          - depo_rate_total(realization,time)
    """
    if (
        "subsidence" not in ds
        or "sea_level" not in ds
        or "initial_bathymetry" not in ds
    ):
        raise ValueError(
            "ds must contain 'subsidence', 'sea_level', 'initial_bathymetry'"
        )

    # Prepare simulators.
    acco_sim = AccommodationSimulator()
    acco_sim.setEustaticCurve(_get_eustatic_curve(scenario))
    acco_sim.prepare()

    prod_sim = AccumulationSimulator()
    prod_sim.setAccumulationModel(scenario.accumulationModel)
    prod_sim.prepare()

    time = np.asarray(ds["time"].values, dtype=np.float64)
    sea_level = np.asarray(ds["sea_level"].values, dtype=np.float64)  # (time,)
    subsidence = np.asarray(ds["subsidence"].values, dtype=np.float64)  # (real,time)
    bathy0 = np.asarray(ds["initial_bathymetry"].values, dtype=np.float64)  # (real,)

    # From AccommodationSimulator docstring:
    # - initial topography = -initialBathymetry
    # - topography(t) = initial topography + subsidence(t)
    topography = (-bathy0[:, None]) + subsidence

    # AccommodationSimulator returns `accommodation = seaLevel + subsidence`.
    accommodation = sea_level[None, :] + subsidence

    # A very simple total deposition rate proxy: sum of element production rates.
    # In more complete models, these rates would depend on environment
    # (e.g. water depth).
    depo_rate_total = np.full_like(accommodation, np.nan, dtype=np.float64)

    for it, t in enumerate(time):
        # state that env_fn can use
        state: dict[str, Any] = {
            "time": float(t),
            "sea_level": float(sea_level[it]),
            "subsidence": subsidence[:, it],
            "topography": topography[:, it],
            "accommodation": accommodation[:, it],
        }
        env = env_fn(state) if env_fn is not None else {}
        element_rates = prod_sim.computeElementAccumulationRate(env)
        total = float(np.nansum(list(element_rates.values())))
        depo_rate_total[:, it] = total

    out = ds.copy()
    out["topography"] = (("realization", "time"), topography)
    out["accommodation"] = (("realization", "time"), accommodation)
    out["depo_rate_total"] = (("realization", "time"), depo_rate_total)

    return out


def gaussian_loglikelihood(
    sim: Any,
    obs: Observation,
    var: str,
) -> float:
    """Gaussian log-likelihood of a simulated variable against an observation.

    Notes:
        This is a *single realization* likelihood helper.
        For ensembles/posteriors you typically compute this for each draw.
    """
    xr = _require_xarray()

    if var not in sim:
        raise KeyError(f"'{var}' not found in simulation dataset")

    series = sim[var]

    # Interpolate to obs times.
    interp = series.interp(time=xr.DataArray(obs.time, dims=("obs_time",)))
    pred = np.asarray(interp.values, dtype=np.float64)
    y = np.asarray(obs.value, dtype=np.float64)

    sigma = np.asarray(obs.sigma, dtype=np.float64)
    if sigma.size == 1:
        sigma = np.full_like(y, float(sigma))

    r = (y - pred) / sigma
    return float(-0.5 * np.sum(r * r + np.log(2.0 * np.pi * sigma * sigma)))


def try_to_inferencedata(
    posterior: Any,
    log_likelihood: Any | None = None,
) -> Any:
    """Convert xarray objects to ArviZ `InferenceData` when available."""
    try:
        import arviz as az  # type: ignore

        kwargs: dict[str, Any] = {"posterior": posterior}
        if log_likelihood is not None:
            kwargs["log_likelihood"] = log_likelihood
        return az.from_dict(**kwargs)
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "arviz is not installed. Install with: pip install arviz"
        ) from exc


if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke example (requires xarray installed):
    # This is not a full forward stratigraphic model; it just demonstrates the
    # ensemble container and hooks.
    _require_xarray()

    t = make_time_axis(0.0, 1.0, 11)

    # Dummy scenario: constant sea-level + constant production.
    # You likely already have a Scenario and production model in your scripts/tests.
    from pywellsfm.model.AccumulationModel import AccumulationModelGaussian
    from pywellsfm.model.Element import Element

    prod = AccumulationModelGaussian("prod")
    prod.addElement(Element("carbonate", accumulationRate=10.0))

    scenario0 = Scenario(
        name="demo",
        accumulationModel=prod,
        eustaticCurve=Curve(
            "Time", "Eustacy", np.array([0.0, 1.0]), np.array([0.0, 0.0])
        ),
    )

    bathy = np.array([20.0, 30.0, 40.0], dtype=np.float64)
    subs_curves = [
        Curve("Time", "Subsidence", np.array([0.0, 1.0]), np.array([0.0, -10.0])),
        Curve("Time", "Subsidence", np.array([0.0, 1.0]), np.array([0.0, -20.0])),
        Curve("Time", "Subsidence", np.array([0.0, 1.0]), np.array([0.0, -5.0])),
    ]

    ds0 = build_ensemble_dataset(scenario0, t, bathy, subs_curves)
    ds1 = run_forward_ensemble(scenario0, ds0)
    print(ds1)
