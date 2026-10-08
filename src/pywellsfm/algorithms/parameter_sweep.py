# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Brute-force exploration of simulation parameters.

A parameter sweep runs the forward model for every combination of parameter
values on a regular grid and evaluates user-defined metrics (e.g., misfits
against an observed well) for each of them.

pyWellSFM distinguishes *scenario* parameters, shared by all realizations of
a simulation (eustasy, accumulation model...), from *realization* parameters
(initial water depth, subsidence...). The sweep exploits this split: for each
combination of scenario parameters, a single :class:`FSSimulator` is built
with one realization per combination of realization parameters, so that all
these realizations share the same time loop.
"""

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from pywellsfm.simulator.FSSimulator import FSSimulator
from pywellsfm.utils import get_logger

logger = get_logger(__name__)

__all__ = ["parameterGrid", "runParameterSweep"]

#: signature of the function building a simulator from parameters
SimulatorBuilder = Callable[
    [dict[str, Any], list[dict[str, Any]]], FSSimulator
]
#: signature of the function evaluating one realization of a simulator
RealizationEvaluator = Callable[
    [FSSimulator, int, dict[str, Any]], Mapping[str, float]
]


def parameterGrid(
    grid: Mapping[str, Sequence[Any]] | None,
) -> list[dict[str, Any]]:
    """List all combinations of parameter values.

    :param Mapping[str, Sequence[Any]] | None grid: parameter names and
        their values.
    :return list[dict[str, Any]]: one dictionary per combination, in
        row-major order of *grid*. An empty or None grid gives a single empty
        combination.
    """
    if not grid:
        return [{}]
    names = list(grid.keys())
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def runParameterSweep(
    scenarioGrid: Mapping[str, Sequence[Any]],
    buildSimulator: SimulatorBuilder,
    evaluate: RealizationEvaluator,
    realizationGrid: Mapping[str, Sequence[Any]] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the forward model over a grid of parameters.

    For each combination of scenario parameters, ``buildSimulator`` is
    called with that combination and the list of all combinations of
    realization parameters. It must return a :class:`FSSimulator` with one
    realization per combination, in the same order. The simulator is then
    prepared, run and finalized, and ``evaluate`` is called for each
    realization.

    :param Mapping[str, Sequence[Any]] scenarioGrid: values of scenario
        parameters.
    :param SimulatorBuilder buildSimulator: function
        ``(scenarioParams, realizationParamsList) -> FSSimulator``.
    :param RealizationEvaluator evaluate: function
        ``(simulator, realizationIndex, params) -> {metric: value}`` where
        ``params`` merges scenario and realization parameters.
    :param Mapping[str, Sequence[Any]] | None realizationGrid: values of
        realization parameters. If None, each simulator has a single
        realization built from an empty parameter set.
    :param bool verbose: log progress at INFO level. Default is False.
    :return pd.DataFrame: one row per (scenario, realization) combination
        with parameter values, metrics, and the wall-clock time of the
        simulation (``runtime_s``, shared by the realizations of a same
        simulator) and per realization (``runtime_per_realization_s``).
    """
    scenarioCombinations = parameterGrid(scenarioGrid)
    realizationCombinations = parameterGrid(realizationGrid)
    rows: list[dict[str, Any]] = []
    for k, scenarioParams in enumerate(scenarioCombinations):
        simulator = buildSimulator(
            dict(scenarioParams), [dict(p) for p in realizationCombinations]
        )
        if simulator.n_real != len(realizationCombinations):
            raise ValueError(
                "buildSimulator must return one realization per combination "
                f"of realization parameters ({len(realizationCombinations)}),"
                f" got {simulator.n_real}."
            )
        start = time.perf_counter()
        simulator.prepare()
        simulator.run()
        simulator.finalize()
        runtime = time.perf_counter() - start
        for r, realizationParams in enumerate(realizationCombinations):
            params = {**scenarioParams, **realizationParams}
            metrics = dict(evaluate(simulator, r, params))
            rows.append(
                {
                    **params,
                    **metrics,
                    "runtime_s": runtime,
                    "runtime_per_realization_s": runtime
                    / len(realizationCombinations),
                }
            )
        if verbose:
            logger.info(
                "[%d/%d] %s - %d realizations in %.2f s",
                k + 1,
                len(scenarioCombinations),
                scenarioParams,
                len(realizationCombinations),
                runtime,
            )
    return pd.DataFrame(rows)
