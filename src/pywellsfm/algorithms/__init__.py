"""Public API.

This package contains numerical algorithms working on model objects and
simulation outputs: statistics of facies successions, comparison of observed
and simulated wells, and parameter sweeps.

The symbols re-exported here form the supported, stable entry points. Callers
should prefer importing from `pywellsfm.algorithms` instead of submodules.
"""

from .facies_statistics import (
    MarkovTestResult,
    bedThicknessStatistics,
    embeddedMarkovChiSquareTest,
    pooledTransitionCountMatrix,
    quasiIndependenceExpectedCounts,
    striplogToSequence,
    thicknessProportionDistance,
    thicknessProportions,
    transitionCountMatrix,
    transitionProbabilityMatrix,
)
from .parameter_sweep import parameterGrid, runParameterSweep
from .well_comparison import (
    discreteLogLabelsAt,
    faciesFromDepositionalEnvironments,
    faciesMismatchFraction,
    markerDepthRmse,
    resampleStriplog,
    simulatedAgesToDepths,
    simulatedStepEndDepths,
)

__all__ = [
    "MarkovTestResult",
    "bedThicknessStatistics",
    "discreteLogLabelsAt",
    "embeddedMarkovChiSquareTest",
    "faciesFromDepositionalEnvironments",
    "faciesMismatchFraction",
    "markerDepthRmse",
    "parameterGrid",
    "pooledTransitionCountMatrix",
    "quasiIndependenceExpectedCounts",
    "resampleStriplog",
    "runParameterSweep",
    "simulatedAgesToDepths",
    "simulatedStepEndDepths",
    "striplogToSequence",
    "thicknessProportionDistance",
    "thicknessProportions",
    "transitionCountMatrix",
    "transitionProbabilityMatrix",
]
