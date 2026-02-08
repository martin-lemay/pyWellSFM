"""Public API.

This package contains data structures.

The symbols re-exported here form the supported, stable entry points. Callers
should prefer importing from `pywellsfm.model` instead of submodules.
"""

from __future__ import annotations

from pywellsfm.model.AccumulationModel import (
    AccumulationModelBase,
    AccumulationModelEnvironmentOptimum,
    AccumulationModelGaussian,
)
from pywellsfm.model.Curve import AccumulationCurve, Curve, UncertaintyCurve
from pywellsfm.model.DepthAgeModel import DepthAgeModel
from pywellsfm.model.Element import Element
from pywellsfm.model.Facies import (
    EnvironmentalFacies,
    Facies,
    FaciesCriteria,
    FaciesCriteriaCollection,
    FaciesCriteriaType,
    FaciesModel,
    PetrophysicalFacies,
    SedimentaryFacies,
)
from pywellsfm.model.Marker import Marker
from pywellsfm.model.SimulationParameters import RealizationData, Scenario
from pywellsfm.model.Well import Well

__all__ = [
    "AccumulationCurve",
    "AccumulationModelBase",
    "AccumulationModelEnvironmentOptimum",
    "AccumulationModelGaussian",
    "Curve",
    "DepthAgeModel",
    "Element",
    "EnvironmentalFacies",
    "Facies",
    "FaciesCriteria",
    "FaciesCriteriaCollection",
    "FaciesCriteriaType",
    "FaciesModel",
    "Marker",
    "PetrophysicalFacies",
    "RealizationData",
    "Scenario",
    "SedimentaryFacies",
    "UncertaintyCurve",
    "Well",
]
