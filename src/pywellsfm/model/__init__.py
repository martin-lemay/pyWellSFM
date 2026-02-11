"""Public API.

This package contains data structures.

The symbols re-exported here form the supported, stable entry points. Callers
should prefer importing from `pywellsfm.model` instead of submodules.
"""

from .AccumulationModel import (
    AccumulationModelBase,
    AccumulationModelEnvironmentOptimum,
    AccumulationModelGaussian,
)
from .AccommodationSpaceWellCalculator import AccommodationSpaceWellCalculator
from .Curve import AccumulationCurve, Curve, UncertaintyCurve
from .DepthAgeModel import DepthAgeModel
from .Element import Element
from .Facies import (
    EnvironmentalFacies,
    Facies,
    FaciesCriteria,
    FaciesCriteriaCollection,
    FaciesCriteriaType,
    FaciesModel,
    PetrophysicalFacies,
    SedimentaryFacies,
)
from .Marker import Marker
from .SimulationParameters import RealizationData, Scenario
from .Well import Well

__all__ = [
    "AccommodationSpaceWellCalculator",
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
