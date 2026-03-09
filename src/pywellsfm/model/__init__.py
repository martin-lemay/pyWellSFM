"""Public API.

This package contains data structures.

The symbols re-exported here form the supported, stable entry points. Callers
should prefer importing from `pywellsfm.model` instead of submodules.
"""

from .AccommodationSpaceWellCalculator import AccommodationSpaceWellCalculator
from .AccumulationModel import (
    AccumulationModel,
    AccumulationModelElementGaussian,
    AccumulationModelElementOptimum,
)
from .Curve import AccumulationCurve, Curve, UncertaintyCurve
from .DepositionalEnvironment import (
    CarbonateOpenRampDepositionalEnvironmentModel,
    CarbonateProtectedRampDepositionalEnvironmentModel,
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)
from .DepthAgeModel import DepthAgeModel
from .Element import Element
from .enums import SubsidenceType
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
from .FSSimulationParameters import RealizationData, Scenario
from .Marker import Marker, StratigraphicSurfaceType
from .Well import Well

__all__ = [
    "AccommodationSpaceWellCalculator",
    "AccumulationCurve",
    "AccumulationModel",
    "AccumulationModelElementOptimum",
    "AccumulationModelElementGaussian",
    "Curve",
    "DepthAgeModel",
    "DepositionalEnvironment",
    "DepositionalEnvironmentModel",
    "CarbonateOpenRampDepositionalEnvironmentModel",
    "CarbonateProtectedRampDepositionalEnvironmentModel",
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
    "StratigraphicSurfaceType",
    "SubsidenceType",
    "SedimentaryFacies",
    "UncertaintyCurve",
    "Well",
]
