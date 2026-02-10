"""PyWellSFM public API."""

from __future__ import annotations

from .io import validate_json_file_against_schema  # noqa: F401
from .model.AccumulationModel import (
    AccumulationModelBase,
    AccumulationModelEnvironmentOptimum,
    AccumulationModelGaussian,
)
from .model.Curve import AccumulationCurve, Curve, UncertaintyCurve
from .model.DepthAgeModel import DepthAgeModel
from .model.Element import Element
from .model.Facies import (
    EnvironmentalFacies,
    Facies,
    FaciesCriteria,
    FaciesCriteriaCollection,
    FaciesCriteriaType,
    FaciesModel,
    PetrophysicalFacies,
    SedimentaryFacies,
)
from .model.Marker import Marker
from .model.SimulationParameters import RealizationData, Scenario
from .model.Well import Well
from .simulator.FSSimulator import FSSimulator
from .simulator.FSSimulatorRunner import FSSimulatorRunner

__version__ = "0.1.1"
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
    "FSSimulator",
    "FSSimulatorRunner",
    "Marker",
    "PetrophysicalFacies",
    "RealizationData",
    "Scenario",
    "SedimentaryFacies",
    "UncertaintyCurve",
    "Well",
    "validate_json_file_against_schema",
]
