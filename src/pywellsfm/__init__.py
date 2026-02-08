"""PyWellSFM public API."""

from __future__ import annotations

from pywellsfm.io import validate_json_file_against_schema  # noqa: F401
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
from pywellsfm.simulator.FSSimulator import FSSimulator
from pywellsfm.simulator.FSSimulatorRunner import FSSimulatorRunner

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
