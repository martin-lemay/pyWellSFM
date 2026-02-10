"""I/O public API.

This package contains file-format specific loaders and utilities.

The symbols re-exported here form the supported, stable entry points. Callers
should prefer importing from `pywellsfm.io` instead of submodules.
"""

from __future__ import annotations

from .accumulation_model_io import (
    loadAccumulationModel,
    loadAccumulationModelGaussianFromCsv,
    saveAccumulationModel,
    saveAccumulationModelEnvironmentOptimumToJson,
    saveAccumulationModelGaussianToCsv,
    saveAccumulationModelGaussianToJson,
)
from .curve_io import (
    loadCurvesFromFile,
    loadEustaticCurve,
    loadSubsidenceCurve,
    loadUncertaintyCurveFromFile,
    saveCurve,
    saveCurveToCsv,
    saveCurveToJson,
)
from .facies_model_io import (
    loadFaciesModel,
    saveFaciesModel,
)
from .json_schema_validation import (
    validate_json_file_against_schema,
    validateAccumulationModelJsonFile,
    validateFaciesModelJsonFile,
    validateScenarioJsonFile,
    validateTabulatedFunctionJsonFile,
)
from .simulation_io import (
    loadRealizationData,
    loadScenario,
    loadSimulationData,
    saveRealizationData,
    saveScenario,
    saveSimulationData,
)
from .striplog_io import importStriplog
from .tabulated_function_io import (
    loadTabulatedFunctionFromFile,
    saveTabulatedFunctionToCsv,
    saveTabulatedFunctionToJson,
)
from .well_io import loadWell, saveWell

__all__ = [
    "importStriplog",
    "loadAccumulationModel",
    "loadAccumulationModelGaussianFromCsv",
    "loadCurvesFromFile",
    "loadEustaticCurve",
    "loadFaciesModel",
    "loadRealizationData",
    "loadScenario",
    "loadSubsidenceCurve",
    "loadSimulationData",
    "loadTabulatedFunctionFromFile",
    "loadUncertaintyCurveFromFile",
    "loadWell",
    "saveAccumulationModel",
    "saveAccumulationModelEnvironmentOptimumToJson",
    "saveAccumulationModelGaussianToCsv",
    "saveAccumulationModelGaussianToJson",
    "saveCurve",
    "saveCurveToCsv",
    "saveCurveToJson",
    "saveFaciesModel",
    "saveRealizationData",
    "saveScenario",
    "saveSimulationData",
    "saveTabulatedFunctionToCsv",
    "saveTabulatedFunctionToJson",
    "saveWell",
    "validateAccumulationModelJsonFile",
    "validateFaciesModelJsonFile",
    "validateScenarioJsonFile",
    "validateTabulatedFunctionJsonFile",
    "validate_json_file_against_schema",
]
