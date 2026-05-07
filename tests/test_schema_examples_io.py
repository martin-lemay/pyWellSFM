# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import json
from pathlib import Path

from pywellsfm.io import (
    loadAccumulationModel,
    loadCurvesFromFile,
    loadDepositionalEnvironmentModel,
    loadDepositionalEnvironmentSimulation,
    loadEnvironmentConditionsModel,
    loadFaciesModel,
    loadFSSimulation,
    loadRealizationData,
    loadScenario,
    loadTabulatedFunctionFromFile,
    loadUncertaintyCurveFromFile,
    loadWell,
)
from pywellsfm.io.environment_condition_model_io import (
    loadEnvironmentConditionModelFromJsonObj,
)

DATA_DIR = Path(__file__).parent / "data" / "schema_examples"


def _read_json(filename: str) -> dict:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


def test_load_accumulation_model_schema_examples() -> None:
    """Load accumulation model examples (inline and CSV-referenced)."""
    gaussian = loadAccumulationModel(
        str(DATA_DIR / "accumulation_model_example.json")
    )
    assert gaussian.name == "AccumulationGaussianExample"
    assert len(gaussian.elements) == 2

    env_optimum = loadAccumulationModel(
        str(DATA_DIR / "accumulation_model_curve_ref_csv_example.json")
    )
    assert env_optimum.name == "AccumulationEnvOptimumCsvRef"
    assert "carbonate" in env_optimum.elements


def test_load_curve_schema_examples() -> None:
    """Load curve examples from JSON and CSV files."""
    curves_json = loadCurvesFromFile(DATA_DIR / "curve_example.json")
    assert len(curves_json) == 1

    curves_csv = loadCurvesFromFile(DATA_DIR / "curve_age_value.csv")
    assert len(curves_csv) == 1


def test_load_depositional_environment_model_schema_example() -> None:
    """Load the depositional environment model schema example."""
    model = loadDepositionalEnvironmentModel(
        str(DATA_DIR / "depositional_environment_model_example.json")
    )
    assert model.name == "DepositionalEnvExample"
    assert model.getEnvironmentCount() == 1


def test_load_de_simulation_schema_example() -> None:
    """Load the depositional environment simulation schema example."""
    sim = loadDepositionalEnvironmentSimulation(
        str(DATA_DIR / "desimulation_example.json")
    )
    assert sim.depositionalEnvironmentModel.getEnvironmentCount() == 1
    assert sim.environment_names == ["shallow"]


def test_load_environment_condition_model_schema_examples() -> None:
    """Load environment condition model examples (inline and CSV-ref)."""
    inline_obj = _read_json("environment_condition_model_example.json")
    inline = loadEnvironmentConditionModelFromJsonObj(
        inline_obj,
        condition_name="temperature",
        base_dir=DATA_DIR,
        ctx="example.inline",
    )
    assert inline.envConditionName == "temperature"

    ref_obj = _read_json(
        "environment_condition_model_curve_ref_csv_example.json"
    )
    ref = loadEnvironmentConditionModelFromJsonObj(
        ref_obj,
        condition_name="oxygen",
        base_dir=DATA_DIR,
        ctx="example.csv_ref",
    )
    assert ref.envConditionName == "oxygen"


def test_load_environment_conditions_model_schema_examples() -> None:
    """Load environment conditions examples (inline and CSV-ref)."""
    inline = loadEnvironmentConditionsModel(
        str(DATA_DIR / "environment_conditions_model_example.json")
    )
    assert set(inline.environmentConditionNames) == {"energy", "temperature"}

    ref = loadEnvironmentConditionsModel(
        str(
            DATA_DIR
            / "environment_conditions_model_curve_ref_csv_example.json"
        )
    )
    assert ref.environmentConditionNames == ["oxygen"]


def test_load_facies_model_schema_example() -> None:
    """Load the facies model schema example."""
    facies_model = loadFaciesModel(str(DATA_DIR / "facies_model_example.json"))
    assert len(facies_model.faciesSet) == 1


def test_load_realization_data_schema_examples() -> None:
    """Load realization examples (inline and CSV-referenced curve)."""
    inline = loadRealizationData(
        str(DATA_DIR / "realization_data_example.json")
    )
    assert inline.well.name == "RealizationWellInline"

    ref = loadRealizationData(
        str(DATA_DIR / "realization_data_curve_ref_csv_example.json")
    )
    assert ref.well.name == "WellSchemaExample"


def test_load_scenario_schema_examples() -> None:
    """Load scenario examples (inline and CSV-referenced curve)."""
    inline = loadScenario(str(DATA_DIR / "scenario_example.json"))
    assert inline.name == "ScenarioInline"

    ref = loadScenario(str(DATA_DIR / "scenario_curve_ref_csv_example.json"))
    assert ref.name == "ScenarioCsvRef"


def test_load_fssimulation_schema_example() -> None:
    """Load the FSSimulation schema example with URL references."""
    simulation = loadFSSimulation(
        str(DATA_DIR / "fssimulation_data_example.json")
    )
    assert simulation.scenario.name == "ScenarioInline"
    assert len(simulation.realizationDataList) == 1


def test_load_tabulated_function_schema_examples() -> None:
    """Load tabulated function examples from JSON and CSV files."""
    _, _, x_json, y_json = loadTabulatedFunctionFromFile(
        DATA_DIR / "tabulated_function_example.json"
    )
    assert x_json.shape == y_json.shape

    _, _, x_csv, y_csv = loadTabulatedFunctionFromFile(
        DATA_DIR / "tabulated_function.csv"
    )
    assert x_csv.shape == y_csv.shape


def test_load_uncertainty_curve_schema_examples() -> None:
    """Load uncertainty curve examples from JSON and CSV files."""
    ucurve_json = loadUncertaintyCurveFromFile(
        DATA_DIR / "uncertainty_curve_example.json"
    )
    assert ucurve_json.name == "Bathymetry"

    ucurve_csv = loadUncertaintyCurveFromFile(
        DATA_DIR / "uncertainty_curve.csv"
    )
    assert ucurve_csv.name == "Age"


def test_load_well_schema_examples() -> None:
    """Load well examples with inline and CSV-referenced logs."""
    inline = loadWell(str(DATA_DIR / "well_example.json"))
    assert inline.name == "WellSchemaExample"
    assert "GR" in inline.getContinuousLogNames()

    ref = loadWell(str(DATA_DIR / "well_ref_csv_example.json"))
    assert ref.name == "WellCsvRefs"
    assert "GR" in ref.getContinuousLogNames()
