# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from striplog import Striplog

from pywellsfm.model.Curve import Curve
from pywellsfm.simulator.FSSimulator import FSSimulator

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.io.fssimulation_io import (
    loadFSSimulation,
    loadRealizationData,
    loadScenario,
    saveFSSimulation,
    saveRealizationData,
    saveScenario,
)
from pywellsfm.model.AccumulationModel import (
    AccumulationModel,
)
from pywellsfm.simulator.DepositionalEnvironmentSimulator import (
    DESimulatorParameters,
)

fileDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(fileDir, "data")


def test_loadRealizationData_from_json(tmp_path: Path) -> None:
    """Loads RealizationData from an embedded-object JSON file."""
    realization_path = tmp_path / "realization.json"

    payload = {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": {
            "format": "pyWellSFM.WellData",
            "version": "1.0",
            "well": {
                "name": "TestWell",
                "location": {"x": 1.0, "y": 2.0, "z": 3.0},
                "depth": 100.0,
            },
        },
        "initialBathymetry": 15.0,
        "subsidenceCurve": {
            "type": "cumulative",
            "curve": {
                "format": "pyWellSFM.CurveData",
                "version": "1.0",
                "curve": {
                    "xAxisName": "Age",
                    "yAxisName": "Subsidence",
                    "interpolationMethod": "linear",
                    "data": [
                        {"x": 0.0, "y": 0.0},
                        {"x": 10.0, "y": 100.0},
                    ],
                },
            },
        },
    }

    realization_path.write_text(json.dumps(payload), encoding="utf-8")

    rd = loadRealizationData(str(realization_path))

    assert rd.well.name == "TestWell"
    assert np.allclose(
        rd.well.wellHeadCoords, np.asarray([1.0, 2.0, 3.0], dtype=float)
    )
    assert rd.well.depth == 100.0
    assert rd.initialBathymetry == 15.0
    assert rd.subsidenceCurve is not None
    assert rd.subsidenceCurve._xAxisName == "Age"
    assert rd.subsidenceCurve._yAxisName == "Subsidence"
    assert rd.subsidenceCurve.getValueAt(0.0) == 0.0


def test_loadRealizationData_from_json_reference(tmp_path: Path) -> None:
    """Loads RealizationData from a JSON with reference to well and curve."""
    realization_path = tmp_path / "realization.json"

    payload = {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": {
            "url": f"{dataDir}/well.json",
        },
        "initialBathymetry": 15.0,
        "subsidenceCurve": {
            "type": "cumulative",
            "curve": {
                "url": f"{dataDir}/subsidence_curve.csv",
            },
        },
    }

    realization_path.write_text(json.dumps(payload), encoding="utf-8")

    rd = loadRealizationData(str(realization_path))

    assert rd.well.name == "Well-B"
    assert np.allclose(
        rd.well.wellHeadCoords, np.asarray([0.0, 0.0, 0.0], dtype=float)
    )
    assert rd.well.depth == 100.0
    assert len(rd.well.getMarkers()) == 1
    litho_log = rd.well.getDepthLog("lithology")
    assert litho_log is not None
    assert isinstance(litho_log, Striplog)
    assert "lithology" in rd.well.getDiscreteLogNames()

    gr_log = rd.well.getDepthLog("GR")
    assert gr_log is not None
    assert isinstance(gr_log, Curve)
    assert "GR" in rd.well.getContinuousLogNames()

    density_log = rd.well.getDepthLog("Density")
    assert density_log is not None
    assert isinstance(density_log, Curve)
    assert "Density" in rd.well.getContinuousLogNames()

    assert rd.subsidenceCurve is not None
    assert rd.subsidenceCurve._xAxisName == "Age"
    assert rd.subsidenceCurve._yAxisName == "Subsidence"
    assert rd.subsidenceCurve.getValueAt(0.0) == 50.0
    assert rd.subsidenceCurve.getValueAt(20.0) == 50.0
    assert rd.subsidenceCurve.getValueAt(60.0) == -30.0
    assert rd.initialBathymetry == 15.0


def test_loadScenario_from_json_minimal(tmp_path: Path) -> None:
    """Loads a Scenario from a schema-like embedded-object JSON file."""
    scenario_path = tmp_path / "scenario.json"

    payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "MyScenario",
        "accumulationModel": {
            "format": "pyWellSFM.AccumulationModelData",
            "version": "1.0",
            "accumulationModel": {
                "name": "AM1",
                "modelType": "Gaussian",
                "elements": {
                    "Carbonate": {
                        "accumulationRate": 100.0,
                        "model": {
                            "modelType": "Gaussian",
                            "stddevFactor": 0.2,
                        },
                    }
                },
            },
        },
    }

    scenario_path.write_text(json.dumps(payload), encoding="utf-8")

    scenario = loadScenario(str(scenario_path))

    assert scenario.name == "MyScenario"
    assert scenario.eustaticCurve is None
    assert scenario.accumulationModel.name == "AM1"
    assert isinstance(scenario.accumulationModel, AccumulationModel)
    assert scenario.accumulationModel.getElementModel("Carbonate") is not None


def test_loadScenario_from_json_with_eustatic_curve(tmp_path: Path) -> None:
    """Loads a Scenario including an embedded eustatic curve."""
    scenario_path = tmp_path / "scenario_with_curve.json"

    payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "ScenarioWithCurve",
        "accumulationModel": {
            "format": "pyWellSFM.AccumulationModelData",
            "version": "1.0",
            "accumulationModel": {
                "name": "AM1",
                "elements": {
                    "Shale": {
                        "accumulationRate": 10.0,
                        "model": {
                            "modelType": "Gaussian",
                            "stddevFactor": 0.1,
                        },
                    }
                },
            },
        },
        "eustaticCurve": {
            "format": "pyWellSFM.CurveData",
            "version": "1.0",
            "curve": {
                "xAxisName": "Age",
                "yAxisName": "SeaLevel",
                "interpolationMethod": "linear",
                "data": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 5.0},
                ],
            },
        },
    }

    scenario_path.write_text(json.dumps(payload), encoding="utf-8")

    scenario = loadScenario(str(scenario_path))

    assert scenario.name == "ScenarioWithCurve"
    assert scenario.eustaticCurve is not None
    assert scenario.eustaticCurve._xAxisName == "Age"
    assert scenario.eustaticCurve._yAxisName == "Eustatism"
    assert scenario.eustaticCurve.getValueAt(0.0) == 0.0
    assert scenario.eustaticCurve.getValueAt(1.0) == 5.0
    assert scenario.accumulationModel.name == "AM1"
    assert isinstance(scenario.accumulationModel, AccumulationModel)
    assert scenario.accumulationModel.getElementModel("Shale") is not None


def test_loadScenario_from_json_with_references(tmp_path: Path) -> None:
    """Loads a Scenario where internal objects are URL references."""
    scenario_path = tmp_path / "scenario_ref.json"
    facies_path = tmp_path / "facies_model.json"
    acc_path = tmp_path / "accumulation_model.json"
    eustatic_path = tmp_path / "eustatic_curve.csv"

    # Reuse repo test data for facies + eustatic curve
    facies_src = Path(dataDir) / "facies_model.json"
    facies_path.write_text(
        facies_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    eustatic_src = Path(dataDir) / "eustatic_curve.csv"
    eustatic_path.write_text(
        eustatic_src.read_text(encoding="utf-8"), encoding="utf-8"
    )

    acc_payload = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "AM_REF",
            "modelType": "Gaussian",
            "elements": {
                "Carbonate": {
                    "accumulationRate": 100.0,
                    "model": {
                        "modelType": "Gaussian",
                        "stddevFactor": 0.2,
                    },
                }
            },
        },
    }
    acc_path.write_text(json.dumps(acc_payload), encoding="utf-8")

    scenario_payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "ScenarioRef",
        "accumulationModel": {"url": "accumulation_model.json"},
        "eustaticCurve": {"url": "eustatic_curve.csv"},
    }
    scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")

    scenario = loadScenario(str(scenario_path))

    assert scenario.name == "ScenarioRef"
    assert scenario.accumulationModel.name == "AM_REF"
    assert scenario.accumulationModel.getElementModel("Carbonate") is not None
    assert scenario.eustaticCurve is not None


def test_loadFSSimulation_from_json_with_references(tmp_path: Path) -> None:
    """FSSimulation data where scenario and realizations are URL references."""
    scenario_path = tmp_path / "scenario.json"
    realization_path = tmp_path / "realization.json"
    simulation_path = tmp_path / "simulation.json"

    # Scenario uses inline internals here; we're validating FSSimulation.
    scenario_payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "Scenario1",
        "accumulationModel": {
            "format": "pyWellSFM.AccumulationModelData",
            "version": "1.0",
            "accumulationModel": {
                "name": "AM1",
                "modelType": "Gaussian",
                "elements": {
                    "Carbonate": {
                        "accumulationRate": 100.0,
                        "model": {
                            "modelType": "Gaussian",
                            "stddevFactor": 0.2,
                        },
                    }
                },
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")

    # Realization references repo data
    # (absolute path is OK; we keep it relative)
    realization_payload = {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": {"url": f"{dataDir}/well.json"},
        "initialBathymetry": 15.0,
        "subsidenceCurve": {
            "type": "cumulative",
            "curve": {"url": f"{dataDir}/subsidence_curve.csv"},
        },
    }
    realization_path.write_text(
        json.dumps(realization_payload), encoding="utf-8"
    )

    simulation_payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {"url": "scenario.json"},
        "realizations": [{"url": "realization.json"}],
    }
    simulation_path.write_text(
        json.dumps(simulation_payload), encoding="utf-8"
    )

    FSSimulationData: FSSimulator = loadFSSimulation(str(simulation_path))
    assert len(FSSimulationData.realizationDataList) == 1
    assert FSSimulationData.scenario.name == "Scenario1"
    assert (
        FSSimulationData.scenario.accumulationModel.getElementModel(
            "Carbonate"
        )
        is not None
    )
    assert FSSimulationData.realizationDataList[0].well.name == "Well-B"
    assert FSSimulationData.realizationDataList[0].subsidenceCurve is not None
    assert FSSimulationData.realizationDataList[0].initialBathymetry == 15.0


def test_loadFSSimulation_from_json_with_desimulator(tmp_path: Path) -> None:
    """FSSimulation data with Depositional Environment Simulator."""
    scenario_path = tmp_path / "scenario.json"
    realization_path = tmp_path / "realization.json"
    fssimulation_path = tmp_path / "fssimulation.json"

    # Scenario uses inline internals here; we're validating FSSimulation.
    scenario_payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "Scenario1",
        "accumulationModel": {
            "format": "pyWellSFM.AccumulationModelData",
            "version": "1.0",
            "accumulationModel": {
                "name": "AM1",
                "modelType": "Gaussian",
                "elements": {
                    "Carbonate": {
                        "accumulationRate": 100.0,
                        "model": {
                            "modelType": "Gaussian",
                            "stddevFactor": 0.2,
                        },
                    }
                },
            },
        },
        "depositionalEnvironmentModel": {
            "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
            "version": "1.0",
            "name": "simple3",
            "environments": [
                {
                    "name": "shallow",
                    "waterDepthModel": {
                        "format": "pyWellSFM.EnvironmentConditionModelData",
                        "version": "1.0",
                        "model": {
                            "modelType": "Uniform",
                            "minValue": 0.0,
                            "maxValue": 10.0,
                        },
                    },
                },
                {
                    "name": "mid",
                    "waterDepthModel": {
                        "format": "pyWellSFM.EnvironmentConditionModelData",
                        "version": "1.0",
                        "model": {
                            "modelType": "Uniform",
                            "minValue": 10.0,
                            "maxValue": 50.0,
                        },
                    },
                },
                {
                    "name": "deep",
                    "waterDepthModel": {
                        "format": "pyWellSFM.EnvironmentConditionModelData",
                        "version": "1.0",
                        "model": {
                            "modelType": "Uniform",
                            "minValue": 50.0,
                            "maxValue": 200.0,
                        },
                    },
                },
            ],
        },
    }
    scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")

    # Realization references repo data
    # (absolute path is OK; we keep it relative)
    realization_payload = {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": {"url": f"{dataDir}/well.json"},
        "initialBathymetry": 15.0,
        "initialEnvironment": "mid",
        "subsidenceCurve": {
            "type": "cumulative",
            "curve": {"url": f"{dataDir}/subsidence_curve.csv"},
        },
    }
    realization_path.write_text(
        json.dumps(realization_payload), encoding="utf-8"
    )

    # fs simulator
    simulation_payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {"url": "scenario.json"},
        "realizations": [{"url": "realization.json"}],
        "use_depositional_environment_simulator": True,
        "deSimulator_weights": {"shallow": 3.0, "mid": 2.0, "deep": 1.0},
        "deSimulator_params": {
            "waterDepth_sigma": 7.0,
            "transition_sigma": 9.0,
            "trend_sigma": 0.5,
            "trend_window": 4,
            "interval_distance_method": "center",
        },
    }
    fssimulation_path.write_text(
        json.dumps(simulation_payload), encoding="utf-8"
    )

    FSSimulationData: FSSimulator = loadFSSimulation(str(fssimulation_path))
    assert len(FSSimulationData.realizationDataList) == 1
    assert FSSimulationData.scenario.name == "Scenario1"
    assert (
        FSSimulationData.scenario.accumulationModel.getElementModel(
            "Carbonate"
        )
        is not None
    )
    assert FSSimulationData.realizationDataList[0].well.name == "Well-B"
    assert FSSimulationData.realizationDataList[0].subsidenceCurve is not None
    assert FSSimulationData.realizationDataList[0].initialBathymetry == 15.0
    assert FSSimulationData.use_deSimulator
    assert FSSimulationData.deSimulator_weights == {
        "shallow": 3.0,
        "mid": 2.0,
        "deep": 1.0,
    }
    deModel = FSSimulationData.scenario.depositionalEnvironmentModel
    assert deModel is not None
    assert deModel.name == "simple3"
    assert [env.name for env in deModel.environments] == [
        "shallow",
        "mid",
        "deep",
    ]


def test_loadSimulationData_from_json_two_realizations() -> None:
    """Loads FSSimulationData and returns one FSSimulator per realization."""
    simulation_path = dataDir + "/simulation.json"
    FSSimulationData: FSSimulator = loadFSSimulation(simulation_path)

    assert len(FSSimulationData.realizationDataList) == 2, (
        "Expected 2 realizations in the loaded FSSimulationData"
    )
    # check scenario data
    assert FSSimulationData.scenario.name == "Scenario1"
    assert (
        FSSimulationData.scenario.accumulationModel.getElementModel(
            "CarbonateShallow"
        )
        is not None
    )

    # realization 1
    assert FSSimulationData.realizationDataList[0].well.name == "Well1"
    assert FSSimulationData.realizationDataList[0].subsidenceCurve is not None
    assert np.isclose(
        FSSimulationData.realizationDataList[0].subsidenceCurve.getValueAt(
            10.0
        ),
        25.0,
    )

    # realization 2
    assert FSSimulationData.realizationDataList[1].well.name == "Well2"
    assert FSSimulationData.realizationDataList[1].subsidenceCurve is not None
    assert np.isclose(
        FSSimulationData.realizationDataList[1].subsidenceCurve.getValueAt(
            10.0
        ),
        20.0,
    )


def test_loadSimulationData_rejects_wrong_format_version(
    tmp_path: Path,
) -> None:
    """Fails fast when the FSSimulationData format/version is wrong."""
    simulation_path = tmp_path / "simulation_bad_format.json"
    payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "9.9",
        "name": "MySimulation",
        "scenario": {},
        "realizations": [],
    }
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"format|version"):
        loadFSSimulation(str(simulation_path))


def test_loadSimulationData_rejects_missing_realizations(
    tmp_path: Path,
) -> None:
    """Rejects FSSimulationData when 'realizations' is missing or not list."""
    simulation_path = tmp_path / "simulation_missing_realizations.json"
    payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {
            "format": "pyWellSFM.ScenarioData",
            "version": "1.0",
            "name": "Scenario1",
            "accumulationModel": {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "AM1",
                    "modelType": "Gaussian",
                    "elements": {
                        "Carbonate": {
                            "accumulationRate": 100.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
        },
    }
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Simulation\.realizations must be a list"
    ):
        loadFSSimulation(str(simulation_path))


def test_loadSimulationData_rejects_empty_realizations(tmp_path: Path) -> None:
    """Rejects FSSimulationData when 'realizations' is an empty list."""
    simulation_path = tmp_path / "simulation_empty_realizations.json"
    payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {
            "format": "pyWellSFM.ScenarioData",
            "version": "1.0",
            "name": "Scenario1",
            "accumulationModel": {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "AM1",
                    "modelType": "Gaussian",
                    "elements": {
                        "Carbonate": {
                            "accumulationRate": 100.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
        },
        "realizations": [],
    }
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Simulation\.realizations must contain at least one item",
    ):
        loadFSSimulation(str(simulation_path))


def test_loadSimulationData_rejects_scenario_with_extra_keys(
    tmp_path: Path,
) -> None:
    """Scenario validation should reject unsupported properties."""
    simulation_path = tmp_path / "simulation_bad_scenario.json"

    payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {
            "format": "pyWellSFM.ScenarioData",
            "version": "1.0",
            "name": "Scenario1",
            "accumulationModel": {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "AM1",
                    "modelType": "Gaussian",
                    "elements": {
                        "Carbonate": {
                            "accumulationRate": 100.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
            "unexpected": 123,
        },
        "realizations": [
            {
                "well": {
                    "format": "pyWellSFM.WellData",
                    "version": "1.0",
                    "well": {
                        "name": "Well1",
                        "location": {"x": 1.0, "y": 2.0, "z": 3.0},
                        "depth": 100.0,
                    },
                },
                "subsidenceCurve": {
                    "type": "cumulative",
                    "curve": {
                        "format": "pyWellSFM.CurveData",
                        "version": "1.0",
                        "curve": {
                            "xAxisName": "Age",
                            "yAxisName": "Subsidence",
                            "interpolationMethod": "linear",
                            "data": [
                                {"x": 0.0, "y": 0.0},
                                {"x": 10.0, "y": 100.0},
                            ],
                        },
                    },
                },
            }
        ],
    }
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Scenario contains unsupported properties"
    ):
        loadFSSimulation(str(simulation_path))


def test_loadFSSimulation_rejects_non_object_realization_item(
    tmp_path: Path,
) -> None:
    """Each item of 'realizations' must be a JSON object."""
    simulation_path = tmp_path / "simulation_bad_realization_item.json"
    payload = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {
            "format": "pyWellSFM.ScenarioData",
            "version": "1.0",
            "name": "Scenario1",
            "accumulationModel": {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "AM1",
                    "modelType": "Gaussian",
                    "elements": {
                        "Carbonate": {
                            "accumulationRate": 100.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
        },
        "realizations": ["not-an-object"],
    }
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Simulation\.realizations\[0\] must be an object",
    ):
        loadFSSimulation(str(simulation_path))


def test_exportScenario_writes_inline_objects(tmp_path: Path) -> None:
    """Exports Scenario with inline objects."""
    scenario_path = tmp_path / "scenario_in.json"
    scenario_out = tmp_path / "scenario_out.json"
    payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "ScenarioExport",
        "accumulationModel": {
            "format": "pyWellSFM.AccumulationModelData",
            "version": "1.0",
            "accumulationModel": {
                "name": "AM1",
                "modelType": "Gaussian",
                "elements": {
                    "Carbonate": {
                        "accumulationRate": 100.0,
                        "model": {
                            "modelType": "Gaussian",
                            "stddevFactor": 0.2,
                        },
                    }
                },
            },
        },
        "eustaticCurve": {
            "format": "pyWellSFM.CurveData",
            "version": "1.0",
            "curve": {
                "xAxisName": "Age",
                "yAxisName": "SeaLevel",
                "interpolationMethod": "linear",
                "data": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 5.0}],
            },
        },
    }
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")

    scenario = loadScenario(str(scenario_path))
    saveScenario(scenario, str(scenario_out))

    out_obj = json.loads(scenario_out.read_text(encoding="utf-8"))
    assert isinstance(out_obj.get("accumulationModel"), dict)
    assert "url" not in out_obj["accumulationModel"]
    assert (
        out_obj["accumulationModel"].get("format")
        == "pyWellSFM.AccumulationModelData"
    )

    assert isinstance(out_obj.get("eustaticCurve"), dict)
    assert "url" not in out_obj["eustaticCurve"]
    assert out_obj["eustaticCurve"].get("format") == "pyWellSFM.CurveData"

    scenario2 = loadScenario(str(scenario_out))
    assert scenario2.name == "ScenarioExport"
    assert scenario2.eustaticCurve is not None
    assert np.isclose(scenario2.eustaticCurve.getValueAt(1.0), 5.0)


def test_exportRealizationData_writes_inline_objects(tmp_path: Path) -> None:
    """Exports RealizationData with inline well and curve."""
    realization_path = tmp_path / "realization_in.json"
    realization_out = tmp_path / "realization_out.json"

    payload = {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": {"url": f"{dataDir}/well.json"},
        "initialBathymetry": 15.0,
        "subsidenceCurve": {
            "type": "cumulative",
            "curve": {"url": f"{dataDir}/subsidence_curve.csv"},
        },
    }
    realization_path.write_text(json.dumps(payload), encoding="utf-8")

    rd = loadRealizationData(str(realization_path))
    saveRealizationData(rd, str(realization_out))

    out_obj = json.loads(realization_out.read_text(encoding="utf-8"))
    assert isinstance(out_obj.get("well"), dict)
    assert "url" not in out_obj["well"]
    assert out_obj["well"].get("format") == "pyWellSFM.WellData"
    assert isinstance(out_obj.get("initialBathymetry"), (int, float))
    assert isinstance(out_obj.get("subsidenceCurve"), dict)
    assert "type" in out_obj["subsidenceCurve"]
    assert out_obj["subsidenceCurve"].get("type") == "cumulative"

    curve_out = out_obj["subsidenceCurve"].get("curve", None)
    assert isinstance(curve_out, dict)
    assert "url" not in curve_out
    assert curve_out.get("format") == "pyWellSFM.CurveData"

    rd2 = loadRealizationData(str(realization_out))
    assert rd2.well.name == rd.well.name
    assert set(rd2.well.getContinuousLogNames()) == set(
        rd.well.getContinuousLogNames()
    )
    assert set(rd2.well.getDiscreteLogNames()) == set(
        rd.well.getDiscreteLogNames()
    )
    assert len(rd2.well.getMarkers()) == len(rd.well.getMarkers())
    assert rd2.subsidenceCurve is not None
    assert np.isclose(rd2.subsidenceCurve.getValueAt(0.0), 50.0)
    assert isinstance(rd2.initialBathymetry, (int, float))
    assert np.isclose(rd2.initialBathymetry, 15.0)


def test_exportSimulationData_writes_inline_objects_and_roundtrips(
    tmp_path: Path,
) -> None:
    """Exports FSSimulationData with inline scenario and realizations."""
    scenario_path = tmp_path / "scenario.json"
    simulation_out = tmp_path / "simulation_out.json"

    scenario_payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "ScenarioForSimulation",
        "accumulationModel": {
            "format": "pyWellSFM.AccumulationModelData",
            "version": "1.0",
            "accumulationModel": {
                "name": "AM1",
                "modelType": "Gaussian",
                "elements": {
                    "Carbonate": {
                        "accumulationRate": 100.0,
                        "model": {
                            "modelType": "Gaussian",
                            "stddevFactor": 0.2,
                        },
                    }
                },
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
    scenario = loadScenario(str(scenario_path))

    # Build realizations from minimal inline JSON (no external URLs).
    r1_path = tmp_path / "r1.json"
    r2_path = tmp_path / "r2.json"
    r1_path.write_text(
        json.dumps(
            {
                "format": "pyWellSFM.RealizationData",
                "version": "1.0",
                "well": {
                    "format": "pyWellSFM.WellData",
                    "version": "1.0",
                    "well": {
                        "name": "Well1",
                        "location": {"x": 1.0, "y": 2.0, "z": 3.0},
                        "depth": 100.0,
                    },
                },
                "initialBathymetry": 15.0,
                "subsidenceCurve": {
                    "type": "cumulative",
                    "curve": {
                        "format": "pyWellSFM.CurveData",
                        "version": "1.0",
                        "curve": {
                            "xAxisName": "Age",
                            "yAxisName": "Subsidence",
                            "interpolationMethod": "linear",
                            "data": [
                                {"x": 0.0, "y": 0.0},
                                {"x": 10.0, "y": 100.0},
                            ],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    r2_path.write_text(
        json.dumps(
            {
                "format": "pyWellSFM.RealizationData",
                "version": "1.0",
                "well": {
                    "format": "pyWellSFM.WellData",
                    "version": "1.0",
                    "well": {
                        "name": "Well2",
                        "location": {"x": 4.0, "y": 5.0, "z": 6.0},
                        "depth": 200.0,
                    },
                },
                "initialBathymetry": 20.0,
                "subsidenceCurve": {
                    "type": "cumulative",
                    "curve": {
                        "format": "pyWellSFM.CurveData",
                        "version": "1.0",
                        "curve": {
                            "xAxisName": "Age",
                            "yAxisName": "Subsidence",
                            "interpolationMethod": "linear",
                            "data": [
                                {"x": 0.0, "y": 0.0},
                                {"x": 10.0, "y": 50.0},
                            ],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rd1 = loadRealizationData(str(r1_path))
    rd2 = loadRealizationData(str(r2_path))

    sim = FSSimulator(scenario=scenario, realizationDataList=[rd1, rd2])
    saveFSSimulation(sim, str(simulation_out), name="MySimulation")

    out_obj = json.loads(simulation_out.read_text(encoding="utf-8"))
    assert isinstance(out_obj.get("scenario"), dict)
    assert "url" not in out_obj["scenario"]
    assert isinstance(out_obj.get("realizations"), list)
    assert len(out_obj["realizations"]) == 2
    assert all(
        isinstance(r, dict) and "url" not in r for r in out_obj["realizations"]
    )

    FSSimulationData: FSSimulator = loadFSSimulation(str(simulation_out))
    assert len(FSSimulationData.realizationDataList) == 2
    assert FSSimulationData.scenario.name == "ScenarioForSimulation"
    assert FSSimulationData.realizationDataList[0].well.name == "Well1"
    assert FSSimulationData.realizationDataList[0].initialBathymetry == 15.0
    assert FSSimulationData.realizationDataList[1].well.name == "Well2"
    assert FSSimulationData.realizationDataList[1].initialBathymetry == 20.0


def _minimal_accumulation_model_obj() -> dict[str, object]:
    return {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "AM_MIN",
            "modelType": "Gaussian",
            "elements": {
                "Carbonate": {
                    "accumulationRate": 100.0,
                    "model": {
                        "modelType": "Gaussian",
                        "stddevFactor": 0.2,
                    },
                }
            },
        },
    }


def _minimal_scenario_obj(name: str = "ScenarioMin") -> dict[str, object]:
    return {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": name,
        "accumulationModel": _minimal_accumulation_model_obj(),
    }


def _minimal_realization_obj() -> dict[str, object]:
    return {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": {
            "format": "pyWellSFM.WellData",
            "version": "1.0",
            "well": {
                "name": "WellMin",
                "location": {"x": 1.0, "y": 2.0, "z": 3.0},
                "depth": 100.0,
            },
        },
        "initialBathymetry": 15.0,
    }


def _minimal_simulation_obj() -> dict[str, object]:
    return {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": "SimMin",
        "scenario": _minimal_scenario_obj(),
        "realizations": [_minimal_realization_obj()],
    }


def _minimal_de_model_obj() -> dict[str, object]:
    return {
        "format": "pyWellSFM.DepositionalEnvironmentModelSchema",
        "version": "1.0",
        "name": "simple",
        "environments": [
            {
                "name": "shallow",
                "waterDepthModel": {
                    "format": "pyWellSFM.EnvironmentConditionModelData",
                    "version": "1.0",
                    "model": {
                        "modelType": "Uniform",
                        "minValue": 0.0,
                        "maxValue": 10.0,
                    },
                },
            }
        ],
    }


def test_loadRealizationData_without_subsidence_curve(tmp_path: Path) -> None:
    """Subsidence curve is optional and defaults to None."""
    realization_path = tmp_path / "realization_no_subsidence.json"
    payload = _minimal_realization_obj()
    realization_path.write_text(json.dumps(payload), encoding="utf-8")

    rd = loadRealizationData(str(realization_path))
    assert rd.subsidenceCurve is None


def test_loadRealizationData_rejects_invalid_bathymetry_type(
    tmp_path: Path,
) -> None:
    """Reject non-numeric initial bathymetry values."""
    payload = _minimal_realization_obj()
    payload["initialBathymetry"] = "fifteen"
    path = tmp_path / "realization_bad_bathy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"initialBathymetry must be a number"
    ):
        loadRealizationData(str(path))


def test_loadRealizationData_rejects_invalid_initial_environment_type(
    tmp_path: Path,
) -> None:
    """Reject non-string initialEnvironment values."""
    payload = _minimal_realization_obj()
    payload["initialEnvironment"] = 10
    path = tmp_path / "realization_bad_initial_env.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"initialEnvironment must be a string"
    ):
        loadRealizationData(str(path))


def test_loadRealizationData_rejects_missing_subsidence_curve_fields(
    tmp_path: Path,
) -> None:
    """Require both curve and type in subsidenceCurve."""
    payload = _minimal_realization_obj()
    payload["subsidenceCurve"] = {
        "curve": {"url": f"{dataDir}/subsidence_curve.csv"}
    }
    path = tmp_path / "realization_bad_subs_fields.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"must contain 'curve' and 'type'"):
        loadRealizationData(str(path))


def test_loadRealizationData_rejects_invalid_subsidence_type(
    tmp_path: Path,
) -> None:
    """Reject subsidenceCurve.type values outside cumulative/rate."""
    payload = _minimal_realization_obj()
    payload["subsidenceCurve"] = {
        "type": "invalid",
        "curve": {"url": f"{dataDir}/subsidence_curve.csv"},
    }
    path = tmp_path / "realization_bad_subs_type.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"must be either 'cumulative' or 'rate'"
    ):
        loadRealizationData(str(path))


def test_loadRealizationData_rejects_unsupported_well_url(
    tmp_path: Path,
) -> None:
    """Wrap unsupported well URL targets with a clear ValueError."""
    bad_well = tmp_path / "well.txt"
    bad_well.write_text("not-a-well", encoding="utf-8")
    payload = _minimal_realization_obj()
    payload["well"] = {"url": "well.txt"}
    path = tmp_path / "realization_bad_well_url.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"well\.url must point"):
        loadRealizationData(str(path))


def test_loadRealizationData_rejects_unsupported_subsidence_url(
    tmp_path: Path,
) -> None:
    """Wrap unsupported subsidence URL targets with a clear ValueError."""
    bad_curve = tmp_path / "subsidence.txt"
    bad_curve.write_text("not-a-curve", encoding="utf-8")
    payload = _minimal_realization_obj()
    payload["subsidenceCurve"] = {
        "type": "cumulative",
        "curve": {"url": "subsidence.txt"},
    }
    path = tmp_path / "realization_bad_subs_url.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"subsidenceCurve\.url must point"):
        loadRealizationData(str(path))


def test_exportRealizationData_null_subsidence_curve_when_missing(
    tmp_path: Path,
) -> None:
    """Exporter writes subsidenceCurve as null when absent in model."""
    in_path = tmp_path / "realization_in.json"
    out_path = tmp_path / "realization_out.json"
    in_path.write_text(
        json.dumps(_minimal_realization_obj()), encoding="utf-8"
    )

    rd = loadRealizationData(str(in_path))
    saveRealizationData(rd, str(out_path))

    out_obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert out_obj["subsidenceCurve"] is None


def test_saveRealizationData_rejects_non_json_extension(
    tmp_path: Path,
) -> None:
    """Reject non-JSON output extension for realization export."""
    in_path = tmp_path / "realization_in.json"
    in_path.write_text(
        json.dumps(_minimal_realization_obj()), encoding="utf-8"
    )
    rd = loadRealizationData(str(in_path))

    with pytest.raises(ValueError, match=r"must have a \.json extension"):
        saveRealizationData(rd, str(tmp_path / "realization_out.txt"))


def test_loadScenario_rejects_empty_name(tmp_path: Path) -> None:
    """Reject blank Scenario.name values."""
    payload = _minimal_scenario_obj(name="   ")
    path = tmp_path / "scenario_empty_name.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Scenario\.name must be a non-empty"
    ):
        loadScenario(str(path))


def test_loadScenario_rejects_unsupported_accumulation_url(
    tmp_path: Path,
) -> None:
    """Wrap unsupported accumulation model URL references."""
    payload = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": "ScenarioBadAM",
        "accumulationModel": {"url": "missing_accumulation_model.csv"},
    }
    path = tmp_path / "scenario_bad_am_url.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"accumulationModel\.url must point"):
        loadScenario(str(path))


def test_loadScenario_rejects_non_object_de_model(tmp_path: Path) -> None:
    """Reject non-object depositionalEnvironmentModel payloads."""
    payload = _minimal_scenario_obj()
    payload["depositionalEnvironmentModel"] = 1
    path = tmp_path / "scenario_bad_demodel_type.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"depositionalEnvironmentModel must be an object"
    ):
        loadScenario(str(path))


def test_loadScenario_rejects_unsupported_de_model_url(
    tmp_path: Path,
) -> None:
    """Wrap unsupported depositional environment model URL references."""
    payload = _minimal_scenario_obj()
    payload["depositionalEnvironmentModel"] = {"url": "missing_demodel.csv"}
    path = tmp_path / "scenario_bad_demodel_url.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"depositionalEnvironmentModel\.url must point"
    ):
        loadScenario(str(path))


def test_loadScenario_rejects_unsupported_eustatic_url(
    tmp_path: Path,
) -> None:
    """Wrap unsupported eustatic curve URL references."""
    payload = _minimal_scenario_obj()
    payload["eustaticCurve"] = {"url": "missing_curve.csv"}
    path = tmp_path / "scenario_bad_eustatic_url.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"eustaticCurve\.url must point"):
        loadScenario(str(path))


def test_saveScenario_rejects_empty_name_on_export(tmp_path: Path) -> None:
    """Export validation rejects blank names on Scenario objects."""
    scenario_in = tmp_path / "scenario_in.json"
    scenario_in.write_text(
        json.dumps(_minimal_scenario_obj()), encoding="utf-8"
    )
    scenario = loadScenario(str(scenario_in))
    invalid = replace(scenario, name="")

    with pytest.raises(
        ValueError, match=r"Scenario\.name must be a non-empty"
    ):
        saveScenario(invalid, str(tmp_path / "scenario_out.json"))


def test_saveScenario_exports_de_model_and_handles_non_null_facies(
    tmp_path: Path,
) -> None:
    """Export includes DE model and tolerates non-null faciesModel."""
    payload = _minimal_scenario_obj(name="ScenarioWithDE")
    payload["depositionalEnvironmentModel"] = _minimal_de_model_obj()
    scenario_in = tmp_path / "scenario_with_de.json"
    scenario_out = tmp_path / "scenario_with_de_out.json"
    scenario_in.write_text(json.dumps(payload), encoding="utf-8")

    scenario = loadScenario(str(scenario_in))
    scenario_with_facies = replace(
        scenario,
        faciesModel=object(),  # type: ignore[arg-type]
    )
    saveScenario(scenario_with_facies, str(scenario_out))

    out_obj = json.loads(scenario_out.read_text(encoding="utf-8"))
    assert isinstance(out_obj.get("depositionalEnvironmentModel"), dict)
    assert "faciesModel" not in out_obj


def test_saveScenario_rejects_non_json_extension(tmp_path: Path) -> None:
    """Reject non-JSON output extension for scenario export."""
    scenario_in = tmp_path / "scenario_in.json"
    scenario_in.write_text(
        json.dumps(_minimal_scenario_obj()), encoding="utf-8"
    )
    scenario = loadScenario(str(scenario_in))

    with pytest.raises(ValueError, match=r"must have a \.json extension"):
        saveScenario(scenario, str(tmp_path / "scenario_out.txt"))


def test_saveFSSimulation_rejects_non_json_extension(tmp_path: Path) -> None:
    """Reject non-JSON output extension for simulation export."""
    scenario_path = tmp_path / "scenario.json"
    realization_path = tmp_path / "realization.json"
    scenario_path.write_text(
        json.dumps(_minimal_scenario_obj()), encoding="utf-8"
    )
    realization_path.write_text(
        json.dumps(_minimal_realization_obj()), encoding="utf-8"
    )

    scenario = loadScenario(str(scenario_path))
    rd = loadRealizationData(str(realization_path))
    sim = FSSimulator(scenario=scenario, realizationDataList=[rd])

    with pytest.raises(ValueError, match=r"must have a \.json extension"):
        saveFSSimulation(sim, str(tmp_path / "simulation.txt"), name="Sim")


def test_saveFSSimulation_rejects_empty_name(tmp_path: Path) -> None:
    """Reject blank simulation names during export."""
    scenario_path = tmp_path / "scenario.json"
    realization_path = tmp_path / "realization.json"
    scenario_path.write_text(
        json.dumps(_minimal_scenario_obj()), encoding="utf-8"
    )
    realization_path.write_text(
        json.dumps(_minimal_realization_obj()), encoding="utf-8"
    )

    scenario = loadScenario(str(scenario_path))
    rd = loadRealizationData(str(realization_path))
    sim = FSSimulator(scenario=scenario, realizationDataList=[rd])

    with pytest.raises(
        ValueError, match=r"Simulation\.name must be a non-empty"
    ):
        saveFSSimulation(sim, str(tmp_path / "simulation.json"), name="")


def test_saveFSSimulation_rejects_empty_realizations(tmp_path: Path) -> None:
    """Reject exports with empty realizationDataList."""
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(_minimal_scenario_obj()), encoding="utf-8"
    )

    scenario = loadScenario(str(scenario_path))
    sim = FSSimulator(scenario=scenario, realizationDataList=[])

    with pytest.raises(
        ValueError,
        match=r"Simulation\.realizations must contain at least one item",
    ):
        saveFSSimulation(sim, str(tmp_path / "simulation.json"), name="Sim")


def test_saveFSSimulation_exports_de_simulator_fields(tmp_path: Path) -> None:
    """Export includes DE simulator fields when enabled."""
    scenario_path = tmp_path / "scenario.json"
    realization_path = tmp_path / "realization.json"
    out_path = tmp_path / "simulation_out.json"
    scenario_path.write_text(
        json.dumps(_minimal_scenario_obj()), encoding="utf-8"
    )
    realization_path.write_text(
        json.dumps(_minimal_realization_obj()), encoding="utf-8"
    )

    scenario = loadScenario(str(scenario_path))
    rd = loadRealizationData(str(realization_path))
    sim = FSSimulator(
        scenario=scenario,
        realizationDataList=[rd],
        use_depositional_environment_simulator=True,
        deSimulator_weights={"shallow": 3.0},
        deSimulator_params=DESimulatorParameters(),
    )
    saveFSSimulation(sim, str(out_path), name="SimWithDE")

    out_obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert out_obj["use_depositional_environment_simulator"] is True
    assert out_obj["deSimulator_weights"] == {"shallow": 3.0}
    assert isinstance(out_obj["deSimulator_params"], dict)


def test_loadFSSimulation_rejects_empty_name(tmp_path: Path) -> None:
    """Reject blank simulation names while loading."""
    payload = _minimal_simulation_obj()
    payload["name"] = " "
    path = tmp_path / "simulation_bad_name.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Simulation\.name must be a non-empty"
    ):
        loadFSSimulation(str(path))


def test_loadFSSimulation_rejects_non_boolean_use_desimulator(
    tmp_path: Path,
) -> None:
    """Require boolean value for DE simulator toggle."""
    payload = _minimal_simulation_obj()
    payload["use_depositional_environment_simulator"] = "yes"
    path = tmp_path / "simulation_bad_use_desimulator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"use_depositional_environment_simulator.*must be a boolean",
    ):
        loadFSSimulation(str(path))


def test_loadFSSimulation_rejects_non_object_deSimulator_weights(
    tmp_path: Path,
) -> None:
    """Require DE simulator weights to be provided as an object."""
    payload = _minimal_simulation_obj()
    payload["use_depositional_environment_simulator"] = True
    payload["deSimulator_weights"] = [1, 2]
    path = tmp_path / "simulation_bad_desim_weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"deSimulator_weights must be an object"
    ):
        loadFSSimulation(str(path))


def test_loadFSSimulation_rejects_non_object_deSimulator_params(
    tmp_path: Path,
) -> None:
    """Require DE simulator params to be provided as an object."""
    payload = _minimal_simulation_obj()
    payload["use_depositional_environment_simulator"] = True
    payload["deSimulator_params"] = 1
    path = tmp_path / "simulation_bad_desim_params.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"deSimulator_params must be an object"
    ):
        loadFSSimulation(str(path))


def test_loadFSSimulation_rejects_non_dict_params(tmp_path: Path) -> None:
    """Require Simulation.params to be a dictionary when provided."""
    payload = _minimal_simulation_obj()
    payload["params"] = "invalid"
    path = tmp_path / "simulation_bad_params.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Simulation\.params must be a dictionary"
    ):
        loadFSSimulation(str(path))
