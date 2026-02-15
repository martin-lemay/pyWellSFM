# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from striplog import Striplog

from pywellsfm.model.Curve import Curve
from pywellsfm.simulator.FSSimulatorRunner import FSSimulatorRunnerData

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.io.simulation_io import (  # noqa: E402
    loadRealizationData,
    loadScenario,
    loadSimulationData,
    saveRealizationData,
    saveScenario,
    saveSimulationData,
)
from pywellsfm.model.AccumulationModel import (
    AccumulationModel,  # noqa: E402
)
from pywellsfm.model.SimulationParameters import SimulationData  # noqa: E402

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


def test_loadSimulationData_from_json_with_references(tmp_path: Path) -> None:
    """SimulationData where scenario and realizations are URL references."""
    scenario_path = tmp_path / "scenario.json"
    realization_path = tmp_path / "realization.json"
    simulation_path = tmp_path / "simulation.json"

    # Scenario uses inline internals here; we're validating SimulationData.
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
        "format": "pyWellSFM.SimulationData",
        "version": "1.0",
        "name": "MySimulation",
        "scenario": {"url": "scenario.json"},
        "realizations": [{"url": "realization.json"}],
    }
    simulation_path.write_text(
        json.dumps(simulation_payload), encoding="utf-8"
    )

    simulationData: FSSimulatorRunnerData = loadSimulationData(
        str(simulation_path)
    )
    assert len(simulationData.realizationDataList) == 1
    assert simulationData.scenario.name == "Scenario1"
    assert (
        simulationData.scenario.accumulationModel.getElementModel("Carbonate")
        is not None
    )
    assert simulationData.realizationDataList[0].well.name == "Well-B"
    assert simulationData.realizationDataList[0].subsidenceCurve is not None
    assert simulationData.realizationDataList[0].initialBathymetry == 15.0


def test_loadSimulationData_from_json_two_realizations() -> None:
    """Loads SimulationData and returns one FSSimulator per realization."""
    simulation_path = dataDir + "/simulation.json"
    simulationData: FSSimulatorRunnerData = loadSimulationData(simulation_path)

    assert len(simulationData.realizationDataList) == 2, (
        "Expected 2 realizations in the loaded SimulationData"
    )
    # check scenario data
    assert simulationData.scenario.name == "Scenario1"
    assert (
        simulationData.scenario.accumulationModel.getElementModel(
            "CarbonateShallow"
        )
        is not None
    )

    # realization 1
    assert simulationData.realizationDataList[0].well.name == "Well1"
    assert simulationData.realizationDataList[0].subsidenceCurve is not None
    assert np.isclose(
        simulationData.realizationDataList[0].subsidenceCurve.getValueAt(10.0),
        25.0,
    )

    # realization 2
    assert simulationData.realizationDataList[1].well.name == "Well2"
    assert simulationData.realizationDataList[1].subsidenceCurve is not None
    assert np.isclose(
        simulationData.realizationDataList[1].subsidenceCurve.getValueAt(10.0),
        20.0,
    )


def test_loadSimulationData_rejects_wrong_format_version(
    tmp_path: Path,
) -> None:
    """Fails fast when the top-level SimulationData format/version is wrong."""
    simulation_path = tmp_path / "simulation_bad_format.json"
    payload = {
        "format": "pyWellSFM.SimulationData",
        "version": "9.9",
        "name": "MySimulation",
        "scenario": {},
        "realizations": [],
    }
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"format|version"):
        loadSimulationData(str(simulation_path))


def test_loadSimulationData_rejects_missing_realizations(
    tmp_path: Path,
) -> None:
    """Rejects SimulationData when 'realizations' is missing or not a list."""
    simulation_path = tmp_path / "simulation_missing_realizations.json"
    payload = {
        "format": "pyWellSFM.SimulationData",
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
        loadSimulationData(str(simulation_path))


def test_loadSimulationData_rejects_empty_realizations(tmp_path: Path) -> None:
    """Rejects SimulationData when 'realizations' is an empty list."""
    simulation_path = tmp_path / "simulation_empty_realizations.json"
    payload = {
        "format": "pyWellSFM.SimulationData",
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
        loadSimulationData(str(simulation_path))


def test_loadSimulationData_rejects_scenario_with_extra_keys(
    tmp_path: Path,
) -> None:
    """Scenario validation should reject unsupported properties."""
    simulation_path = tmp_path / "simulation_bad_scenario.json"

    payload = {
        "format": "pyWellSFM.SimulationData",
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
        loadSimulationData(str(simulation_path))


def test_loadSimulationData_rejects_non_object_realization_item(
    tmp_path: Path,
) -> None:
    """Each item of 'realizations' must be a JSON object."""
    simulation_path = tmp_path / "simulation_bad_realization_item.json"
    payload = {
        "format": "pyWellSFM.SimulationData",
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
        loadSimulationData(str(simulation_path))


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
    """Exports SimulationData with inline scenario and realizations."""
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

    sim = SimulationData(scenario=scenario, realizationsData=[rd1, rd2])
    saveSimulationData(sim, str(simulation_out), name="MySimulation")

    out_obj = json.loads(simulation_out.read_text(encoding="utf-8"))
    assert isinstance(out_obj.get("scenario"), dict)
    assert "url" not in out_obj["scenario"]
    assert isinstance(out_obj.get("realizations"), list)
    assert len(out_obj["realizations"]) == 2
    assert all(
        isinstance(r, dict) and "url" not in r for r in out_obj["realizations"]
    )

    simulationData: FSSimulatorRunnerData = loadSimulationData(
        str(simulation_out)
    )
    assert len(simulationData.realizationDataList) == 2
    assert simulationData.scenario.name == "ScenarioForSimulation"
    assert simulationData.realizationDataList[0].well.name == "Well1"
    assert simulationData.realizationDataList[0].initialBathymetry == 15.0
    assert simulationData.realizationDataList[1].well.name == "Well2"
    assert simulationData.realizationDataList[1].initialBathymetry == 20.0
