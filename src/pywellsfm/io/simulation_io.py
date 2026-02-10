# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for Simulation data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pywellsfm.io._common import (
    load_inline_or_url,
    reject_extra_keys,
)
from pywellsfm.io.accumulation_model_io import (
    accumulationModelToJsonObj,
    loadAccumulationModel,
    loadAccumulationModelFromJsonObj,
)
from pywellsfm.io.curve_io import (
    curveToJsonObj,
    loadEustaticCurve,
    loadEustaticCurveFromJsonObj,
    loadSubsidenceCurve,
    loadSubsidenceCurveFromJsonObj,
)
from pywellsfm.io.facies_model_io import (
    faciesModelToJsonObj,
    loadFaciesModel,
    loadFaciesModelFromJsonObj,
)
from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.io.well_io import loadWell, loadWellFromJsonObj, wellToJsonObj
from pywellsfm.model import Curve
from pywellsfm.model.SimulationParameters import (
    RealizationData,
    Scenario,
    SimulationData,
)
from pywellsfm.simulator.FSSimulator import FSSimulator


def loadRealizationData(filepath: str) -> RealizationData:
    """Load realization data from json file.

    json file schema is defined by jsonSchemas/RealizationData.json

    :param str filepath: path to realization data json file
    :return RealizationData: realization data object
    """
    path = Path(filepath)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _loadRealizationDataFromJsonObj(
        data, base_dir=path.resolve().parent
    )


def _loadRealizationDataFromJsonObj(
    obj: dict[str, Any], base_dir: Path
) -> RealizationData:
    """Helper to load RealizationData from json dict.

    :param dict[str, Any] obj: json dict representing RealizationData
    :param Path base_dir: base directory for resolving relative file paths
    :raises ValueError: if well property is missing or not an object
    :raises ValueError: if subsidenceCurve property is missing or not an object
    :return RealizationData: realization data object
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.RealizationData",
        expected_version="1.0",
        kind="realization",
    )

    reject_extra_keys(
        obj=obj,
        allowed_keys={"format", "version", "well", "subsidenceCurve"},
        ctx="RealizationData",
    )

    # --- Well ---
    well_obj: Any = obj.get("well")

    def _load_inline_well(well_json: dict[str, Any]) -> Any: # noqa: ANN401
        return loadWellFromJsonObj(well_json, base_dir=base_dir)

    def _load_well_file(path: Path) -> Any: # noqa: ANN401
        try:
            return loadWell(str(path))
        except ValueError as exc:
            raise ValueError(
                "RealizationData.well.url must point to a supported well file "
                "(.json or .las)."
            ) from exc

    well = load_inline_or_url(
        well_obj,
        base_dir=base_dir,
        ctx="RealizationData.well",
        load_inline=_load_inline_well,
        load_file=_load_well_file,
    )

    # --- Subsidence curve (optional) ---
    subsidence_curve: Curve | None
    subs_curve_obj: Any = obj.get("subsidenceCurve")
    if subs_curve_obj is None:
        subsidence_curve = None
    else:

        def _load_inline_subs_curve(curve_json: dict[str, Any]) -> Curve:
            return loadSubsidenceCurveFromJsonObj(curve_json)

        def _load_subs_curve_file(path: Path) -> Curve:
            try:
                return loadSubsidenceCurve(str(path))
            except ValueError as exc:
                raise ValueError(
                    "RealizationData.subsidenceCurve.url must point to a "
                    "supported curve file (.json or .csv)."
                ) from exc

        subsidence_curve = load_inline_or_url(
            subs_curve_obj,
            base_dir=base_dir,
            ctx="RealizationData.subsidenceCurve",
            load_inline=_load_inline_subs_curve,
            load_file=_load_subs_curve_file,
        )

    return RealizationData(well=well, subsidenceCurve=subsidence_curve)


def exportRealizationDataToJsonObj(
    realizationData: RealizationData,
) -> dict[str, Any]:
    """Export RealizationData object to json object.

    json format conforms to json/RealizationDataSchema.json.

    :param RealizationData realizationData: RealizationData object
    :raises ValueError: if realizationData.well is not defined
    :return dict[str, Any]: JSON-serializable dictionary representing the
        realization data
    """
    payload: dict[str, Any] = {
        "format": "pyWellSFM.RealizationData",
        "version": "1.0",
        "well": wellToJsonObj(realizationData.well),
    }

    if realizationData.subsidenceCurve is not None:
        payload["subsidenceCurve"] = curveToJsonObj(
            realizationData.subsidenceCurve
        )
    else:
        payload["subsidenceCurve"] = None
    return payload


def saveRealizationData(
    realizationData: RealizationData, filepath: str
) -> None:
    """Save RealizationData object to json file.

    json file format conforms to json/RealizationDataSchema.json.

    :param RealizationData realizationData: RealizationData object
    :param str filepath: file path to write JSON output
    :raises ValueError: if filepath does not have .json extension
    """
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "RealizationData output file must have a .json extension."
        )
    out = exportRealizationDataToJsonObj(realizationData)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def loadScenario(filepath: str) -> Scenario:
    """Load scenario from json file.

    json file schema is defined by jsonSchemas/ScenarioSchema.json

    :param str filepath: path to scenario json file
    :return Scenario: scenario object
    """
    path = Path(filepath)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _loadScenarioFromJsonObj(data, base_dir=path.resolve().parent)


def _loadScenarioFromJsonObj(obj: dict[str, Any], base_dir: Path) -> Scenario:
    """Helper to load Scenario from json dict.

    Supports inline or URL-referenced objects for:

    - faciesModel
    - accumulationModel
    - eustaticCurve

    Relative URLs are resolved against ``base_dir``.

    :param dict[str, Any] obj: json dict representing Scenario
    :param Path base_dir: base directory for resolving relative file paths
    :raises ValueError: if required properties are missing/invalid
    :return Scenario: scenario object
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.ScenarioData",
        expected_version="1.0",
        kind="scenario",
    )

    reject_extra_keys(
        obj=obj,
        allowed_keys={
            "format",
            "version",
            "name",
            "faciesModel",
            "accumulationModel",
            "eustaticCurve",
        },
        ctx="Scenario",
    )

    name = obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Scenario.name must be a non-empty string.")

    # --- Facies model ---
    facies_model_obj: Any = obj.get("faciesModel")

    def _load_facies_file(path: Path) -> Any: # noqa: ANN401
        try:
            return loadFaciesModel(str(path))
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise ValueError(
                "Scenario.faciesModel.url must point to a supported facies "
                "model file (.json)."
            ) from exc

    facies_model = load_inline_or_url(
        facies_model_obj,
        base_dir=base_dir,
        ctx="Scenario.faciesModel",
        load_inline=loadFaciesModelFromJsonObj,
        load_file=_load_facies_file,
    )

    # --- Accumulation model ---
    accumulation_model_obj: Any = obj.get("accumulationModel")

    def _load_inline_am(am_json: dict[str, Any]) -> Any: # noqa: ANN401
        return loadAccumulationModelFromJsonObj(am_json, base_dir=base_dir)

    def _load_am_file(path: Path) -> Any: # noqa: ANN401
        try:
            return loadAccumulationModel(str(path))
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise ValueError(
                "Scenario.accumulationModel.url must point to a supported "
                "accumulation model file (.json or .csv)."
            ) from exc

    accumulation_model = load_inline_or_url(
        accumulation_model_obj,
        base_dir=base_dir,
        ctx="Scenario.accumulationModel",
        load_inline=_load_inline_am,
        load_file=_load_am_file,
    )

    # --- Eustatic curve (optional) ---
    eustatic_curve: Curve | None
    eustatic_curve_obj: Any = obj.get("eustaticCurve")
    if eustatic_curve_obj is None:
        eustatic_curve = None
    else:

        def _load_inline_eustatic(curve_json: dict[str, Any]) -> Curve:
            return loadEustaticCurveFromJsonObj(curve_json)

        def _load_eustatic_file(path: Path) -> Curve:
            try:
                return loadEustaticCurve(str(path))
            except (ValueError, FileNotFoundError, OSError) as exc:
                raise ValueError(
                    "Scenario.eustaticCurve.url must point to a supported "
                    "curve file (.json or .csv)."
                ) from exc

        eustatic_curve = load_inline_or_url(
            eustatic_curve_obj,
            base_dir=base_dir,
            ctx="Scenario.eustaticCurve",
            load_inline=_load_inline_eustatic,
            load_file=_load_eustatic_file,
        )

    return Scenario(
        name=name,
        accumulationModel=accumulation_model,
        eustaticCurve=eustatic_curve,
        faciesModel=facies_model,
    )


def exportScenarioToJsonObj(scenario: Scenario) -> dict[str, Any]:
    """Export Scenario object to json object.

    json format conforms to json/ScenarioSchema.json.

    :param Scenario scenario: Scenario object
    :raises ValueError: if Scenario.name is not a non-empty string
    :return dict[str, Any]: JSON-serializable dictionary representing the
        scenario
    """
    if not isinstance(scenario.name, str) or scenario.name.strip() == "":
        raise ValueError("Scenario.name must be a non-empty string.")

    payload: dict[str, Any] = {
        "format": "pyWellSFM.ScenarioData",
        "version": "1.0",
        "name": scenario.name,
        "faciesModel": faciesModelToJsonObj(scenario.faciesModel),
        "accumulationModel": accumulationModelToJsonObj(
            scenario.accumulationModel
        ),
    }
    if scenario.eustaticCurve is not None:
        payload["eustaticCurve"] = curveToJsonObj(scenario.eustaticCurve)
    else:
        payload["eustaticCurve"] = None
    return payload


def saveScenario(scenario: Scenario, filepath: str) -> None:
    """Save Scenario object to json file.

    json file format conforms to json/ScenarioSchema.json.

    :param Scenario scenario: Scenario object
    :param str filepath: file path to write JSON output
    :raises ValueError: if filepath does not have .json extension
    """
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Scenario output file must have a .json extension.")
    out = exportScenarioToJsonObj(scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def exportSimulationDataToJsonObj(
    simulationData: SimulationData,
    *,
    name: str,
) -> dict[str, Any]:
    """Export SimulationData object to json object.

    json format conforms to json/SimulationDataSchema.json.

    :param SimulationData simulationData: SimulationData object
    :param str name: name of the simulation data
    :raises ValueError: if name is not a non-empty string
    :raises ValueError: if realizationsData is not a non-empty list
    :return dict[str, Any]: JSON-serializable dictionary representing the
        simulation data
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Simulation.name must be a non-empty string.")
    if (
        not isinstance(simulationData.realizationsData, list)
        or len(simulationData.realizationsData) < 1
    ):
        raise ValueError(
            "Simulation.realizations must contain at least one item."
        )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.SimulationData",
        "version": "1.0",
        "name": str(name),
        "scenario": exportScenarioToJsonObj(simulationData.scenario),
        "realizations": [
            exportRealizationDataToJsonObj(r)
            for r in simulationData.realizationsData
        ],
    }
    return payload


def saveSimulationData(
    simulationData: SimulationData,
    filepath: str,
    *,
    name: str,
) -> None:
    """Save SimulationData object to json file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "SimulationData output file must have a .json extension."
        )
    out = exportSimulationDataToJsonObj(simulationData, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def loadSimulationData(filepath: str) -> list[FSSimulator]:
    """Load scenario from json file.

    json file schema is defined by jsonSchemas/ScenarioSchema.json

    :param str filepath: path to scenario json file
    :return Scenario: scenario object
    """
    path = Path(filepath)
    data = json.loads(path.read_text(encoding="utf-8"))

    expect_format_version(
        data,
        expected_format="pyWellSFM.SimulationData",
        expected_version="1.0",
        kind="scenario",
    )

    reject_extra_keys(
        obj=data,
        allowed_keys={"format", "version", "name", "scenario", "realizations"},
        ctx="SimulationData",
    )

    base_dir = path.resolve().parent

    name = data.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Simulation.name must be a non-empty string.")

    # --- Scenario ---
    scenario_obj: Any = data.get("scenario")

    def _load_inline_scenario(scenario_json: dict[str, Any]) -> Scenario:
        return _loadScenarioFromJsonObj(scenario_json, base_dir=base_dir)

    def _load_scenario_file(path: Path) -> Scenario:
        return loadScenario(str(path))

    scenario = load_inline_or_url(
        scenario_obj,
        base_dir=base_dir,
        ctx="Simulation.scenario",
        load_inline=_load_inline_scenario,
        load_file=_load_scenario_file,
    )

    # --- Realizations ---
    realizations_obj: Any = data.get("realizations")
    if not isinstance(realizations_obj, list):
        raise ValueError("Simulation.realizations must be a list.")
    if len(realizations_obj) == 0:
        raise ValueError(
            "Simulation.realizations must contain at least one item."
        )

    realizations_data: list[RealizationData] = []
    for idx, realization_item in enumerate(realizations_obj):
        ctx = f"Simulation.realizations[{idx}]"

        def _load_inline_realization(
            real_json: dict[str, Any],
        ) -> RealizationData:
            return _loadRealizationDataFromJsonObj(
                real_json, base_dir=base_dir
            )

        def _load_realization_file(path: Path) -> RealizationData:
            return loadRealizationData(str(path))

        realizations_data.append(
            load_inline_or_url(
                realization_item,
                base_dir=base_dir,
                ctx=ctx,
                load_inline=_load_inline_realization,
                load_file=_load_realization_file,
            )
        )

    simulators = [
        FSSimulator(scenario=scenario, realizationData=realization_data)
        for realization_data in realizations_data
    ]
    return simulators
