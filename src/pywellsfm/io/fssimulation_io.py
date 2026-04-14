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
from pywellsfm.io.depositional_environment_model_io import (
    depositionalEnvironmentModelToJsonObj,
    loadDepositionalEnvironmentModel,
    loadDepositionalEnvironmentModelFromJsonObj,
)
from pywellsfm.io.depositional_environment_simulation_io import (
    loadSimulatorParametersFromJsonObj,
    loadSimulatorWeights,
    simulatorParametersToJsonObj,
)
from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.io.well_io import loadWell, loadWellFromJsonObj, wellToJsonObj
from pywellsfm.model import Curve
from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironmentModel,
)
from pywellsfm.model.enums import SubsidenceType
from pywellsfm.model.Facies import FaciesModel
from pywellsfm.model.FSSimulationParameters import (
    RealizationData,
    Scenario,
)
from pywellsfm.simulator.DepositionalEnvironmentSimulator import (
    DESimulatorParameters,
)
from pywellsfm.simulator.FSSimulator import FSSimulator, FSSimulatorParameters


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
        allowed_keys={
            "format",
            "version",
            "well",
            "subsidenceCurve",
            "initialEnvironment",
            "initialBathymetry",
        },
        ctx="RealizationData",
    )

    # --- Well ---
    well_obj: Any = obj.get("well")

    def _load_inline_well(well_json: dict[str, Any]) -> Any:  # noqa: ANN401
        return loadWellFromJsonObj(well_json, base_dir=base_dir)

    def _load_well_file(path: Path) -> Any:  # noqa: ANN401
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

    # --- Initial bathymetry ---
    initial_bathymetry_obj: Any = obj.get("initialBathymetry")
    if not isinstance(initial_bathymetry_obj, (int, float)):
        raise ValueError("RealizationData.initialBathymetry must be a number.")
    initial_bathymetry = float(initial_bathymetry_obj)

    # --- Initial environment ---
    initial_environment_obj: Any = obj.get("initialEnvironment")
    if initial_environment_obj is not None and not isinstance(
        initial_environment_obj, str
    ):
        raise ValueError(
            "RealizationData.initialEnvironment must be a string."
        )
    initial_environment = initial_environment_obj

    # --- Subsidence curve (optional) ---
    subsidence_curve: Curve | None
    subs_type: SubsidenceType = SubsidenceType.CUMULATIVE
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

        curve_obj = subs_curve_obj.get("curve", None)
        subs_type = subs_curve_obj.get("type", None)
        if curve_obj is None or subs_type is None:
            raise ValueError(
                "RealizationData.subsidenceCurve must contain 'curve' and "
                "'type' properties."
            )
        if subs_type not in {"cumulative", "rate"}:
            raise ValueError(
                "RealizationData.subsidenceCurve.type must be either "
                "'cumulative' or 'rate'."
            )
        print(curve_obj)
        subsidence_curve = load_inline_or_url(
            curve_obj,
            base_dir=base_dir,
            ctx="RealizationData.subsidenceCurve",
            load_inline=_load_inline_subs_curve,
            load_file=_load_subs_curve_file,
        )

    return RealizationData(
        well=well,
        initialBathymetry=initial_bathymetry,
        initialEnvironmentName=initial_environment,
        subsidenceCurve=subsidence_curve,
        subsidenceType=SubsidenceType(subs_type),
    )


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
        "initialBathymetry": realizationData.initialBathymetry,
    }

    if realizationData.subsidenceCurve is not None:
        subs_payload: dict[str, Any] = {
            "type": realizationData.subsidenceType.value,
            "curve": curveToJsonObj(realizationData.subsidenceCurve),
        }
        payload["subsidenceCurve"] = subs_payload
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
            "accumulationModel",
            "depositionalEnvironmentModel",
            "faciesModel",
            "eustaticCurve",
        },
        ctx="Scenario",
    )

    name = obj.get("name")
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Scenario.name must be a non-empty string.")

    # --- Accumulation model ---
    accumulation_model_obj: Any = obj.get("accumulationModel")

    def _load_inline_am(am_json: dict[str, Any]) -> Any:  # noqa: ANN401
        return loadAccumulationModelFromJsonObj(
            am_json, base_dir=str(base_dir)
        )

    def _load_am_file(path: Path) -> Any:  # noqa: ANN401
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

    # --- Depositional Environment Model (optional) ---
    depositional_environment_model: DepositionalEnvironmentModel | None
    deModel_obj = obj.get("depositionalEnvironmentModel")
    if deModel_obj is None:
        depositional_environment_model = None
    else:
        if not isinstance(deModel_obj, dict):
            raise ValueError(
                "Scenario.depositionalEnvironmentModel must be " + "an object."
            )

        # url or inline definition
        def _load_inline_deModel(
            deModel_json: dict[str, Any],
        ) -> DepositionalEnvironmentModel:
            return loadDepositionalEnvironmentModelFromJsonObj(
                deModel_json, base_dir=str(base_dir)
            )

        def _load_deModel_file(path: Path) -> DepositionalEnvironmentModel:
            try:
                return loadDepositionalEnvironmentModel(str(path))
            except (ValueError, FileNotFoundError, OSError) as exc:
                raise ValueError(
                    "Scenario.depositionalEnvironmentModel.url must point "
                    + "to a supported depositional environment model file "
                    + "(.json or .csv)."
                ) from exc

        depositional_environment_model = load_inline_or_url(
            deModel_obj,
            base_dir=base_dir,
            ctx="Scenario.depositionalEnvironmentModel",
            load_inline=_load_inline_deModel,
            load_file=_load_deModel_file,
        )

    # --- Facies model (optional) ---
    facies_model: FaciesModel | None = None

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
        depositionalEnvironmentModel=depositional_environment_model,
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
        "accumulationModel": accumulationModelToJsonObj(
            scenario.accumulationModel
        ),
    }
    # depositional environment model
    if scenario.depositionalEnvironmentModel is not None:
        payload["depositionalEnvironmentModel"] = (
            depositionalEnvironmentModelToJsonObj(
                scenario.depositionalEnvironmentModel
            )
        )
    else:
        payload["depositionalEnvironmentModel"] = None
    # facies model
    if scenario.faciesModel is not None:
        pass
        # payload["faciesModel"] = faciesModelToJsonObj(scenario.faciesModel)
    else:
        payload["faciesModel"] = None
    # eustatic curve
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
    fsSimulator: FSSimulator,
    *,
    name: str,
) -> dict[str, Any]:
    """Export FSSimulator object to json object.

    json format conforms to json/FSSimulationDataSchema.json.

    :param FSSimulationData FSSimulationData: FSSimulationData object
    :param str name: name of the simulation data
    :raises ValueError: if name is not a non-empty string
    :raises ValueError: if realizationsData is not a non-empty list
    :return dict[str, Any]: JSON-serializable dictionary representing the
        simulation data
    """
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError("Simulation.name must be a non-empty string.")
    if (
        not isinstance(fsSimulator.realizationDataList, list)
        or len(fsSimulator.realizationDataList) < 1
    ):
        raise ValueError(
            "Simulation.realizations must contain at least one item."
        )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.FSSimulationData",
        "version": "1.0",
        "name": str(name),
        "scenario": exportScenarioToJsonObj(fsSimulator.scenario),
        "realizations": [
            exportRealizationDataToJsonObj(r)
            for r in fsSimulator.realizationDataList
        ],
        "params": {
            "max_waterDepth_change_per_step": (
                fsSimulator.params.max_waterDepth_change_per_step
            ),
            "dt_min": fsSimulator.params.dt_min,
            "dt_max": fsSimulator.params.dt_max,
            "safety": fsSimulator.params.safety,
            "max_steps": fsSimulator.params.max_steps,
        },
    }
    if fsSimulator.use_deSimulator:
        payload["use_depositional_environment_simulator"] = True
        if fsSimulator.deSimulator_weights is not None:
            payload["deSimulator_weights"] = fsSimulator.deSimulator_weights
        if fsSimulator.deSimulator_params is not None:
            payload["deSimulator_params"] = simulatorParametersToJsonObj(
                fsSimulator.deSimulator_params
            )
    return payload


def saveFSSimulation(
    fsSimulator: FSSimulator,
    filepath: str,
    *,
    name: str,
) -> None:
    """Save FSSimulator object to json file."""
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "FSSimulator output file must have a .json extension."
        )
    out = exportSimulationDataToJsonObj(fsSimulator, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def loadFSSimulation(filepath: str) -> FSSimulator:
    """Load scenario from json file.

    json file schema is defined by jsonSchemas/ScenarioSchema.json

    :param str filepath: path to scenario json file
    :return FSSimulatorData: data object containing scenario and
        realizations data for running
    """
    path = Path(filepath)
    data = json.loads(path.read_text(encoding="utf-8"))

    expect_format_version(
        data,
        expected_format="pyWellSFM.FSSimulationData",
        expected_version="1.0",
        kind="scenario",
    )

    reject_extra_keys(
        obj=data,
        allowed_keys={
            "format",
            "version",
            "name",
            "scenario",
            "realizations",
            "use_depositional_environment_simulator",
            "deSimulator_weights",
            "deSimulator_params",
            "params",
        },
        ctx="FSSimulationData",
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

    # --- Depostional Environment Simulator ---
    use_deSimulator = data.get("use_depositional_environment_simulator", False)
    if not isinstance(use_deSimulator, bool):
        raise ValueError(
            "Simulation.use_depositional_environment_simulator "
            + "must be a boolean."
        )

    deSimulator_weights: dict[str, float] | None = None
    deSimulator_params: DESimulatorParameters | None = None
    if use_deSimulator:
        # get weights
        desimulatorWeights_obj = data.get("deSimulator_weights", None)
        if desimulatorWeights_obj is not None:
            if not isinstance(desimulatorWeights_obj, dict):
                raise ValueError(
                    "Simulation.deSimulator_weights must be an "
                    + "object if provided."
                )
            deSimulator_weights = loadSimulatorWeights(desimulatorWeights_obj)
        # get parameters
        desimulatorParams_obj = data.get("deSimulator_params", None)
        if desimulatorParams_obj is not None:
            if not isinstance(desimulatorParams_obj, dict):
                raise ValueError(
                    "Simulation.deSimulator_params must be an "
                    + "object if provided."
                )
            deSimulator_params = loadSimulatorParametersFromJsonObj(
                desimulatorParams_obj
            )

    # --- Simulation Parameters ---
    params_obs = data.get("params", None)
    defaultSimulationParams = FSSimulatorParameters()
    simulationParams: FSSimulatorParameters
    if params_obs is None:
        simulationParams = defaultSimulationParams
    else:
        if not isinstance(params_obs, dict):
            raise ValueError("Simulation.params must be a dictionary.")
        max_waterDepth_change = params_obs.get(
            "max_waterDepth_change_per_step",
            defaultSimulationParams.max_waterDepth_change_per_step,
        )
        dt_min = params_obs.get("dt_min", defaultSimulationParams.dt_min)
        dt_max = params_obs.get("dt_max", defaultSimulationParams.dt_max)
        safety = params_obs.get("safety", defaultSimulationParams.safety)
        max_steps = params_obs.get(
            "max_steps", defaultSimulationParams.max_steps
        )
        simulationParams = FSSimulatorParameters(
            max_waterDepth_change_per_step=max_waterDepth_change,
            dt_min=dt_min,
            dt_max=dt_max,
            safety=safety,
            max_steps=max_steps,
        )

    return FSSimulator(
        scenario,
        realizations_data,
        use_depositional_environment_simulator=use_deSimulator,
        deSimulator_weights=deSimulator_weights,
        deSimulator_params=deSimulator_params,
        fsSimulator_params=simulationParams,
    )
