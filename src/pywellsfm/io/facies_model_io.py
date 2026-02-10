# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""I/O utilities for FaciesModel."""

from __future__ import annotations

import json
import math
from typing import Any

from pywellsfm.io.json_schema_validation import expect_format_version
from pywellsfm.model.Facies import (
    EnvironmentalFacies,
    Facies,
    FaciesCriteria,
    FaciesCriteriaType,
    FaciesModel,
    PetrophysicalFacies,
    SedimentaryFacies,
)


def loadFaciesModelFromJsonObj(obj: dict[str, Any]) -> FaciesModel:
    """Parse a FaciesModel JSON object into a FaciesModel.

    The JSON must match the on-disk format used by this project:

    - format: "pyWellSFM.FaciesModelData"
    - version: "1.0"

    :param dict[str, Any] obj: Parsed facies model JSON object.
    :return FaciesModel: Loaded facies model
    """
    expect_format_version(
        obj,
        expected_format="pyWellSFM.FaciesModelData",
        expected_version="1.0",
        kind="facies model",
    )

    facies_items = obj.get("faciesModel")
    if not isinstance(facies_items, list):
        raise ValueError("'faciesModel' must be a list.")

    facies_set: set[Facies] = set()
    seen_names: set[str] = set()

    for idx, facies_def in enumerate(facies_items):
        if not isinstance(facies_def, dict):
            raise ValueError(f"faciesModel[{idx}] must be an object.")

        facies_name = facies_def.get("name")
        if not isinstance(facies_name, str) or facies_name.strip() == "":
            raise ValueError(
                f"faciesModel[{idx}].name must be a non-empty string."
            )
        if facies_name in seen_names:
            raise ValueError(
                f"Duplicate facies name '{facies_name}' in faciesModel."
            )
        seen_names.add(facies_name)

        facies_type_raw = facies_def.get(
            "criteriaType", FaciesCriteriaType.UNCATEGORIZED.value
        )
        if not isinstance(facies_type_raw, str):
            raise ValueError(
                f"faciesModel[{idx}].criteriaType must be a string "
                "when provided."
            )
        try:
            facies_type = FaciesCriteriaType(facies_type_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid faciesModel[{idx}].criteriaType '{facies_type_raw}'."
            ) from exc

        criteria_list = facies_def.get("criteria")
        if not isinstance(criteria_list, list) or len(criteria_list) < 1:
            raise ValueError(
                f"faciesModel[{idx}].criteria must be a non-empty list."
            )

        criteria_set: set[FaciesCriteria] = set()

        for jdx, crit_def in enumerate(criteria_list):
            if not isinstance(crit_def, dict):
                raise ValueError(
                    f"faciesModel[{idx}].criteria[{jdx}] must be an object."
                )

            crit_name = crit_def.get("name")
            if not isinstance(crit_name, str) or crit_name.strip() == "":
                raise ValueError(
                    f"faciesModel[{idx}].criteria[{jdx}].name must be "
                    "a non-empty string."
                )

            # If criterion type is missing, default to the facies criteriaType.
            crit_type_raw = crit_def.get("type")
            if crit_type_raw is None:
                crit_type_raw = facies_type.value
            if not isinstance(crit_type_raw, str):
                raise ValueError(
                    f"faciesModel[{idx}].criteria[{jdx}].type must be a "
                    "string when provided."
                )
            try:
                crit_type = FaciesCriteriaType(crit_type_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid faciesModel[{idx}].criteria[{jdx}].type "
                    f"'{crit_type_raw}'."
                ) from exc

            min_range = crit_def.get("minRange", -float("inf"))
            max_range = crit_def.get("maxRange", float("inf"))

            if (min_range is not None) and (
                not isinstance(min_range, (int, float))
            ):
                raise ValueError(
                    f"faciesModel[{idx}].criteria[{jdx}].minRange must be "
                    "a number when provided."
                )
            if (max_range is not None) and (
                not isinstance(max_range, (int, float))
            ):
                raise ValueError(
                    f"faciesModel[{idx}].criteria[{jdx}].maxRange must be "
                    "a number when provided."
                )

            criteria_set.add(
                FaciesCriteria(
                    name=crit_name,
                    minRange=float(min_range),  # type: ignore[arg-type]
                    maxRange=float(max_range),  # type: ignore[arg-type]
                    type=crit_type,
                )
            )

        if facies_type == FaciesCriteriaType.SEDIMENTOLOGICAL:
            facies_obj: Facies = SedimentaryFacies(
                name=facies_name, criteria=criteria_set
            )
        elif facies_type == FaciesCriteriaType.PETROPHYSICAL:
            facies_obj = PetrophysicalFacies(
                name=facies_name, criteria=criteria_set
            )
        elif facies_type == FaciesCriteriaType.ENVIRONMENTAL:
            facies_obj = EnvironmentalFacies(
                name=facies_name, criteria=criteria_set
            )
        else:
            facies_obj = Facies(
                name=facies_name,
                criteria=criteria_set,
                criteriaType=facies_type,
            )

        facies_set.add(facies_obj)

    return FaciesModel(faciesSet=facies_set)


def loadFaciesModel(filepath: str) -> FaciesModel:
    """Load a facies model from a JSON file.

    The JSON must match the on-disk format used by this project:

    - format: "pyWellSFM.FaciesModelData"
    - version: "1.0"

    :param str filepath: Path to the facies model JSON file.
    :return FaciesModel: Loaded facies model
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return loadFaciesModelFromJsonObj(data)


def faciesModelToJsonObj(faciesModel: FaciesModel) -> dict[str, Any]:
    """Serialize a FaciesModel to a JSON object.

    The returned object conforms to the on-disk format used by this project:

    - format: "pyWellSFM.FaciesModelData"
    - version: "1.0"

    .. Note::

        JSON doesn't support +/-inf; infinite ranges are omitted.
    """
    payload: dict[str, Any] = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [],
    }

    facies_list = sorted(faciesModel.faciesSet, key=lambda f: f.name)
    for facies in facies_list:
        facies_type = facies.criteriaCollection.type
        facies_obj: dict[str, Any] = {
            "name": facies.name,
            "criteria": [],
        }
        # Keep file concise: only write facies-level criteriaType when
        # meaningful.
        if facies_type != FaciesCriteriaType.UNCATEGORIZED:
            facies_obj["criteriaType"] = facies_type.value

        criteria_list = sorted(
            facies.criteriaCollection.getAllCriteria(), key=lambda c: c.name
        )
        for crit in criteria_list:
            crit_obj: dict[str, Any] = {"name": crit.name}

            # Only write criterion type when it differs from the facies default
            if crit.type != facies_type:
                crit_obj["type"] = crit.type.value

            if crit.minRange is not None and not math.isinf(crit.minRange):
                crit_obj["minRange"] = float(crit.minRange)
            if crit.maxRange is not None and not math.isinf(crit.maxRange):
                crit_obj["maxRange"] = float(crit.maxRange)

            facies_obj["criteria"].append(crit_obj)

        if len(facies_obj["criteria"]) == 0:
            raise ValueError(
                f"At least one criteria must be defined for the "
                f"facies '{facies.name}'"
            )

        payload["faciesModel"].append(facies_obj)

    return payload


def saveFaciesModel(faciesModel: FaciesModel, filepath: str) -> None:
    """Save a facies model to a JSON file."""
    payload = faciesModelToJsonObj(faciesModel)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
