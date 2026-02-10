# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Unit tests for Facies and related IO methods."""

import json
import math
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

import pytest

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

from pywellsfm.io import loadFaciesModel, saveFaciesModel  # noqa: E402
from pywellsfm.model import (  # noqa: E402
    EnvironmentalFacies,
    Facies,
    FaciesCriteria,
    FaciesCriteriaCollection,
    FaciesCriteriaType,
    FaciesModel,
    PetrophysicalFacies,
    SedimentaryFacies,
)

fileDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(fileDir, "data")

out_path = Path(fileDir) / ".out"
if os.path.exists(out_path) is False:
    os.mkdir(out_path)


def _write_json(
    tmp_path: Path,
    payload: dict[str, Any],
    filename: str = "facies_model.json",
) -> str:
    """Helper: write JSON to a temp file and return its filesystem path."""
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _delete_temp_file(file_path: str) -> None:
    """Helper function to delete a temporary file."""
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Error deleting temporary file {file_path}: {e}")


# ------------------------------
# FaciesCriteriaType (StrEnum)
# ------------------------------


def test_FaciesCriteriaType_values_are_stable() -> None:
    """Test FaciesCriteriaType enum values are stable.

    Objective:
    - Ensure the enum values used by JSON IO are stable.
    Input data:
    - The enum members.
    Expected outputs:
    - Each member has the exact expected string value.
    """
    assert FaciesCriteriaType.SEDIMENTOLOGICAL.value == "sedimentological"
    assert FaciesCriteriaType.PETROPHYSICAL.value == "petrophysical"
    assert FaciesCriteriaType.ENVIRONMENTAL.value == "environmental"
    assert FaciesCriteriaType.UNCATEGORIZED.value == "uncategorized"


# ------------------------------
# FaciesCriteria
# ------------------------------


def test_FaciesCriteria_init_defaults_and_fields() -> None:
    """Test FaciesCriteria initialization with defaults and specified fields.

    Objective:
    - Check that FaciesCriteria is correctly instantiated with default range
      and type.

    Input data:
    - name="Porosity" only.

    Expected outputs:
    - minRange=-inf, maxRange=+inf, type=UNCATEGORIZED, name is set.
    """
    crit = FaciesCriteria(name="Porosity")
    assert crit.name == "Porosity"
    assert crit.type == FaciesCriteriaType.UNCATEGORIZED
    assert math.isinf(crit.minRange) and crit.minRange < 0
    assert math.isinf(crit.maxRange) and crit.maxRange > 0


def test_FaciesCriteria_repr() -> None:
    """Test FaciesCriteria __repr__ method.

    Objective:
    - Validate __repr__ formatting (useful for debugging/logging).

    Input data:
    - name="Gamma", minRange=10, maxRange=20

    Expected outputs:
    - "Gamma [10.0, 20.0]" (exact string).
    """
    crit = FaciesCriteria(name="Gamma", minRange=10.0, maxRange=20.0)
    assert repr(crit) == "Gamma [10.0, 20.0]"


def test_FaciesCriteria_hash_is_based_on_name_only() -> None:
    """Test FaciesCriteria __hash__ method.

    Objective:
    - Document current hashing behavior (hash uses only name).

    Input data:
    - Two criteria with same name but different type.

    Expected outputs:
    - Hashes are equal.
    """
    c1 = FaciesCriteria(name="Porosity", type=FaciesCriteriaType.PETROPHYSICAL)
    c2 = FaciesCriteria(name="Porosity", type=FaciesCriteriaType.ENVIRONMENTAL)
    assert hash(c1) == hash(c2)


def test_FaciesCriteria_equality_with_criteria_and_with_str() -> None:
    """Test FaciesCriteria __eq__ method for criteria and string comparison.

    Objective:
    - Verify __eq__ supports comparing to another FaciesCriteria and to a
      string.

    Input data:
    - Two criteria with same name/type; a third with same name/different type.

    Expected outputs:
    - Same name & same type => equal.
    - Same name & different type => not equal.
    - Comparing to string matches by name.
    """
    c1 = FaciesCriteria(name="Porosity", type=FaciesCriteriaType.PETROPHYSICAL)
    c2 = FaciesCriteria(name="Porosity", type=FaciesCriteriaType.PETROPHYSICAL)
    c3 = FaciesCriteria(name="Porosity", type=FaciesCriteriaType.ENVIRONMENTAL)

    assert c1 == c2
    assert c1 != c3
    assert c1 == "Porosity"
    assert c1 != "Other"


def test_FaciesCriteria_isNamed_is_case_insensitive() -> None:
    """Test FaciesCriteria isNamed method for case-insensitive name matching.

    Objective:
    - Ensure isNamed compares case-insensitively.

    Input data:
    - Criteria name "Gamma".

    Expected outputs:
    - "gamma" matches, "GAMMA" matches, "Other" does not.
    """
    crit = FaciesCriteria(name="Gamma")
    assert crit.isNamed("gamma")
    assert crit.isNamed("GAMMA")
    assert not crit.isNamed("Other")


def test_FaciesCriteria_hasType() -> None:
    """Test FaciesCriteria hasType method.

    Objective:
    - Verify hasType returns True only for the stored type.

    Input data:
    - Criteria type PETROPHYSICAL.

    Expected outputs:
    - hasType(PETROPHYSICAL)=True, hasType(ENVIRONMENTAL)=False.
    """
    crit = FaciesCriteria(
        name="Porosity", type=FaciesCriteriaType.PETROPHYSICAL
    )
    assert crit.hasType(FaciesCriteriaType.PETROPHYSICAL)
    assert not crit.hasType(FaciesCriteriaType.ENVIRONMENTAL)


# ------------------------------
# FaciesCriteriaCollection
# ------------------------------


def test_FaciesCriteriaCollection_init_defaults() -> None:
    """Test FaciesCriteriaCollection initialization with defaults.

    Objective:
    - Check default construction: empty set and UNCATEGORIZED type.

    Input data:
    - No inputs.

    Expected outputs:
    - isEmpty() is True; getCriteriaCount()==0; type==UNCATEGORIZED.
    """
    col = FaciesCriteriaCollection()
    assert col.type == FaciesCriteriaType.UNCATEGORIZED
    assert col.isEmpty()
    assert col.getCriteriaCount() == 0


def test_FaciesCriteriaCollection_addCriteria_single_and_duplicate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test addCriteria with single and duplicate criteria.

    Objective:
    - Add a criterion; adding a duplicate by name should be rejected.

    Input data:
    - Two criteria objects with the same name.

    Expected outputs:
    - Collection count remains 1 after attempting duplicate add.
    """
    col = FaciesCriteriaCollection()
    col.addCriteria(FaciesCriteria(name="Porosity"))
    col.addCriteria(FaciesCriteria(name="Porosity"))
    out = capsys.readouterr().out
    assert col.getCriteriaCount() == 1
    assert "already exists" in out


def test_FaciesCriteriaCollection_addCriteria_set() -> None:
    """Test FaciesCriteriaCollection addCriteria with a set of criteria.

    Objective:
    - Ensure addCriteria accepts a set and adds each criterion.

    Input data:
    - A set of two criteria.

    Expected outputs:
    - Collection count == 2.
    """
    col = FaciesCriteriaCollection()
    col.addCriteria({FaciesCriteria(name="A"), FaciesCriteria(name="B")})
    assert col.getCriteriaCount() == 2


def test_FaciesCriteriaCollection_addCriteria_wrong_type_raises() -> None:
    """Test addCriteria rejects unsupported input types.

    Objective:
    - Ensure type checking: addCriteria rejects unsupported input types.

    Input data:
    - criteria=123

    Expected outputs:
    - TypeError.
    """
    col = FaciesCriteriaCollection()
    with pytest.raises(TypeError):
        col.addCriteria(123)  # type: ignore[arg-type]


def test_FaciesCriteriaCollection_criteriaIsAllowed_rules() -> None:
    """Test criteriaIsAllowed method.

    Objective:
    - Validate criteriaIsAllowed behavior.

    Input data:
    - A PETROPHYSICAL collection; one PETROPHYSICAL criterion;
      one ENVIRONMENTAL criterion.

    Expected outputs:
    - Allowed for PETROPHYSICAL; not allowed for ENVIRONMENTAL.
    """
    col = FaciesCriteriaCollection(
        criteriaType=FaciesCriteriaType.PETROPHYSICAL
    )
    pet = FaciesCriteria(
        name="Porosity", type=FaciesCriteriaType.PETROPHYSICAL
    )
    env = FaciesCriteria(
        name="WaterDepth", type=FaciesCriteriaType.ENVIRONMENTAL
    )
    assert col.criteriaIsAllowed(pet)
    assert not col.criteriaIsAllowed(env)


def test_FaciesCriteriaCollection_addCriteria_disallowed_type_is_ignored(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test addCriteria ignores disallowed criterion types.

    Objective:
    - Document behavior when adding a criterion of disallowed type: it is not
      added.

    Input data:
    - A PETROPHYSICAL collection; an ENVIRONMENTAL criterion.

    Expected outputs:
    - Collection remains empty; a message is printed.
    """
    col = FaciesCriteriaCollection(
        criteriaType=FaciesCriteriaType.PETROPHYSICAL
    )
    col.addCriteria(
        FaciesCriteria(
            name="WaterDepth", type=FaciesCriteriaType.ENVIRONMENTAL
        )
    )
    out = capsys.readouterr().out
    assert col.getCriteriaCount() == 0
    assert "is not allowed" in out


def test_FaciesCriteriaCollection_criteriaExists_case_insensitive() -> None:
    """Test criteriaExists method with case-insensitive match.

    Objective:
    - Ensure criteriaExists uses case-insensitive matching via isNamed().

    Input data:
    - Add criterion named "Gamma".

    Expected outputs:
    - criteriaExists("gamma") == True.
    """
    col = FaciesCriteriaCollection()
    col.addCriteria(FaciesCriteria(name="Gamma"))
    assert col.criteriaExists("gamma")


def test_FaciesCriteriaCollection_removeCriteria_str_and_set() -> None:
    """Test removeCriteria method with string and set inputs.

    Objective:
    - Remove criteria by a single name and by a set of names.

    Input data:
    - Criteria A, B, C; remove "B" then remove {"A","C"}.

    Expected outputs:
    - Counts go from 3 -> 2 -> 0.
    """
    col = FaciesCriteriaCollection()
    col.addCriteria(
        {FaciesCriteria("A"), FaciesCriteria("B"), FaciesCriteria("C")}
    )
    assert col.getCriteriaCount() == 3
    col.removeCriteria("B")
    assert col.getCriteriaCount() == 2
    col.removeCriteria({"A", "C"})
    assert col.getCriteriaCount() == 0


def test_FaciesCriteriaCollection_removeCriteria_wrong_type_raises() -> None:
    """Test removeCriteria rejects unsupported input types.

    Objective:
    - Ensure removeCriteria rejects unsupported input types.

    Input data:
    - criteriaNames=123

    Expected outputs:
    - TypeError.
    """
    col = FaciesCriteriaCollection()
    with pytest.raises(TypeError):
        col.removeCriteria(123)  # type: ignore[arg-type]


def test_FaciesCriteriaCollection_getters_and_clear_methods() -> None:
    """Test getters and clear methods.

    Objective:
    - Exercise: getAllCriteria, getCriteriaByName, getCriteriaSetByType,
      clearCriteriaByType, clearAllCriteria.

    Input data:
    - Add one PETROPHYSICAL and one ENVIRONMENTAL criterion.

    Expected outputs:
    - getCriteriaByName finds by case-insensitive name.
    - getCriteriaSetByType returns correct subset.
    - clearCriteriaByType removes only the requested type.
    - clearAllCriteria empties the collection.
    """
    col = FaciesCriteriaCollection()
    pet = FaciesCriteria("Porosity", type=FaciesCriteriaType.PETROPHYSICAL)
    env = FaciesCriteria("WaterDepth", type=FaciesCriteriaType.ENVIRONMENTAL)
    col.addCriteria({pet, env})

    assert col.getAllCriteria()  # non-empty
    assert col.getCriteriaByName("porosity") == pet

    pet_set = col.getCriteriaSetByType(FaciesCriteriaType.PETROPHYSICAL)
    assert pet in pet_set and env not in pet_set

    removed = col.clearCriteriaByType(FaciesCriteriaType.PETROPHYSICAL)
    assert removed == 1
    assert col.getCriteriaCount() == 1

    col.clearAllCriteria()
    assert col.isEmpty()


# ------------------------------
# Facies + subclasses
# ------------------------------


def test_Facies_init_requires_at_least_one_criteria_when_collection_provided(
) -> None :
    """Test Facies __init__ rejects empty collections for criteria parameter.

    Objective:
    - Ensure Facies rejects empty collections (set/list/tuple).

    Input data:
    - criteria=set()

    Expected outputs:
    - ValueError.
    """
    with pytest.raises(ValueError):
        Facies(name="Test", criteria=set())


def test_Facies_init_accepts_single_FaciesCriteria() -> None:
    """Test Facies __init__ accepts a single FaciesCriteria object.

    Objective:
    - Ensure Facies accepts a single FaciesCriteria object.

    Input data:
    - criteria=FaciesCriteria("Gamma")

    Expected outputs:
    - Facies is created and contains 1 criterion.
    """
    facies = Facies(name="F", criteria=FaciesCriteria("Gamma"))
    assert facies.name == "F"
    assert facies.criteriaCollection.getCriteriaCount() == 1


@pytest.mark.parametrize(
    "cls, expected_type",
    [
        (SedimentaryFacies, FaciesCriteriaType.SEDIMENTOLOGICAL),
        (PetrophysicalFacies, FaciesCriteriaType.PETROPHYSICAL),
        (EnvironmentalFacies, FaciesCriteriaType.ENVIRONMENTAL),
    ],
)
def test_Facies_subclass_sets_collection_type(
    cls: type[Facies], expected_type: FaciesCriteriaType
) -> None:
    """Test Facies subclasses set the expected criteriaType for collection.

    Objective:
    - Verify each Facies subclass sets the expected criteriaType for its
      collection.

    Input data:
    - Instantiate each subclass with one criterion of matching type.

    Expected outputs:
    - facies.criteria.type == expected_type.
    """
    crit = FaciesCriteria("X", type=expected_type)
    facies = cls(name="F", criteria={crit})
    assert facies.criteriaCollection.type == expected_type


def test_Facies_getCriteria() -> None:
    """Test Facies getCriteria method.

    Objective:
    - Ensure getCriteria returns the correct criterion by name.

    Input data:
    - Facies with criteria "A" and "B".

    Expected outputs:
    - getCriteria("A") returns that object.
    - getCriteria("Missing") returns None.
    """
    c1 = FaciesCriteria("A")
    c2 = FaciesCriteria("B")
    facies = Facies(name="F", criteria={c1, c2})
    assert facies.getCriteria("A") is c1
    assert facies.getCriteria("Missing") is None


def test_Facies_addCriteria() -> None:
    """Test Facies addCriteria method.

    Objective:
    - Ensure addCriteria adds a new criterion to the facies.

    Input data:
    - Facies with initial criterion "A"; add criterion "B".

    Expected outputs:
    - After addition, criteria count is 2 and "B" is present.
    """
    c1 = FaciesCriteria("A")
    facies = Facies(name="F", criteria={c1})
    c2 = FaciesCriteria("B")
    facies.addCriteria(c2)
    assert facies.criteriaCollection.getCriteriaCount() == 2
    assert facies.criteriaCollection.criteriaExists("B")


# ------------------------------
# FaciesModel
# ------------------------------


def test_FaciesModel_getFaciesByName_found_and_not_found() -> None:
    """Test FaciesModel getFaciesByName method for found and not found cases.

    Objective:
    - Ensure getFaciesByName returns a Facies when present, else None.

    Input data:
    - Model with facies named "A" and "B".

    Expected outputs:
    - getFaciesByName("A") returns that object.
    - getFaciesByName("Missing") returns None.
    """
    f1 = Facies("A", FaciesCriteria("c1"))
    f2 = Facies("B", FaciesCriteria("c2"))
    model = FaciesModel(faciesSet={f1, f2})
    assert model.getFaciesByName("A") is f1
    assert model.getFaciesByName("Missing") is None


def test_FaciesModel_getCriteriaRangeForFacies_happy_path() -> None:
    """Test FM getCriteriaRangeForFacies method in happy path scenario.

    Objective:
    - Verify getCriteriaRangeForFacies returns (min,max) for existing
      facies/criteria.

    Input data:
    - One facies "A" with criterion "Gamma" range [10, 20].

    Expected outputs:
    - (10.0, 20.0)
    """
    crit = FaciesCriteria("Gamma", minRange=10.0, maxRange=20.0)
    f1 = Facies("A", crit)
    model = FaciesModel(faciesSet={f1})
    assert model.getCriteriaRangeForFacies("A", "Gamma") == (10.0, 20.0)


def test_FM_getCriteriaRangeForFacies_missing_facies_prints_and_returns_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test FaciesModel getCriteriaRangeForFacies when facies is missing.

    Objective:
    - When facies name is missing, method returns None and prints a message.

    Input data:
    - Model without requested facies.

    Expected outputs:
    - Returns None; stdout contains a "not found" message.
    """
    model = FaciesModel(faciesSet=set())
    assert model.getCriteriaRangeForFacies("Missing", "Gamma") is None
    out = capsys.readouterr().out
    assert "not found" in out


def test_FM_getCriteriaRangeForFacies_missing_criteria_prints_and_returns_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test FaciesModel getCriteriaRangeForFacies when criterion is missing.

    Objective:
    - When criteria name is missing for a valid facies, returns None and prints

    Input data:
    - Model with facies "A" but no "Gamma" criterion.

    Expected outputs:
    - Returns None; stdout contains a "Criteria with name" message.
    """
    f1 = Facies("A", FaciesCriteria("Other"))
    model = FaciesModel(faciesSet={f1})
    assert model.getCriteriaRangeForFacies("A", "Gamma") is None
    out = capsys.readouterr().out
    assert "Criteria with name" in out


# ------------------------------
# ioHelpers.loadFaciesModel
# ------------------------------


def test_loadFaciesModel_happy_path_creates_expected_facies_types(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel happy path with various facies types.

    Objective:
    - Ensure loadFaciesModel parses valid JSON and instantiates the correct
      Facies subclass.

    Input data:
    - JSON with 4 facies definitions: sedimentological, petrophysical,
      environmental, uncategorized.

    Expected outputs:
    - Returned object is a FaciesModel with 4 facies.
    - Each facies is instantiated as the correct class.
    """
    model = loadFaciesModel(str(Path(dataDir) / "facies_model.json"))

    assert isinstance(model, FaciesModel)
    assert len(model.faciesSet) == 7
    assert isinstance(model.getFaciesByName("Sand"), SedimentaryFacies)
    assert isinstance(model.getFaciesByName("Reservoir"), PetrophysicalFacies)
    assert isinstance(model.getFaciesByName("Shelf"), EnvironmentalFacies)
    assert isinstance(model.getFaciesByName("Other"), Facies)


def test_loadFaciesModel_defaults_criteria_type_and_ranges(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel defaults for missing criteriaType and ranges.

    Objective:
    - Verify defaulting behavior:
      - missing facies.criteriaType defaults to UNCATEGORIZED
      - missing criterion.type defaults to the facies criteriaType
      - missing minRange/maxRange default to (-inf, +inf)

    Input data:
    - JSON with one facies, no criteriaType, one criterion with only name.

    Expected outputs:
    - Facies created with collection type UNCATEGORIZED.
    - Criterion type is UNCATEGORIZED.
    - Criterion ranges are infinite.
    """
    payload = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [{"name": "X", "criteria": [{"name": "Gamma"}]}],
    }
    path = _write_json(tmp_path, payload)
    model = loadFaciesModel(path)
    facies = model.getFaciesByName("X")
    assert facies is not None

    assert facies.criteriaCollection.type == FaciesCriteriaType.UNCATEGORIZED
    gamma = facies.criteriaCollection.getCriteriaByName("Gamma")
    assert gamma is not None
    assert gamma.type == FaciesCriteriaType.UNCATEGORIZED
    assert math.isinf(gamma.minRange) and gamma.minRange < 0
    assert math.isinf(gamma.maxRange) and gamma.maxRange > 0

    # delete temp file
    _delete_temp_file(path)


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        ([], "must be an object"),
        (
            {"format": "x", "version": "1.0", "faciesModel": []},
            "Invalid facies model format",
        ),
        (
            {
                "format": "pyWellSFM.FaciesModelData",
                "version": "x",
                "faciesModel": [],
            },
            "Invalid facies model version",
        ),
        (
            {
                "format": "pyWellSFM.FaciesModelData",
                "version": "1.0",
                "faciesModel": {},
            },
            "'faciesModel' must be a list",
        ),
    ],
)
def test_loadFaciesModel_rejects_invalid_top_level_structures(
    tmp_path: pathlib.Path, payload: dict[str, Any], expected_message: str
) -> None:
    """Test loadFaciesModel rejects invalid top-level JSON structures.

    Objective:
    - Ensure loadFaciesModel validates top-level JSON structure.

    Input data:
    - Various malformed top-level payloads.

    Expected outputs:
    - ValueError with a helpful message.
    """
    path = _write_json(tmp_path, payload, filename="bad.json")
    with pytest.raises(ValueError, match=expected_message):
        loadFaciesModel(path)
    _delete_temp_file(path)


def test_loadFaciesModel_rejects_duplicate_facies_names(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel rejects duplicate facies names.

    Objective:
    - Ensure duplicate facies names are rejected.

    Input data:
    - JSON with two facies entries having the same name.

    Expected outputs:
    - ValueError mentioning duplicate name.
    """
    payload = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [
            {"name": "A", "criteria": [{"name": "C"}]},
            {"name": "A", "criteria": [{"name": "D"}]},
        ],
    }
    path = _write_json(tmp_path, payload)
    with pytest.raises(ValueError, match="Duplicate facies name"):
        loadFaciesModel(path)
    _delete_temp_file(path)


def test_loadFaciesModel_rejects_invalid_facies_definition_shapes(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel rejects invalid facies definition shapes.

    Objective:
    - Validate per-facies structure checks.

    Input data:
    - faciesModel contains an item that is not an object, and an object with
      empty name.

    Expected outputs:
    - ValueError.
    """
    payload_non_object = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": ["not-an-object"],
    }
    with pytest.raises(
        ValueError, match="faciesModel\\[0\\] must be an object"
    ):
        loadFaciesModel(
            _write_json(tmp_path, payload_non_object, filename="bad1.json")
        )

    payload_empty_name = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [{"name": "", "criteria": [{"name": "C"}]}],
    }
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        loadFaciesModel(
            _write_json(tmp_path, payload_empty_name, filename="bad2.json")
        )


def test_loadFaciesModel_rejects_invalid_criteriaType_values(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel rejects invalid facies-level criteriaType values.

    Objective:
    - Ensure invalid facies-level criteriaType values are rejected.

    Input data:
    - criteriaType="not-a-type".

    Expected outputs:
    - ValueError mentioning invalid criteriaType.
    """
    payload = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [
            {
                "name": "A",
                "criteriaType": "not-a-type",
                "criteria": [{"name": "C"}],
            }
        ],
    }
    with pytest.raises(
        ValueError, match="Invalid faciesModel\\[0\\]\\.criteriaType"
    ):
        loadFaciesModel(_write_json(tmp_path, payload))


def test_loadFaciesModel_rejects_invalid_criteria_list_and_items(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel rejects invalid criteria list and items.

    Objective:
    - Validate criteria list constraints and per-criterion shape checks.

    Input data:
    - criteria is empty and criterion items that are not objects.

    Expected outputs:
    - ValueError with informative messages.
    """
    payload_empty_list = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [{"name": "A", "criteria": []}],
    }
    with pytest.raises(ValueError, match="criteria must be a non-empty list"):
        path = _write_json(tmp_path, payload_empty_list, filename="bad3.json")
        loadFaciesModel(path)
        _delete_temp_file(path)

    payload_crit_not_object = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [{"name": "A", "criteria": ["nope"]}],
    }
    with pytest.raises(ValueError, match="criteria\\[0\\] must be an object"):
        path = _write_json(
            tmp_path, payload_crit_not_object, filename="bad4.json"
        )
        loadFaciesModel(path)
        _delete_temp_file(path)


def test_loadFaciesModel_rejects_invalid_criterion_fields(
    tmp_path: pathlib.Path,
) -> None:
    """Test loadFaciesModel rejects invalid criterion fields.

    Objective:
    - Ensure invalid criterion fields are rejected:
      - name empty
      - type invalid
      - minRange/maxRange not numeric

    Input data:
    - JSON variants with invalid criterion fields.

    Expected outputs:
    - ValueError.
    """
    base = {
        "format": "pyWellSFM.FaciesModelData",
        "version": "1.0",
        "faciesModel": [
            {
                "name": "A",
                "criteriaType": "petrophysical",
                "criteria": [{"name": "X"}],
            }
        ],
    }

    payload_bad_name = json.loads(json.dumps(base))
    payload_bad_name["faciesModel"][0]["criteria"][0] = {"name": ""}
    with pytest.raises(
        ValueError, match="criteria\\[0\\]\\.name must be a non-empty"
    ):
        path = _write_json(tmp_path, payload_bad_name, filename="bad5.json")
        loadFaciesModel(path)
        _delete_temp_file(path)

    payload_bad_type = json.loads(json.dumps(base))
    payload_bad_type["faciesModel"][0]["criteria"][0]["type"] = "not-a-type"
    with pytest.raises(
        ValueError, match="Invalid faciesModel\\[0\\]\\.criteria\\[0\\]\\.type"
    ):
        path = _write_json(tmp_path, payload_bad_type, filename="bad6.json")
        loadFaciesModel(path)
        _delete_temp_file(path)

    payload_bad_min = json.loads(json.dumps(base))
    payload_bad_min["faciesModel"][0]["criteria"][0]["minRange"] = "NaN"
    with pytest.raises(ValueError, match="minRange must be a number"):
        path = _write_json(tmp_path, payload_bad_min, filename="bad7.json")
        loadFaciesModel(path)
        _delete_temp_file(path)

    payload_bad_max = json.loads(json.dumps(base))
    payload_bad_max["faciesModel"][0]["criteria"][0]["maxRange"] = "NaN"
    with pytest.raises(ValueError, match="maxRange must be a number"):
        path = _write_json(tmp_path, payload_bad_max, filename="bad8.json")
        loadFaciesModel(path)
        _delete_temp_file(path)


# ------------------------------
# ioHelpers.saveFaciesModel
# ------------------------------


def _model_signature(
    model: FaciesModel,
) -> dict[str, dict[str, tuple[str, float, float]]]:
    """Build a deterministic signature for equality checks in tests.

    Signature = {faciesName:
      {criteriaName: (criteriaType, minRange, maxRange)}
    }
    """
    sig: dict[str, dict[str, tuple[str, float, float]]] = {}
    for facies in model.faciesSet:
        crits: dict[str, tuple[str, float, float]] = {}
        for crit in facies.criteriaCollection.getAllCriteria():
            crits[crit.name] = (crit.type.value, crit.minRange, crit.maxRange)
        sig[facies.name] = crits
    return sig


def test_saveFaciesModel_round_trip_preserves_model(
    tmp_path: pathlib.Path,
) -> None:
    """Test saveFaciesModel round-trip with loadFaciesModel.

    Objective:
    - Ensure a model exported to JSON can be loaded back without losing facies
      or criteria information.

    Input data:
    - A valid facies model loaded from tests/data/facies_model.json.

    Expected results:
    - The reloaded model has the same facies/criteria content
      (names, types, ranges).
    - Known facies names map to expected Facies subclasses on reload.
    """
    model_in = loadFaciesModel(str(Path(dataDir) / "facies_model.json"))

    out_file = tmp_path / "facies_model_export.json"
    saveFaciesModel(model_in, str(out_file))
    model_out = loadFaciesModel(str(out_file))

    assert _model_signature(model_out) == _model_signature(model_in)
    assert isinstance(model_out.getFaciesByName("Sand"), SedimentaryFacies)
    assert isinstance(
        model_out.getFaciesByName("Reservoir"), PetrophysicalFacies
    )
    assert isinstance(model_out.getFaciesByName("Shelf"), EnvironmentalFacies)
    assert isinstance(model_out.getFaciesByName("Other"), Facies)


def test_saveFaciesModel_omits_defaults_and_infinite_ranges(
    tmp_path: pathlib.Path,
) -> None:
    """Test saveFaciesModel omits default fields for schema-friendly JSON.

    Objective:
    - Verify the exporter writes schema-friendly JSON and stays concise:
      - criteriaType omitted for UNCATEGORIZED facies
      - criterion.type omitted when it matches the facies default
      - minRange/maxRange omitted when they are +/-inf
        (JSON can't represent inf)

    Input data:
    - A simple UNCATEGORIZED facies with one default criterion
      (infinite ranges).

    Expected results:
    - Exported JSON has no criteriaType/type/minRange/maxRange for that
      criterion.
    - Loading back recreates UNCATEGORIZED type and infinite ranges.
    """
    facies = Facies(name="X", criteria={FaciesCriteria(name="Gamma")})
    model = FaciesModel(faciesSet={facies})

    out_file = tmp_path / "defaults.json"
    saveFaciesModel(model, str(out_file))

    exported = json.loads(out_file.read_text(encoding="utf-8"))
    assert exported["format"] == "pyWellSFM.FaciesModelData"
    assert exported["version"] == "1.0"
    assert len(exported["faciesModel"]) == 1

    facies_json = exported["faciesModel"][0]
    assert facies_json["name"] == "X"
    assert "criteriaType" not in facies_json
    assert len(facies_json["criteria"]) == 1

    crit_json = facies_json["criteria"][0]
    assert crit_json == {"name": "Gamma"}

    model_out = loadFaciesModel(str(out_file))
    faciesOut = model_out.getFaciesByName("X")
    assert faciesOut is not None
    gamma = faciesOut.criteriaCollection.getCriteriaByName("Gamma")
    assert gamma is not None
    assert gamma.type == FaciesCriteriaType.UNCATEGORIZED
    assert math.isinf(gamma.minRange) and gamma.minRange < 0
    assert math.isinf(gamma.maxRange) and gamma.maxRange > 0


def test_saveFaciesModel_sorts_facies_and_criteria(
    tmp_path: pathlib.Path,
) -> None:
    """Test saveFaciesModel outputs deterministic facies/criteria order.

    Objective:
    - Ensure export is deterministic: facies and criteria are sorted by name.
    Input data:
    - Two facies with out-of-order names and out-of-order criteria names.
    Expected results:
    - JSON faciesModel list is ordered by facies name.
    - Each facies.criteria list is ordered by criterion name.
    """
    a = Facies(
        name="A", criteria={FaciesCriteria(name="b"), FaciesCriteria(name="a")}
    )
    b = Facies(
        name="B", criteria={FaciesCriteria(name="d"), FaciesCriteria(name="c")}
    )
    model = FaciesModel(faciesSet={b, a})

    out_file = tmp_path / "sorted.json"
    saveFaciesModel(model, str(out_file))
    exported = json.loads(out_file.read_text(encoding="utf-8"))

    facies_names = [f["name"] for f in exported["faciesModel"]]
    assert facies_names == ["A", "B"]

    a_criteria_names = [
        c["name"] for c in exported["faciesModel"][0]["criteria"]
    ]
    b_criteria_names = [
        c["name"] for c in exported["faciesModel"][1]["criteria"]
    ]
    assert a_criteria_names == ["a", "b"]
    assert b_criteria_names == ["c", "d"]


if __name__ == "__main__":
    pytest.main(
        [
            os.path.abspath(__file__),
        ]
    )
