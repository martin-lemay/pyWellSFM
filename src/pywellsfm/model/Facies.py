# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from enum import StrEnum
from typing import Any, Optional, Self


class FaciesCriteriaType(StrEnum):
    """Criteria types.

    Criteria types include:
        - sedimentatological criteria (e.g., grain size, element composition,
          classifications)
        - petrophysical criteria (e.g., porosity, permeability, density)
        - environmental conditions (e.g., water depth, energy level,
          temperature)
        - uncategorized criteria (e.g., any other property used to define
          facies)
    """

    SEDIMENTOLOGICAL = "sedimentological"
    PETROPHYSICAL = "petrophysical"
    ENVIRONMENTAL = "environmental"
    UNCATEGORIZED = "uncategorized"


class FaciesCriteria:
    def __init__(
        self: Self,
        name: str,
        minRange: float = -float("inf"),
        maxRange: float = float("inf"),
        type: FaciesCriteriaType = FaciesCriteriaType.UNCATEGORIZED,
    ) -> None:
        """Defines a criteria to classify rocks based on a range of a property.

        :param str name: property name
        :param float minRange: minimum value of the property.
            Default is -infinity.
        :param float maxRange: maximum value of the property.
            Default is infinity.
        :param FaciesCriteriaType type: type of the criteria.
            Default is UNCATEGORIZED.
        """
        self.name: str = name
        self.type: FaciesCriteriaType = type
        self.minRange: float = minRange
        self.maxRange: float = maxRange

    def __repr__(self: Self) -> str:
        """Redefines __repr__.

        :return str: repr string
        """
        return f"{self.name} [{self.minRange}, {self.maxRange}]"

    def __hash__(self: Self) -> int:
        """Defines __hash__ method.

        :return int: object hash
        """
        return hash(self.name)

    def __eq__(self: Self, other: Any) -> bool:  # noqa: ANN401
        """Defines __eq__ method.

        :return bool: True if input object is an Element with the same name
            and type.
        """
        if isinstance(other, FaciesCriteria):
            return (other.name == self.name) & (other.type == self.type)
        if isinstance(other, str):
            return other == self.name
        return False

    def isNamed(self: Self, name: str) -> bool:
        """Compare criteria name against input name.

        Name comparison is case insensitive.

        :param str name: input name
        :return bool: True if the criteria name equals input name
        """
        return self.name.lower() == name.lower()

    def hasType(self: Self, type: FaciesCriteriaType) -> bool:
        """Check if criteria is of given type.

        :param FaciesCriteriaType type: criteria type
        :return bool: True if criteria is of given type
        """
        return self.type == type


class FaciesCriteriaCollection:
    def __init__(
        self: Self,
        criteriaType: FaciesCriteriaType = FaciesCriteriaType.UNCATEGORIZED,
    ) -> None:
        """Collection of facies criteria.

        Criteria types include:

        - sedimentatological criteria (e.g., grain size, element composition,
          classifications)
        - petrophysical criteria (e.g., porosity, permeability, density)
        - environmental conditions (e.g., water depth, energy level,
          temperature)
        - uncategorized criteria (e.g., any other property used to define
          facies)

        All criteria of the collection must be unique by name.

        :param FaciesCriteriaType criteriaType: type of criteria allowed in the
            collection. Default is UNCATEGORIZED that means that any criteria
            type can be added.
        """
        self.criteria: set[FaciesCriteria] = set()
        self.type: FaciesCriteriaType = criteriaType

    def addCriteria(
        self: Self, criteria: FaciesCriteria | set[FaciesCriteria]
    ) -> None:
        """Add a criteria or set of criteria to the collection.

        If a criteria with the same name already exists in the collection, it
        is not added.

        :param FaciesCriteria | set[FaciesCriteria] criteria: criteria or set
            of criteria to add
        """
        if isinstance(criteria, FaciesCriteria):
            if not self.criteriaIsAllowed(criteria):
                print(
                    f"Criteria with name '{criteria.name}' and type "
                    f"'{criteria.type.value}' is not allowed in this "
                    f"collection of {self.type.value} criteria."
                )
                return
            if self.criteriaExists(criteria.name):
                print(
                    f"Criteria with name '{criteria.name}' already exists, "
                    f"cannot add a duplicate."
                )
                return
            self.criteria.add(criteria)
        elif isinstance(criteria, set):
            for crit in criteria:
                self.addCriteria(crit)
        else:
            raise TypeError(
                "criteria must be a FaciesCriteria or a set of FaciesCriteria"
            )

    def criteriaIsAllowed(self: Self, criteria: FaciesCriteria) -> bool:
        """Check if a criteria can be added to the collection.

        A criteria can be added if its type matches the collection type, or if
        the collection type is UNCATEGORIZED.

        :param FaciesCriteria criteria: criteria to check
        :return bool: True if the criteria can be added to the collection
        """
        return (self.type == FaciesCriteriaType.UNCATEGORIZED) | (
            criteria.type == self.type
        )

    def criteriaExists(self: Self, criteriaName: str) -> bool:
        """Check if a criteria exists in the collection by name.

        :param str criteriaName: name of the criteria to check
        :return bool: True if the criteria exists in the collection
        """
        return any(crit.isNamed(criteriaName) for crit in self.criteria)

    def removeCriteria(self: Self, criteriaNames: str | set[str]) -> None:
        """Remove a criteria or set of criteria from the collection by name.

        :param str | set[str] criteriaNames: name or set of names of criteria
            to remove
        """
        if isinstance(criteriaNames, str):
            for critSet in (self.criteria,):
                critToRemove = None
                for crit in critSet:
                    if crit.isNamed(criteriaNames):
                        critToRemove = crit
                        break
                if critToRemove is not None:
                    critSet.remove(critToRemove)
        elif isinstance(criteriaNames, set):
            for critName in criteriaNames:
                self.removeCriteria(critName)
        else:
            raise TypeError("criteriaNames must be a str or a set of str")

    def getAllCriteria(self: Self) -> set[FaciesCriteria]:
        """Get all criteria in the collection.

        :return set[FaciesCriteria]: set of all criteria
        """
        return self.criteria

    def getCriteriaByName(
        self: Self, criteriaName: str
    ) -> Optional[FaciesCriteria]:
        """Get criteria from the collection by name.

        :param str criteriaName: name of the criteria to get
        :return FaciesCriteria | None: criteria with the given name, or None if
            not found
        """
        for crit in self.criteria:
            if crit.isNamed(criteriaName):
                return crit
        return None

    def getCriteriaSetByType(
        self: Self, criteriaType: FaciesCriteriaType
    ) -> set[FaciesCriteria]:
        """Get a subset of criteria from the collection by type.

        :param FaciesCriteriaType criteriaType: type of criteria to get
        :return set[FaciesCriteria]: set of criteria of the given type
        """
        return {crit for crit in self.criteria if crit.hasType(criteriaType)}

    def clearAllCriteria(self: Self) -> None:
        """Remove all criteria from the collection."""
        self.criteria.clear()

    def clearCriteriaByType(
        self: Self, criteriaType: FaciesCriteriaType
    ) -> int:
        """Remove all criteria of a given type from the collection.

        :param FaciesCriteriaType criteriaType: type of criteria to remove
        :return int: number of criteria removed
        """
        critsToRemove = self.getCriteriaSetByType(criteriaType)
        for crit in critsToRemove:
            self.criteria.remove(crit)
        return len(critsToRemove)

    def getCriteriaCount(self: Self) -> int:
        """Get the number of criteria in the collection.

        :return int: number of criteria
        """
        return len(self.criteria)

    def isEmpty(self: Self) -> bool:
        """Check if the collection is empty.

        :return bool: True if the collection is empty
        """
        return len(self.criteria) == 0


class Facies:
    def __init__(
        self: Self,
        name: str,
        criteria: FaciesCriteria | set[FaciesCriteria],
        criteriaType: FaciesCriteriaType = FaciesCriteriaType.UNCATEGORIZED,
    ) -> None:
        """A facies is a category of rock defined from some criteria.

        :param str name: name of the facies
        :param FaciesCriteria | set[FaciesCriteria] criteria: criteria or set
            of criteria used to define the facies
        :param FaciesCriteriaType criteriaType: type of criteria used to define
            the facies. Default is UNCATEGORIZED that means that any criteria
            type can be used.
        """
        if isinstance(criteria, (set, list, tuple)) and len(criteria) == 0:
            raise ValueError(
                f"At least one criteria must be defined for the facies {name}"
            )
        self.name: str = name
        self.criteriaCollection: FaciesCriteriaCollection = (
            FaciesCriteriaCollection(criteriaType)
        )
        self.addCriteria(criteria)

    def addCriteria(self: Self, criteria: FaciesCriteria) -> None:
        """Add a criteria to the facies.

        :param FaciesCriteria criteria: criteria to add
        """
        self.criteriaCollection.addCriteria(criteria)

    def getCriteriaCount(self: Self) -> int:
        """Get the number of criteria defining the facies.

        :return int: number of criteria
        """
        return self.criteriaCollection.getCriteriaCount()

    def getCriteria(self: Self, criteriaName: str) -> Optional[FaciesCriteria]:
        """Get a criteria defining the facies by name.

        :param str criteriaName: name of the criteria to get
        :return FaciesCriteria | None: criteria with the given name, or None
            if not found
        """
        return self.criteriaCollection.getCriteriaByName(criteriaName)


class PetrophysicalFacies(Facies):
    def __init__(self: Self, name: str, criteria: set[FaciesCriteria]) -> None:
        """A facies defined from petrophysical criteria only.

        A petrophysical facies contains the criteria from which it is defined.

        :param str name: name of the facies
        :param set[FaciesCriteria] criteria: set of criteria used to define
            the facies
        """
        super().__init__(name, criteria, FaciesCriteriaType.PETROPHYSICAL)


class SedimentaryFacies(Facies):
    def __init__(self: Self, name: str, criteria: set[FaciesCriteria]) -> None:
        """A facies defined from sedimentological criteria.

        A sedimentary facies contains the criteria and optionnaly the
        environmental conditions from which it is deposited.

        :param str name: name of the facies
        :param set[FaciesCriteria] criteria: set of criteria used to define
            the facies
        """
        super().__init__(name, criteria, FaciesCriteriaType.SEDIMENTOLOGICAL)


class EnvironmentalFacies(Facies):
    def __init__(self: Self, name: str, criteria: set[FaciesCriteria]) -> None:
        """A facies defined from environmental criteria.

        An environmental facies contains the criteria from which it is defined.

        :param str name: name of the facies
        :param set[FaciesCriteria] criteria: set of criteria used to define
            the facies
        """
        super().__init__(name, criteria, FaciesCriteriaType.ENVIRONMENTAL)


class FaciesModel:
    def __init__(self: Self, faciesSet: set[Facies]) -> None:
        """Defines a facies model containing a set of facies.

        :param set[Facies] faciesSet: set of facies in the model
        """
        self.faciesSet: set[Facies] = faciesSet

    def getFaciesByName(self: Self, faciesName: str) -> Optional[Facies]:
        """Get a facies from the model by name.

        :param str faciesName: name of the facies to get
        :return Facies | None: facies with the given name, or None if not found
        """
        for facies in self.faciesSet:
            if facies.name == faciesName:
                return facies
        return None

    def getCriteriaRangeForFacies(
        self: Self, faciesName: str, criteriaName: str
    ) -> Optional[tuple[float, float]]:
        """Get the range of a criteria for a given facies.

        :param str faciesName: name of the facies
        :param str criteriaName: name of the criteria
        :return tuple[float, float] | None: (min, max) range of the criteria
            for the facies, or None if not found
        """
        facies = self.getFaciesByName(faciesName)
        if facies is None:
            print(
                f"Facies with name '{faciesName}' not found in the "
                "facies model."
            )
            return None
        criteria = facies.getCriteria(criteriaName)
        if criteria is None:
            print(
                f"Criteria with name '{criteriaName}' not found for the facies"
                f" '{faciesName}'."
            )
            return None
        return (criteria.minRange, criteria.maxRange)
