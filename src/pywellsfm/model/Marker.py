# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from enum import StrEnum
from typing import Any, Self

import numpy as np


class StratigraphicSurfaceType(StrEnum):
    """Define surface stratigraphic type."""

    CONFORM = "Conform"
    EROSIVE = "Erosive"
    TOPLAP = "Toplap"
    BASELAP = "Baselap"
    UNKNOWN = "Unknown"


class Marker:
    def __init__(
        self: Self,
        name: str,
        depth: float,
        age: float = np.nan,
        stratigraphicType: StratigraphicSurfaceType = StratigraphicSurfaceType.UNKNOWN,
    ):
        """Defines stratigraphic markers and associated properties.

        :param str name: name of the marker
        :param float depth: marker depth
        :param float age: marker age, defaults to np.nan
        :param StratigraphiSurfaceType stratigraphicType: stratigraphic type of the
            marker, defaults to StratigraphicSurfaceType.UNKNOWN.
        """
        #: name of the marker
        self.name: str = name
        #: depth of the marker
        self.depth: float = depth
        #: age of the marker
        # TODO: need to manage age uncertainty
        self.age: float = age
        #: stratigraphic relationship between above and bottom units
        self.stratigraphicType: StratigraphicSurfaceType = stratigraphicType

    def __eq__(self: Self, other: Any) -> bool:
        """Two markers are equal if all their properties are equal.

        :param Any other: other object
        :return bool: True if all their properties are equal
        """
        if isinstance(other, Marker):
            return (
                (self.name == other.name)
                and (self.depth == other.depth)
                and (self.age == other.age)
                and (self.stratigraphicType == other.stratigraphicType)
            )
        return False

    def __hash__(self: Self) -> int:
        return hash((self.name, self.depth, self.age, self.stratigraphicType))

    def areCollocated(self: Self, other: Any) -> bool:
        """Two markers are collocated if they are at the same depth.

        :param Any other: other object
        :return bool: True if at the same depth
        """
        if isinstance(other, Marker):
            return self.depth == other.depth
        return False

    def areSynchrone(self: Self, other: Any) -> bool:
        """Two markers are synchrone if they are at the same age.

        :param Any other: other object
        :return bool: True if at the same age
        """
        if isinstance(other, Marker):
            return self.age == other.age
        return False

    def areFromSameHorizon(self: Self, other: Any) -> bool:
        """Two markers belong to a same horizon if they have the same name and age.

        :param Any other: other object
        :return bool: True if same name and synchrone
        """
        if isinstance(other, Marker):
            return (self.name == other.name) and self.areSynchrone(other)
        return False
