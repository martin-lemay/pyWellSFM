# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Self


class Element:
    def __init__(self: Self, name: str, accumulationRate: float) -> None:
        """Defines an Element which is a sediment that is accumulated.

        The element is accumulated (produced/deposited) at a rate that may be
        modulated by external conditions depending on the accumulation model
        used.

        :param str name: name of the element
        :param float accumulationRate: Accumulation rate (m/My)
        """
        self.name: str = name
        self.accumulationRate: float = accumulationRate

    def __repr__(self: Self) -> str:
        """Defines __repr__ method.

        :return str: repr string
        """
        return self.name

    def __hash__(self: Self) -> int:
        """Defines __hash__ method.

        :return int: object hash
        """
        return hash(self.name)

    def __eq__(self: Self, other: Any) -> bool: # noqa: ANN401
        """Defines __eq__ method.

        :return bool: True if input object is an Element with the same name.
        """
        if isinstance(other, Element):
            return other.name == self.name
        return False
