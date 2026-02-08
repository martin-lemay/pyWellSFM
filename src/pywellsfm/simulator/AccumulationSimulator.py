# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Self

import numpy as np

from pywellsfm.model.AccumulationModel import AccumulationModelBase


class AccumulationSimulator:
    def __init__(
        self: Self,
    ) -> None:
        """Simulate sediment accumulation in wells based on a sedimentationModel."""
        self.accumulationModel: AccumulationModelBase | None = None

    def setAccumulationModel(
        self: Self, accumulationModel: AccumulationModelBase
    ) -> None:
        """Set the accumulation model used by the simulator.

        :param AccumulationModelBase accumulationModel: accumulation model to set.
        """
        self.accumulationModel = accumulationModel

    def prepare(self: Self) -> None:
        """Prepare the simulator for computations."""
        if self.accumulationModel is None:
            raise ValueError("Accumulation model is not set.")

    def computeAccumulationRatesForAllElements(
        self: Self, environmentConditions: dict[str, float]
    ) -> dict[str, float]:
        """Compute the accumulation rate of each element in the accumulation model.

        :param dict[str, float] environmentConditions: dictionary of environmental
        conditions where keys are the names of the environmental factors and values are
        the corresponding values.

        :return dict[str, float]: dictionary where keys are element names and values are
        the corresponding production rates.
        """
        if self.accumulationModel is None:
            raise ValueError("Accumulation model is not set in the simulator.")

        accumulationRates: dict[str, float] = {}
        for element in self.accumulationModel.elements:
            try:
                accumulationRates[element.name] = (
                    self.accumulationModel.getElementAccumulationAt(
                        element, environmentConditions
                    )
                )
            except ValueError:
                # Handle models that require env conditions
                accumulationRates[element.name] = np.nan

        return accumulationRates

    def computeElementAccumulationRate(
        self: Self, elementName: str, environmentConditions: dict[str, float]
    ) -> float:
        """Compute the accumulation rate of a given element in the accumulation model.

        :param str elementName: name of the element to compute the accumulation rate for.
        :param dict[str, float] environmentConditions: dictionary of environmental
        conditions where keys are the names of the environmental factors and values are
        the corresponding values.

        :return float: accumulation rate of the given element.
        """
        elementsAccumulationRates = self.computeAccumulationRatesForAllElements(
            environmentConditions
        )
        return elementsAccumulationRates.get(elementName, 0.0)

    def computeTotalAccumulationRate(
        self: Self, environmentConditions: dict[str, float]
    ) -> float:
        """Compute the total accumulation rate in the accumulation model.

        :param dict[str, float] environmentConditions: dictionary of environmental
        conditions where keys are the names of the environmental factors and values are
        the corresponding values.

        :return float: total accumulation rate.
        """
        elementsAccumulationRates = self.computeAccumulationRatesForAllElements(
            environmentConditions
        )
        return float(np.nansum(list(elementsAccumulationRates.values())))
