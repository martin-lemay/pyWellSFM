# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Self

import numpy as np

from pywellsfm.model.AccumulationModel import AccumulationModel


class AccumulationSimulator:
    def __init__(
        self: Self,
    ) -> None:
        """Simulate accumulation in wells based on an accumulation model."""
        self.accumulationModel: AccumulationModel | None = None

    def setAccumulationModel(
        self: Self, accumulationModel: AccumulationModel
    ) -> None:
        """Set the accumulation model used by the simulator.

        :param AccumulationModel accumulationModel: accumulation model
            to set.
        """
        self.accumulationModel = accumulationModel

    def prepare(self: Self) -> None:
        """Prepare the simulator for computations."""
        if self.accumulationModel is None:
            raise ValueError("Accumulation model is not set.")

    def computeAccumulationRatesForAllElements(
        self: Self,
        environmentConditions: dict[str, float],
        age: float | None = None,
    ) -> dict[str, float]:
        """Compute the accumulation rate of each element from the model.

        :param dict[str, float] environmentConditions: dictionary of
            environmental conditions where keys are the names of the
            environmental factors and values are the corresponding values.
        :param float | None age: age of the accumulation (in My). If None, age
            is not used in the computation of the accumulation rate.
        :return dict[str, float]: dictionary where keys are element names and
            values are the corresponding accumulation rates.
        """
        if self.accumulationModel is None:
            raise ValueError("Accumulation model is not set in the simulator.")
        accumulationRates: dict[str, float] = {}
        for elementName in self.accumulationModel.elements:
            try:
                accumulationRates[elementName] = (
                    self.accumulationModel.getElementAccumulationAt(
                        elementName, environmentConditions, age
                    )
                )
            except ValueError:
                # Handle models that require env conditions
                accumulationRates[elementName] = np.nan

        return accumulationRates

    def computeElementAccumulationRate(
        self: Self,
        elementName: str,
        environmentConditions: dict[str, float],
        age: float | None = None,
    ) -> float:
        """Compute the accumulation rate of a given element from the model.

        :param str elementName: name of the element to compute the accumulation
            rate for.
        :param dict[str, float] environmentConditions: dictionary of
            environmental conditions where keys are the names of the
            environmental factors and values are the corresponding values.
        :param float | None age: age of the accumulation (in My). If None, age
            is not used in the computation of the accumulation rate.
        :return float: accumulation rate of the given element, or 0.0 if the
            element is not in the model.
        """
        if self.accumulationModel is None:
            raise ValueError("Accumulation model is not set in the simulator.")
        return self.accumulationModel.getElementAccumulationAt(
            elementName, environmentConditions, age
        )

    def computeTotalAccumulationRate(
        self: Self,
        environmentConditions: dict[str, float],
        age: float | None = None,
    ) -> float:
        """Compute the total accumulation rate in the accumulation model.

        :param dict[str, float] environmentConditions: dictionary of
            environmental conditions where keys are the names of the
            environmental factors and values are the corresponding values.
        :param float | None age: age of the accumulation (in My). If None, age
            is not used in the computation of the accumulation rate.
        :return float: total accumulation rate.
        """
        elementsAccumulationRates = (
            self.computeAccumulationRatesForAllElements(
                environmentConditions, age
            )
        )
        return float(np.nansum(list(elementsAccumulationRates.values())))
