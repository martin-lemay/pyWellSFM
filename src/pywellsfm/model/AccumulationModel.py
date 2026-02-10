# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from pywellsfm.model.Curve import AccumulationCurve
from pywellsfm.model.Element import Element


class AccumulationModelBase(ABC):
    def __init__(
        self: Self,
        name: str,
        elements: set[Element] | None = None,
    ) -> None:
        """Defines the base class for sediment accumulation models.

        An accumulation model defines a list of elements and the rules that
        govern their accumulation through time.

        :param str name: name of the accumulation model
        :param set[Element] elements: set of elements in the accumulation model
        """
        self.name = name
        self.elements = elements if elements is not None else set()

    def addElement(self: Self, element: Element) -> None:
        """Add an element to the accumulation model.

        :param Element element: element to add
        """
        self.elements.add(element)

    def removeElement(self: Self, elementName: str) -> None:
        """Remove an element from the model.

        :param str elementName: name of the element to remove
        """
        element_to_remove: Element | None = None
        for elt in self.elements:
            if elt.name == elementName:
                element_to_remove = elt
                break

        if element_to_remove is not None:
            self.elements.remove(element_to_remove)

    def getElement(self: Self, elementName: str) -> Element | None:
        """Get an element of the model from its name.

        :param str elementName: name of the element to get
        :return Element | None: element with the given name, or None if not
            found
        """
        for elt in self.elements:
            if elt.name == elementName:
                return elt
        return None

    @abstractmethod
    def getElementAccumulationAt(
        self: Self,
        element: Element,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Compute the accumulation rate of an element in the model.

        This method should be implemented in derived classes.

        :param Element element: element to compute the accumulation rate for
        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :return float: accumulation rate (m/My)
        :raise NotImplementedError: if the method is not implemented in derived
            class
        """
        raise NotImplementedError(
            "getElementAccumulationAt() must be implemented "
            "in derived classes."
        )


class AccumulationModelGaussian(AccumulationModelBase):
    def __init__(
        self: Self,
        name: str,
        elements: set[Element] | None = None,
        std_dev_factors: dict[str, float] | None = None,
    ) -> None:
        """Defines an accumulation model based on a probabilistic approach.

        In this accumulation model, the accumulation rate of each element
        follows a Gaussian distribution centered around the reference
        accumulation rate of the element, with a standard deviation defined as
        twice the reference rate.

        :param str name: name of the accumulation model
        :param set[Element] elements: set of elements in the accumulation model
        :param float std_dev_factor: factor to multiply the standard deviation
            by
        """
        super().__init__(name, elements)

        # default standard deviation factor
        self.defaultStdDev = 0.2  # 20% of mean
        if std_dev_factors is None:
            self.std_dev_factors = dict.fromkeys(
                [elt.name for elt in self.elements], self.defaultStdDev
            )
        else:
            self.std_dev_factors = std_dev_factors

    def addElement(
        self: Self, element: Element, std_dev_factor: float | None = None
    ) -> None:
        """Add an element together with the standard deviation factor.

        :param Element element: element to add
        :param float | None std_dev_factor: standard deviation factor,
            defaults to None
        """
        super().addElement(element)
        factor = (
            self.defaultStdDev if std_dev_factor is None else std_dev_factor
        )
        self.std_dev_factors[element.name] = factor

    def getElementAccumulationAt(
        self: Self,
        element: Element,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Get the accumulation rate according to the Gaussian distribution.

        :param Element element: element to get the accumulation rate for
        :param dict[str, float] | None environmentConditions: environmental
            conditions (ignored by this model, accepted for API consistency)
        :return float: accumulation rate (m/My)
        """
        mean = element.accumulationRate
        stdDev_factor = self.std_dev_factors.get(
            element.name, self.defaultStdDev
        )
        stddev = stdDev_factor * mean
        return float(np.random.normal(mean, stddev))


class AccumulationModelEnvironmentOptimum(AccumulationModelBase):
    def __init__(
        self: Self,
        name: str,
        elements: set[Element] | None = None,
    ) -> None:
        """Defines an accumulation model based on environmental optimums.

        For a given element, the accumulation rate is maximal if all
        environmental conditions are at their optimum value. The accumulation
        rate decreases as the environmental values deviate from their optimum.
        The rate equals the reference accumulation rate of the element
        multiplied by the product of all the reduction coefficients defined by
        the accumulation curves.

        :param str name: name of the accumulation model
        :param set[Element] elements: set of elements in the accumulation model
        """
        if elements is None:
            elements = set()
        super().__init__(name, elements)
        self.prodCurves: dict[str, AccumulationCurve] = {}

    def addAccumulationCurve(self: Self, curve: AccumulationCurve) -> None:
        """Add a reduction coefficient curve that modulate the accumulation.

        The name of the environmental factor is the name of x axis of the
        curve.

        :param AccumulationCurve curve: reduction coefficient curve
        """
        self.prodCurves[curve._xAxisName] = curve

    def removeAccumulationCurve(self: Self, curveName: str) -> None:
        """Remove an accumulation curve from the model.

        :param str curveName: name of the accumulation curve to remove
        """
        self.prodCurves.pop(curveName, None)

    def getAccumulationCurve(self: Self, name: str) -> AccumulationCurve:
        """Get the reduction coefficient curve corresponding to the name.

        :param str name: name of the environmental factor
        :return AccumulationCurve: reduction coefficient curve
        """
        return self.prodCurves[name]

    def getElementAccumulationAt(
        self: Self,
        element: Element,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Get accumulation rate of an element from environmental condition.

        :param Element element: element to get the accumulation rate for
        :param dict[str, float] | None environmentConditions: environment
            conditions. The keys are the name of the curves, the values are
            the corresponding conditions. Required for this model type.
        :return float: accumulation rate (m/My)
        :raise ValueError: if environmentConditions is None or empty
        """
        if environmentConditions is None:
            raise ValueError(
                f"{self.__class__.__name__} requires environmentConditions "
                f"to compute accumulation rate for element '{element.name}'"
            )

        values = [
            self.prodCurves[name](value)
            for name, value in environmentConditions.items()
            if name in self.prodCurves
        ]
        return element.accumulationRate * float(np.prod(values))
