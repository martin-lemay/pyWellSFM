# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from .Curve import AccumulationCurve


class AccumulationModelElementBase(ABC):
    def __init__(
        self: Self,
        elementName: str,
        accumulationRate: float,
    ) -> None:
        """Defines the base class for accumulation models based on elements.

        :param str elementName: name of the element the model applies to
        :param float accumulationRate: reference accumulation rate of the
            element (m/My)
        """
        self.elementName = elementName
        self.accumulationRate: float = accumulationRate

    @abstractmethod
    def getElementAccumulationAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Compute the accumulation rate of the element.

        This method should be implemented in derived classes.

        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :return float: accumulation rate (m/My)
        """
        pass


class AccumulationModelElementGaussian(AccumulationModelElementBase):
    def __init__(
        self: Self,
        elementName: str,
        accumulationRate: float,
        std_dev_factor: float | None = None,
    ) -> None:
        """Defines an accumulation model based on a probabilistic approach.

        In this accumulation model, the accumulation rate of the element
        follows a Gaussian distribution centered around the reference
        accumulation rate of the element, with a standard deviation defined as
        twice the reference rate.

        :param str elementName: name of the element the model applies to
        :param float accumulationRate: reference accumulation rate of the
            element (m/My)
        :param float | None std_dev_factor: factor to multiply the standard
            deviation by
        """
        super().__init__(elementName, accumulationRate)

        # default standard deviation factor is 0.2
        self.std_dev_factor = (
            std_dev_factor if std_dev_factor is not None else 0.2
        )

    def getElementAccumulationAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Get the accumulation rate according to the Gaussian distribution.

        :param dict[str, float] | None environmentConditions: environmental
            conditions (ignored by this model, accepted for API consistency)
        :return float: accumulation rate (m/My)
        """
        stddev = self.std_dev_factor * self.accumulationRate
        return float(np.random.normal(self.accumulationRate, stddev))


class AccumulationModelElementEnvironmentOptimum(AccumulationModelElementBase):
    def __init__(
        self: Self,
        elementName: str,
        accumulationRate: float,
        accumulationCurves: dict[str, AccumulationCurve] | None = None,
    ) -> None:
        """Defines an accumulation model based on environmental optimums.

        The accumulation rate is maximal if all
        environmental conditions are at their optimum value. The accumulation
        rate decreases as the environmental values deviate from their optimum.
        The rate equals the reference accumulation rate of the element
        multiplied by the product of all the reduction coefficients defined by
        the accumulation curves.

        :param str elementName: name of the element
        :param float accumulationRate: reference accumulation rate of the
            element (m/My)
        :param dict[str, AccumulationCurve] | None accumulationCurves: element
            accumulation reduction curves
        """
        super().__init__(elementName, accumulationRate)
        self.accumulationCurves: dict[str, AccumulationCurve] = (
            accumulationCurves if accumulationCurves is not None else {}
        )

    def addAccumulationCurve(self: Self, curve: AccumulationCurve) -> None:
        """Add a reduction coefficient curve that modulate the accumulation.

        The name of the environmental factor is the name of x axis of the
        curve.

        :param AccumulationCurve curve: reduction coefficient curve
        """
        self.accumulationCurves[curve._xAxisName] = curve

    def removeAccumulationCurve(self: Self, curveName: str) -> None:
        """Remove an accumulation curve from the model.

        :param str curveName: name of the accumulation curve to remove
        """
        if curveName in self.accumulationCurves:
            self.accumulationCurves.pop(curveName)

    def getAccumulationCurve(
        self: Self, curveName: str
    ) -> AccumulationCurve | None:
        """Get the reduction coefficient curve corresponding to the name.

        :param str curveName: name of the accumulation curve
        :return AccumulationCurve | None: reduction coefficient curve
        """
        return self.accumulationCurves.get(curveName, None)

    def getElementAccumulationAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Get accumulation rate of the element from environmental conditions.

        :param dict[str, float] | None environmentConditions: environment
            conditions. The keys are the name of the curves, the values are
            the corresponding conditions. Required for this model type.
        :return float: accumulation rate (m/My)
        :raise ValueError: if environmentConditions is None or empty
        """
        if environmentConditions is None:
            raise ValueError(
                f"{self.__class__.__name__} requires environmentConditions "
                f"to compute accumulation rate."
            )
        # get reduction coefficients for each environmental condition
        values = []
        for curveName, value in environmentConditions.items():
            curve = self.getAccumulationCurve(curveName)
            if curve is not None:
                values.append(curve.getValueAt(value))
        product: float = float(np.prod(values)) if len(values) > 0 else 1.0
        return self.accumulationRate * product


class AccumulationModel:
    def __init__(
        self: Self,
        name: str,
        elementAccumulationModels: dict[str, AccumulationModelElementBase]
        | None = None,
    ) -> None:
        """Defines the accumulation model for sediments.

        An accumulation model defines a list of elements and the rules that
        govern their accumulation through time.

        :param str name: name of the accumulation model
        :param dict | None elementAccumulationModels: dictionary
            of element names to their corresponding accumulation models
        """
        self.name: str = name
        self.elements: dict[str, AccumulationModelElementBase] = {}
        if elementAccumulationModels is not None:
            for elementName, model in elementAccumulationModels.items():
                self.addElement(elementName, model)

    def addElement(
        self: Self,
        elementName: str,
        accumulationModel: AccumulationModelElementBase,
    ) -> None:
        """Add an element to the accumulation model.

        :param str elementName: name of the element to add
        :param AccumulationModelElementBase accumulationModel: accumulation
            model associated to the element
        """
        self.elements[elementName] = accumulationModel

    def removeElement(self: Self, elementName: str) -> None:
        """Remove an element from the model.

        :param str elementName: name of the element to remove
        """
        self.elements.pop(elementName, None)

    def getElementModel(
        self: Self, elementName: str
    ) -> AccumulationModelElementBase | None:
        """Get an element of the model from its name.

        :param str elementName: name of the element to get
        :return AccumulationModelElementBase | None: element with the given
            name, or None if not found
        """
        for name, model in self.elements.items():
            if name == elementName:
                return model
        return None

    def getElementAccumulationAt(
        self: Self,
        elementName: str,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Compute the accumulation rate of an element in the model.

        This method should be implemented in derived classes.

        :param str elementName: name of the element to compute the accumulation
            rate of
        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :return float: accumulation rate of the element (m/My)
        """
        elementModel = self.getElementModel(elementName)
        if elementModel is not None:
            return elementModel.getElementAccumulationAt(environmentConditions)
        return 0.0

    def getTotalAccumulationAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
    ) -> float:
        """Compute the total accumulation rate from element models.

        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :return float: total accumulation rate (m/My)
        """
        return sum(
            elementModel.getElementAccumulationAt(environmentConditions)
            for elementModel in self.elements.values()
        )
