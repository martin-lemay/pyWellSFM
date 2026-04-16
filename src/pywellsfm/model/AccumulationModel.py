# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from pywellsfm.utils import get_logger

from .Curve import AccumulationCurve

logger = get_logger(__name__)


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

    def getAccumulationAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Compute the accumulation rate of the element.

        This method should be implemented in derived classes.

        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :param float | None age: age of the accumulation.
        :return float: accumulation rate (m/My)
        """
        return self.accumulationRate * self.getAccumulationCoefficientAt(
            environmentConditions, age
        )

    @abstractmethod
    def getAccumulationCoefficientAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Compute the accumulation coefficient of the element.

        This method should be implemented in derived classes.

        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :param float | None age: age of the accumulation.
        :return float: accumulation coefficient
        """
        pass


class AccumulationModelCombination(AccumulationModelElementBase):
    def __init__(
        self: Self, models: list[AccumulationModelElementBase]
    ) -> None:
        """Defines a model that combines multiple accumulation models.

        The accumulation rate of the element is the combination of the
        accumulation rates of the individual models.

        The combination is performed by:

        - computing the accumulation rate of each model
        - compute the ratio of accumulation rate of each model to the reference
          accumulation rate
        - multiply this ratio by the reference accumulation rate

        Ideally, the reference accumulation rate of the element should be
        equal over all models. If not, the reference accumulation rate of the
        element is the average of the reference accumulation rates of the
        models.

        :param list[AccumulationModelElementBase] models: list of accumulation
            models to combine
        """
        self.models: list[AccumulationModelElementBase] = models
        self.checkModelsConsistency()

    def checkModelsConsistency(self: Self) -> bool:
        """Check if accumulation models are consistent.

        The reference accumulation rates of the models should be similar to
        ensure a meaningful combination. If the reference accumulation rates
        are very different, the combination may not be meaningful.

        :return bool: True if the accumulation models are consistent (name and
            reference accumulation rate), False otherwise.
        """
        if len(self.models) == 0:
            logger.warning("No accumulation model in the combination.")
            return False

        # check list of elementName in models is consistent
        elementNames = [model.elementName for model in self.models]
        if len(set(elementNames)) != 1:
            logger.warning(
                "The models in the combination have different element "
                "names: %s. This may lead to a meaningless combination.",
                set(elementNames),
            )
        self.elementName = elementNames[0]

        # check reference accumulation rates are consistent
        accumulationRates = [model.accumulationRate for model in self.models]
        if len(set(accumulationRates)) != 1:
            logger.warning(
                "The models in the combination have different reference "
                "accumulation rates. This may lead to a meaningless "
                "combination."
            )
        self.accumulationRate = float(np.mean(accumulationRates))
        return True

    def getAccumulationCoefficientAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Get the accumulation rate by combining multiple models.

        :param dict[str, float] | None environmentConditions: environmental
            conditions (ignored by this model, accepted for API consistency)
        :param float | None age: age of the accumulation (ignored by this
            model, accepted for API consistency)
        :return float: accumulation rate (m/My)
        """
        if len(self.models) == 0:
            return 0.0

        accumulationCoeffs = np.array(
            [
                model.getAccumulationCoefficientAt(environmentConditions, age)
                for model in self.models
            ]
        )

        if (accumulationCoeffs.size == 0) or (
            np.max(accumulationCoeffs) < 1e-6
        ):
            return 0.0

        return float(np.prod(accumulationCoeffs))


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
        a fraction of the reference rate.

        :param str elementName: name of the element the model applies to
        :param float accumulationRate: reference accumulation rate of the
            element (m/My)
        :param float | None std_dev_factor: ratio of standard deviation to the
            reference accumulation rate. If None, the default value is 0.2.
        """
        super().__init__(elementName, accumulationRate)

        # default standard deviation factor is 0.2
        self.std_dev_factor = (
            std_dev_factor if std_dev_factor is not None else 0.2
        )

    def getAccumulationCoefficientAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Get the accumulation coefficient from the Gaussian distribution.

        :param dict[str, float] | None environmentConditions: environmental
            conditions (ignored by this model, accepted for API consistency)
        :param float | None age: age of the accumulation (ignored by this
            model, accepted for API consistency)
        :return float: accumulation coefficient
        """
        return float(np.random.normal(1.0, self.std_dev_factor))


class AccumulationModelElementOptimum(AccumulationModelElementBase):
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
        self.accumulationCurves: dict[str, AccumulationCurve] = {}
        for curve in (
            accumulationCurves if accumulationCurves is not None else {}
        ).values():
            self.addAccumulationCurve(curve)

    def addAccumulationCurve(self: Self, curve: AccumulationCurve) -> None:
        """Add a reduction coefficient curve that modulate the accumulation.

        The name of the environmental factor is the name of x axis of the
        curve.

        :param AccumulationCurve curve: reduction coefficient curve
        """
        self.accumulationCurves[curve._xAxisName.lower()] = curve

    def removeAccumulationCurve(self: Self, curveName: str) -> None:
        """Remove an accumulation curve from the model.

        :param str curveName: name of the accumulation curve to remove
        """
        if curveName.lower() in self.accumulationCurves:
            self.accumulationCurves.pop(curveName.lower())

    def getAccumulationCurve(
        self: Self, curveName: str
    ) -> AccumulationCurve | None:
        """Get the reduction coefficient curve corresponding to the name.

        :param str curveName: name of the accumulation curve
        :return AccumulationCurve | None: reduction coefficient curve
        """
        return self.accumulationCurves.get(curveName.lower(), None)

    def getAccumulationCoefficientAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Get accumulation coefficient from environmental conditions.

        :param dict[str, float] | None environmentConditions: environment
            conditions. The keys are the name of the curves, the values are
            the corresponding conditions. Required for this model type.
        :param float | None age: age of the accumulation.
        :return float: accumulation coefficient (0-1)
        :raise ValueError: if environmentConditions is None or empty
        """
        values = []
        # get reduction coefficients for the given age if age curve is defined
        if age is not None:
            curveAge: AccumulationCurve | None = self.getAccumulationCurve(
                "Age"
            )
            if curveAge is not None:
                values.append(curveAge.getValueAt(age))

        # get reduction coefficients for each environmental condition
        if environmentConditions is not None:
            for curveName, value in environmentConditions.items():
                curve: AccumulationCurve | None = self.getAccumulationCurve(
                    curveName
                )
                if curve is not None:
                    values.append(curve.getValueAt(value))
        product: float = float(np.prod(values)) if len(values) > 0 else 1.0
        return product


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
        age: float | None = None,
    ) -> float:
        """Compute the accumulation rate of an element in the model.

        This method should be implemented in derived classes.

        :param str elementName: name of the element to compute the accumulation
            rate of
        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :param float | None age: age of the accumulation.
        :return float: accumulation rate of the element (m/My)
        """
        elementModel = self.getElementModel(elementName)
        if elementModel is not None:
            return elementModel.getAccumulationAt(environmentConditions, age)
        return 0.0

    def getTotalAccumulationAt(
        self: Self,
        environmentConditions: dict[str, float] | None = None,
        age: float | None = None,
    ) -> float:
        """Compute the total accumulation rate from element models.

        :param dict[str, float] | None environmentConditions: optional
            environmental conditions. Keys are environmental factor names,
            values are the conditions.
        :param float | None age: age of the accumulation.
        :return float: total accumulation rate (m/My)
        """
        return sum(
            elementModel.getAccumulationAt(environmentConditions, age)
            for elementModel in self.elements.values()
        )
