# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np

from .Curve import Curve


class EnvironmentConditionModelBase(ABC):
    def __init__(
        self: Self,
        environmentConditionName: str,
    ) -> None:
        """Defines the base class for environment condition models.

        An environment condition model defines the laws that govern the
        evolution of a given environmental condition (e.g., temperature,
        energy, salinity, etc.) in a given depositional environment.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        """
        self.envConditionName = environmentConditionName

    @abstractmethod
    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Compute the environmental condition of the environment.

        This method should be implemented in derived classes.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location.
        :return float: environmental condition
        """
        pass


class EnvironmentConditionModelStats(EnvironmentConditionModelBase):
    def __init__(
        self: Self,
        environmentConditionName: str,
        minValue: float,
        maxValue: float,
    ) -> None:
        """Base class for environment condition models based on statistics.

        In these models, the environmental condition follows a statistical
        distribution (e.g., uniform, Gaussian) defined by the provided
        parameters.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        :param float minValue: minimum value of the environmental condition
        :param float maxValue: maximum value of the environmental condition
        """
        super().__init__(environmentConditionName)

        self.minValue = minValue
        self.maxValue = maxValue

    @property
    def rangeWidth(self: Self) -> float:
        """Width of the range."""
        return self.maxValue - self.minValue

    @property
    def rangeMid(self: Self) -> float:
        """Mid-point of the range."""
        return (self.maxValue + self.minValue) / 2.0

    @abstractmethod
    def getReferenceValue(self: Self) -> float:
        """Reference value for the environmental condition.

        This is the value used to normalize the distribution. It may be the
        mean value for a Gaussian distribution, or the mid-point for a uniform
        distribution.

        :return float: reference value for the environmental condition
        """
        pass

    @abstractmethod
    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the env condition coefficient from the uniform distribution.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location (ignored by this model,
            accepted for API consistency).
        :return float: environmental condition
        """
        pass

    def getEnvironmentConditionCoefficientAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the env condition coefficient from the Gaussian distribution.

        Valid only if the mean value is not zero.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location (ignored by this model,
            accepted for API consistency).
        :return float: env condition coefficient (mean value)
        """
        val: float = self.getEnvironmentConditionAt(relatedCondition)
        refValue: float = self.getReferenceValue()
        if refValue != 0:
            val /= refValue
        return val


class EnvironmentConditionModelUniform(EnvironmentConditionModelStats):
    def __init__(
        self: Self,
        environmentConditionName: str,
        minValue: float,
        maxValue: float,
    ) -> None:
        """Defines an env condition model based on a uniform distribution.

        In this model, the environmental condition follows a uniform
        distribution centered around the mean value, with a standard deviation
        defined as a fraction of the mean value.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        :param float minValue: minimum value of the environmental condition
        :param float maxValue: maximum value of the environmental condition
        """
        super().__init__(environmentConditionName, minValue, maxValue)

    def getReferenceValue(self: Self) -> float:
        """Reference value for the environmental condition.

        For a uniform distribution, the reference value is the mid-point of the
        range.

        :return float: reference value for the environmental condition
        """
        return self.rangeMid

    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the env condition coefficient from the uniform distribution.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location (ignored by this model,
            accepted for API consistency).
        :return float: environmental condition
        """
        return float(np.random.uniform(self.minValue, self.maxValue))


class EnvironmentConditionModelTriangular(EnvironmentConditionModelStats):
    def __init__(
        self: Self,
        environmentConditionName: str,
        modeValue: float,
        minValue: float,
        maxValue: float,
    ) -> None:
        """Defines an env condition model based on a triangular distribution.

        In this model, the environmental condition follows a triangular
        distribution centered around the mode.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        :param float minValue: minimum value of the environmental condition
        :param float maxValue: maximum value of the environmental condition
        """
        super().__init__(environmentConditionName, minValue, maxValue)
        self.mode = modeValue

    def getReferenceValue(self: Self) -> float:
        """Reference value for the environmental condition.

        For a triangular distribution, the reference value is the mode.

        :return float: reference value for the environmental condition
        """
        return self.mode

    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the env condition coefficient from the triangular distribution.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location (ignored by this model,
            accepted for API consistency).
        :return float: environmental condition
        """
        return float(
            np.random.triangular(self.minValue, self.mode, self.maxValue)
        )


class EnvironmentConditionModelGaussian(EnvironmentConditionModelStats):
    def __init__(
        self: Self,
        environmentConditionName: str,
        meanValue: float,
        stdDev: float | None = None,
        minValue: float = -float(np.inf),
        maxValue: float = float(np.inf),
    ) -> None:
        """Defines an env condition model based on a Gaussian distribution.

        In this model, the environmental condition follows a Gaussian
        distribution centered around the mean value, with a standard deviation
        defined as a fraction of the mean value.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        :param float meanValue: mean value of the environmental condition
        :param float | None stdDev: standard deviation of the environmental
            condition. If None, the default value is 0.2.
        """
        super().__init__(environmentConditionName, minValue, maxValue)

        self.meanValue = meanValue
        # default standard deviation is 0.2 * mean value
        self.stdDev = stdDev if stdDev is not None else 0.2 * self.meanValue

    def getReferenceValue(self: Self) -> float:
        """Reference value for the environmental condition.

        For a Gaussian distribution, the reference value is the mean value.

        :return float: reference value for the environmental condition
        """
        return self.meanValue

    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the env condition coefficient from the Gaussian distribution.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location (ignored by this model,
            accepted for API consistency).
        :return float: environmental condition
        """
        val: float = float(np.random.normal(self.meanValue, self.stdDev))
        # clamp to min and max values
        val = max(val, self.minValue)
        val = min(val, self.maxValue)
        return val


class EnvironmentConditionModelConstant(EnvironmentConditionModelStats):
    def __init__(
        self: Self, environmentConditionName: str, value: float
    ) -> None:
        """Defines an env condition model based on a constant value.

        In this model, the environmental condition is constant and equal to the
        provided value.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        :param float value: value of the environmental condition
        """
        super().__init__(environmentConditionName, value, value)

        self.value = value

    def getReferenceValue(self: Self) -> float:
        """Reference value for the environmental condition.

        For a constant value, the reference value is the constant value itself.

        :return float: reference value for the environmental condition
        """
        return self.value

    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the constant value of the environmental condition.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location (ignored by this model,
            accepted for API consistency).
        :return float: environmental condition
        """
        return self.value


class EnvironmentConditionModelCurve(EnvironmentConditionModelBase):
    def __init__(
        self: Self, environmentConditionName: str, curve: Curve
    ) -> None:
        """Defines an env condition model based on a curve.

        In this model, the environmental condition is defined by a curve that
        relates the condition to another condition like water depth or age.

        :param str environmentConditionName: name of the environmental
            condition the model applies to
        :param Curve curve: curve defining the relationship between the
            environmental condition and another condition like water depth or
            age.
        """
        super().__init__(environmentConditionName)

        self.curve = curve

    @property
    def relatedConditionName(self: Self) -> str:
        """Get the name of the related condition (x-axis) from the curve.

        :return str: name of the related condition (x-axis)
        """
        return self.curve._xAxisName

    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the value of the environmental condition from the curve.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location.
        :return float: environmental condition
        """
        if relatedCondition is None:
            raise ValueError(
                "relatedCondition must be provided for"
                + " EnvironmentConditionModelCurve"
            )
        return self.curve.getValueAt(relatedCondition)


class EnvironmentConditionModelCombination(EnvironmentConditionModelBase):
    def __init__(
        self: Self, models: list[EnvironmentConditionModelBase]
    ) -> None:
        """Defines a model that combines multiple env condition models.

        The environment condition is the combination of the values from
        the individual models. Basic usage include the use of a curve model
        combined with a statistical model to add variability to the curve. In
        this case the statistical model can be defined around 1.0 to represent
        a coefficient that modifies the curve value.

        The combination is performed by:

        - computing the env condition of each model
        - compute the ratios of the given value to the reference
          env condition for each model
        - multiply these ratios by the reference env condition

        Ideally, the reference env condition should be
        equal over all models (e.g., same average if multiple statistical
        models). If not, the reference env condition of the
        combination is the average of the reference env conditions of the
        models.

        :param list[EnvironmentConditionModelBase] models: list of environment
            condition models to combine
        """
        self.models: list[EnvironmentConditionModelBase] = models
        # default value, will be updated in checkModelsConsistency
        self._referenceEnvCondition = 1.0
        self.checkModelsConsistency()

    def checkModelsConsistency(self: Self) -> bool:
        """Check if env conditions models are consistent.

        The reference environment conditions of the models should be similar to
        ensure a meaningful combination. If the reference environment
        conditions are very different, the combination may not be meaningful.

        :return bool: True if the environment condition models are consistent
            (name and reference environment condition), False otherwise.
        """
        if len(self.models) == 0:
            print(
                "Warning: no environment condition model in the "
                + " combination."
            )
            return False

        # check list of env condition name in models is consistent
        envCondNames = [model.envConditionName for model in self.models]
        if len(set(envCondNames)) != 1:
            print(
                "Warning: the models in the combination have different "
                f"environment condition names: {set(envCondNames)}. "
                + "This may lead to a meaningless combination."
            )
        self.envConditionName = envCondNames[0]

        # check reference env condition and ranges for statistical models
        statModels = [
            model
            for model in self.models
            if isinstance(model, EnvironmentConditionModelStats)
        ]
        referenceEnvConditions = [
            model.getReferenceValue() for model in statModels
        ]
        if len(set(referenceEnvConditions)) != 1:
            print(
                "Warning: the stat models in the combination have different "
                "reference environment conditions. This may lead to a "
                + "meaningless combination."
            )
        minValues = [model.minValue for model in statModels]
        if len(set(minValues)) != 1:
            print(
                "Warning: the stat models in the combination have different "
                "minimum environment conditions. This may lead to a "
                + "meaningless combination."
            )
        maxValues = [model.maxValue for model in statModels]
        if len(set(maxValues)) != 1:
            print(
                "Warning: the stat models in the combination have different "
                "maximum environment conditions. This may lead to a "
                + "meaningless combination."
            )
        self._referenceEnvCondition = float(np.mean(referenceEnvConditions))
        return True

    def getEnvironmentConditionAt(
        self: Self,
        relatedCondition: float | None = None,
    ) -> float:
        """Get the value of the environmental condition from the curve.

        :param float | None relatedCondition: value of the related condition
            (e.g., water depth or age) at the location.
        :return float: environmental condition
        """
        if len(self.models) == 0:
            return 0.0

        statModels = [
            model
            for model in self.models
            if isinstance(model, EnvironmentConditionModelStats)
        ]
        envCondCoeff = 1.0
        if len(statModels) > 0:
            envCondCoeffs = np.array(
                [
                    model.getEnvironmentConditionCoefficientAt(
                        relatedCondition
                    )
                    for model in statModels
                ]
            )
            envCondCoeff = float(np.prod(envCondCoeffs))
        otherModels = [
            model
            for model in self.models
            if not isinstance(model, EnvironmentConditionModelStats)
        ]

        # by default use the mean from stat models
        refValue: float = self._referenceEnvCondition
        if len(otherModels) > 0:
            # if other models are present, use the average value from these
            # models as reference value
            values = np.array(
                [
                    model.getEnvironmentConditionAt(relatedCondition)
                    for model in otherModels
                ]
            )
            refValue = float(np.mean(values))
        return envCondCoeff * refValue


class EnvironmentConditionsModel:
    def __init__(
        self: Self,
        envConditionModels: list[EnvironmentConditionModelBase] | None = None,
    ) -> None:
        """Defines the model for environment conditions.

        This model contains a set of environment condition models. The model
        can be used to compute the environmental conditions at a water depth
        and age.
        """
        self.envConditionModels: dict[str, EnvironmentConditionModelBase] = {}
        if envConditionModels is not None:
            for model in envConditionModels:
                self.addEnvironmentConditionModel(
                    model.envConditionName, model
                )

    def addEnvironmentConditionModel(
        self: Self, name: str, model: EnvironmentConditionModelBase
    ) -> None:
        """Add an environment condition model to the model.

        :param str name: name of the environment condition.
        :param EnvironmentConditionModelBase model: environment condition model
            to add.
        """
        if self.isEnvironmentConditionModelPresent(name):
            print(
                f"Environment condition model '{name}' is already present, "
                + "it will be overwritten. You may use a composite model to "
                + "combine multiple models for the same condition if needed."
            )
        self.envConditionModels[name] = model

    @property
    def environmentConditionNames(self: Self) -> list[str]:
        """Get the list of environment condition names defined in the model.

        :return list[str]: list of environment condition names.
        """
        return list(self.envConditionModels.keys())

    def removeEnvironmentConditionModel(self: Self, name: str) -> None:
        """Remove an environment condition model from the model.

        :param str name: name of the environment condition to remove.
        """
        if name in self.envConditionModels:
            del self.envConditionModels[name]

    def isEnvironmentConditionModelPresent(self: Self, name: str) -> bool:
        """Check if an environment condition model is present in the model.

        :param str name: name of the environment condition to check.
        :return bool: True if the environment condition model is present, False
            otherwise.
        """
        return name in self.envConditionModels

    def getEnvironmentConditionsAt(
        self: Self, waterDepth: float, age: Optional[float] = None
    ) -> dict[str, float]:
        """Get all environment condition values from known waterDepth and age.

        Resolution strategy:

        - Build the condition universe from ``envConditionModels``.
        - Infer dependency edges from curve-based models
          (target condition depends on source condition).
        - Resolve independent conditions first via
          ``model.getEnvironmentConditionAt()``.
        - Resolve conditions that depend directly on ``waterDepth``
          and, when available, on ``age``.
        - Iteratively resolve remaining conditions once their source
          value is available.
        - Raise ``ValueError`` if unresolved conditions remain (missing
          or cyclic dependencies).

        This is effectively a dependency-ordered (topological)
        resolution, supporting both independent models and
        condition-to-condition models.

        :param float waterDepth: water depth at the location.
        :param float | None age: age at the location (optional, only needed if
            some conditions depend on age).
        :return dict[str, float]: environment condition dict.
        """
        dependencies: dict[str, str] = self.getCurveModelDependencies()

        env_conds: dict[str, float] = {}
        # resolve independent environment conditions first
        for condName, model in sorted(self.envConditionModels.items()):
            if condName not in dependencies:
                env_conds[condName] = model.getEnvironmentConditionAt()

        # resolve properties directly dependent on waterDepth and age
        for condName, sourceCond in sorted(dependencies.items()):
            if sourceCond.lower() == "waterdepth":
                env_conds[condName] = self.envConditionModels[
                    condName
                ].getEnvironmentConditionAt(waterDepth)
            elif sourceCond.lower() == "age" and age is not None:
                env_conds[condName] = self.envConditionModels[
                    condName
                ].getEnvironmentConditionAt(age)

        # resolve the other env conditions from dependencies iteratively
        unresolved: set[str] = set(self.envConditionModels.keys()) - set(
            env_conds.keys()
        )
        progress = (
            True  # True as long as a condition is resolved at each iteration
        )
        while unresolved and progress:
            progress = False
            for condName in sorted(unresolved):
                relatedCondName = dependencies.get(condName)
                if (
                    relatedCondName is not None
                    and relatedCondName in env_conds
                ):
                    relatedCondValue: float = env_conds[relatedCondName]
                    env_conds[condName] = self.envConditionModels[
                        condName
                    ].getEnvironmentConditionAt(relatedCondValue)
                    progress = True
            # update unresolved environment conditions
            unresolved = set(self.envConditionModels.keys()) - set(
                env_conds.keys()
            )

        if unresolved:
            unresolved_with_sources = {
                condName: dependencies.get(condName)
                for condName in sorted(unresolved)
            }
            raise ValueError(
                "Could not resolve all environment conditions. "
                + f"Unresolved dependencies: {unresolved_with_sources}."
            )

        return env_conds

    def getCurveModelDependencies(self: Self) -> dict[str, str]:
        """Get the condition dependencies from models based on a curve.

        :return dict[str, str]: dict mapping target condition (y-axis) to
            source condition (x-axis).
        """

        def _curve_x_axes(model: EnvironmentConditionModelBase) -> list[str]:
            if isinstance(model, EnvironmentConditionModelCurve):
                return [model.relatedConditionName]
            if isinstance(model, EnvironmentConditionModelCombination):
                out: list[str] = []
                for sub in model.models:
                    out.extend(_curve_x_axes(sub))
                return out
            return []

        dependencies: dict[str, str] = {}
        for cond_name, model in self.envConditionModels.items():
            x_axes = _curve_x_axes(model)
            if not x_axes:
                continue

            unique_x_axes = sorted({x for x in x_axes if isinstance(x, str)})
            if len(unique_x_axes) != 1:
                raise ValueError(
                    f"Condition '{cond_name}' has ambiguous curve "
                    + f"dependencies: {set(unique_x_axes)}"
                )
            x_axis = unique_x_axes[0]

            if cond_name in dependencies and dependencies[cond_name] != x_axis:
                raise ValueError(
                    f"Condition '{cond_name}' has multiple source "
                    + "dependencies "
                    + f"({dependencies[cond_name]!r}, {x_axis!r})."
                )
            dependencies[cond_name] = x_axis

        return dependencies
