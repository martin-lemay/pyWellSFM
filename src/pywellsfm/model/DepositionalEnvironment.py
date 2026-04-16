# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Optional, Self

import numpy as np

from pywellsfm.utils import get_logger

from .EnvironmentConditionModel import (
    EnvironmentConditionModelStats,
    EnvironmentConditionModelUniform,
    EnvironmentConditionsModel,
)

logger = get_logger(__name__)


class DepositionalEnvironment:
    def __init__(
        self: Self,
        name: str,
        waterDepthModel: EnvironmentConditionModelStats,
        envConditionsModel: EnvironmentConditionsModel | None = None,
        distality: float | None = None,
    ) -> None:
        """Defines a depositional environment.

        The environment is defined from a waterDepth range and optionaly other
        property ranges including energy, temperature, salinity, etc.

        Curves can be set to define relationships between properties,
        e.g. energy vs waterDepth, temperature vs age.

        .. NOTE::

            Properties can be related to a single other property,
            e.g. temperature can be defined as a function of waterDepth, but
            not as a function of both waterDepth and age.

        :param str name: name of the environment
        :param EnvironmentConditionModelStats waterDepthModel: model for the
            water depth of the environment. It must be based on a statistical
            distribution, either constant, uniform, triangular or Gaussian.
        :param EnvironmentConditionsModel | None environmentConditionsModel:
            model for the evolution of environment conditions, including
            energy, temperature, salinity, etc. If None, a default model with
            no conditions is used.
        :param float distality: distality of the environment, defined as the
            distance from the shoreline, in km.
        """
        self.name: str = name
        self.waterDepthModel: EnvironmentConditionModelStats = waterDepthModel
        # evolution of environment conditions
        self.envConditionsModel: EnvironmentConditionsModel = (
            EnvironmentConditionsModel()
            if envConditionsModel is None
            else envConditionsModel
        )
        self.distality: float | None = distality

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

    def __eq__(self: Self, other: Any) -> bool:  # noqa: ANN401
        """Defines __eq__ method.

        :return bool: True if input object is a DepositionalEnvironment with
            the same name, same distality, same waterDepth and other property
            ranges.
        """
        if isinstance(other, DepositionalEnvironment):
            return (
                other.name == self.name
                and other.distality == self.distality
                and other.waterDepth_range == self.waterDepth_range
                # and other.other_property_ranges == self.other_property_ranges
                and (
                    other.envConditionsModel.environmentConditionNames
                    == self.envConditionsModel.environmentConditionNames
                )
            )
        return False

    @property
    def waterDepth_range(self: Self) -> tuple[float, float]:
        """WaterDepth range of the environment."""
        return (self.waterDepthModel.minValue, self.waterDepthModel.maxValue)

    @property
    def waterDepth_min(self: Self) -> float:
        """Minimum waterDepth of the environment."""
        return self.waterDepthModel.minValue

    @property
    def waterDepth_max(self: Self) -> float:
        """Maximum waterDepth of the environment."""
        return self.waterDepthModel.maxValue

    @property
    def waterDepth_rangeRef(self: Self) -> float:
        """Reference value of the waterDepth range."""
        return self.waterDepthModel.getReferenceValue()

    @property
    def waterDepth_rangeWidth(self: Self) -> float:
        """Width of the waterDepth range."""
        return self.waterDepthModel.rangeWidth

    def getEnvironmentConditions(
        self: Self,
        waterDepth: float,
        age: float,
    ) -> dict[str, float]:
        """Get environment conditions corresponding to this environment.

        :param float waterDepth: water depth value.
        :param float age: age at the location (only needed if some conditions
            depend on age).

        :return dict[str, float]: dictionary containing environment conditions
            for the given water depth and age.
        """
        return self.envConditionsModel.getEnvironmentConditionsAt(
            waterDepth, age
        )


class DepositionalEnvironmentModel:
    def __init__(
        self: Self, name: str, environments: list[DepositionalEnvironment]
    ) -> None:
        """Defines a depositional environment model.

        The model is defined as a list of depositional environments. These
        environments defined the spatial organization of the depositional
        system.

        :param str name: name of the depositional system
        :param list[DepositionalEnvironment] environments: list of depositional
            environments defining the model.
        """
        self.name: str = name
        self.environments: list[DepositionalEnvironment] = environments

    def __eq__(self: Self, other: Any) -> bool:  # noqa: ANN401
        """Defines __eq__ method.

        :return bool: True if input object is a DepositionalEnvironmentModel
            with the same name and same environments.
        """
        if not isinstance(other, DepositionalEnvironmentModel):
            return False

        # check model name equality
        if other.name != self.name:
            return False

        # check environment list equality based on environment names and
        # properties, regardless of order
        return set(other.environments) == set(self.environments)

    def addEnvironment(
        self: Self,
        environment: DepositionalEnvironment | set[DepositionalEnvironment],
    ) -> None:
        """Add an environment or set of environments to the model.

        If an environment with the same name already exists in the model, it
        is not added.

        :param DepositionalEnvironment|set[] environment: environment or set
            of environments to add
        """
        if isinstance(environment, DepositionalEnvironment):
            if self.environmentExists(environment.name):
                logger.warning(
                    "Environment with name '%s' already exists; cannot add "
                    "a duplicate.",
                    environment.name,
                )
                return
            self.environments.append(environment)
        elif isinstance(environment, set):
            for env in environment:
                self.addEnvironment(env)
        else:
            raise TypeError(
                "environment must be a DepositionalEnvironment or a set of "
                + "DepositionalEnvironment"
            )

    def environmentExists(self: Self, environmentName: str) -> bool:
        """Check if an environment exists in the collection by name.

        :param str environmentName: name of the environment to check
        :return bool: True if the environment exists in the collection
        """
        return any(
            env.name.lower() == environmentName.lower()
            for env in self.environments
        )

    def removeEnvironment(
        self: Self, environmentNames: str | set[str]
    ) -> None:
        """Remove an environment or set of environments from the list by name.

        :param str | set[str] environmentNames: name or set of names of
            environments to remove
        """
        if isinstance(environmentNames, str):
            for env in self.environments:
                if env.name.lower() == environmentNames.lower():
                    self.environments.remove(env)
                    break
        elif isinstance(environmentNames, set):
            for envName in environmentNames:
                self.removeEnvironment(envName)
        else:
            raise TypeError("environmentNames must be a str or a set of str")

    def getEnvironmentByName(
        self: Self, environmentName: str
    ) -> Optional[DepositionalEnvironment]:
        """Get environment from the collection by name.

        :param str environmentName: name of the environment to get
        :return DepositionalEnvironment | None: environment with the given
            name, or None if not found
        """
        for env in self.environments:
            if env.name.lower() == environmentName.lower():
                return env
        return None

    def clearAllEnvironments(self: Self) -> None:
        """Remove all environments from the collection."""
        self.environments.clear()

    def getEnvironmentCount(self: Self) -> int:
        """Get the number of environments in the collection.

        :return int: number of environments
        """
        return len(self.environments)

    def isEmpty(self: Self) -> bool:
        """Check if the collection is empty.

        :return bool: True if the collection is empty
        """
        return len(self.environments) == 0


class CarbonateOpenRampDepositionalEnvironmentModel(
    DepositionalEnvironmentModel
):
    def __init__(
        self: Self,
        tidal_range: float = 2.0,
        fairweather_wave_breaking_waterDepth: float = 5.0,
        fairweather_wave_base_waterDepth: float = 20.0,
        storm_wave_base_waterDepth: float = 50.0,
        shelf_break_waterDepth: float = 200.0,
        slope_toe_max_waterDepth: float = 1000.0,
    ) -> None:
        """Defines an open carbonate ramp depositional environment model.

        The open carbonate ramp depositional environment is characterized by a
        gently sloping ramp with no significant break in slope. The inner
        plateform zone is typically dominated by patch reefs and other
        buildups, but is not protected from wave energy by a barrier.
        The outer ramp is characterized by a lower energy.
        The model has a pre-defined list of environmnents, but waterDepth
        ranges are parameterized based on input parameters.
        The list of pre-defined environmnets includes:

        - Continent: terrestrial environment, above tidal limit.
        - SupraTidal: supratidal zone where carbonate/salt precipitation may
          occur.
        - Inner Ramp Upper Shoreface: 0 to fairweather wave-breaking depth,
          where energy is high
        - Inner Ramp Lower Shoreface: fairweather wave-breaking depth to
          fairweather wave-base where energy is lower than the shoreface zone
        - Buildup: patch reefs and other buildups creating locally low
          waterDepth () and high energy () environment.
        - Outer Ramp: fairweather wave-base to storm wave-base (offshore zone),
          where energy is low
        - Shelf Slope: Continental slope
        - Basin: Deep basin (intra-shelf or open ocean)

        Energy is given between 0.0 (no energy) and 1.0 (high energy).
        Distality is here given as the distance from the shoreline in km. The
        most significant is the relative distality between environments.

        :param float tidal_range: tidal range in meters (default 2 m).
        :param float fairweather_wave_breaking_waterDepth: fairweather
            wave-breaking depth (default 5 m).
        :param float fairweather_wave_base_waterDepth: fairweather
            wave-base depth (default 20 m).
        :param float storm_wave_base_waterDepth: storm wave-base depth
            (default 50 m).
        :param float shelf_break_waterDepth: shelf-break depth (default 200 m).
        :param float slope_toe_max_waterDepth: base of the slope maximum
            waterDepth (default 1000 m).
        """
        name = "Carbonate Open Ramp"
        environments = [
            DepositionalEnvironment(
                name="Continent",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", -np.inf, -tidal_range
                ),
                distality=-2.0,
            ),
            DepositionalEnvironment(
                name="SupraTidal",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", -tidal_range, 0.0
                ),
                distality=-1.0,
            ),
            DepositionalEnvironment(
                name="InnerRampUpperShoreface",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, fairweather_wave_breaking_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.5, 1.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 25.0, 30.0
                        ),
                    ]
                ),
                distality=0.0,
            ),
            DepositionalEnvironment(
                name="InnerRampLowerShoreface",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    fairweather_wave_breaking_waterDepth,
                    fairweather_wave_base_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.2, 0.5),
                        EnvironmentConditionModelUniform(
                            "temperature", 15.0, 25.0
                        ),
                    ]
                ),
                distality=0.5,
            ),
            DepositionalEnvironment(
                name="Buildup",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, storm_wave_base_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.7, 1.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 25.0, 30.0
                        ),
                    ]
                ),
                distality=0.1,  # on inner ramp but more distal than shoreline
            ),
            DepositionalEnvironment(
                name="OuterRamp",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    fairweather_wave_base_waterDepth,
                    storm_wave_base_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.1),
                        EnvironmentConditionModelUniform(
                            "temperature", 10.0, 15.0
                        ),
                    ]
                ),
                distality=2.0,
            ),
            DepositionalEnvironment(
                name="ShelfSlope",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    shelf_break_waterDepth,
                    slope_toe_max_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 4.0, 10.0
                        ),
                    ]
                ),
                distality=100.0,
            ),
            DepositionalEnvironment(
                name="Basin",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", slope_toe_max_waterDepth, 10000.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 4.0, 6.0
                        ),
                    ]
                ),
                distality=200.0,
            ),
        ]
        super().__init__(name, environments)


class CarbonateProtectedRampDepositionalEnvironmentModel(
    DepositionalEnvironmentModel
):
    def __init__(
        self: Self,
        tidal_range: float = 2.0,
        lagoon_max_waterDepth: float = 10.0,
        fairweather_wave_base_waterDepth: float = 20.0,
        storm_wave_base_waterDepth: float = 50.0,
        shelf_break_waterDepth: float = 200.0,
        slope_toe_max_waterDepth: float = 1000.0,
    ) -> None:
        """Defines a carbonate ramp depositional environment model.

        The model is defined as a list of depositional environments. The
        model has a pre-defined list of environmnents, but waterDepth ranges
        are parameterized based on input parameters.
        The list of pre-defined environmnets includes:

        - Continent: terrestrial environment, above tidal limit.
        - SupraTidal: supratidal zone where carbonate/salt precipitation may
          occur.
        - Inner Ramp Upper Shoreface: 0 to fairweather wave-breaking depth,
          where energy is high
        - Inner Ramp Lower Shoreface: fairweather wave-breaking depth to
          fairweather wave-base where energy is lower than the shoreface zone
        - Buildup: patch reefs and other buildups creating locally low
          waterDepth () and high energy () environment.
        - Outer Ramp: fairweather wave-base to storm wave-base (offshore zone),
          where energy is low
        - Shelf Slope: Continental slope
        - Basin: Deep basin (intra-shelf or open ocean)

        :param float tidal_range: tidal range in meters (default 2 m).
        :param float lagoon_max_waterDepth: maximum depth of the lagoon
            (default 10 m).

        :param float fairweather_wave_base_waterDepth: fairweather
            wave-base depth (default 20 m).
        :param float storm_wave_base_waterDepth: storm wave-base depth
            (default 50 m).
        :param float shelf_break_waterDepth: shelf-break depth
            (default 200 m).
        :param float slope_toe_max_waterDepth: base of the slope maximum
            waterDepth (default 1000 m).
        """
        name = "Carbonate Protected Ramp"
        environments = [
            DepositionalEnvironment(
                name="Continent",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", -10000, -tidal_range
                ),
                distality=-2.0,
            ),
            DepositionalEnvironment(
                name="SupraTidal",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", -tidal_range, 0.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform(
                            "salinity", 0.5, 1.0
                        ),  # hypersaline conditions, no unit
                    ]
                ),
                distality=-1.0,
            ),
            DepositionalEnvironment(
                name="Shore",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, 2.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.1, 0.5),
                        EnvironmentConditionModelUniform(
                            "temperature", 20.0, 30.0
                        ),
                    ]
                ),
                distality=0.0,
            ),
            DepositionalEnvironment(
                # deepest part of the lagoon
                name="Lagoon",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 2.0, lagoon_max_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.1),
                        EnvironmentConditionModelUniform(
                            "temperature", 20.0, 30.0
                        ),
                    ]
                ),
                distality=0.01,
            ),
            DepositionalEnvironment(
                name="Buildup",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, lagoon_max_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.5),
                        EnvironmentConditionModelUniform(
                            "temperature", 25.0, 30.0
                        ),
                    ]
                ),
                distality=0.01,
            ),
            DepositionalEnvironment(
                name="BackReef",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 1.0, 2.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.1, 0.2),
                        EnvironmentConditionModelUniform(
                            "temperature", 20.0, 30.0
                        ),
                    ]
                ),
                distality=0.4,
            ),
            DepositionalEnvironment(
                name="ReefCrest",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, 1.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.7, 1.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 20.0, 30.0
                        ),
                    ]
                ),
                distality=0.5,
            ),
            DepositionalEnvironment(
                name="ForeReef",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 1.0, fairweather_wave_base_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.2, 0.7),
                        EnvironmentConditionModelUniform(
                            "temperature", 15.0, 20.0
                        ),
                    ]
                ),
                distality=0.6,
            ),
            DepositionalEnvironment(
                name="OuterRamp",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    fairweather_wave_base_waterDepth,
                    storm_wave_base_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.2),
                        EnvironmentConditionModelUniform(
                            "temperature", 10.0, 15.0
                        ),
                    ]
                ),
                distality=2.0,
            ),
            DepositionalEnvironment(
                name="ShelfSlope",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    shelf_break_waterDepth,
                    slope_toe_max_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 4.0, 10.0
                        ),
                    ]
                ),
                distality=100.0,
            ),
            DepositionalEnvironment(
                name="Basin",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", slope_toe_max_waterDepth, 10000.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.0),
                        EnvironmentConditionModelUniform(
                            "temperature", 4.0, 6.0
                        ),
                    ]
                ),
                distality=200.0,
            ),
        ]
        super().__init__(name, environments)
