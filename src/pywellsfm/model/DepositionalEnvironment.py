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
        weight: float = 1.0,
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
        :param float weight: relative weight of the environment in the prior
            probabilities of the depositional environment simulator, e.g., to
            make an environment less frequent than others sharing the same
            water depth. Must be non-negative. Default is 1.0.
        """
        if weight < 0.0:
            raise ValueError("Environment weight must be non-negative.")
        self.name: str = name
        self.waterDepthModel: EnvironmentConditionModelStats = waterDepthModel
        # evolution of environment conditions
        self.envConditionsModel: EnvironmentConditionsModel = (
            EnvironmentConditionsModel()
            if envConditionsModel is None
            else envConditionsModel
        )
        self.distality: float | None = distality
        #: relative weight of the environment in prior probabilities
        self.weight: float = float(weight)

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
        fairweather_wave_base_waterDepth: float = 30.0,
        storm_wave_base_waterDepth: float = 40.0,
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
        - Outer Ramp: fairweather wave-base to shelf-break depth (offshore
          zone, including the zone below storm wave-base), where energy is
          low
        - Shelf Slope: Continental slope
        - Basin: Deep basin (intra-shelf or open ocean)

        Energy is given between 0.0 (no energy) and 1.0 (high energy).
        Distality is here given as the distance from the shoreline in km. The
        most significant is the relative distality between environments.

        :param float tidal_range: tidal range in meters (default 2 m).
        :param float fairweather_wave_breaking_waterDepth: fairweather
            wave-breaking depth (default 5 m).
        :param float fairweather_wave_base_waterDepth: fairweather
            wave-base depth (default 30 m).
        :param float storm_wave_base_waterDepth: storm wave-base depth
            (default 40 m).
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
                # extends down to the shelf break so that water depth ranges
                # are contiguous with the shelf slope
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    fairweather_wave_base_waterDepth,
                    shelf_break_waterDepth,
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
        reef_flat_max_waterDepth: float = 8.0,
        fairweather_wave_breaking_waterDepth: float = 5.0,
        fairweather_wave_base_waterDepth: float = 30.0,
        storm_wave_base_waterDepth: float = 40.0,
        shelf_break_waterDepth: float = 200.0,
        buildup_weight: float = 0.1,
    ) -> None:
        """Defines a protected (rimmed) carbonate platform model.

        The platform interior (tidal flat and lagoon) is protected from open
        sea waves by a reef margin. The model has a pre-defined list of
        environments whose water depth ranges are parameterized. From
        proximal to distal:

        - Continent: subaerial platform, above the tidal flat.
        - TidalFlat: intertidal zone (-tidal_range/2 to +tidal_range/2),
          low energy, restricted (high salinity).
        - Lagoon: restricted inner platform, from the low tide level to the
          lagoon maximum depth, very low energy, moderately restricted.
        - Buildup: patch reefs within the lagoon, with the same water depth
          range and distality as the lagoon but higher energy and normal
          salinity. Its weight in the prior probabilities is lower than the
          other environments so that buildups remain occasional.
        - BackReef: open inner platform behind the margin (e.g., rudist
          shoals), from sea level to the lagoon maximum depth, moderate
          energy.
        - ReefFlat: reef margin (e.g., coral-rudist buildups), from sea level
          to the reef flat maximum depth, high energy, open marine.
        - ForeReef: reef front and upper slope, from the fair-weather
          wave-breaking depth to the storm wave base, moderate energy.
        - OuterPlatform: from the fair-weather wave base to the shelf break,
          low energy.
        - Basin: below the shelf break, no energy.

        Lagoon, buildup, back-reef and reef flat share the same water depths:
        they can only be told apart by their energy and salinity (which
        control the accumulation of environment-specific elements) and by
        their position along the platform profile (distality), which the
        transition and trend likelihoods of the depositional environment
        simulator use.
        Fore-reef and outer platform overlap between the fair-weather wave
        base and the storm wave base.

        Energy and salinity (restriction) are given between 0.0 and 1.0.
        Distality is a rank along the platform profile; only the relative
        distality between environments matters.

        :param float tidal_range: tidal range in meters (default 2 m).
        :param float lagoon_max_waterDepth: maximum depth of the lagoon and
            back-reef (default 10 m).
        :param float reef_flat_max_waterDepth: maximum depth of the reef flat
            (default 8 m).
        :param float fairweather_wave_breaking_waterDepth: fair-weather
            wave-breaking depth, top of the fore-reef (default 5 m).
        :param float fairweather_wave_base_waterDepth: fair-weather wave-base
            depth, top of the outer platform (default 30 m).
        :param float storm_wave_base_waterDepth: storm wave-base depth, base
            of the fore-reef (default 40 m).
        :param float shelf_break_waterDepth: shelf-break depth, base of the
            outer platform and top of the basin (default 200 m).
        :param float buildup_weight: weight of the buildup environment in the
            prior probabilities of the depositional environment simulator,
            relative to the weight of other environments (1.0) (default 0.1).
        """
        name = "Carbonate Protected Ramp"
        low_tide = 0.5 * tidal_range
        environments = [
            DepositionalEnvironment(
                name="Continent",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", -10000.0, -low_tide
                ),
                distality=-1.0,
            ),
            DepositionalEnvironment(
                name="TidalFlat",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", -low_tide, low_tide
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.1, 0.3),
                        EnvironmentConditionModelUniform("salinity", 0.6, 1.0),
                    ]
                ),
                distality=0.0,
            ),
            DepositionalEnvironment(
                name="Lagoon",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", low_tide, lagoon_max_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.2),
                        EnvironmentConditionModelUniform("salinity", 0.3, 0.7),
                    ]
                ),
                distality=1.0,
            ),
            DepositionalEnvironment(
                name="Buildup",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", low_tide, lagoon_max_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.4, 0.8),
                        EnvironmentConditionModelUniform("salinity", 0.1, 0.3),
                    ]
                ),
                distality=1.0,  # same as the lagoon
                weight=buildup_weight,
            ),
            DepositionalEnvironment(
                name="BackReef",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, lagoon_max_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.3, 0.6),
                        EnvironmentConditionModelUniform("salinity", 0.1, 0.3),
                    ]
                ),
                distality=2.0,
            ),
            DepositionalEnvironment(
                name="ReefFlat",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", 0.0, reef_flat_max_waterDepth
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.7, 1.0),
                        EnvironmentConditionModelUniform("salinity", 0.0, 0.1),
                    ]
                ),
                distality=3.0,
            ),
            DepositionalEnvironment(
                name="ForeReef",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    fairweather_wave_breaking_waterDepth,
                    storm_wave_base_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.3, 0.6),
                        EnvironmentConditionModelUniform("salinity", 0.0, 0.1),
                    ]
                ),
                distality=4.0,
            ),
            DepositionalEnvironment(
                name="OuterPlatform",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth",
                    fairweather_wave_base_waterDepth,
                    shelf_break_waterDepth,
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.2),
                        EnvironmentConditionModelUniform("salinity", 0.0, 0.1),
                    ]
                ),
                distality=5.0,
            ),
            DepositionalEnvironment(
                name="Basin",
                waterDepthModel=EnvironmentConditionModelUniform(
                    "waterDepth", shelf_break_waterDepth, 10000.0
                ),
                envConditionsModel=EnvironmentConditionsModel(
                    [
                        EnvironmentConditionModelUniform("energy", 0.0, 0.0),
                        EnvironmentConditionModelUniform("salinity", 0.0, 0.1),
                    ]
                ),
                distality=6.0,
            ),
        ]
        super().__init__(name, environments)
