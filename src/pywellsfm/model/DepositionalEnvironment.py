# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Any, Optional, Self

from .Curve import Curve


class DepositionalEnvironment:
    def __init__(
        self: Self,
        name: str,
        waterDepth_range: tuple[float, float],
        other_property_ranges: dict[str, tuple[float, float]] | None = None,
        distality: float | None = None,
    ) -> None:
        """Defines a depositional environment.

        The environment is defined from a waterDepth range and optionaly other
        property ranges including energy, temperature, salinity, etc.

        Curves can be set to define relationships between properties,
        e.g. energy vs waterDepth.

        .. NOTE::

            Properties can be related to a single other property,
            e.g. temperature can be defined as a function of waterDepth, but
            not as a function of both waterDepth and energy.

        :param str name: name of the environment
        :param tuple[float, float] waterDepth_range: waterDepth range of the
            environment
        :param dict[str, tuple[float, float]] other_property_ranges: ranges of
            any other properties of the environment
        :param float distality: distality of the environment, defined as the
            distance from the shoreline, in km.
        """
        self.name: str = name
        self.waterDepth_range: tuple[float, float] = (
            min(waterDepth_range),
            max(waterDepth_range),
        )
        self.other_property_ranges: dict[str, tuple[float, float]] = {}
        if other_property_ranges is not None:
            self.other_property_ranges = {
                k: (min(v), max(v)) for k, v in other_property_ranges.items()
            }
        # curves defining relationships between properties, e.g. energy vs
        # waterDepth
        self.property_curves: dict[str, Curve] = {}
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
                and other.other_property_ranges == self.other_property_ranges
            )
        return False

    @property
    def waterDepth_min(self: Self) -> float:
        """Minimum waterDepth of the environment."""
        return self.waterDepth_range[0]

    @property
    def waterDepth_max(self: Self) -> float:
        """Maximum waterDepth of the environment."""
        return self.waterDepth_range[1]

    @property
    def waterDepth_mid(self: Self) -> float:
        """Mid-point of the waterDepth range."""
        return (self.waterDepth_range[0] + self.waterDepth_range[1]) / 2.0

    @property
    def waterDepth_range_width(self: Self) -> float:
        """Width of the waterDepth range."""
        return self.waterDepth_range[1] - self.waterDepth_range[0]

    def getPropertyMid(self: Self, property_name: str) -> float:
        """Get the mid-point of a property range.

        :param str property_name: name of the property
        :return float: mid-point of the property range
        """
        if property_name not in self.other_property_ranges:
            raise ValueError(
                f"Property {property_name} not defined for environment "
                f"{self.name}."
            )
        prop_range = self.other_property_ranges[property_name]
        return (prop_range[0] + prop_range[1]) / 2.0

    def getPropertyRangeWidth(self: Self, property_name: str) -> float:
        """Get the width of a property range.

        :param str property_name: name of the property
        :return float: width of the property range
        """
        if property_name not in self.other_property_ranges:
            raise ValueError(
                f"Property {property_name} not defined for environment "
                f"{self.name}."
            )
        prop_range = self.other_property_ranges[property_name]
        return prop_range[1] - prop_range[0]

    def setOtherPropertyRange(
        self: Self, property_name: str, property_range: tuple[float, float]
    ) -> None:
        """Set the range of a property for the environment.

        :param str property_name: name of the property
        :param tuple[float, float] property_range: range of the property
        """
        if property_name in self.other_property_ranges:
            print(
                f"Warning: overwriting existing range for property "
                f"{property_name} in environment {self.name}."
            )
        self.other_property_ranges[property_name] = property_range

    def _resolve_name(self: Self, x_axis_name: str, y_axis_name: str) -> str:
        """Resolve curve name from axes names.

        :param str x_axis_name: name of the x-axis property
        :param str y_axis_name: name of the y-axis property
        :return str: curve name
        """
        return f"{y_axis_name}_vs_{x_axis_name}"

    def setPropertyCurve(self: Self, curve: Curve) -> None:
        """Set a curve for the environment.

        The curve allows to define a relationship between environmental
        properties. For example, a curve can be used to define the a function
        of the energy versus the depth.
        The curve is defined as a list of (property1, property2) points.

        .. WARNING::

            Curve axis names must match the keys defined in the
            other_property_ranges dictionary.

        :param Curve curve: curve defining the relationship between 2
            properties.
        """
        name = self._resolve_name(curve._xAxisName, curve._yAxisName)
        if name in self.property_curves:
            print(
                f"Warning: overwriting existing curve for properties "
                f"{curve._yAxisName} vs {curve._xAxisName} in environment "
                f"{self.name}."
            )

        if curve._xAxisName not in self.other_property_ranges:
            print(
                f"Warning: x-axis property {curve._xAxisName} not defined "
                + f"in other_property_ranges for environment {self.name}."
            )
        if curve._yAxisName not in self.other_property_ranges:
            print(
                f"Warning: y-axis property {curve._yAxisName} not defined "
                + f"in other_property_ranges for environment {self.name}."
            )
        self.property_curves[name] = curve

    def getValueFromCurveAt(
        self: Self, x_property_name: str, y_property_name: str, x_value: float
    ) -> float:
        """Get the value of a property at a given waterDepth.

        :param str x_property_name: name of the x-axis property
        :param str y_property_name: name of the y-axis property
        :param float x_value: value of the x-axis property at which to get the
            y-axis property value
        :return float: value of the y-axis property at the given x-axis
            property value
        """
        name = self._resolve_name(x_property_name, y_property_name)
        if name not in self.property_curves:
            raise ValueError(
                f"The curve {x_property_name} vs {y_property_name} is not "
                + f"defined for environment {self.name}."
            )
        if x_property_name not in self.other_property_ranges:
            raise ValueError(
                f"Property {x_property_name} not defined in "
                + f"other_property_ranges for environment {self.name}."
            )
        if y_property_name not in self.other_property_ranges:
            raise ValueError(
                f"Property {y_property_name} not defined in "
                + f"other_property_ranges for environment {self.name}."
            )
        y_range = self.other_property_ranges[y_property_name]
        y_width = self.getPropertyRangeWidth(y_property_name)
        if y_width == 0.0:
            return y_range[0]

        # linear interpolation between property range limits based on
        # waterDepth range limits
        x_width = self.getPropertyRangeWidth(x_property_name)
        if x_width == 0.0:
            # return mean value
            return self.getPropertyMid(y_property_name)

        value = (
            y_range[0]
            + y_width * (x_value - self.waterDepth_range[0]) / x_width
        )
        # clamp value to property range limits
        if value < y_range[0]:
            value = y_range[0]
        elif value > y_range[1]:
            value = y_range[1]
        return value

    def getCurveDependencies(self: Self) -> dict[str, str]:
        """Get the dependencies of the curves defined for the environment.

        :return dict[str, str]: mapping of curve name to a tuple of the x and
            y property names.
        """
        dependencies: dict[str, str] = {}
        for curve in self.property_curves.values():
            x_axis = curve._xAxisName
            y_axis = curve._yAxisName
            if y_axis not in self.other_property_ranges:
                raise ValueError(
                    f"Curve target property '{y_axis}' is not defined in "
                    f"other_property_ranges for environment {self.name}."
                )
            if (
                x_axis != "waterDepth"
                and x_axis not in self.other_property_ranges
            ):
                raise ValueError(
                    f"Curve source property '{x_axis}' is not defined in "
                    f"other_property_ranges for environment {self.name}."
                )
            if y_axis in dependencies and dependencies[y_axis] != x_axis:
                raise ValueError(
                    f"Property '{y_axis}' has multiple source dependencies "
                    f"({dependencies[y_axis]!r}, {x_axis!r}) in environment "
                    f"{self.name}."
                )
            dependencies[y_axis] = x_axis
        return dependencies


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
        return set(other.environments) != set(self.environments)

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
                print(
                    f"Environment with name '{environment.name}' already "
                    f"exists, cannot add a duplicate."
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
        return any(env.name == environmentName for env in self.environments)

    def removeEnvironment(
        self: Self, environmentNames: str | set[str]
    ) -> None:
        """Remove an environment or set of environments from the list by name.

        :param str | set[str] environmentNames: name or set of names of
            environments to remove
        """
        if isinstance(environmentNames, str):
            for env in self.environments:
                if env.name == environmentNames:
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
            if env.name == environmentName:
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

        - Sabkha: supratidal zone where carbonate/salt precipitation may occur.
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
                name="Sabkha",
                waterDepth_range=(-tidal_range, 0.0),
                distality=-1.0,
            ),
            DepositionalEnvironment(
                name="InnerRampUpperShoreface",
                waterDepth_range=(0.0, fairweather_wave_breaking_waterDepth),
                other_property_ranges={
                    "energy": (0.5, 1.0),
                    "temperature": (25.0, 30.0),
                },
                distality=0.0,
            ),
            DepositionalEnvironment(
                name="InnerRampLowerShoreface",
                waterDepth_range=(
                    fairweather_wave_breaking_waterDepth,
                    fairweather_wave_base_waterDepth,
                ),
                other_property_ranges={
                    "energy": (0.2, 0.5),
                    "temperature": (15.0, 25.0),
                },
                distality=0.5,
            ),
            DepositionalEnvironment(
                name="Buildup",
                waterDepth_range=(0.0, storm_wave_base_waterDepth),
                other_property_ranges={
                    "energy": (0.7, 1.0),
                    "temperature": (25.0, 30.0),
                },
                distality=0.0,
            ),
            DepositionalEnvironment(
                name="OuterRamp",
                waterDepth_range=(
                    fairweather_wave_base_waterDepth,
                    storm_wave_base_waterDepth,
                ),
                other_property_ranges={
                    "energy": (0.0, 0.1),
                    "temperature": (10.0, 15.0),
                },
                distality=2.0,
            ),
            DepositionalEnvironment(
                name="ShelfSlope",
                waterDepth_range=(
                    shelf_break_waterDepth,
                    slope_toe_max_waterDepth,
                ),
                other_property_ranges={
                    "energy": (0.0, 0.0),
                    "temperature": (4.0, 10.0),
                },
                distality=100.0,
            ),
            DepositionalEnvironment(
                name="Basin",
                waterDepth_range=(slope_toe_max_waterDepth, 10000.0),
                other_property_ranges={
                    "energy": (0.0, 0.0),
                    "temperature": (4.0, 6.0),
                },
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

        - Sabkha: supratidal zone where carbonate/salt precipitation may occur.
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
                name="Sabkha",
                waterDepth_range=(-tidal_range, 0.0),
                other_property_ranges={
                    "salinity": (0.5, 1.0),  # hypersaline conditions, no unit
                },
                distality=-1.0,
            ),
            DepositionalEnvironment(
                name="Shore",
                waterDepth_range=(0.0, 2.0),
                other_property_ranges={
                    "energy": (0.1, 0.5),
                    "temperature": (20.0, 30.0),
                },
                distality=0.0,
            ),
            DepositionalEnvironment(
                # deepest part of the lagoon
                name="Lagoon",
                waterDepth_range=(2.0, lagoon_max_waterDepth),
                other_property_ranges={
                    "energy": (0.0, 0.1),
                    "temperature": (20.0, 30.0),
                },
                distality=0.01,
            ),
            DepositionalEnvironment(
                name="Buildup",
                waterDepth_range=(0.0, lagoon_max_waterDepth),
                other_property_ranges={
                    "energy": (0.0, 0.5),
                    "temperature": (25.0, 30.0),
                },
                distality=0.0,
            ),
            DepositionalEnvironment(
                name="BackReef",
                waterDepth_range=(1.0, 2.0),
                other_property_ranges={
                    "energy": (0.1, 0.2),
                    "temperature": (20.0, 30.0),
                },
                distality=0.4,
            ),
            DepositionalEnvironment(
                name="ReefCrest",
                waterDepth_range=(0.0, 1.0),
                other_property_ranges={
                    "energy": (0.7, 1.0),
                    "temperature": (20.0, 30.0),
                },
                distality=0.5,
            ),
            DepositionalEnvironment(
                name="ForeReef",
                waterDepth_range=(1.0, fairweather_wave_base_waterDepth),
                other_property_ranges={
                    "energy": (0.2, 0.7),
                    "temperature": (15.0, 20.0),
                },
                distality=0.6,
            ),
            DepositionalEnvironment(
                name="OuterRamp",
                waterDepth_range=(
                    fairweather_wave_base_waterDepth,
                    storm_wave_base_waterDepth,
                ),
                other_property_ranges={
                    "energy": (0.0, 0.2),
                    "temperature": (10.0, 15.0),
                },
                distality=2.0,
            ),
            DepositionalEnvironment(
                name="ShelfSlope",
                waterDepth_range=(
                    shelf_break_waterDepth,
                    slope_toe_max_waterDepth,
                ),
                other_property_ranges={
                    "energy": (0.0, 0.0),
                    "temperature": (4.0, 10.0),
                },
                distality=100.0,
            ),
            DepositionalEnvironment(
                name="Basin",
                waterDepth_range=(slope_toe_max_waterDepth, 10000.0),
                other_property_ranges={
                    "energy": (0.0, 0.0),
                    "temperature": (4.0, 6.0),
                },
                distality=200.0,
            ),
        ]
        super().__init__(name, environments)
