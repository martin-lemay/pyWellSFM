# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from dataclasses import dataclass
from typing import Self

import numpy as np

from pywellsfm.model.Curve import Curve


@dataclass(frozen=True)
class AccommodationStorage:
    """Storage class for accommodation computation results.

    Elevation datum is given by sea level at the start of the simulation.

    :param float topography: topography value
    :param float seaLevel: sea level value
    :param float accommodation: accommodation value
    """

    topography: float
    seaLevel: float
    accommodation: float


class AccommodationSimulator:
    def __init__(self: Self) -> None:
        """Simulate the accommodation space in wells based on a scenario.

        The simulator uses the eustatism at the start of the simulation as the
        datum. All elevations are computed relatively to this datum.

        At the start of the simulation, we have:
          - initial sea level elevation is 0
          - initial topography is -initial bathymetry
          - initial accommodation is initial bathymetry

        At a given time t, we have:
          - sea level(t) = eustatism(t) - eustatism(0)
          - topography(t) = initial topography + subsidence(t)
          - accommodation(t) = sea level + subsidence(t)
        """
        #: subsidence curve used by the simulator
        self.subsidenceCurve: Curve | None = None
        #: eustatic curve used by the simulator
        self.eustaticCurve: Curve | None = None
        #: Eustatism at the start of the simulation
        self.eustatismStart: float = 0.0
        #: initial topography value at the start of the simulation
        self.topographyStart: float = 0.0

    def setSubsidenceCurve(self: Self, subsidenceCurve: Curve | None) -> None:
        """Set the subsidence curve used by the simulator.

        :param Curve | None subsidenceCurve: subsidence curve to set.
        """
        self.subsidenceCurve = subsidenceCurve

    def setEustaticCurve(self: Self, eustaticCurve: Curve) -> None:
        """Set the eustatic curve used by the simulator.

        :param Curve eustaticCurve: eustatic curve to set.
        """
        self.eustaticCurve = eustaticCurve

    def setInitialBathymetry(self: Self, bathymetry: float) -> None:
        """Set the initial bathymetry used by the simulator.

        :param float bathymetry: initial bathymetry to set.
        """
        # considers that sea level at start is 0, so topography is -bathymetry
        self.topographyStart = -1.0 * bathymetry

    def prepare(self: Self) -> None:
        """Prepare the simulator for computations."""
        if self.subsidenceCurve is None:
            # if no subsidence curve, consider no variations
            self.subsidenceCurve = Curve(
                "Time", "Subsidence", np.array([0, 1]), np.array([0.0, 0.0])
            )
            print("Subsidence curve not set. There is no subsidence.")

        if self.eustaticCurve is None:
            # if no eustatic curve, consider no variations
            self.eustaticCurve = Curve(
                "Time", "Eustacy", np.array([0, 1]), np.array([0.0, 0.0])
            )
            print("Eustatic curve not set. There is no sea level variation.")

        self.eustatismStart = self.eustaticCurve(0.0)

    def getEustatismAt(self: Self, age: float) -> float:
        """Get the eustatism value at a given age.

        :param float age: age to get the eustatism for.
        :return float: eustatism value at the given age.
        """
        if self.eustaticCurve is None:
            raise ValueError("Eustatic curve is not set.")
        return self.eustaticCurve(age)

    def getSubsidenceAt(self: Self, age: float) -> float:
        """Get the subsidence value at a given age.

        :param float age: age to get the subsidence for.
        :return float: subsidence value at the given age.
        """
        if self.subsidenceCurve is None:
            raise ValueError("Subsidence curve is not set.")
        return self.subsidenceCurve(age)

    def getSeaLevelAt(self: Self, age: float) -> float:
        """Get the sea level value at a given age.

        :param float age: age to get the sea level for.
        :return float: sea level value at the given age.
        """
        if self.eustaticCurve is None:
            raise ValueError("Eustatic curve is not set.")
        return self.eustaticCurve(age) - self.eustatismStart

    def getBasementElevationAt(self: Self, age: float) -> float:
        """Get the basement elevation at a given age.

        :param float age: age to get the basement elevation for.
        :return float: basement elevation at the given age.
        """
        if self.subsidenceCurve is None:
            raise ValueError("Subsidence curve is not set.")
        subsidence = self.subsidenceCurve(age)
        return self.topographyStart + subsidence

    def getAccommodationAt(self: Self, age: float) -> float:
        """Get the cumulative accommodation at a given age from the start.

        :param float age: age to compute the accommodation for.
        :return: accommodation at the given age.
        """
        if self.subsidenceCurve is None:
            raise ValueError("Subsidence curve is not set.")
        if self.eustaticCurve is None:
            raise ValueError("Eustatic curve is not set.")

        subsidence = self.subsidenceCurve(age)
        seaLevel = self.getSeaLevelAt(age)
        accommodation = seaLevel + subsidence
        return accommodation
