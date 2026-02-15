# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Optional, Self

import numpy as np

from pywellsfm.model.enums import SubsidenceType
from pywellsfm.model.SimulationParameters import RealizationData, Scenario
from pywellsfm.model.Well import Well
from pywellsfm.simulator.AccommodationSimulator import AccommodationSimulator
from pywellsfm.simulator.AccumulationSimulator import AccumulationSimulator


class FSSimulator:
    def __init__(
        self: Self,
        scenario: Scenario,
        realizationData: RealizationData,
    ) -> None:
        """Defines the Forward Stratigraphic Simulator.

        The FS simulator parameters include a scenario and realization data:

          - A scenario defines global parameters for the simulation shared by
            multiple realizations, including eustatic curve and the
            accumulation model.
          - A realization corresponds to a well location with its specific
            conditions such as the well log and the subsidence curve.

        :param Scenario scenario: name of the scenario to simulate
        :param RealizationData realizationData: realization data to use.
        """
        self.scenario: Scenario = scenario
        self.realizationData: RealizationData = realizationData
        self.simulatedWell: Optional[Well] = None

        # initialize simulators
        # one single accummulation simulator applicable to all realizations
        self.accumulationSimulator = AccumulationSimulator()
        # one accommodation simulator for each realization
        self.accommodationSimulator = AccommodationSimulator()

    def prepare(self: Self) -> None:
        """Prepare the simulation run."""
        # reset simulated well
        del self.simulatedWell
        # shallow copy the well from realization data
        prefix = "Sim_" + self.scenario.name
        self.simulatedWell = self.realizationData.well.shallowCopy(
            f"{prefix}_{self.realizationData.well.name}",
            copyMarkers=True,
            copyLogs=False,
        )

        # set accummulation model global to all realizations
        self.accumulationSimulator.setAccumulationModel(
            self.scenario.accumulationModel
        )
        self.accumulationSimulator.prepare()

        # update simulators with models
        self.accommodationSimulator.setSubsidenceCurve(
            self.realizationData.subsidenceCurve
        )
        # set initial bathymetry
        self.accommodationSimulator.setInitialBathymetry(
            self.realizationData.initialBathymetry
        )

        if self.scenario.eustaticCurve is not None:
            # set the eustatic curve if defined, otherwise uses a flat curve
            # (no eustatism variations)
            self.accommodationSimulator.setEustaticCurve(
                self.scenario.eustaticCurve
            )
        self.accommodationSimulator.prepare()

    def getSubsidenceType(self: Self) -> SubsidenceType:
        """Get the subsidence type used by the simulator.

        :return SubsidenceType: subsidence type used by the simulator.
        """
        return self.realizationData.subsidenceType

    def getSubsidenceAtAge(self: Self, age: float) -> float:
        """Get the subsidence at a given age for the current realization.

        :param float age: age at which to get the subsidence.
        :return float: subsidence at the given age.
        """
        return self.accommodationSimulator.getSubsidenceAt(age)

    def getSeaLevelAtAge(self: Self, age: float) -> float:
        """Get the sea level at a given age for the current realization.

        :param float age: age at which to get the sea level.
        :return float: sea level at the given age.
        """
        return self.accommodationSimulator.getSeaLevelAt(age)

    def getElementAccumulationRate(
        self: Self, envCond: dict[str, float], elementName: str
    ) -> float:
        """Get the accumulation rate of a given element.

        :param float age: age at which to get the total accumulation rate.
        :param str elementName: name of the element to get the accumulation
            rate for.
        :return float: accumulation rate of the given element.
        """
        return self.accumulationSimulator.computeElementAccumulationRate(
            elementName, envCond
        )

    def getTotalAccumulationRate(
        self: Self, envCond: dict[str, float]
    ) -> float:
        """Get the total accumulation rate for the current realization.

        :param float age: age at which to get the total accumulation rate.
        :return float: total accumulation rate.
        """
        return self.accumulationSimulator.computeTotalAccumulationRate(envCond)

    def finalize(
        self: Self,
    ) -> None:
        """Finalize the simulation run.

        Create the simulated well logs.
        """
        # TODO: create simulated well log
        pass

    def getFirstMarkerAge(self: Self) -> float:
        """Get the first (oldest) marker age.

        :return float: last marker age
        """
        ageMax: float = -np.inf
        for marker in self.realizationData.well.getMarkers():
            if marker.age > ageMax:
                ageMax = marker.age
        return ageMax

    def getLastMarkerAge(self: Self) -> float:
        """Get the last (youngest) marker age.

        :return float: last marker age
        """
        ageMin: float = np.inf
        for marker in self.realizationData.well.getMarkers():
            if marker.age < ageMin:
                ageMin = marker.age
        return ageMin
