# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Optional, Self, cast

import numpy as np
from striplog import Interval, Striplog

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
          - A scenario defines global parameters for the simulation shared by multiple
          realizations, including eustatic curve, facies model, accumulation model.
          - A realization corresponds to a well location with its specific conditions
          such as the well log and the subsidence curve.

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
        self.accumulationSimulator.setAccumulationModel(self.scenario.accumulationModel)
        self.accumulationSimulator.prepare()

        # update simulators with models
        self.accommodationSimulator.setSubsidenceCurve(
            self.realizationData.subsidenceCurve
        )
        # set initial bathymetry from facies at well
        initialBathy = self.getInitialBathymetry()
        self.accommodationSimulator.setInitialBathymetry(initialBathy)

        if self.scenario.eustaticCurve is not None:
            # set the eustatic curve if defined, otherwise uses a flat curve
            # (no eustatism variations)
            self.accommodationSimulator.setEustaticCurve(self.scenario.eustaticCurve)
        self.accommodationSimulator.prepare()

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
        """Get the accumulation rate of a given element for the current realization.

        :param float age: age at which to get the total accumulation rate.
        :param str elementName: name of the element to get the accumulation rate for.
        :return float: accumulation rate of the given element.
        """
        return self.accumulationSimulator.computeElementAccumulationRate(
            elementName, envCond
        )

    def getTotalAccumulationRate(self: Self, envCond: dict[str, float]) -> float:
        """Get the total accumulation rate for the current realization.

        :param float age: age at which to get the total accumulation rate.
        :return float: total accumulation rate.
        """
        return self.accumulationSimulator.computeTotalAccumulationRate(envCond)

    # TODO: set accommodation parameters at age?
    # TODO: set accumulation parameters at age?

    # def run(self: Self, markerEnd: Optional[Marker]) -> None:
    #     """Run the simulation until a given marker or to the top.

    #     :param Optional[Marker] markerEnd: marker until which to run the
    #         simulation. If None, the simulation runs to the top of wells.
    #     """
    #     # simulated ages
    #     # get end age from marker
    #     # TODO: implement single-realization forward simulation
    #     # This method is for running a single FSSimulator instance.
    #     # For ensemble runs, use FSSimulatorRunner instead.

    #     # overall deposition rate
    #     # depoRate = np.zeros_like(ages)
    #     # rate per element

    #     # thickness

    #     # log per depth (element proportions, main element)

    #     # log per time (subsidence, sea level, cumul deposited thickness,
    #     # deposition rates)

    #     # depth-age model

    #     # TODO: input start and end ages, but adapt time step such as
    #     # overall deposition thickness < 1m per time step
    #     # for i, age in enumerate(ages):
    #     #     # compute and store environment conditions
    #     #     envCond = {}
    #     #     elementsAccumulationRates = (
    #     #         self.accumulationSimulator.computeElementAccumulationRate(
    #     #             envCond
    #     #         )
    #     #     )
    #     #     for elt in self.scenario.accumulationModel.elements:
    #     #         # TODO: store this elt prod rate
    #     #         depoRate[i] += elementsAccumulationRates[elt.name]
    #     pass

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

    def getInitialBathymetry(self: Self) -> float:
        """Get the initial bathymetry.

        Take the middle value of the range as initial bathymetry get from facies log.

        :return float: initial bathymetry value.
        """
        bathyRange: tuple[float, float] = self._getInitialBathymetryRange()
        return 0.5 * (bathyRange[0] + bathyRange[1])

    # Helper functions
    def _getInitialBathymetryRange(self: Self) -> tuple[float, float]:
        """Get the initial bathymetry range.

        The bathymetry is retreive from the facies log at and facies conditions.

        :return tuple[float, float]: min and max initial bathymetry values.
        """
        well = self.realizationData.well
        faciesLogNames: set[str] = well.getDiscreteLogNames()
        if len(faciesLogNames) == 0:
            raise ValueError(
                f"No discrete log found in well '{well.name}' "
                "to get initial bathymetry."
            )
        faciesLog: Striplog = cast(Striplog, well.getDepthLog(faciesLogNames.pop()))
        if faciesLog is None:
            raise ValueError(
                f"Facies log not found in well '{well.name}' to get initial bathymetry."
            )

        interval: Interval = cast(Interval, faciesLog[-1])
        faciesName: str = interval.primary["lithology"]  # type: ignore
        bathyRange: Optional[tuple[float, float]] = (
            self.scenario.faciesModel.getCriteriaRangeForFacies(
                faciesName, "Bathymetry"
            )
        )
        if bathyRange is None:
            raise ValueError(f"Bathymetry condition not found for facies {faciesName}.")
        return bathyRange
