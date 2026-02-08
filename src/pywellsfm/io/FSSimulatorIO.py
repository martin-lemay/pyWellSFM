# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Self

from pywellsfm.io import (
    loadAccumulationModelGaussianFromJson,
    loadEustaticCurve,
    loadFaciesModel,
    loadSubsidenceCurve,
    loadWell,
)
from pywellsfm.model.SimulationParameters import RealizationData, Scenario
from pywellsfm.model.Well import Well


class FSSimulatorIO:
    def __init__(self: Self) -> None:
        """Defines the Forward Stratigraphic Simulator I/O handler."""
        pass

    def createScenario(
        self: Self,
        scenarioName: str,
        accumulationModelFilepath: str,
        faciesModelFilepath: str,
        eustaticCurveFilepath: str,
    ) -> Scenario:
        """Create Scenario object from files.

        :param str scenarioName: name of the scenario
        :param str accumulationModelFilepath: path to accumulation model file
        :param str faciesModelFilepath: path to facies model file
        :param str eustaticCurveFilepath: path to eustatic curve file
        :return Scenario: scenario object
        """
        faciesModel = loadFaciesModel(faciesModelFilepath)
        accumulationModel = loadAccumulationModelGaussianFromJson(
            accumulationModelFilepath
        )
        eustaticCurve = loadEustaticCurve(eustaticCurveFilepath)
        return Scenario(
            name=scenarioName,
            accumulationModel=accumulationModel,
            eustaticCurve=eustaticCurve,
            faciesModel=faciesModel,
        )

    def createRealizationData(
        self: Self,
        wellFilepath: str,
        subsidenceCurveFilepath: str,
    ) -> RealizationData:
        """Create RealizationData object from files.

        :param str wellFilepath: path to well file
        :param str subsidenceCurveFilepath: path to subsidence curve file
        :return RealizationData: realization data object
        """
        well = loadWell(wellFilepath)
        subsidenceCurve = loadSubsidenceCurve(subsidenceCurveFilepath)
        return RealizationData(
            well=well,
            subsidenceCurve=subsidenceCurve,
        )

    def saveSimulatedWell(self: Self, well: Well, filepath: str) -> None:
        """Save simulated well to file.

        :param Well well: simulated well object
        :param str filepath: path to output well file
        """
        pass
