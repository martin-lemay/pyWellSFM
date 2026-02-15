# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from dataclasses import dataclass
from typing import Optional

from .AccumulationModel import AccumulationModel
from .Curve import Curve
from .enums import SubsidenceType
from .Facies import FaciesModel
from .Well import Well


@dataclass(frozen=True)
class Scenario:
    """Defines a simulation scenario.

    A scenario defines all the parameters that are supposed to be uniform over
    a given area. It includes the accumulation model and the eustatic curve.

    :param str name: name of the scenario
    :param AccumulationModel accumulationModel: accumulation model used in
        the scenario. It defines the list of elements and their default
        accumulation rates.
    :param Curve | None eustaticCurve: Eustatic curve, defaults to None
    :param FaciesModel | None faciesModel: facies model used in the scenario.
        It defines the list of facies and their default proportions.
    """

    #: name of the scenario
    name: str
    #: accumulation model used in the scenario
    accumulationModel: AccumulationModel
    #: eustatic curve used in the scenario
    eustaticCurve: Optional[Curve]
    #: facies model used in the scenario
    faciesModel: Optional[FaciesModel] = None


@dataclass(frozen=True)
class RealizationData:
    """Data class for to define realization-specific parameters.

    :param Well well: well object
    :param float initialBathymetry: initial bathymetry for the realization.
    :param Curve subsidenceCurve: subsidence curve for the realization.
    """

    well: Well
    initialBathymetry: float
    subsidenceCurve: Optional[Curve]
    subsidenceType: SubsidenceType


@dataclass(frozen=True)
class SimulationData:
    scenario: Scenario
    realizationsData: list[RealizationData]
