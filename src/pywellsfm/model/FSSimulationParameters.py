# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from dataclasses import dataclass
from typing import Optional

from .AccumulationModel import AccumulationModel
from .Curve import Curve
from .DepositionalEnvironment import (
    DepositionalEnvironmentModel,
)
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
    :param DepositionalEnvironmentModel depositionalEnvironmentModel:
        depositional environment model used in the scenario. It defines the
        list of depositional environments and their corresponding environmental
        conditions.
    :param FaciesModel | None faciesModel: facies model used in the scenario.
        It defines the list of facies and their default proportions.
    """

    #: name of the scenario
    name: str
    #: accumulation model used in the scenario
    accumulationModel: AccumulationModel
    #: eustatic curve used in the scenario
    eustaticCurve: Optional[Curve]
    #: depositional environment model used in the scenario
    depositionalEnvironmentModel: Optional[DepositionalEnvironmentModel] = None
    #: facies model used in the scenario
    faciesModel: Optional[FaciesModel] = None


@dataclass(frozen=True)
class RealizationData:
    """Data class for to define realization-specific parameters.

    :param Well well: well object
    :param float initialBathymetry: initial bathymetry for the realization.
    :param Curve | None subsidenceCurve: subsidence curve for the realization.
    :param SubsidenceType subsidenceType: subsidence type for the realization.
    :param DepositionalEnvironment | None initialEnvironment: initial
        depositional environment for the realization.
    """

    well: Well
    initialBathymetry: float
    initialEnvironmentName: Optional[str]
    subsidenceCurve: Optional[Curve]
    subsidenceType: SubsidenceType
