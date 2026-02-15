# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from enum import StrEnum


class SubsidenceType(StrEnum):
    """Subsidence curve type."""

    #: cumulative subsidence curve, where values are cumulative subsidence
    #: at a given age
    CUMULATIVE = "cumulative"
    #: subsidence rate curve, where values are subsidence rate at a given
    #: age. Subsidence at a given age is then computed by multiplying the
    #: rate by the time interval
    RATE = "rate"
