# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

import math
from enum import StrEnum

# ------------------------------------------------------------------
# Interval-distance helper
# ------------------------------------------------------------------


class IntervalDistanceMethod(StrEnum):
    GAP = "gap"
    CENTER = "center"
    HAUSDORFF = "hausdorff"
    WASSERSTEIN2 = "wasserstein2"
    GAP_TIMES_CENTER = "gap_times_center"
    GAP_OVERLAPPING_WIDTH = "gap_overlapping_width"


def gap_times_center_distance(
    min1: float, max1: float, min2: float, max2: float
) -> float:
    r"""Compute the gap times center distance between two intervals.

    .. math::

        \delta = \delta_{\text{gap}} \cdot \delta_{\text{center}}

    where :math:`\delta_{\text{gap}}` is the gap distance and
    :math:`\delta_{\text{center}}` is the center distance between the
    intervals.

    Returns 0 when the two intervals overlap.
    """
    d_gap = gap_distance(min1, max1, min2, max2)
    d_center = center_distance(min1, max1, min2, max2)
    return d_gap * d_center


def gap_overlapping_width_distance(
    min1: float, max1: float, min2: float, max2: float
) -> float:
    r"""Compute the distance between two intervals based on gap function.

    Considering two intervals :math:`[min_1, max_1]` and :math:`[min_2, max_2]`
    if they do not overlap, the distance is defined as the gap distance + 1.
    If they overlap, the distance is defined as 1 minus the overlapping width
    divided by the width of the union of the two intervals such as it is
    comprised between 0 (same intervals) and 1.

    Returns 0 when the two intervals are identical.
    """
    d_gap = gap_distance(min1, max1, min2, max2)
    if d_gap > 0.0:
        return 1.0 + d_gap

    # if min==max for one of the intervals,
    # then we consider a distance of 0 if gap is 0.
    if (max1 - min1 == 0.0 or max2 - min2 == 0.0) and d_gap == 0.0:
        return 0.0

    # otherwise, get overlapping fraction
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    overlap_width = max(0.0, overlap_max - overlap_min)
    if overlap_width == 0.0:
        return 1.0  # just touching
    # by default, first interval is the reference
    den = max1 - min1
    # but if one interval is fully contained in the other, use the width of the
    # containing interval as reference
    if min2 <= min1 and max2 >= max1:
        den = max2 - min2
    if den == 0.0:
        den = 1.0
    return 1.0 - overlap_width / den


def gap_distance(min1: float, max1: float, min2: float, max2: float) -> float:
    r"""Compute the gap distance between two intervals.

    .. math::

        \delta = \max(0, \text{min}_2 -\text{max}_1, \text{min}_1-\text{max}_2)

    Returns 0 when the two intervals overlap.
    """
    return max(0, min2 - max1, min1 - max2)


def center_distance(
    min1: float, max1: float, min2: float, max2: float
) -> float:
    r"""Compute the center distance between two intervals.

    .. math::

        \delta = \left|\frac{\text{min}_1 + \text{max}_1}{2} -
        \frac{\text{min}_2 + \text{max}_2}{2}\right|

    Returns 0 when the two intervals are identical.
    """
    return abs((max1 + min1) / 2.0 - (max2 + min2) / 2.0)


def hausdorff_distance(
    min1: float, max1: float, min2: float, max2: float
) -> float:
    r"""Compute the Hausdorff distance between two intervals.

    .. math::

        \delta = \max\left(
            |\text{min}_1 - \text{min}_2|,
            |\text{max}_1 - \text{max}_2|
        \right)

    Returns 0 when the two intervals are identical.
    """
    return max(abs(min1 - min2), abs(max1 - max2))


def wasserstein2_distance(
    min1: float, max1: float, min2: float, max2: float
) -> float:
    r"""Compute the Wasserstein-2 distance between two intervals.

    .. math::

        \delta = \max\left(
            |\text{min}_1 - \text{min}_2|,
            |\text{max}_1 - \text{max}_2|
        \right)

    Returns 0 when the two intervals are identical.
    """
    d1 = hausdorff_distance(min1, max1, min2, max2)
    width1 = max1 - min1
    width2 = max2 - min2
    d2 = width1 - width2
    return math.sqrt(d1**2 + 1.0 / 3.0 * d2**2)
