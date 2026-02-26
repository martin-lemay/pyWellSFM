# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

import math
import os
import sys
from typing import Self

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)

import pytest

from pywellsfm.utils.geometry import (
    center_distance,
    gap_distance,
    gap_overlapping_width_distance,
    gap_times_center_distance,
    hausdorff_distance,
    wasserstein2_distance,
)


class TestGapDistance:
    def test_gap_distance_overlap_is_zero(self: Self) -> None:
        """Return zero when two intervals overlap."""
        d = gap_distance(0.0, 10.0, 5.0, 20.0)
        assert d == 0.0

    def test_gap_distance_disjoint(self: Self) -> None:
        """Return the gap length when intervals are disjoint."""
        d = gap_distance(0.0, 10.0, 20.0, 30.0)
        assert d == 10.0


class TestCenterDistance:
    def test_center_distance_value(self: Self) -> None:
        """Compute center distance for separated intervals."""
        d = center_distance(0.0, 10.0, 20.0, 30.0)
        assert d == 20.0

    def test_center_distance_symmetric(self: Self) -> None:
        """Be symmetric when swapping interval order."""
        d1 = center_distance(0.0, 10.0, 5.0, 20.0)
        d2 = center_distance(5.0, 20.0, 0.0, 10.0)
        assert d1 == d2


class TestHausdorffDistance:
    def test_hausdorff_distance_identical(self: Self) -> None:
        """Return zero for identical intervals."""
        d = hausdorff_distance(0.0, 10.0, 0.0, 10.0)
        assert d == 0.0

    def test_hausdorff_distance_value(self: Self) -> None:
        """Match expected Hausdorff distance for overlapping intervals."""
        d = hausdorff_distance(0.0, 10.0, 5.0, 20.0)
        assert d == 10.0


class TestWasserstein2Distance:
    def test_wasserstein2_distance_identical(self: Self) -> None:
        """Return zero for identical intervals."""
        d = wasserstein2_distance(0.0, 10.0, 0.0, 10.0)
        assert d == 0.0

    def test_wasserstein2_distance_value(self: Self) -> None:
        """Match expected Wasserstein-2 distance for a known case."""
        d = wasserstein2_distance(0.0, 10.0, 5.0, 20.0)
        expected = math.sqrt(10.0**2 + (1.0 / 3.0) * (-5.0) ** 2)
        assert math.isclose(d, expected)


class TestGapOverlappingWidthDistance:
    def test_gap_overlapping_width_distance_same(self: Self) -> None:
        """Return zero for identical intervals."""
        d = gap_overlapping_width_distance(0.0, 10.0, 0.0, 10.0)
        assert d == 0.0  # overlap width is equal to the interval width

    def test_gap_overlapping_width_distance_1_value(self: Self) -> None:
        """Handle degenerate second interval at the left boundary."""
        d = gap_overlapping_width_distance(0.0, 10.0, 0.0, 0.0)
        print(d)
        assert d == 0.0  # overlap width is equal to the interval width

    def test_gap_overlapping_width_distance_disjoint_right(self: Self) -> None:
        """Increase above one by gap when second interval is right."""
        d = gap_overlapping_width_distance(0.0, 10.0, 20.0, 30.0)
        assert d == 11.0  # 1 + gap(=10)

    def test_gap_overlapping_width_distance_disjoint_left(self: Self) -> None:
        """Increase above one by gap when second interval is left."""
        d = gap_overlapping_width_distance(20.0, 30.0, 0.0, 10.0)
        assert d == 11.0  # 1 + gap(=10)

    def test_gap_overlapping_width_distance_touching_intervals(
        self: Self,
    ) -> None:
        """Return one when intervals only touch at a boundary."""
        d = gap_overlapping_width_distance(0.0, 10.0, 10.0, 20.0)
        assert d == 1.0  # overlap width is zero when just touching

    def test_gap_overlapping_width_distance_partial_overlap(
        self: Self,
    ) -> None:
        """Return normalized non-zero value for partial overlap."""
        d = gap_overlapping_width_distance(0.0, 10.0, 5.0, 20.0)
        assert math.isclose(d, 0.5)  # overlap=5, width1=10

    def test_gap_overlapping_width_distance_second_inside_first(
        self: Self,
    ) -> None:
        """Return reduced value when second interval lies inside first."""
        d = gap_overlapping_width_distance(0.0, 10.0, 2.0, 8.0)
        assert math.isclose(d, 0.4)  # overlap=6, width1=10: 1-0.6=0.4

    def test_gap_overlapping_width_distance_symmetric(self: Self) -> None:
        """Be symmetric when swapping interval order."""
        d1 = gap_overlapping_width_distance(0.0, 10.0, 2.0, 8.0)
        d2 = gap_overlapping_width_distance(2.0, 8.0, 0.0, 10.0)
        assert d1 == d2

    @pytest.mark.parametrize(
        "min1, max1, min2, max2",
        [
            (0.0, 10.0, 0.0, 10.0),
            (0.0, 10.0, 2.0, 8.0),
            (0.0, 10.0, 5.0, 20.0),
            (5.0, 30.0, 10.0, 20.0),
        ],
    )
    def test_gap_overlapping_width_distance_overlap_range(
        self: Self, min1: float, max1: float, min2: float, max2: float
    ) -> None:
        """Stay in [0, 1] for overlapping interval pairs."""
        d = gap_overlapping_width_distance(min1, max1, min2, max2)
        assert 0.0 <= d <= 1.0


class TestCompareDistances:
    @pytest.mark.parametrize(
        "min1, max1, min2, max2",
        [
            (0.0, 10.0, 5.0, 30.0),
            (0.0, 10.0, 10.01, 20.0),
            (5.0, 30.0, 10.0, 20.0),
        ],
    )
    def test_compare_distances(
        self: Self,
        min1: float,
        max1: float,
        min2: float,
        max2: float,
    ) -> None:
        """Preserve expected ordering between distance metrics."""
        d_gap = gap_distance(min1, max1, min2, max2)
        d_center = center_distance(min1, max1, min2, max2)
        d_hausdorff = hausdorff_distance(min1, max1, min2, max2)
        d_wasserstein2 = wasserstein2_distance(min1, max1, min2, max2)
        d_gap_times_center = gap_times_center_distance(min1, max1, min2, max2)
        d_gap_overlapping_width = gap_overlapping_width_distance(
            min1, max1, min2, max2
        )
        print(
            f"(min1, max1, min2, max2) = ({min1}, {max1}, {min2}, {max2}), "
            f"gap: {d_gap}, center: {d_center}, "
            f"Hausdorff: {d_hausdorff}, Wasserstein-2: {d_wasserstein2}, "
            f"gap_times_center: {d_gap_times_center}, "
            f"gap_overlapping_width: {d_gap_overlapping_width}"
        )
        assert d_gap <= d_gap_overlapping_width <= d_center <= d_hausdorff
        assert d_hausdorff <= d_wasserstein2
