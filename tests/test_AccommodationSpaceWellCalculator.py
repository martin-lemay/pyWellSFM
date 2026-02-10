# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
import os
import sys

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

# import matplotlib.pyplot as plt
import numpy as np
import pytest
from striplog import Legend, Striplog

from pywellsfm.model import (
    FaciesCriteria,
    FaciesCriteriaType,
    SedimentaryFacies,
    UncertaintyCurve,
    Well,
)
from pywellsfm.model.AccommodationSpaceWellCalculator import (
    AccommodationSpaceWellCalculator,
)

fileDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(fileDir, "data")

wellCoords = np.array((0.0, 0.0, 0.0))
depth: float = 100.0

well0: Well = Well("Well0", wellCoords, depth)

# add lithology log
lithoLog0Txt = """top,base,comp lithology
0.0,50.0,sandstone
"""

lithoLogName = "lithology"
lithoLog0 = Striplog.from_csv(text=lithoLog0Txt)
well0.addLog(lithoLogName, lithoLog0)

# well from Barbier et al. 2024
wellBarbier: Well = Well("Well0", wellCoords, depth)

# add lithology log
lithoLogBarbierTxt = """top,base,comp lithology
0.0,15.0,sandstone
15.0,30.0,siltstone
30.0,55.0,shale
"""

lithoLogName = "lithology"
lithoLogBarbier = Striplog.from_csv(text=lithoLogBarbierTxt)
wellBarbier.addLog(lithoLogName, lithoLogBarbier)

# defines facies bathymetry
sandstoneFac1 = SedimentaryFacies(
    "sandstone", {
        FaciesCriteria(
            "Bathymetry", 0.0, 20.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    }
)
sandstoneFac1.addCriteria(
    FaciesCriteria(
        "Bathymetry", 0.0, 20.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
siltstoneFac1 = SedimentaryFacies(
    "siltstone", {
        FaciesCriteria(
            "Bathymetry", 20.0, 50.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    }
)
siltstoneFac1.addCriteria(
    FaciesCriteria(
        "Bathymetry", 20.0, 50.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
shaleFac1 = SedimentaryFacies(
    "shale", {
        FaciesCriteria(
            "Bathymetry", 40.0, 100.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    }
)
shaleFac1.addCriteria(
    FaciesCriteria(
        "Bathymetry", 40.0, 100.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
faciesList1: list[SedimentaryFacies] = [
    sandstoneFac1,
    siltstoneFac1,
    shaleFac1,
]

# defines facies bathymetry
sandstoneFac2 = SedimentaryFacies(
    "sandstone", {
        FaciesCriteria(
            "Bathymetry", 5.0, 5.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    }
)
sandstoneFac2.addCriteria(
    FaciesCriteria(
        "Bathymetry", 5.0, 5.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
siltstoneFac2 = SedimentaryFacies(
    "siltstone", {
        FaciesCriteria(
            "Bathymetry", 10.0, 10.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    }
)
siltstoneFac2.addCriteria(
    FaciesCriteria(
        "Bathymetry", 10.0, 10.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
shaleFac2 = SedimentaryFacies(
    "shale", {
        FaciesCriteria(
            "Bathymetry", 60.0, 60.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    }
)
shaleFac2.addCriteria(
    FaciesCriteria(
        "Bathymetry", 60.0, 60.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
faciesList2: list[SedimentaryFacies] = [
    sandstoneFac2,
    siltstoneFac2,
    shaleFac2,
]

# legend for litho log plot
leg_txt = """colour,comp lithology
gold,sandstone
brown,siltstone
gray,shale
"""
legend = Legend.from_csv(text=leg_txt)


def test_ASPC_init() -> None:
    """Test of AccommodationSpaceWellCalculator class init method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    assert aspc._well is not None, "Well is undefined."
    assert aspc._faciesDict is not None, "facies map is undefined."
    assert (aspc.bathymetryCurve is not None) and (
        isinstance(aspc.bathymetryCurve, UncertaintyCurve)
    ), "Bathymetry curve is undefined."
    assert (aspc.accommodationChangeCurve is not None) and (
        isinstance(aspc.bathymetryCurve, UncertaintyCurve)
    ), "Accommodation curve is undefined."


@pytest.mark.parametrize("facies", faciesList1)
def test_getBathymetryRangeFromFaciesName(facies: SedimentaryFacies) -> None:
    """Test of getBathymetryRangeFromFaciesName method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    (minBathy, maxBathy) = aspc._getBathymetryRangeFromFaciesName(facies.name)
    fac: FaciesCriteria | None = facies.getCriteria("Bathymetry")
    assert fac is not None, "Facies criteria is undefined."
    assert minBathy == fac.minRange, "Min value is wrong"
    assert maxBathy == fac.maxRange, "Max value is wrong"


def test_computeBathymetryStepCurve() -> None:
    """Test of computeBathymetryStepCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    bathymetryStepCurves = aspc._computeBathymetryStepCurve(
        lithoLogBarbier, depth, 0
    )
    # np.savetxt(
    #     os.path.join(fileDir, "bathyStepCurves.csv"),
    #     bathymetryStepCurves,
    #     fmt="%.3e",
    #     delimiter=",",
    # )
    expArray = np.loadtxt(
        os.path.join(dataDir, "bathyStepCurves.csv"), delimiter=","
    )
    eps = 1e-6
    assert array_equal(expArray, bathymetryStepCurves, eps), (
        "Bathymetry step curve array is wrong."
    )


def test_computeBathymetryCurve() -> None:
    """Test of computeBathymetryCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    bathymetryCurve = aspc.computeBathymetryCurve(lithoLogName)
    assert bathymetryCurve is not None, "Bathymetry curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        bathymetryCurve._medianCurve._abscissa,
        bathymetryCurve._minCurve._abscissa,
        eps,
    ) and array_equal(
        bathymetryCurve._medianCurve._abscissa,
        bathymetryCurve._maxCurve._abscissa,
        eps,
    ), "Bathymetry curves do not have same abscissa values."

    # plot bathymetry curves to check results
    # bathymetryStepCurves = aspc._computeBathymetryStepCurve(
    #     lithoLog, depth, 0
    # )
    # fig, (ax0, ax1) = plt.subplots(figsize=(5, 10), ncols=2, sharey=True)
    # lithoLog.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in bathymetryStepCurves:
    #     ax1.plot((row[2], row[2]), row[:2], ":", color="grey")
    #     ax1.plot((row[3], row[3]), row[:2], ":k")
    # ax1.plot(
    #     bathymetryCurve.getMedianValues(),
    #     bathymetryCurve.getAbscissa(),
    #     "--r"
    # )
    # ax1.plot(
    #     bathymetryCurve.getMinValues(),
    #     bathymetryCurve.getAbscissa(),
    #     "--b"
    # )
    # ax1.plot(
    #     bathymetryCurve.getMaxValues(),
    #     bathymetryCurve.getAbscissa(),
    #     "--g"
    # )
    # ax1.set_xlabel("Bathymetry")
    # plt.show()

    # outputArray = np.column_stack(
    #     (
    #         bathymetryCurve._medianCurve._abscissa,
    #         bathymetryCurve._medianCurve._ordinate,
    #         bathymetryCurve._minCurve._ordinate,
    #         bathymetryCurve._maxCurve._ordinate,
    #     )
    # )
    # np.savetxt(
    #     os.path.join(fileDir, "bathyUncertaintyCurves.csv"),
    #     outputArray,
    #     fmt="%.5e",
    #     delimiter=",",
    # )

    # check curve values

    expArray = np.loadtxt(
        os.path.join(dataDir, "bathyUncertaintyCurves.csv"), delimiter=","
    )
    assert array_equal(bathymetryCurve.getAbscissa(), expArray[:, 0], eps), (
        "Abscissa values of bathymetry curve are wrong."
    )
    assert array_equal(
        bathymetryCurve.getMedianValues(), expArray[:, 1], eps
    ), "Ordinate values of median bathymetry curve are wrong."
    assert array_equal(bathymetryCurve.getMinValues(), expArray[:, 2], eps), (
        "Ordinate values of min bathymetry curve are wrong."
    )
    assert array_equal(bathymetryCurve.getMaxValues(), expArray[:, 3], eps), (
        "Ordinate values of max bathymetry curve are wrong."
    )


def test_computeAccommodationArray() -> None:
    """Test of _computeAccommodationStepCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    accommodationArray = aspc._computeAccommodationArray(
        lithoLogBarbier, 55.0, 0
    )
    # np.savetxt(
    #     os.path.join(fileDir, "accomodationArray.csv"),
    #     accommodationArray,
    #     fmt="%.3e",
    #     delimiter=",",
    # )
    expArray = np.loadtxt(
        os.path.join(dataDir, "accomodationArray.csv"), delimiter=","
    )
    eps = 1e-6
    assert array_equal(expArray, accommodationArray, eps), (
        "Accommodation step curve array is wrong."
    )


# No facies variation, no bathymetry uncertainty
def test_computeAccommodationCurve01() -> None:
    """Test of computeAccommodationCurve method."""
    aspc = AccommodationSpaceWellCalculator(well0, faciesList2)
    accoCurve: UncertaintyCurve = aspc.computeAccommodationCurve(lithoLogName)
    assert accoCurve is not None, "Accommodation curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._minCurve._abscissa, eps
    ) and array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._maxCurve._abscissa, eps
    ), "Accommodation curves do not have same abscissa values."

    # plot bathymetry curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(figsize=(5, 10), ncols=3,
    #     sharey=True)
    # lithoLog0.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._bathymetryStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("Bathymetry")
    # ax2.plot(accoCurve.getMedianValues(), accoCurve.getAbscissa(), "--r")
    # ax2.plot(accoCurve.getMinValues(), accoCurve.getAbscissa(), "--b")
    # ax2.plot(accoCurve.getMaxValues(), accoCurve.getAbscissa(), "--g")
    # ax2.set_xlabel("Accommodation")

    # interval: Interval
    # for interval in lithoLog0:
    #     ax1.hlines(interval.top.middle, 0, 10, colors="k",
    #         linestyles="--", linewidth=1)
    #     ax2.hlines(interval.top.middle, 0, 50, colors="k",
    #         linestyles="--", linewidth=1)
    # plt.show()

    # outputArray1 = np.column_stack(
    #     (
    #         accoCurve._medianCurve._abscissa,
    #         accoCurve._medianCurve._ordinate,
    #         accoCurve._minCurve._ordinate,
    #         accoCurve._maxCurve._ordinate,
    #     )
    # )
    # np.savetxt(
    #     os.path.join(fileDir, "accommodationUncertaintyCurves01.csv"),
    #     outputArray1,
    #     fmt="%.3e",
    #     delimiter=",",
    # )

    # check curve values
    expArray = np.loadtxt(
        os.path.join(dataDir, "accommodationUncertaintyCurves01.csv"),
        delimiter=",",
    )
    print(accoCurve.getAbscissa(), expArray[:, 0])
    assert array_equal(accoCurve.getAbscissa(), expArray[:, 0], eps), (
        "Abscissa values of accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMedianValues(), expArray[:, 1], eps), (
        "Ordinate values of median accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMinValues(), expArray[:, 2], eps), (
        "Ordinate values of min accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMaxValues(), expArray[:, 3], eps), (
        "Ordinate values of max accommodation curve are wrong."
    )


# No facies variation, no bathymetry uncertainty
def test_computeAccommodationCurve02() -> None:
    """Test of computeAccommodationCurve method."""
    aspc = AccommodationSpaceWellCalculator(well0, faciesList1)
    accoCurve: UncertaintyCurve = aspc.computeAccommodationCurve(lithoLogName)
    assert accoCurve is not None, "Accommodation curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._minCurve._abscissa, eps
    ) and array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._maxCurve._abscissa, eps
    ), "Accommodation curves do not have same abscissa values."

    # plot bathymetry curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(figsize=(5, 10), ncols=3,
    # lithoLog0.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._bathymetryStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("Bathymetry")
    # ax2.plot(accoCurve.getMedianValues(), accoCurve.getAbscissa(), "--r")
    # ax2.plot(accoCurve.getMinValues(), accoCurve.getAbscissa(), "--b")
    # ax2.plot(accoCurve.getMaxValues(), accoCurve.getAbscissa(), "--g")
    # ax2.set_xlabel("Accommodation")

    # interval: Interval
    # for interval in lithoLog0:
    #     ax1.hlines(interval.top.middle, 0, 10, colors="k",
    #          linestyles="--", linewidth=1)
    #     ax2.hlines(interval.top.middle, 0, 50, colors="k",
    #          linestyles="--", linewidth=1)
    # plt.show()

    # outputArray1 = np.column_stack(
    #     (
    #         accoCurve._medianCurve._abscissa,
    #         accoCurve._medianCurve._ordinate,
    #         accoCurve._minCurve._ordinate,
    #         accoCurve._maxCurve._ordinate,
    #     )
    # )
    # np.savetxt(
    #     os.path.join(fileDir, "accommodationUncertaintyCurves02.csv"),
    #     outputArray1,
    #     fmt="%.3e",
    #     delimiter=",",
    # )

    # check curve values
    expArray = np.loadtxt(
        os.path.join(dataDir, "accommodationUncertaintyCurves02.csv"),
        delimiter=",",
    )
    assert array_equal(accoCurve.getAbscissa(), expArray[:, 0], eps), (
        "Abscissa values of accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMedianValues(), expArray[:, 1], eps), (
        "Ordinate values of median accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMinValues(), expArray[:, 2], eps), (
        "Ordinate values of min accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMaxValues(), expArray[:, 3], eps), (
        "Ordinate values of max accommodation curve are wrong."
    )


def test_computeAccommodationCurve1() -> None:
    """Test of computeAccommodationCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    accoCurve: UncertaintyCurve = aspc.computeAccommodationCurve(lithoLogName)
    assert accoCurve is not None, "Accommodation curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._minCurve._abscissa, eps
    ) and array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._maxCurve._abscissa, eps
    ), "Accommodation curves do not have same abscissa values."

    # plot bathymetry curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(
    #     figsize=(5, 10), ncols=3, sharey=True
    # )
    # lithoLog.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._bathymetryStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("Bathymetry")
    # ax2.plot(accoCurve.getMedianValues(), accoCurve.getAbscissa(), "--r")
    # ax2.plot(accoCurve.getMinValues(), accoCurve.getAbscissa(), "--b")
    # ax2.plot(accoCurve.getMaxValues(), accoCurve.getAbscissa(), "--g")
    # ax2.set_xlabel("Accommodation")

    # interval: Interval
    # for interval in lithoLog:
    #     ax1.hlines(
    #         interval.top.middle, 0, 100, colors="k",
    #         linestyles="--", linewidth=1
    #     )
    #     ax2.hlines(
    #         interval.top.middle, -50, 50, colors="k",
    #         linestyles="--", linewidth=1
    #     )
    # plt.show()

    # outputArray1 = np.column_stack(
    #     (
    #         accoCurve._medianCurve._abscissa,
    #         accoCurve._medianCurve._ordinate,
    #         accoCurve._minCurve._ordinate,
    #         accoCurve._maxCurve._ordinate,
    #     )
    # )
    # np.savetxt(
    #     os.path.join(fileDir, "accommodationUncertaintyCurves1.csv"),
    #     outputArray1,
    #     fmt="%.3e",
    #     delimiter=",",
    # )

    # check curve values
    expArray = np.loadtxt(
        os.path.join(dataDir, "accommodationUncertaintyCurves1.csv"),
        delimiter=",",
    )
    print(accoCurve.getAbscissa(), expArray[:, 0])
    assert array_equal(accoCurve.getAbscissa(), expArray[:, 0], eps), (
        "Abscissa values of accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMedianValues(), expArray[:, 1], eps), (
        "Ordinate values of median accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMinValues(), expArray[:, 2], eps), (
        "Ordinate values of min accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMaxValues(), expArray[:, 3], eps), (
        "Ordinate values of max accommodation curve are wrong."
    )


def test_computeAccommodationCurve2() -> None:
    """Test of computeAccommodationCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList2)
    accoCurve: UncertaintyCurve = aspc.computeAccommodationCurve(lithoLogName)
    assert accoCurve is not None, "Accommodation curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._minCurve._abscissa, eps
    ) and array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._maxCurve._abscissa, eps
    ), "Accommodation curves do not have same abscissa values."

    # plot bathymetry curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(
    #     figsize=(5, 10), ncols=3, sharey=True)
    # lithoLogBarbier.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._bathymetryStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("Bathymetry")
    # ax2.plot(accoCurve.getMedianValues(), accoCurve.getAbscissa(), "--r")
    # ax2.plot(accoCurve.getMinValues(), accoCurve.getAbscissa(), "--b")
    # ax2.plot(accoCurve.getMaxValues(), accoCurve.getAbscissa(), "--g")
    # ax2.set_xlabel("Accommodation")

    # interval: Interval
    # for interval in lithoLogBarbier:
    #     ax1.hlines(
    #         interval.top.middle, 0, 100, colors="k",
    #         linestyles="--", linewidth=1
    #     )
    #     ax2.hlines(
    #         interval.top.middle, -50, 50, colors="k",
    #         linestyles="--", linewidth=1
    #     )
    # plt.show()

    # outputArray1 = np.column_stack(
    #     (
    #         accoCurve._medianCurve._abscissa,
    #         accoCurve._medianCurve._ordinate,
    #         accoCurve._minCurve._ordinate,
    #         accoCurve._maxCurve._ordinate,
    #     )
    # )
    # np.savetxt(
    #     os.path.join(fileDir, "accommodationUncertaintyCurves2.csv"),
    #     outputArray1,
    #     fmt="%.3e",
    #     delimiter=",",
    # )

    # check curve values
    expArray = np.loadtxt(
        os.path.join(dataDir, "accommodationUncertaintyCurves2.csv"),
        delimiter=",",
    )
    print(accoCurve.getAbscissa(), expArray[:, 0])
    assert array_equal(accoCurve.getAbscissa(), expArray[:, 0], eps), (
        "Abscissa values of accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMedianValues(), expArray[:, 1], eps), (
        "Ordinate values of median accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMinValues(), expArray[:, 2], eps), (
        "Ordinate values of min accommodation curve are wrong."
    )
    assert array_equal(accoCurve.getMaxValues(), expArray[:, 3], eps), (
        "Ordinate values of max accommodation curve are wrong."
    )


def array_equal(array1: np.ndarray, array2: np.ndarray, tol: float) -> bool:
    """Helper to compare arrays.

    :param np.ndarray array1: numpy array
    :param np.ndarray array2: numpy array
    :param float tol: tolerance for comparison
    :return bool: True if arrays are equal within tolerance, False otherwise
    """
    __test__ = False # noqa: F841
    if array1.shape != array2.shape:
        return False
    diff = np.abs(array1 - array2)
    return bool(np.all(diff[np.isfinite(diff)] < tol))


if __name__ == "__main__":
    pytest.main(
        [
            os.path.abspath(__file__),
        ]
    )
