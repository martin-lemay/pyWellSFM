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
    Marker,
    SedimentaryFacies,
    UncertaintyCurve,
    Well,
)
from pywellsfm.model.AccommodationSpaceWellCalculator import (
    AccommodationBoundaryRule,
    AccommodationEstimateMethod,
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

# defines facies waterDepth
sandstoneFac1 = SedimentaryFacies(
    "sandstone",
    {
        FaciesCriteria(
            "WaterDepth", 0.0, 20.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    },
)
sandstoneFac1.addCriteria(
    FaciesCriteria(
        "WaterDepth", 0.0, 20.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
siltstoneFac1 = SedimentaryFacies(
    "siltstone",
    {
        FaciesCriteria(
            "WaterDepth", 20.0, 50.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    },
)
siltstoneFac1.addCriteria(
    FaciesCriteria(
        "WaterDepth", 20.0, 50.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
shaleFac1 = SedimentaryFacies(
    "shale",
    {
        FaciesCriteria(
            "WaterDepth", 40.0, 100.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    },
)
shaleFac1.addCriteria(
    FaciesCriteria(
        "WaterDepth", 40.0, 100.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
faciesList1: list[SedimentaryFacies] = [
    sandstoneFac1,
    siltstoneFac1,
    shaleFac1,
]

# defines facies waterDepth
sandstoneFac2 = SedimentaryFacies(
    "sandstone",
    {
        FaciesCriteria(
            "WaterDepth", 5.0, 5.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    },
)
sandstoneFac2.addCriteria(
    FaciesCriteria("WaterDepth", 5.0, 5.0, FaciesCriteriaType.SEDIMENTOLOGICAL)
)
siltstoneFac2 = SedimentaryFacies(
    "siltstone",
    {
        FaciesCriteria(
            "WaterDepth", 10.0, 10.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    },
)
siltstoneFac2.addCriteria(
    FaciesCriteria(
        "WaterDepth", 10.0, 10.0, FaciesCriteriaType.SEDIMENTOLOGICAL
    )
)
shaleFac2 = SedimentaryFacies(
    "shale",
    {
        FaciesCriteria(
            "WaterDepth", 60.0, 60.0, FaciesCriteriaType.SEDIMENTOLOGICAL
        )
    },
)
shaleFac2.addCriteria(
    FaciesCriteria(
        "WaterDepth", 60.0, 60.0, FaciesCriteriaType.SEDIMENTOLOGICAL
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
    assert (aspc.waterDepthCurve is not None) and (
        isinstance(aspc.waterDepthCurve, UncertaintyCurve)
    ), "WaterDepth curve is undefined."
    assert (aspc.accommodationChangeCurve is not None) and (
        isinstance(aspc.waterDepthCurve, UncertaintyCurve)
    ), "Accommodation curve is undefined."


@pytest.mark.parametrize("facies", faciesList1)
def test_getWaterDepthRangeFromFaciesName(facies: SedimentaryFacies) -> None:
    """Test of getWaterDepthRangeFromFaciesName method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    (minBathy, maxBathy) = aspc._getWaterDepthRangeFromFaciesName(facies.name)
    fac: FaciesCriteria | None = facies.getCriteria("WaterDepth")
    assert fac is not None, "Facies criteria is undefined."
    assert minBathy == fac.minRange, "Min value is wrong"
    assert maxBathy == fac.maxRange, "Max value is wrong"


def test_computeWaterDepthStepCurve() -> None:
    """Test of computeWaterDepthStepCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    waterDepthStepCurves = aspc._computeWaterDepthStepCurve(
        lithoLogBarbier, depth, 0
    )
    # np.savetxt(
    #     os.path.join(fileDir, "waterDepthStepCurves.csv"),
    #     waterDepthStepCurves,
    #     fmt="%.3e",
    #     delimiter=",",
    # )
    expArray = np.loadtxt(
        os.path.join(dataDir, "bathyStepCurves.csv"), delimiter=","
    )
    eps = 1e-6
    assert array_equal(expArray, waterDepthStepCurves, eps), (
        "Water Depth step curve array is wrong."
    )


def test_computeWaterDepthCurve() -> None:
    """Test of computeWaterDepthCurve method."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    waterDepthCurve = aspc.computeWaterDepthCurve(lithoLogName)
    assert waterDepthCurve is not None, "WaterDepth curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        waterDepthCurve._medianCurve._abscissa,
        waterDepthCurve._minCurve._abscissa,
        eps,
    ) and array_equal(
        waterDepthCurve._medianCurve._abscissa,
        waterDepthCurve._maxCurve._abscissa,
        eps,
    ), "WaterDepth curves do not have same abscissa values."

    # plot waterDepth curves to check results
    # waterDepthStepCurves = aspc._computeWaterDepthStepCurve(
    #     lithoLog, depth, 0
    # )
    # fig, (ax0, ax1) = plt.subplots(figsize=(5, 10), ncols=2, sharey=True)
    # lithoLog.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in waterDepthStepCurves:
    #     ax1.plot((row[2], row[2]), row[:2], ":", color="grey")
    #     ax1.plot((row[3], row[3]), row[:2], ":k")
    # ax1.plot(
    #     waterDepthCurve.getMedianValues(),
    #     waterDepthCurve.getAbscissa(),
    #     "--r"
    # )
    # ax1.plot(
    #     waterDepthCurve.getMinValues(),
    #     waterDepthCurve.getAbscissa(),
    #     "--b"
    # )
    # ax1.plot(
    #     waterDepthCurve.getMaxValues(),
    #     waterDepthCurve.getAbscissa(),
    #     "--g"
    # )
    # ax1.set_xlabel("WaterDepth")
    # plt.show()

    # outputArray = np.column_stack(
    #     (
    #         waterDepthCurve._medianCurve._abscissa,
    #         waterDepthCurve._medianCurve._ordinate,
    #         waterDepthCurve._minCurve._ordinate,
    #         waterDepthCurve._maxCurve._ordinate,
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
    assert array_equal(waterDepthCurve.getAbscissa(), expArray[:, 0], eps), (
        "Abscissa values of waterDepth curve are wrong."
    )
    assert array_equal(
        waterDepthCurve.getMedianValues(), expArray[:, 1], eps
    ), "Ordinate values of median waterDepth curve are wrong."
    assert array_equal(waterDepthCurve.getMinValues(), expArray[:, 2], eps), (
        "Ordinate values of min waterDepth curve are wrong."
    )
    assert array_equal(waterDepthCurve.getMaxValues(), expArray[:, 3], eps), (
        "Ordinate values of max waterDepth curve are wrong."
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


# No facies variation, no waterDepth uncertainty
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

    # plot waterDepth curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(figsize=(5, 10), ncols=3,
    #     sharey=True)
    # lithoLog0.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._waterDepthStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("WaterDepth")
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


# No facies variation, no waterDepth uncertainty
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

    # plot waterDepth curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(figsize=(5, 10), ncols=3,
    # lithoLog0.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._waterDepthStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("WaterDepth")
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
    """Test of computeAccommodationCurve method (intersection rule)."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    accoCurve: UncertaintyCurve = aspc.computeAccommodationCurve(
        lithoLogName, boundaryRule=AccommodationBoundaryRule.INTERSECTION
    )
    assert accoCurve is not None, "Accommodation curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._minCurve._abscissa, eps
    ) and array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._maxCurve._abscissa, eps
    ), "Accommodation curves do not have same abscissa values."

    # plot waterDepth curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(
    #     figsize=(5, 10), ncols=3, sharey=True
    # )
    # lithoLog.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._waterDepthStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("WaterDepth")
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
    """Test of computeAccommodationCurve method (intersection rule)."""
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList2)
    accoCurve: UncertaintyCurve = aspc.computeAccommodationCurve(
        lithoLogName, boundaryRule=AccommodationBoundaryRule.INTERSECTION
    )
    assert accoCurve is not None, "Accommodation curve is undefined"

    # check curves coherency
    eps = 1e-4
    assert array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._minCurve._abscissa, eps
    ) and array_equal(
        accoCurve._medianCurve._abscissa, accoCurve._maxCurve._abscissa, eps
    ), "Accommodation curves do not have same abscissa values."

    # plot waterDepth curves to check results
    # fig, (ax0, ax1, ax2) = plt.subplots(
    #     figsize=(5, 10), ncols=3, sharey=True)
    # lithoLogBarbier.plot(legend, ax=ax0)
    # ax0.set_ylabel("Depth")
    # for row in aspc._waterDepthStepCurve:
    #     ax1.plot((row[2], row[2]), row[:2], ":b")
    #     ax1.plot((row[3], row[3]), row[:2], ":b")
    # ax1.set_xlabel("WaterDepth")
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


def _make_facies(
    name: str,
    min_depth: float,
    max_depth: float,
    crit_name: str = "WaterDepth",
) -> SedimentaryFacies:
    """Build one facies with a single criterion."""
    criterion = FaciesCriteria(
        crit_name,
        min_depth,
        max_depth,
        FaciesCriteriaType.SEDIMENTOLOGICAL,
    )
    return SedimentaryFacies(name, {criterion})


def _make_well(log_text: str, name: str = "W") -> tuple[Well, Striplog]:
    """Build a well and attach a lithology striplog."""
    coords = np.array((0.0, 0.0, 0.0))
    well = Well(name, coords, 120.0)
    log = Striplog.from_csv(text=log_text)
    well.addLog("lithology", log)
    return well, log


def test_get_initial_water_depth_midpoint() -> None:
    """Return midpoint from initial facies water depth range."""
    txt = "top,base,comp lithology\n0.0,20.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 10.0, 30.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    assert calc.getInitialwaterDepth() == 20.0


def test_get_initial_water_depth_range_no_discrete_log() -> None:
    """Raise when no discrete log exists in the well."""
    coords = np.array((0.0, 0.0, 0.0))
    well = Well("NoLog", coords, 100.0)
    facies = [_make_facies("sandstone", 0.0, 10.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    with pytest.raises(ValueError, match="No discrete log"):
        calc._getInitialwaterDepthRange()


def test_get_initial_water_depth_range_missing_log_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise when the facies log name exists but log object is None."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 2.0, 8.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    monkeypatch.setattr(well, "getDepthLog", lambda _name: None)

    with pytest.raises(ValueError, match="Facies log not found"):
        calc._getInitialwaterDepthRange()


def test_get_initial_water_depth_range_none_from_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise if facies lookup helper returns no depth range."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 1.0, 2.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    monkeypatch.setattr(
        calc,
        "_getWaterDepthRangeFromFaciesName",
        lambda _name: None,
    )

    with pytest.raises(ValueError, match="condition not found"):
        calc._getInitialwaterDepthRange()


def test_get_water_depth_range_unknown_facies() -> None:
    """Raise when facies name is not available."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 0.0, 20.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    with pytest.raises(ValueError, match="not in the facies list"):
        calc._getWaterDepthRangeFromFaciesName("missing")


def test_get_water_depth_range_missing_water_depth_criteria() -> None:
    """Raise when facies has no waterDepth criterion."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 0.0, 1.0, crit_name="GrainSize")]
    calc = AccommodationSpaceWellCalculator(well, facies)

    with pytest.raises(ValueError, match="waterDepth is undefined"):
        calc._getWaterDepthRangeFromFaciesName("sandstone")


def test_compute_water_depth_step_curve_skips_outside_interval() -> None:
    """Leave rows as NaN when intervals are outside marker bounds."""
    txt = (
        "top,base,comp lithology\n"
        "0.0,10.0,sandstone\n"
        "10.0,20.0,sandstone\n"
        "20.0,30.0,sandstone\n"
    )
    well, log = _make_well(txt)
    facies = [_make_facies("sandstone", 5.0, 15.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    arr = calc._computeWaterDepthStepCurve(log, baseDepth=30.0, topDepth=10.0)

    assert np.isnan(arr[0, 0])
    assert np.allclose(arr[1, 2:], np.array((5.0, 15.0)))


def test_compute_accommodation_value_swaps_inverted_bounds() -> None:
    """Swap min and max when computed values are inverted."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 0.0, 10.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    acco_min, acco_max = calc._computeAccommodationValue(
        thickness=0.0,
        waterDepthBase=(10.0, 0.0),
        waterDepthTop=(0.0, -10.0),
    )

    assert acco_min <= acco_max
    assert (acco_min, acco_max) == (-20.0, 0.0)


def test_compute_accommodation_array_handles_partial_base_depth() -> None:
    """Skip deeper strata when base depth is inside the log."""
    txt = (
        "top,base,comp lithology\n"
        "0.0,10.0,sandstone\n"
        "10.0,20.0,sandstone\n"
        "20.0,30.0,sandstone\n"
    )
    well, log = _make_well(txt)
    facies = [_make_facies("sandstone", 2.0, 4.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    calc._computeWaterDepthStepCurve(log, baseDepth=30.0, topDepth=0.0)

    arr = calc._computeAccommodationArray(
        log,
        baseDepth=25.0,
        topDepth=0.0,
        accommodationAtBase=1.5,
    )

    assert arr.shape == (4, 5)
    assert np.isfinite(arr[0, 1])
    assert np.isnan(arr[3, 0])
    assert np.allclose(arr[-1, 1:], np.array((1.5, 1.5, 1.5, 1.5)))


def test_compute_accommodation_step_curve_auto_builds_water_depth() -> None:
    """Build water depth first when missing before step computation."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n10.0,20.0,siltstone\n"
    well, log = _make_well(txt)
    facies = [
        _make_facies("sandstone", 5.0, 6.0),
        _make_facies("siltstone", 1.0, 2.0),
    ]
    calc = AccommodationSpaceWellCalculator(well, facies)

    arr = calc._computeAccommodationStepCurve(
        log,
        baseDepth=20.0,
        topDepth=0.0,
    )

    assert calc._waterDepthStepCurve is not None
    assert arr.shape == (2, 4)
    assert arr[0, 2] <= arr[0, 3]
    assert arr[1, 2] <= arr[1, 3]


def test_compute_accommodation_step_curve_swap_branch() -> None:
    """Execute swap path in step curve min max ordering."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n10.0,20.0,siltstone\n"
    well, log = _make_well(txt)
    facies = [
        _make_facies("sandstone", 0.0, 1.0),
        _make_facies("siltstone", 0.0, 1.0),
    ]
    calc = AccommodationSpaceWellCalculator(well, facies)
    calc._waterDepthStepCurve = np.array(
        [
            [10.0, 0.0, 5.0, 1.0],
            [20.0, 10.0, 0.0, -2.0],
        ]
    )

    arr = calc._computeAccommodationStepCurve(
        log,
        baseDepth=20.0,
        topDepth=0.0,
    )

    assert arr[1, 2] <= arr[1, 3]


def test_compute_accommodation_curve0_with_markers() -> None:
    """Run legacy curve method with explicit top and base markers."""
    txt = (
        "top,base,comp lithology\n"
        "0.0,10.0,sandstone\n"
        "10.0,20.0,siltstone\n"
        "20.0,30.0,shale\n"
    )
    well, _ = _make_well(txt)
    facies = [
        _make_facies("sandstone", 0.0, 5.0),
        _make_facies("siltstone", 5.0, 15.0),
        _make_facies("shale", 15.0, 30.0),
    ]
    calc = AccommodationSpaceWellCalculator(well, facies)

    curve = calc.computeAccommodationCurve0(
        "lithology",
        fromMarker=Marker("base", 30.0),
        toMarker=Marker("top", 0.0),
    )

    assert curve is calc.accommodationCurve
    assert len(curve.getAbscissa()) > 0


def test_compute_accommodation_curve0_missing_log_assertion() -> None:
    """Raise assertion for unknown facies log name."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 0.0, 20.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)

    with pytest.raises(ValueError, match="does not exist"):
        calc.computeAccommodationCurve0("unknown_log")


def test_compute_accommodation_curve0_uses_cached_step_curve() -> None:
    """Use cached accommodation step curve when already available."""
    txt = "top,base,comp lithology\n0.0,10.0,sandstone\n"
    well, _ = _make_well(txt)
    facies = [_make_facies("sandstone", 0.0, 20.0)]
    calc = AccommodationSpaceWellCalculator(well, facies)
    calc._accommodationStepCurve = np.array([[10.0, 0.0, 1.0, 2.0]])

    curve = calc.computeAccommodationCurve0("lithology")

    assert curve is calc.accommodationCurve
    assert len(calc.accommodationChangeCurve.getAbscissa()) > 0


def test_computeWaterDepthThicknessRatioCurve() -> None:
    """Test WD/thickness ratio curve computation."""
    simpleLithoLog = """top,base,comp lithology
0.0,15.0,sandstone
15.0,30.0,siltstone
30.0,55.0,shale
"""
    wellCoords = np.array((0.0, 0.0, 0.0))
    depth = 100.0
    well = Well("TestWell", wellCoords, depth)
    lithoLog = Striplog.from_csv(text=simpleLithoLog)
    well.addLog("lithology", lithoLog)

    sandFac = SedimentaryFacies(
        "sandstone",
        {
            FaciesCriteria(
                "WaterDepth",
                0.0,
                5.0,
                FaciesCriteriaType.SEDIMENTOLOGICAL,
            )
        },
    )
    siltFac = SedimentaryFacies(
        "siltstone",
        {
            FaciesCriteria(
                "WaterDepth",
                5.0,
                30.0,
                FaciesCriteriaType.SEDIMENTOLOGICAL,
            )
        },
    )
    shaleFac = SedimentaryFacies(
        "shale",
        {
            FaciesCriteria(
                "WaterDepth",
                20.0,
                50.0,
                FaciesCriteriaType.SEDIMENTOLOGICAL,
            )
        },
    )

    calc = AccommodationSpaceWellCalculator(well, [sandFac, siltFac, shaleFac])
    calc.computeAccommodationCurve("lithology")
    ratio = calc.computeWaterDepthThicknessRatioCurve("lithology")

    assert isinstance(ratio, UncertaintyCurve)
    # 2 points per interval (base-eps, top+eps) = 6 points
    assert len(ratio.getAbscissa()) == 6
    # Min and max values should be finite
    assert np.all(np.isfinite(ratio.getMinValues()))
    assert np.all(np.isfinite(ratio.getMaxValues()))


def test_computeWaterDepthThicknessRatioCurve_raises_without_wd() -> None:
    """Test that ratio curve raises if WD not computed."""
    wellCoords = np.array((0.0, 0.0, 0.0))
    well = Well("TestWell", wellCoords, 100.0)
    lithoLog = Striplog.from_csv(
        text="top,base,comp lithology\n0.0,50.0,sandstone\n"
    )
    well.addLog("lithology", lithoLog)
    sandFac = SedimentaryFacies(
        "sandstone",
        {
            FaciesCriteria(
                "WaterDepth",
                0.0,
                5.0,
                FaciesCriteriaType.SEDIMENTOLOGICAL,
            )
        },
    )
    calc = AccommodationSpaceWellCalculator(well, [sandFac])
    with pytest.raises(RuntimeError, match="water depth"):
        calc.computeWaterDepthThicknessRatioCurve("lithology")


def _curveValuesAt(curve: UncertaintyCurve, depths: list[float]) -> np.ndarray:
    """Get (min, median, max) of an uncertainty curve at sampled depths."""
    abscissa = curve.getAbscissa()
    rows = []
    for depth in depths:
        idx = np.flatnonzero(
            np.isclose(abscissa, depth) & np.isfinite(curve.getMinValues())
        )
        assert idx.size > 0, f"No defined sample at depth {depth}."
        i = idx[0]
        rows.append(
            (
                curve.getMinValues()[i],
                curve.getMedianValues()[i],
                curve.getMaxValues()[i],
            )
        )
    return np.array(rows)


def test_computeAccommodationCurve_union_is_default() -> None:
    """Union rule spans both estimates at facies boundaries (default).

    Barbier well with exact facies water depths: shale (30-55 m, 60 m wd),
    siltstone (15-30 m, 10 m wd), sandstone (0-15 m, 5 m wd).
    """
    aspc = AccommodationSpaceWellCalculator(wellBarbier, faciesList2)
    accoCurve = aspc.computeAccommodationCurve(lithoLogName)
    values = _curveValuesAt(accoCurve, [55.0, 30.0, 15.0, 0.0])
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            # 25 m deposited; water depth 60 m (shale) or 10 m (siltstone)
            [-25.0, 0.0, 25.0],
            # 40 m deposited; water depth 10 m (siltstone) or 5 m (sandstone)
            [-15.0, -12.5, -10.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(values, expected, atol=1e-6)


def test_computeAccommodationCurve_union_contains_intersection() -> None:
    """Union bounds always contain intersection bounds."""
    union = AccommodationSpaceWellCalculator(
        wellBarbier, faciesList1
    ).computeAccommodationCurve(lithoLogName)
    inter = AccommodationSpaceWellCalculator(
        wellBarbier, faciesList1
    ).computeAccommodationCurve(
        lithoLogName, boundaryRule=AccommodationBoundaryRule.INTERSECTION
    )
    ok = np.isfinite(union.getMinValues()) & np.isfinite(inter.getMinValues())
    assert np.all(union.getMinValues()[ok] <= inter.getMinValues()[ok] + 1e-9)
    assert np.all(union.getMaxValues()[ok] >= inter.getMaxValues()[ok] - 1e-9)


def test_computeAccommodationCurve_waterDepthAtBase() -> None:
    """Water depth at base given by the user overrides the basal facies."""
    depths = [30.0, 15.0, 0.0]
    reference = _curveValuesAt(
        AccommodationSpaceWellCalculator(
            wellBarbier, faciesList2
        ).computeAccommodationCurve(lithoLogName),
        depths,
    )
    # same value as the basal facies: identical result
    same = _curveValuesAt(
        AccommodationSpaceWellCalculator(
            wellBarbier, faciesList2
        ).computeAccommodationCurve(lithoLogName, waterDepthAtBase=60.0),
        depths,
    )
    np.testing.assert_allclose(same, reference, atol=1e-6)
    # 50 m shallower at the base: accommodation is 50 m larger everywhere
    shallower = _curveValuesAt(
        AccommodationSpaceWellCalculator(
            wellBarbier, faciesList2
        ).computeAccommodationCurve(lithoLogName, waterDepthAtBase=10.0),
        depths,
    )
    np.testing.assert_allclose(shallower, reference + 50.0, atol=1e-6)
    # a range at the base widens the curve by the range width
    ranged = _curveValuesAt(
        AccommodationSpaceWellCalculator(
            wellBarbier, faciesList2
        ).computeAccommodationCurve(
            lithoLogName, waterDepthAtBase=(12.0, 8.0)
        ),
        depths,
    )
    np.testing.assert_allclose(ranged[:, 0], reference[:, 0] + 48.0)
    np.testing.assert_allclose(ranged[:, 2], reference[:, 2] + 52.0)


def test_computeAccommodationCurve_waterDepthAtBase_narrows_range() -> None:
    """A known base water depth removes the basal facies uncertainty."""
    facies = AccommodationSpaceWellCalculator(
        wellBarbier, faciesList1
    ).computeAccommodationCurve(lithoLogName)
    known = AccommodationSpaceWellCalculator(
        wellBarbier, faciesList1
    ).computeAccommodationCurve(lithoLogName, waterDepthAtBase=70.0)
    ok = np.isfinite(facies.getMinValues()) & np.isfinite(known.getMinValues())
    width_facies = facies.getMaxValues()[ok] - facies.getMinValues()[ok]
    width_known = known.getMaxValues()[ok] - known.getMinValues()[ok]
    # shale range at the base is 40-100 m: 60 m narrower except at the base
    assert np.all(width_known <= width_facies + 1e-9)
    assert np.max(width_facies - width_known) == pytest.approx(60.0)


def _alternatingWell() -> Well:
    """Well with frequent facies changes, as cyclic shallowing upward."""
    logTxt = """top,base,comp lithology
0.0,8.0,sandstone
8.0,20.0,siltstone
20.0,40.0,shale
40.0,55.0,siltstone
55.0,70.0,shale
70.0,75.0,siltstone
75.0,78.0,sandstone
78.0,85.0,siltstone
85.0,95.0,shale
"""
    well = Well("Alternating", wellCoords, 120.0)
    well.addLog(lithoLogName, Striplog.from_csv(text=logTxt))
    return well


def _definedSamples(
    curve: UncertaintyCurve,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Depth, min, median, max of the defined samples of a curve."""
    ok = np.isfinite(curve.getMedianValues())
    return (
        curve.getAbscissa()[ok],
        curve.getMinValues()[ok],
        curve.getMedianValues()[ok],
        curve.getMaxValues()[ok],
    )


@pytest.mark.parametrize(
    "boundaryRule",
    [AccommodationBoundaryRule.UNION, AccommodationBoundaryRule.INTERSECTION],
)
def test_computeAccommodationCurve_smooth_in_range(
    boundaryRule: AccommodationBoundaryRule,
) -> None:
    """Smooth estimate lies in the range, which is the midpoint one."""
    well = _alternatingWell()
    midpoint = AccommodationSpaceWellCalculator(
        well, faciesList1
    ).computeAccommodationCurve(lithoLogName, boundaryRule=boundaryRule)
    smooth = AccommodationSpaceWellCalculator(
        well, faciesList1
    ).computeAccommodationCurve(
        lithoLogName,
        boundaryRule=boundaryRule,
        estimateMethod=AccommodationEstimateMethod.SMOOTH,
        smoothingLength=20.0,
    )
    _, midMin, _, midMax = _definedSamples(midpoint)
    depth, accoMin, accoMed, accoMax = _definedSamples(smooth)
    np.testing.assert_allclose(accoMin, midMin)
    np.testing.assert_allclose(accoMax, midMax)
    low, high = np.minimum(accoMin, accoMax), np.maximum(accoMin, accoMax)
    assert np.all(accoMed >= low - 1e-6)
    assert np.all(accoMed <= high + 1e-6)
    # accommodation is 0 at the base
    assert accoMed[np.argmax(depth)] == pytest.approx(0.0, abs=1e-6)


def test_computeAccommodationCurve_smooth_reduces_roughness() -> None:
    """Smooth estimate is smoother than the midpoint, more with length."""

    def roughness(curve: UncertaintyCurve) -> float:
        depth, _, med, _ = _definedSamples(curve)
        slope = np.diff(med) / np.diff(depth)
        return float(np.sum(np.diff(slope) ** 2))

    well = _alternatingWell()
    values = [
        roughness(
            AccommodationSpaceWellCalculator(
                well, faciesList1
            ).computeAccommodationCurve(lithoLogName)
        )
    ]
    for length in (5.0, 20.0, 80.0):
        values.append(
            roughness(
                AccommodationSpaceWellCalculator(
                    well, faciesList1
                ).computeAccommodationCurve(
                    lithoLogName,
                    estimateMethod=AccommodationEstimateMethod.SMOOTH,
                    smoothingLength=length,
                )
            )
        )
    assert all(b < a for a, b in zip(values[:-1], values[1:], strict=True))


def test_computeAccommodationCurve_smooth_single_interval() -> None:
    """Without curvature, the smooth estimate is the midpoint."""
    midpoint = AccommodationSpaceWellCalculator(
        well0, faciesList1
    ).computeAccommodationCurve(lithoLogName)
    smooth = AccommodationSpaceWellCalculator(
        well0, faciesList1
    ).computeAccommodationCurve(
        lithoLogName, estimateMethod=AccommodationEstimateMethod.SMOOTH
    )
    np.testing.assert_allclose(
        _definedSamples(smooth)[2], _definedSamples(midpoint)[2], atol=1e-6
    )


def test_computeAccommodationCurve_smooth_known_base() -> None:
    """A known water depth at the base and an offset are honored."""
    well = _alternatingWell()
    curve = AccommodationSpaceWellCalculator(
        well, faciesList1
    ).computeAccommodationCurve(
        lithoLogName,
        accommodationAtBase=10.0,
        waterDepthAtBase=45.0,
        estimateMethod=AccommodationEstimateMethod.SMOOTH,
    )
    depth, accoMin, accoMed, accoMax = _definedSamples(curve)
    assert accoMed[np.argmax(depth)] == pytest.approx(10.0, abs=1e-6)
    assert np.all((accoMed >= accoMin - 1e-6) & (accoMed <= accoMax + 1e-6))


def test_computeAccommodationCurve_smooth_invalid_length() -> None:
    """Smoothing length must be strictly positive."""
    calc = AccommodationSpaceWellCalculator(wellBarbier, faciesList1)
    with pytest.raises(ValueError, match="smoothingLength"):
        calc.computeAccommodationCurve(
            lithoLogName,
            estimateMethod=AccommodationEstimateMethod.SMOOTH,
            smoothingLength=0.0,
        )


def _faciesListWithModes() -> list[SedimentaryFacies]:
    """Facies of faciesList1 with a most likely water depth."""
    return [
        SedimentaryFacies(
            name,
            {
                FaciesCriteria(
                    "WaterDepth",
                    wdMin,
                    wdMax,
                    FaciesCriteriaType.SEDIMENTOLOGICAL,
                    mode=mode,
                )
            },
        )
        for name, wdMin, wdMax, mode in (
            ("sandstone", 0.0, 20.0, 5.0),
            ("siltstone", 20.0, 50.0, 30.0),
            ("shale", 40.0, 100.0, 60.0),
        )
    ]


def test_computeAccommodationCurve_mode_uniform() -> None:
    """Without mode, MODE uses the middle of each facies range.

    Exact facies water depths: the mean of both facies at boundaries is the
    middle of the union range, as for MIDPOINT.
    """
    curve = AccommodationSpaceWellCalculator(
        wellBarbier, faciesList2
    ).computeAccommodationCurve(
        lithoLogName, estimateMethod=AccommodationEstimateMethod.MODE
    )
    values = _curveValuesAt(curve, [55.0, 30.0, 15.0, 0.0])
    np.testing.assert_allclose(values[:, 1], [0.0, 0.0, -12.5, 0.0], atol=1e-6)


def test_computeAccommodationCurve_mode_given() -> None:
    """Facies modes set the estimate; boundaries average both facies.

    Base water depth is the shale mode (60 m). At 30 m: 25 m deposited and
    water depth (30 + 60) / 2 = 45 m. At 15 m: 40 m deposited and
    (5 + 30) / 2 = 17.5 m. At the top: 55 m deposited and 5 m.
    """
    curve = AccommodationSpaceWellCalculator(
        wellBarbier, _faciesListWithModes()
    ).computeAccommodationCurve(
        lithoLogName, estimateMethod=AccommodationEstimateMethod.MODE
    )
    values = _curveValuesAt(curve, [55.0, 30.0, 15.0, 0.0])
    np.testing.assert_allclose(values[:, 1], [0.0, 10.0, -2.5, 0.0], atol=1e-6)
    assert np.all(values[:, 1] >= values[:, 0] - 1e-6)
    assert np.all(values[:, 1] <= values[:, 2] + 1e-6)


def test_computeAccommodationCurve_mode_limited_to_range() -> None:
    """With intersection, the mean of the modes is limited to the range.

    At 15 m, sandstone (0-20 m) and siltstone (20-50 m) only share 20 m:
    17.5 m is limited to 20 m, so accommodation is 40 + 20 - 60 = 0 m.
    """
    curve = AccommodationSpaceWellCalculator(
        wellBarbier, _faciesListWithModes()
    ).computeAccommodationCurve(
        lithoLogName,
        boundaryRule=AccommodationBoundaryRule.INTERSECTION,
        estimateMethod=AccommodationEstimateMethod.MODE,
    )
    values = _curveValuesAt(curve, [30.0, 15.0])
    np.testing.assert_allclose(values[:, 1], [10.0, 0.0], atol=1e-6)


def test_computeAccommodationCurve_mode_known_base() -> None:
    """A known base water depth replaces the mode of the basal facies."""
    curve = AccommodationSpaceWellCalculator(
        wellBarbier, _faciesListWithModes()
    ).computeAccommodationCurve(
        lithoLogName,
        waterDepthAtBase=50.0,
        estimateMethod=AccommodationEstimateMethod.MODE,
    )
    values = _curveValuesAt(curve, [30.0, 0.0])
    np.testing.assert_allclose(values[:, 1], [20.0, 10.0], atol=1e-6)


def test_computeAccommodationCurve_smooth_targets_mode() -> None:
    """Without smoothing, the smooth estimate is the mode estimate."""
    well = _alternatingWell()
    mode = AccommodationSpaceWellCalculator(
        well, _faciesListWithModes()
    ).computeAccommodationCurve(
        lithoLogName, estimateMethod=AccommodationEstimateMethod.MODE
    )
    smooth = AccommodationSpaceWellCalculator(
        well, _faciesListWithModes()
    ).computeAccommodationCurve(
        lithoLogName,
        estimateMethod=AccommodationEstimateMethod.SMOOTH,
        smoothingLength=1e-3,
    )
    np.testing.assert_allclose(
        _definedSamples(smooth)[2], _definedSamples(mode)[2], atol=1e-3
    )


def array_equal(array1: np.ndarray, array2: np.ndarray, tol: float) -> bool:
    """Helper to compare arrays.

    :param np.ndarray array1: numpy array
    :param np.ndarray array2: numpy array
    :param float tol: tolerance for comparison
    :return bool: True if arrays are equal within tolerance, False otherwise
    """
    __test__ = False  # noqa: F841
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
