import numpy as np
import pytest
from striplog import Striplog

from pywellsfm.model import (
    AccommodationSpaceWellCalculator,
    FaciesCriteria,
    FaciesCriteriaType,
    SedimentaryFacies,
    Well,
)
from pywellsfm.utils.plot import (
    DEFAULT_LITHO_COLORS,
    plot_accommodation,
    plot_litho_log,
    plot_water_depth,
    plot_wd_thickness_ratio,
    plot_well_analysis,
    plot_well_comparison,
)


@pytest.fixture
def well_and_calc() -> tuple[Well, AccommodationSpaceWellCalculator]:
    """Set up a well and calculator for plot tests."""
    litho_txt = """top,base,comp lithology
0.0,15.0,sandstone
15.0,30.0,siltstone
30.0,55.0,shale
"""
    coords = np.array((0.0, 0.0, 0.0))
    well = Well("TestWell", coords, 100.0)
    litho = Striplog.from_csv(text=litho_txt)
    well.addLog("lithology", litho)

    sand = SedimentaryFacies(
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
    silt = SedimentaryFacies(
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
    shale = SedimentaryFacies(
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
    calc = AccommodationSpaceWellCalculator(well, [sand, silt, shale])
    calc.computeAccommodationCurve("lithology")
    return well, calc


def test_default_litho_colors() -> None:
    """Test DEFAULT_LITHO_COLORS contains expected keys."""
    assert isinstance(DEFAULT_LITHO_COLORS, dict)
    assert "sandstone" in DEFAULT_LITHO_COLORS
    assert "shale" in DEFAULT_LITHO_COLORS


def test_plot_litho_log(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_litho_log returns a Figure with traces."""
    well, _ = well_and_calc
    import plotly.graph_objects as go

    fig = plot_litho_log(well, "lithology")
    assert isinstance(fig, go.Figure)
    # 3 intervals -> 6 traces (rectangle + label per interval)
    assert len(fig.data) >= 6


def test_plot_litho_log_custom_colors(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_litho_log accepts a custom color map."""
    well, _ = well_and_calc
    colors = {"sandstone": "red", "siltstone": "blue"}
    fig = plot_litho_log(well, "lithology", color_map=colors)
    assert fig is not None


def test_plot_water_depth(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_water_depth returns a Figure with at least 2 traces."""
    _, calc = well_and_calc
    import plotly.graph_objects as go

    fig = plot_water_depth(calc)
    assert isinstance(fig, go.Figure)
    # At least 2 traces (min line + max line with fill)
    assert len(fig.data) >= 2


def test_plot_accommodation(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_accommodation returns a Figure with at least 2 traces."""
    _, calc = well_and_calc
    import plotly.graph_objects as go

    fig = plot_accommodation(calc)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_plot_wd_thickness_ratio(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_wd_thickness_ratio returns a Figure with at least 2 traces."""
    _, calc = well_and_calc
    import plotly.graph_objects as go

    fig = plot_wd_thickness_ratio(calc, "lithology")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_plot_litho_log_depth_range(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_litho_log applies depth_range to y-axis."""
    well, _ = well_and_calc
    fig = plot_litho_log(well, "lithology", depth_range=(0.0, 80.0))
    yaxis = fig.layout.yaxis
    assert yaxis.range is not None


def test_plot_well_analysis(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_well_analysis returns a 4-column subplot Figure."""
    _, calc = well_and_calc
    import plotly.graph_objects as go

    fig = plot_well_analysis(calc, "lithology")
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis4 is not None


def test_plot_well_comparison_accommodation(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_well_comparison with accommodation track."""
    _, calc = well_and_calc
    import plotly.graph_objects as go

    calcs = {"TestWell": calc}
    logs = {"TestWell": "lithology"}
    fig = plot_well_comparison(calcs, logs, track="accommodation")
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis2 is not None


def test_plot_well_comparison_water_depth(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_well_comparison with water_depth track."""
    _, calc = well_and_calc
    calcs = {"TestWell": calc}
    logs = {"TestWell": "lithology"}
    fig = plot_well_comparison(calcs, logs, track="water_depth")
    assert len(fig.data) >= 2


def test_plot_well_comparison_ratio(
    well_and_calc: tuple[Well, AccommodationSpaceWellCalculator],
) -> None:
    """Test plot_well_comparison with wd_thickness_ratio track."""
    _, calc = well_and_calc
    calcs = {"TestWell": calc}
    logs = {"TestWell": "lithology"}
    fig = plot_well_comparison(calcs, logs, track="wd_thickness_ratio")
    assert len(fig.data) >= 2
