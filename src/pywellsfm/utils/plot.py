"""Plotly plot functions for well accommodation analysis."""

from __future__ import annotations

from collections.abc import Callable as _Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from striplog import Interval, Striplog

if TYPE_CHECKING:
    from pywellsfm.model.AccommodationSpaceWellCalculator import (
        AccommodationSpaceWellCalculator,
    )
    from pywellsfm.model.Curve import UncertaintyCurve
    from pywellsfm.model.Marker import Marker
    from pywellsfm.model.Well import Well

DEFAULT_LITHO_COLORS: dict[str, str] = {
    "sandstone": "gold",
    "siltstone": "brown",
    "shale": "gray",
    "limestone": "lightblue",
    "dolomite": "mediumpurple",
    "conglomerate": "orange",
    "marl": "olive",
    "chalk": "white",
    "coal": "black",
}

_FALLBACK_COLOR = "lightgray"


def _apply_depth_range(
    fig: go.Figure,
    depth_range: tuple[float, float] | None,
) -> None:
    """Apply depth range to y-axis if provided."""
    if depth_range is not None:
        fig.update_yaxes(
            range=[depth_range[1], depth_range[0]],
        )


def _add_interval_hlines(
    fig: go.Figure,
    facies_log: Striplog,
    x_min: float,
    x_max: float,
) -> None:
    """Add horizontal dashed lines at interval boundaries."""
    depths: set[float] = set()
    interval: Interval
    for interval in facies_log:
        depths.add(interval.top.middle)
        depths.add(interval.base.middle)
    for d in sorted(depths):
        fig.add_shape(
            type="line",
            x0=x_min,
            x1=x_max,
            y0=d,
            y1=d,
            line={"color": "black", "width": 0.5, "dash": "dash"},
        )


def _build_litho_traces(
    facies_log: Striplog,
    color_map: dict[str, str],
    legend_group: str = "",
) -> list[go.Scatter]:
    """Build scatter traces for lithology rectangles.

    Each interval becomes a filled polygon trace so it can be
    toggled via ``updatemenus`` without losing zoom.

    Args:
        facies_log: Striplog with lithology intervals.
        color_map: Lithology name to CSS color.
        legend_group: Optional legend group prefix.

    Returns:
        List of Scatter traces (one per interval).
    """
    traces: list[go.Scatter] = []
    interval: Interval
    for interval in facies_log:
        litho = interval.primary["lithology"]
        color = color_map.get(litho, _FALLBACK_COLOR)
        top = interval.top.middle
        base = interval.base.middle
        y_mid = (top + base) / 2
        traces.append(
            go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[top, top, base, base, top],
                fill="toself",
                fillcolor=color,
                opacity=0.7,
                line={"color": "black", "width": 0.5},
                mode="lines",
                showlegend=False,
                legendgroup=legend_group,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter(
                x=[0.5],
                y=[y_mid],
                mode="text",
                text=[litho],
                textfont={"size": 10},
                showlegend=False,
                legendgroup=legend_group,
                hoverinfo="skip",
            )
        )
    return traces


def _build_marker_traces(
    markers: list[Marker],
    x_range: tuple[float, float] = (0.0, 1.0),
) -> list[go.Scatter]:
    """Build hidden scatter traces for well markers.

    Each marker becomes a horizontal line + text label,
    initially hidden (``visible=False``).

    Args:
        markers: List of well markers.
        x_range: (min, max) x extent for horizontal lines.

    Returns:
        List of Scatter traces (2 per marker: line + label).
    """
    traces: list[go.Scatter] = []
    for marker in markers:
        traces.append(
            go.Scatter(
                x=[x_range[0], x_range[1]],
                y=[marker.depth, marker.depth],
                mode="lines",
                line={"color": "red", "width": 1.5, "dash": "dot"},
                showlegend=False,
                visible=False,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter(
                x=[x_range[1]],
                y=[marker.depth],
                mode="text",
                text=[marker.name],
                textposition="middle left",
                textfont={"size": 9, "color": "red"},
                showlegend=False,
                visible=False,
                hoverinfo="skip",
            )
        )
    return traces


def plot_litho_log(
    well: Well,
    facies_log_name: str,
    color_map: dict[str, str] | None = None,
    depth_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot lithology log as colored rectangles.

    Args:
        well: Well containing the striplog.
        facies_log_name: Name of the discrete log.
        color_map: Lithology name to CSS color. Falls back
            to ``DEFAULT_LITHO_COLORS``.
        depth_range: Optional (top, base) depth range.

    Returns:
        Plotly Figure with colored rectangles.
    """
    colors = color_map or DEFAULT_LITHO_COLORS
    facies_log: Striplog = cast(Striplog, well.getDepthLog(facies_log_name))

    fig = go.Figure()
    for trace in _build_litho_traces(facies_log, colors):
        fig.add_trace(trace)

    fig.update_yaxes(autorange="reversed", title_text="Depth")
    fig.update_xaxes(
        showticklabels=False,
        title_text="Lithology",
        range=[0, 1],
    )
    fig.update_layout(
        margin={"l": 5, "r": 5, "t": 30, "b": 30},
        showlegend=False,
    )
    _apply_depth_range(fig, depth_range)
    return fig


def _plot_uncertainty_fill(
    fig: go.Figure,
    curve: UncertaintyCurve,
    color: str,
    x_label: str,
) -> None:
    """Plot min/max fill for an UncertaintyCurve."""
    abscissa = curve.getAbscissa()
    min_vals = curve.getMinValues()
    max_vals = curve.getMaxValues()

    fig.add_trace(
        go.Scatter(
            x=min_vals,
            y=abscissa,
            mode="lines",
            line={"color": color, "width": 1},
            name=f"{x_label} min",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=max_vals,
            y=abscissa,
            mode="lines",
            line={"color": color, "width": 1},
            fill="tonextx",
            fillcolor=color.replace(")", ", 0.2)").replace("rgb", "rgba")
            if color.startswith("rgb")
            else color,
            opacity=0.2,
            name=f"{x_label} max",
            showlegend=False,
        )
    )


def _plot_step_fill(
    fig: go.Figure,
    step_curve: np.ndarray,
    color: str,
    x_label: str,
) -> None:
    """Plot step curves with fill from waterDepthStepCurve.

    Each row: [base_depth, top_depth, val_min, val_max].
    """
    y_min_pts: list[float] = []
    x_min_pts: list[float] = []
    y_max_pts: list[float] = []
    x_max_pts: list[float] = []

    for row in step_curve:
        if not np.isfinite(row[2]):
            continue
        # Step: constant value across interval
        y_min_pts.extend([row[1], row[0]])
        x_min_pts.extend([row[2], row[2]])
        y_max_pts.extend([row[1], row[0]])
        x_max_pts.extend([row[3], row[3]])

    fig.add_trace(
        go.Scatter(
            x=x_min_pts,
            y=y_min_pts,
            mode="lines",
            line={"color": color, "width": 1},
            name=f"{x_label} min",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_max_pts,
            y=y_max_pts,
            mode="lines",
            line={"color": color, "width": 1},
            fill="tonextx",
            fillcolor=color,
            opacity=0.2,
            name=f"{x_label} max",
            showlegend=False,
        )
    )


def plot_water_depth(
    calculator: AccommodationSpaceWellCalculator,
    depth_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot water depth step curves with min/max fill.

    Args:
        calculator: Calculator with computed water depth.
        depth_range: Optional (top, base) depth range.

    Returns:
        Plotly Figure with water depth tracks.
    """
    if calculator._waterDepthStepCurve is None:
        raise RuntimeError(
            "Water depth step curve not computed. Call"
            " computeAccommodationCurve() first."
        )
    fig = go.Figure()
    step_curve = calculator._waterDepthStepCurve
    _plot_step_fill(fig, step_curve, "blue", "Water Depth")
    facies_log = cast(
        Striplog,
        calculator._well.getDepthLog(
            next(iter(calculator._well.getDiscreteLogNames()))
        ),
    )
    x_max = float(np.nanmax(step_curve[:, 3]))
    _add_interval_hlines(fig, facies_log, 0, x_max)
    fig.update_yaxes(autorange="reversed", title_text="Depth")
    fig.update_xaxes(title_text="Water Depth (m)")
    fig.update_layout(
        margin={"l": 5, "r": 5, "t": 30, "b": 30},
        showlegend=False,
    )
    _apply_depth_range(fig, depth_range)
    return fig


def plot_accommodation(
    calculator: AccommodationSpaceWellCalculator,
    depth_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot accommodation curve with min/max fill.

    Args:
        calculator: Calculator with computed accommodation.
        depth_range: Optional (top, base) depth range.

    Returns:
        Plotly Figure with accommodation tracks.
    """
    fig = go.Figure()
    _plot_uncertainty_fill(
        fig,
        calculator.accommodationCurve,
        "red",
        "App. Accommodation",
    )
    facies_log = cast(
        Striplog,
        calculator._well.getDepthLog(
            next(iter(calculator._well.getDiscreteLogNames()))
        ),
    )
    acco = calculator.accommodationCurve
    x_min = float(np.nanmin(acco.getMinValues()))
    x_max = float(np.nanmax(acco.getMaxValues()))
    _add_interval_hlines(fig, facies_log, x_min, x_max)
    fig.update_yaxes(autorange="reversed", title_text="Depth")
    fig.update_xaxes(title_text="App. Accommodation")
    fig.update_layout(
        margin={"l": 5, "r": 5, "t": 30, "b": 30},
        showlegend=False,
    )
    _apply_depth_range(fig, depth_range)
    return fig


def plot_wd_thickness_ratio(
    calculator: AccommodationSpaceWellCalculator,
    facies_log_name: str,
    depth_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot water depth / thickness ratio with min/max fill.

    Args:
        calculator: Calculator with computed water depth.
        facies_log_name: Name of the facies log.
        depth_range: Optional (top, base) depth range.

    Returns:
        Plotly Figure with ratio tracks.
    """
    ratio_curve = calculator.computeWaterDepthThicknessRatioCurve(
        facies_log_name
    )
    fig = go.Figure()
    _plot_uncertainty_fill(fig, ratio_curve, "green", "WD/Thickness")
    facies_log = cast(
        Striplog,
        calculator._well.getDepthLog(facies_log_name),
    )
    r_min = float(np.nanmin(ratio_curve.getMinValues()))
    r_max = float(np.nanmax(ratio_curve.getMaxValues()))
    _add_interval_hlines(fig, facies_log, r_min, r_max)
    fig.update_yaxes(autorange="reversed", title_text="Depth")
    fig.update_xaxes(title_text="WaterDepth / Thickness")
    fig.update_layout(
        margin={"l": 5, "r": 5, "t": 30, "b": 30},
        showlegend=False,
    )
    _apply_depth_range(fig, depth_range)
    return fig


def _transfer_to_subplot(
    source: go.Figure,
    target: go.Figure,
    col: int,
) -> None:
    """Copy traces and shapes from source into target subplot.

    Args:
        source: Standalone figure to copy from.
        target: Subplot figure to copy into.
        col: Column index in the target subplot (1-based).
    """
    for trace in source.data:
        target.add_trace(trace, row=1, col=col)
    x_ref = f"x{col}" if col > 1 else "x"
    # With shared_yaxes=True all subplots share the first y-axis.
    y_ref = "y"
    for shape in source.layout.shapes:
        shape_dict = shape.to_plotly_json()
        shape_dict["xref"] = x_ref
        shape_dict["yref"] = y_ref
        target.add_shape(**shape_dict)
    for ann in source.layout.annotations:
        ann_dict = ann.to_plotly_json()
        ann_dict["xref"] = x_ref
        ann_dict["yref"] = y_ref
        target.add_annotation(**ann_dict)


def _add_marker_traces_to_subplots(
    fig: go.Figure,
    markers: list[Marker],
    num_cols: int,
    x_ranges: dict[int, tuple[float, float]] | None = None,
) -> int:
    """Add hidden marker traces across all subplot columns.

    Args:
        fig: Target subplot figure.
        markers: List of well markers.
        num_cols: Number of subplot columns.
        x_ranges: Per-column (1-based) x range overrides.
            Defaults to (0, 1) for each column.

    Returns:
        Number of marker traces added.
    """
    count = 0
    for col in range(1, num_cols + 1):
        x_range = x_ranges.get(col, (0.0, 1.0)) if x_ranges else (0.0, 1.0)
        traces = _build_marker_traces(markers, x_range)
        for trace in traces:
            fig.add_trace(trace, row=1, col=col)
            count += 1
    return count


def _build_marker_buttons(
    total_traces: int,
    marker_count: int,
) -> list[dict]:  # type: ignore[type-arg]
    """Build Show/Hide Markers buttons using ``restyle``.

    Uses trace-index targeting so marker visibility is
    independent of other toggle states (e.g. log selection).

    Args:
        total_traces: Total number of traces in the figure.
        marker_count: Number of marker traces (at the end).

    Returns:
        List of two button dicts (show / hide).
    """
    base_count = total_traces - marker_count
    marker_indices = list(range(base_count, total_traces))
    return [
        {
            "label": "Show Markers",
            "method": "restyle",
            "args": [{"visible": True}, marker_indices],
        },
        {
            "label": "Hide Markers",
            "method": "restyle",
            "args": [{"visible": False}, marker_indices],
        },
    ]


def _build_log_buttons(
    litho_groups: dict[str, list[int]],
) -> list[dict]:  # type: ignore[type-arg]
    """Build log-switching buttons using ``restyle``.

    Each button hides all litho groups then shows only the
    selected one, without affecting marker visibility.

    Args:
        litho_groups: Mapping of log name to trace indices.

    Returns:
        List of button dicts (one per log).
    """
    all_litho_indices: list[int] = []
    for indices in litho_groups.values():
        all_litho_indices.extend(indices)

    buttons: list[dict] = []  # type: ignore[type-arg]
    for log_name in sorted(litho_groups.keys()):
        # Build per-trace visible values: False for all
        # litho traces, then True for the selected group.
        vis = dict.fromkeys(all_litho_indices, False)
        for idx in litho_groups[log_name]:
            vis[idx] = True
        ordered_indices = sorted(vis.keys())
        ordered_vis = [vis[i] for i in ordered_indices]
        buttons.append(
            {
                "label": f"Log: {log_name}",
                "method": "restyle",
                "args": [{"visible": ordered_vis}, ordered_indices],
            }
        )
    return buttons


def plot_well_analysis(
    calculator: AccommodationSpaceWellCalculator,
    facies_log_name: str,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """Plot 4-track well analysis figure.

    Tracks: lithology, water depth, accommodation,
    WD/thickness ratio. Shared Y-axis (depth, inverted).
    Includes buttons to show/hide markers and switch
    between available discrete logs.

    Args:
        calculator: Calculator with computed results.
        facies_log_name: Name of the facies log.
        color_map: Optional lithology color map.

    Returns:
        Plotly Figure with 4 subplot columns and toggle
        buttons.
    """
    well = calculator._well
    colors = color_map or DEFAULT_LITHO_COLORS
    num_cols = 4

    fig = make_subplots(
        rows=1,
        cols=num_cols,
        shared_yaxes=True,
        column_titles=[
            "Lithology",
            "Water Depth (m)",
            "App. Accommodation",
            "WD / Thickness",
        ],
        horizontal_spacing=0.03,
    )

    # -- Litho column: build traces for all available logs --
    all_log_names = sorted(well.getDiscreteLogNames())
    litho_groups: dict[str, list[int]] = {}
    trace_idx = 0
    for log_name in all_log_names:
        log_data = cast(Striplog, well.getDepthLog(log_name))
        traces = _build_litho_traces(log_data, colors, log_name)
        indices: list[int] = []
        for trace in traces:
            # Only the active log is visible initially.
            trace.visible = log_name == facies_log_name
            fig.add_trace(trace, row=1, col=1)
            indices.append(trace_idx)
            trace_idx += 1
        litho_groups[log_name] = indices

    # -- Data tracks --
    wd_fig = plot_water_depth(calculator)
    _transfer_to_subplot(wd_fig, fig, 2)

    acco_fig = plot_accommodation(calculator)
    _transfer_to_subplot(acco_fig, fig, 3)

    ratio_fig = plot_wd_thickness_ratio(calculator, facies_log_name)
    _transfer_to_subplot(ratio_fig, fig, 4)

    # -- Marker traces (hidden by default) --
    markers = well.getMarkers()
    marker_count = 0
    if markers:
        marker_count = _add_marker_traces_to_subplots(fig, markers, num_cols)

    # -- Toggle buttons --
    menus: list[dict] = []  # type: ignore[type-arg]
    if markers:
        menus.append(
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": _build_marker_buttons(len(fig.data), marker_count),
            }
        )
    if len(all_log_names) > 1:
        menus.append(
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.5,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": _build_log_buttons(litho_groups),
            }
        )

    fig.update_yaxes(
        autorange="reversed",
        title_text="Depth",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showticklabels=False,
        range=[0, 1],
        row=1,
        col=1,
    )
    fig.update_layout(
        height=600,
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        updatemenus=menus if menus else None,
    )
    return fig


_TRACK_FUNCS: dict[str, _Callable[..., go.Figure]] = {
    "water_depth": plot_water_depth,
    "accommodation": plot_accommodation,
}


def plot_well_comparison(
    calculators: dict[str, AccommodationSpaceWellCalculator],
    facies_log_name_per_well: dict[str, str],
    color_map: dict[str, str] | None = None,
    track: str = "accommodation",
) -> go.Figure:
    """Plot wells side by side for comparison.

    Each well gets two columns: lithology + selected track.
    Shared Y-axis across all wells, scaled to max depth.
    Includes a button to show/hide markers.

    Args:
        calculators: Calculator per well name.
        facies_log_name_per_well: Facies log name per well.
        color_map: Optional lithology color map.
        track: One of ``"water_depth"``,
            ``"accommodation"``, ``"wd_thickness_ratio"``.

    Returns:
        Plotly Figure with 2*N subplot columns and marker
        toggle button.
    """
    well_names = sorted(calculators.keys())
    n = len(well_names)
    if n == 0:
        return go.Figure()

    num_cols = 2 * n
    titles: list[str] = []
    for name in well_names:
        titles.extend([name, track.replace("_", " ").title()])

    fig = make_subplots(
        rows=1,
        cols=num_cols,
        shared_yaxes=True,
        column_titles=titles,
        horizontal_spacing=0.02,
    )

    max_depth = 0.0
    for name in well_names:
        calc = calculators[name]
        max_depth = max(max_depth, calc._well.depth)
    depth_range = (0.0, max_depth)

    for i, name in enumerate(well_names):
        calc = calculators[name]
        log_name = facies_log_name_per_well.get(name, "")
        col_litho = 2 * i + 1
        col_track = 2 * i + 2

        litho_fig = plot_litho_log(
            calc._well, log_name, color_map, depth_range=depth_range
        )
        _transfer_to_subplot(litho_fig, fig, col_litho)

        if track == "wd_thickness_ratio":
            track_fig: go.Figure = plot_wd_thickness_ratio(
                calc, log_name, depth_range=depth_range
            )
        elif track in _TRACK_FUNCS:
            track_fig = _TRACK_FUNCS[track](calc, depth_range=depth_range)
        else:
            track_fig = plot_accommodation(calc, depth_range=depth_range)
        _transfer_to_subplot(track_fig, fig, col_track)

        fig.update_xaxes(
            showticklabels=False,
            range=[0, 1],
            row=1,
            col=col_litho,
        )

    # -- Marker traces (hidden by default) across all wells --
    all_markers: list[Marker] = []
    for name in well_names:
        all_markers.extend(calculators[name]._well.getMarkers())
    marker_count = 0
    if all_markers:
        # Add per-well markers to that well's columns only.
        for i, name in enumerate(well_names):
            markers = calculators[name]._well.getMarkers()
            if markers:
                col_litho = 2 * i + 1
                col_track = 2 * i + 2
                for col in (col_litho, col_track):
                    x_range = (0.0, 1.0)
                    traces = _build_marker_traces(markers, x_range)
                    for tr in traces:
                        fig.add_trace(tr, row=1, col=col)
                        marker_count += 1

    menus: list[dict] = []  # type: ignore[type-arg]
    if marker_count:
        menus.append(
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": _build_marker_buttons(len(fig.data), marker_count),
            }
        )

    fig.update_yaxes(
        autorange="reversed",
        title_text="Depth",
        row=1,
        col=1,
    )
    fig.update_layout(
        height=600,
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        updatemenus=menus if menus else None,
    )
    return fig
