from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib.axes import Axes
from shapely.geometry import LineString, Point, Polygon

from commute_analysis.visualization import (
    _format_time_range,
    _normalize_mode_name,
    plot_commute_map,
)


def _probability_grid(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    geoms = [
        Polygon([(10.0, 53.5), (10.01, 53.5), (10.01, 53.51), (10.0, 53.51)]),
        Polygon([(10.01, 53.5), (10.02, 53.5), (10.02, 53.51), (10.01, 53.51)]),
    ]
    return gpd.GeoDataFrame(
        {
            "sampling_probability": [0.1, 0.9],
            "probability_density_km2": [100.0, 900.0],
        },
        geometry=geoms,
        crs=crs,
    )


def _employees(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "employee_id": ["EMP-001", "EMP-002"],
        },
        geometry=[Point(10.005, 53.505), Point(10.015, 53.505)],
        crs=crs,
    )


def _target_area(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    geom = Polygon([(9.99, 53.49), (10.03, 53.49), (10.03, 53.52), (9.99, 53.52)])
    return gpd.GeoDataFrame({"name": ["target"]}, geometry=[geom], crs=crs)


def _route(
    option_values: list[int] | None = None,
    segments: int = 2,
    crs: str | None = "EPSG:4326",
) -> gpd.GeoDataFrame:
    options = option_values if option_values is not None else [0] * segments
    rows = []
    geoms = []
    for i in range(segments):
        geoms.append(LineString([(10.0 + i * 0.005, 53.5), (10.005 + i * 0.005, 53.505)]))
        rows.append(
            {
                "option": options[i],
                "segment": i + 1,
                "leg_index": i,
                "transport_mode": "walk" if i % 2 == 0 else "bus",
                "leg_mode": "walk" if i % 2 == 0 else "bus",
                "departure_time": pd.Timestamp("2026-07-21 07:30") + pd.Timedelta(minutes=i * 10),
                "arrival_time": pd.Timestamp("2026-07-21 07:35") + pd.Timedelta(minutes=i * 10),
                "travel_time": pd.Timedelta(minutes=5),
                "distance": 500.0,
            }
        )
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def test_matplotlib_return_type() -> None:
    ax = plot_commute_map(probability_grid=_probability_grid(), add_basemap=False)
    assert isinstance(ax, Axes)


def test_invalid_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        plot_commute_map(probability_grid=_probability_grid(), backend="bad", add_basemap=False)


def test_missing_crs() -> None:
    with pytest.raises(ValueError, match="must have a CRS"):
        plot_commute_map(probability_grid=_probability_grid(crs=None), add_basemap=False)


def test_missing_probability_column() -> None:
    grid = _probability_grid().drop(columns=["sampling_probability"])
    with pytest.raises(ValueError, match="must contain"):
        plot_commute_map(probability_grid=grid, add_basemap=False)


def test_empty_enabled_layer() -> None:
    with pytest.raises(ValueError, match="No enabled layers"):
        plot_commute_map(add_basemap=False)


def test_probability_grid_only_plot() -> None:
    ax = plot_commute_map(probability_grid=_probability_grid(), show_colorbar=False, add_basemap=False)
    assert isinstance(ax, Axes)


def test_employee_only_plot() -> None:
    ax = plot_commute_map(
        employees=_employees(),
        show_probability_grid=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_route_only_plot() -> None:
    ax = plot_commute_map(
        routes=_route(segments=1),
        show_probability_grid=False,
        show_employees=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_combined_plot() -> None:
    ax = plot_commute_map(
        probability_grid=_probability_grid(),
        employees=_employees(),
        target_area=_target_area(),
        workplace=(53.51, 10.02),
        routes=_route(segments=2),
        route_origins=(53.5, 10.0),
        route_destinations=(53.51, 10.02),
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_one_segment_route() -> None:
    ax = plot_commute_map(
        routes=_route(segments=1),
        show_probability_grid=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_multi_segment_route() -> None:
    ax = plot_commute_map(
        routes=_route(segments=3),
        show_probability_grid=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_multiple_routes() -> None:
    route_a = _route(segments=2)
    route_b = _route(segments=2)
    ax = plot_commute_map(
        routes=[route_a, route_b],
        route_origins=[(53.5, 10.0), (53.5, 10.002)],
        route_destinations=(53.51, 10.02),
        show_probability_grid=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_multiple_route_options_without_selection_raises() -> None:
    route = _route(option_values=[0, 1], segments=2)
    with pytest.raises(ValueError, match="multiple itinerary options"):
        plot_commute_map(routes=route, show_probability_grid=False, add_basemap=False)


def test_route_option_selection_valid() -> None:
    route = _route(option_values=[0, 1], segments=2)
    ax = plot_commute_map(
        routes=route,
        route_options=1,
        show_probability_grid=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_route_origin_destination_length_mismatch() -> None:
    route_a = _route(segments=2)
    route_b = _route(segments=2)
    with pytest.raises(ValueError, match="route_origins length"):
        plot_commute_map(
            routes=[route_a, route_b],
            route_origins=[(53.5, 10.0), (53.5, 10.01), (53.5, 10.02)],
            show_probability_grid=False,
            add_basemap=False,
        )


def test_hide_legend() -> None:
    ax = plot_commute_map(
        routes=_route(segments=2),
        show_probability_grid=False,
        show_legend=False,
        add_basemap=False,
    )
    assert ax.get_legend() is None


def test_hide_route_origins() -> None:
    ax = plot_commute_map(
        routes=_route(segments=2),
        route_origins=(53.5, 10.0),
        show_probability_grid=False,
        show_route_origins=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_hide_route_destinations() -> None:
    ax = plot_commute_map(
        routes=_route(segments=2),
        route_destinations=(53.51, 10.02),
        show_probability_grid=False,
        show_route_destinations=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_hide_route_times() -> None:
    ax = plot_commute_map(
        routes=_route(segments=2),
        show_probability_grid=False,
        show_route_times=False,
        add_basemap=False,
    )
    assert isinstance(ax, Axes)


def test_mode_name_normalization() -> None:
    assert _normalize_mode_name("public_transport") == "Public transport"
    assert _normalize_mode_name("bicycle") == "Bike"
    assert _normalize_mode_name("tram") == "Tram"


def test_route_time_formatting() -> None:
    formatted = _format_time_range("2026-07-21T07:30:00", "2026-07-21T07:35:00")
    assert formatted == "07:30\u201307:35"


def test_reuse_existing_matplotlib_axis() -> None:
    _, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
    first = plot_commute_map(probability_grid=_probability_grid(), ax=ax, show_colorbar=False, add_basemap=False)
    second = plot_commute_map(
        employees=_employees(),
        routes=_route(segments=1),
        show_probability_grid=False,
        add_basemap=False,
        ax=ax,
    )
    assert first is ax
    assert second is ax


def test_folium_return_type() -> None:
    folium = pytest.importorskip("folium")
    fmap = plot_commute_map(probability_grid=_probability_grid(), backend="folium")
    assert isinstance(fmap, folium.Map)


def test_reuse_existing_folium_map() -> None:
    folium = pytest.importorskip("folium")
    fmap = plot_commute_map(probability_grid=_probability_grid(), backend="folium")
    fmap2 = plot_commute_map(
        employees=_employees(),
        routes=_route(segments=1),
        backend="folium",
        show_probability_grid=False,
        folium_map=fmap,
    )
    assert isinstance(fmap2, folium.Map)
    assert fmap2 is fmap
