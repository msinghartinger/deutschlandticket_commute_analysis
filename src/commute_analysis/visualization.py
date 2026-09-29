from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString, Point

from . import config

if TYPE_CHECKING:
    import folium
    from matplotlib.axes import Axes


PointLike = Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame
"""Type alias for point-like geometry inputs (Point, tuple, GeoSeries, GeoDataFrame)."""

RouteInput = gpd.GeoDataFrame | Iterable[gpd.GeoDataFrame]
"""Type alias for route input: single GeoDataFrame or iterable of GeoDataFrames."""

RouteColorStrategy = Literal["segment", "mode", "route"]
"""Literal type for route coloring strategies.

Options:
- "segment": Color by route segment index
- "mode": Color by transport mode
- "route": Color by route index
"""

Backend = Literal["matplotlib", "folium"]
"""Literal type for rendering backends.

Options:
- "matplotlib": Use Matplotlib for rendering
- "folium": Use Folium for interactive maps
"""

_OSM_TILE_HEADERS = {"User-Agent": "deutschlandticket-commute-analysis/0.1.0"}


def _normalize_backend(backend: str) -> Backend:
    """Normalize backend string to lowercase and validate.
    
    Parameters
    ----------
    backend : str
        The backend name to normalize.
    
    Returns
    -------
    Backend
        Normalized backend name as a literal.
    
    Raises
    ------
    ValueError
        If the backend is not "matplotlib" or "folium".
    """
    normalized = backend.lower().strip()
    if normalized not in {"matplotlib", "folium"}:
        raise ValueError("Unsupported backend. Use 'matplotlib' or 'folium'.")
    return normalized  # type: ignore[return-value]


def _require_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Ensure a GeoDataFrame has a valid Coordinate Reference System.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to check.
    name : str
        Description of the GeoDataFrame for error messages.
    
    Raises
    ------
    ValueError
        If the GeoDataFrame has no CRS set.
    """
    if gdf.crs is None:
        raise ValueError(f"{name} must have a CRS.")


def _to_geodataframe(value: gpd.GeoDataFrame | gpd.GeoSeries, name: str) -> gpd.GeoDataFrame:
    """Convert GeoSeries or GeoDataFrame to a proper GeoDataFrame.
    
    Parameters
    ----------
    value : gpd.GeoDataFrame | gpd.GeoSeries
        The input geometry data.
    name : str
        Description of the input for error messages.
    
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the input geometry.
    
    Raises
    ------
    TypeError
        If the input is not a GeoDataFrame or GeoSeries.
    ValueError
        If the input does not contain a geometry column or has no CRS.
    """
    if isinstance(value, gpd.GeoDataFrame):
        gdf = value.copy()
    elif isinstance(value, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=value.copy(), crs=value.crs)
    else:
        raise TypeError(f"{name} must be a GeoDataFrame or GeoSeries.")

    if "geometry" not in gdf.columns:
        raise ValueError(f"{name} must contain a geometry column.")
    _require_crs(gdf, name)
    return gdf


def _is_point_like(value: object) -> bool:
    """Check if a value is point-like (Point, tuple, GeoSeries, or GeoDataFrame).
    
    Parameters
    ----------
    value : object
        The value to check.
    
    Returns
    -------
    bool
        True if the value is point-like, False otherwise.
    """
    return isinstance(value, (Point, tuple, gpd.GeoSeries, gpd.GeoDataFrame))


def _point_to_gdf(value: PointLike, target_crs: Any, name: str) -> gpd.GeoDataFrame:
    """Convert a point-like input to a GeoDataFrame with the target CRS.
    
    Parameters
    ----------
    value : PointLike
        The point-like input (Point, tuple, GeoSeries, or GeoDataFrame).
    target_crs : Any
        The target Coordinate Reference System.
    name : str
        Description of the input for error messages.
    
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the point(s) in the target CRS.
    
    Raises
    ------
    ValueError
        If the input is invalid (empty, wrong dimensions, etc.).
    TypeError
        If the input type is not supported.
    """
    if isinstance(value, Point):
        gdf = gpd.GeoDataFrame(geometry=[value], crs=config.CRS_WGS84)
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} tuple must be (latitude, longitude).")
        lat, lon = value
        gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=config.CRS_WGS84)
    elif isinstance(value, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=value.copy(), crs=value.crs)
    elif isinstance(value, gpd.GeoDataFrame):
        gdf = value.copy()
    else:
        raise TypeError(f"Unsupported point input for {name}.")

    _require_crs(gdf, name)
    if gdf.empty:
        raise ValueError(f"{name} is empty.")
    geom_types = set(gdf.geometry.geom_type.dropna().unique())
    if not geom_types.issubset({"Point", "MultiPoint"}):
        raise ValueError(f"{name} must contain Point geometries.")
    return gdf.to_crs(target_crs)


def _normalize_routes(routes: RouteInput | None) -> list[gpd.GeoDataFrame]:
    """Normalize route input to a list of GeoDataFrames.
    
    Parameters
    ----------
    routes : RouteInput | None
        The route input to normalize.
    
    Returns
    -------
    list[gpd.GeoDataFrame]
        A list of normalized GeoDataFrames.
    
    Raises
    ------
    ValueError
        If any route is missing a geometry column or has no CRS.
    """
    if routes is None:
        return []
    if isinstance(routes, gpd.GeoDataFrame):
        route_list = [routes.copy()]
    else:
        route_list = [route.copy() for route in routes]

    for idx, route in enumerate(route_list):
        if "geometry" not in route.columns:
            raise ValueError(f"Route {idx + 1} is missing a geometry column.")
        _require_crs(route, f"Route {idx + 1}")
    return route_list


def _expand_point_inputs(
    values: PointLike | Iterable[PointLike] | None,
    count: int,
    name: str,
) -> list[PointLike | None]:
    """Expand point inputs to match the required count.
    
    Parameters
    ----------
    values : PointLike | Iterable[PointLike] | None
        The point-like values to expand.
    count : int
        The number of values needed.
    name : str
        Description of the input for error messages.
    
    Returns
    -------
    list[PointLike | None]
        A list of point-like values, possibly repeated or filled with None.
    
    Raises
    ------
    ValueError
        If the input length doesn't match the required count.
    """
    if count == 0:
        return []
    if values is None:
        return [None] * count

    if _is_point_like(values):
        return [values] * count

    items = list(values)  # type: ignore[arg-type]
    if len(items) == 1 and count > 1:
        return items * count
    if len(items) != count:
        raise ValueError(f"{name} length must match number of routes.")
    return items


def _expand_route_options(
    route_options: int | Iterable[int | None] | None,
    count: int,
) -> list[int | None]:
    """Expand route options to match the number of routes.
    
    Parameters
    ----------
    route_options : int | Iterable[int | None] | None
        The route options to expand.
    count : int
        The number of routes.
    
    Returns
    -------
    list[int | None]
        A list of route options, possibly repeated or filled with None.
    
    Raises
    ------
    ValueError
        If the options length doesn't match the number of routes.
    """
    if count == 0:
        return []
    if route_options is None:
        return [None] * count
    if isinstance(route_options, int):
        return [route_options] * count
    options = list(route_options)
    if len(options) != count:
        raise ValueError("route_options length must match number of routes.")
    return options


def _select_route_option(route: gpd.GeoDataFrame, option: int | None, route_index: int) -> gpd.GeoDataFrame:
    """Select a specific route option from a GeoDataFrame or return all rows.
    
    Parameters
    ----------
    route : gpd.GeoDataFrame
        The route GeoDataFrame containing an 'option' column.
    option : int | None
        The option index to select, or None to keep all options.
    route_index : int
        The route index for error messages.
    
    Returns
    -------
    gpd.GeoDataFrame
        The filtered GeoDataFrame with the selected option, or a copy if no filtering needed.
    
    Raises
    ------
    ValueError
        If multiple options exist but no option is specified, or if the option is not found.
    """
    if "option" not in route.columns:
        return route.copy()

    unique_options = [opt for opt in pd.Series(route["option"]).dropna().unique().tolist()]
    if len(unique_options) <= 1:
        return route.copy()
    if option is None:
        raise ValueError(
            f"Route {route_index + 1} contains multiple itinerary options. "
            "Specify route_options to select one option per route."
        )

    selected = route.loc[route["option"] == option].copy()
    if selected.empty:
        raise ValueError(f"Selected route option {option} not found in route {route_index + 1}.")
    return selected


def _sorted_route_segments(route: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sort route segments by segment and leg_index columns if they exist.
    
    Parameters
    ----------
    route : gpd.GeoDataFrame
        The route GeoDataFrame to sort.
    
    Returns
    -------
    gpd.GeoDataFrame
        A sorted copy of the route GeoDataFrame, or the original if no sorting columns.
    """
    sort_cols: list[str] = []
    for col in ["segment", "leg_index"]:
        if col in route.columns:
            sort_cols.append(col)
    if sort_cols:
        return route.sort_values(sort_cols).copy()
    return route.copy()


def _normalize_mode_name(mode_value: object) -> str:
    """Normalize transport mode names to standard capitalization.
    
    Parameters
    ----------
    mode_value : object
        The mode value to normalize.
    
    Returns
    -------
    str
        The normalized mode name with proper capitalization.
    """
    if mode_value is None or (isinstance(mode_value, float) and pd.isna(mode_value)):
        return "Unknown"

    text = str(mode_value).strip().lower()
    if "." in text:
        text = text.split(".")[-1]

    mapping = {
        "walk": "Walk",
        "walking": "Walk",
        "foot": "Walk",
        "bike": "Bike",
        "bicycle": "Bike",
        "car": "Car",
        "bus": "Bus",
        "rail": "Rail",
        "train": "Rail",
        "subway": "Subway",
        "metro": "Subway",
        "tram": "Tram",
        "ferry": "Ferry",
        "public_transport": "Public transport",
        "public transit": "Public transport",
        "transit": "Public transport",
    }
    return mapping.get(text, text.replace("_", " ").title())


def _to_timestamp(value: object) -> pd.Timestamp | None:
    """Convert a value to a pandas Timestamp, returning None if conversion fails.
    
    Parameters
    ----------
    value : object
        The value to convert to a Timestamp.
    
    Returns
    -------
    pd.Timestamp | None
        The Timestamp if successful, None otherwise.
    """
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _format_time_range(departure: object, arrival: object, travel_time: object = None) -> str | None:
    """Format a time range string from departure, arrival, and/or travel time.
    
    Parameters
    ----------
    departure : object
        The departure time value.
    arrival : object
        The arrival time value.
    travel_time : object, optional
        The travel time in time units (hours, minutes, etc.).
    
    Returns
    -------
    str | None
        Formatted time range string, or None if unable to format.
    """
    dep = _to_timestamp(departure)
    arr = _to_timestamp(arrival)

    if dep is not None and arr is None and travel_time is not None:
        td = pd.to_timedelta(travel_time, errors="coerce")
        if pd.notna(td):
            arr = dep + pd.Timedelta(td)

    if dep is None or arr is None:
        return None
    return f"{dep.strftime('%H:%M')}\u2013{arr.strftime('%H:%M')}"


def _segment_mode(row: pd.Series) -> str:
    """Extract and normalize the transport mode from a route row.
    
    Parameters
    ----------
    row : pd.Series
        A row from a route GeoDataFrame.
    
    Returns
    -------
    str
        The normalized transport mode name.
    """
    if "leg_mode" in row and pd.notna(row.get("leg_mode")):
        return _normalize_mode_name(row.get("leg_mode"))
    if "transport_mode" in row and pd.notna(row.get("transport_mode")):
        return _normalize_mode_name(row.get("transport_mode"))
    return "Unknown"


def _segment_label(
    row: pd.Series,
    route_index: int,
    show_route_ids: bool,
    show_route_modes: bool,
    show_route_times: bool,
) -> str | None:
    """Generate a label for a route segment combining route info, mode, and times.
    
    Parameters
    ----------
    row : pd.Series
        A row from a route GeoDataFrame.
    route_index : int
        The route index for labeling.
    show_route_ids : bool
        Whether to include route IDs in the label.
    show_route_modes : bool
        Whether to include transport modes in the label.
    show_route_times : bool
        Whether to include time information in the label.
    
    Returns
    -------
    str | None
        The formatted label string, or None if no components are shown.
    """
    pieces: list[str] = []
    if show_route_ids:
        pieces.append(f"Route {route_index + 1}")

    mode_text = _segment_mode(row)
    if show_route_modes:
        pieces.append(mode_text)

    time_text = _format_time_range(
        row.get("departure_time"),
        row.get("arrival_time"),
        row.get("travel_time"),
    )
    if show_route_times and time_text is not None:
        pieces.append(time_text)

    if not pieces:
        return None
    return " \u00b7 ".join(pieces)


def _validate_probability_grid(probability_grid: gpd.GeoDataFrame, probability_column: str) -> None:
    """Validate a probability grid GeoDataFrame.
    
    Parameters
    ----------
    probability_grid : gpd.GeoDataFrame
        The probability grid to validate.
    probability_column : str
        The name of the probability column.
    
    Raises
    ------
    ValueError
        If the grid is empty or missing the required column.
    """
    _require_crs(probability_grid, "probability_grid")
    if probability_grid.empty:
        raise ValueError("probability_grid is empty.")
    if probability_column not in probability_grid.columns:
        raise ValueError(
            f"probability_grid must contain '{probability_column}'."
        )


def _validate_employee_points(employees: gpd.GeoDataFrame) -> None:
    """Validate employee points GeoDataFrame.
    
    Parameters
    ----------
    employees : gpd.GeoDataFrame
        The employee points to validate.
    
    Raises
    ------
    ValueError
        If the GeoDataFrame is empty or doesn't contain Point geometries.
    """
    _require_crs(employees, "employees")
    if employees.empty:
        raise ValueError("employees is empty.")
    geom_types = set(employees.geometry.geom_type.dropna().unique())
    if not geom_types.issubset({"Point", "MultiPoint"}):
        raise ValueError("employees must contain Point geometries.")


def _get_target_crs(
    backend: Backend,
    add_basemap: bool,
    first_crs: Any,
) -> Any:
    """Determine the target CRS based on backend and basemap settings.
    
    Parameters
    ----------
    backend : Backend
        The rendering backend.
    add_basemap : bool
        Whether to add a basemap layer.
    first_crs : Any
        The first CRS found in the input data.
    
    Returns
    -------
    Any
        The target CRS string.
    """
    if backend == "folium":
        return config.CRS_WGS84
    if backend == "matplotlib" and add_basemap:
        return config.CRS_WEB_MERCATOR
    return first_crs


def _series_bounds(geometries: Sequence[gpd.GeoDataFrame]) -> tuple[float, float, float, float] | None:
    """Calculate bounding box from multiple GeoDataFrames.
    
    Parameters
    ----------
    geometries : Sequence[gpd.GeoDataFrame]
        A sequence of GeoDataFrames to get bounds from.
    
    Returns
    -------
    tuple[float, float, float, float] | None
        The bounding box as (minx, miny, maxx, maxy), or None if no data.
    """
    bounds: list[tuple[float, float, float, float]] = []
    for gdf in geometries:
        if gdf is None or gdf.empty:
            continue
        b = gdf.total_bounds
        if not pd.isna(b).any():
            bounds.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    if not bounds:
        return None

    minx = min(b[0] for b in bounds)
    miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds)
    maxy = max(b[3] for b in bounds)
    return minx, miny, maxx, maxy


def _has_data_for_enabled_layers(
    *,
    show_probability_grid: bool,
    probability_grid: gpd.GeoDataFrame | None,
    show_employees: bool,
    employees: gpd.GeoDataFrame | None,
    show_target_area: bool,
    target_area: gpd.GeoDataFrame | gpd.GeoSeries | None,
    show_workplace: bool,
    workplace: PointLike | None,
    show_routes: bool,
    route_count: int,
    show_route_origins: bool,
    route_origins: PointLike | Iterable[PointLike] | None,
    show_route_destinations: bool,
    route_destinations: PointLike | Iterable[PointLike] | None,
) -> bool:
    """Check if any enabled layer has data.
    
    Parameters
    ----------
    show_probability_grid : bool
        Whether to show the probability grid layer.
    probability_grid : gpd.GeoDataFrame | None
        The probability grid data.
    show_employees : bool
        Whether to show employee points.
    employees : gpd.GeoDataFrame | None
        The employee points data.
    show_target_area : bool
        Whether to show the target area.
    target_area : gpd.GeoDataFrame | gpd.GeoSeries | None
        The target area data.
    show_workplace : bool
        Whether to show the workplace point.
    workplace : PointLike | None
        The workplace point data.
    show_routes : bool
        Whether to show routes.
    route_count : int
        The number of routes.
    show_route_origins : bool
        Whether to show route origins.
    route_origins : PointLike | Iterable[PointLike] | None
        The route origins data.
    show_route_destinations : bool
        Whether to show route destinations.
    route_destinations : PointLike | Iterable[PointLike] | None
        The route destinations data.
    
    Returns
    -------
    bool
        True if at least one enabled layer has data.
    """
    return any((
        show_probability_grid and probability_grid is not None,
        show_employees and employees is not None,
        show_target_area and target_area is not None,
        show_workplace and workplace is not None,
        show_routes and route_count > 0,
        show_route_origins and route_origins is not None and route_count > 0,
        show_route_destinations and route_destinations is not None and route_count > 0,
    ))


def _require_optional_dependency(module_name: str, install_hint: str) -> Any:
    """Require an optional module and raise ImportError with hint if missing.
    
    Parameters
    ----------
    module_name : str
        The name of the module to import.
    install_hint : str
        Installation hint to include in the error message.
    
    Returns
    -------
    Any
        The imported module object.
    
    Raises
    ------
    ImportError
        If the module is not installed.
    """
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(install_hint) from exc


def _color_for_segment(
    *,
    route_index: int,
    segment_index: int,
    mode_name: str,
    strategy: RouteColorStrategy,
    cmap_name: str = "tab20",
) -> tuple[float, float, float, float]:
    """Calculate the RGBA color for a route segment.
    
    Parameters
    ----------
    route_index : int
        The route index.
    segment_index : int
        The segment index within the route.
    mode_name : str
        The transport mode name.
    strategy : RouteColorStrategy
        The coloring strategy to use.
    cmap_name : str, default "tab20"
        The colormap name to use.
    
    Returns
    -------
    tuple[float, float, float, float]
        The RGBA color tuple.
    """
    cmap = plt.get_cmap(cmap_name)
    if strategy == "route":
        value = (route_index % 20) / 20
    elif strategy == "mode":
        mode_key = hash(mode_name) % 20
        value = mode_key / 20
    else:
        value = (segment_index % 20) / 20
    return cmap(value)


def _line_midpoint(geom: object) -> Point | None:
    """Get the midpoint of a line geometry.
    
    Parameters
    ----------
    geom : object
        The geometry to get the midpoint from.
    
    Returns
    -------
    Point | None
        The midpoint Point, or None if not applicable.
    """
    if isinstance(geom, LineString):
        return geom.interpolate(0.5, normalized=True)
    if isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
        return geom.geoms[0].interpolate(0.5, normalized=True)
    return None


def plot_transport_score_routes(
    *,
    summary_gdf: gpd.GeoDataFrame,
    route_gdf: gpd.GeoDataFrame | None = None,
    workplace: PointLike,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 10),
    ax: Axes | None = None,
    add_basemap: bool = True,
    show_axis: bool = False,
    fit_bounds: bool = True,
    fit_bounds_padding_fraction: float = 0.02,
    home_markersize: float = 28,
    workplace_markersize: float = 140,
    route_linewidth: float = 4,
    route_alpha: float = 0.95,
    home_alpha: float = 0.95,
    score_cmap: str = "viridis",
    show_colorbar: bool = True,
    colorbar_label: str = "Transport score",
    basemap_alpha: float = 0.5,
    show_home_component_labels: bool = True,
    label_every_nth_home: int = 10,
    label_fontsize: float = 7.5,
    label_alpha: float = 0.9,
) -> Axes:
    """Plot employee homes and optional routes colored by transport score.
    
    This function creates a map visualization showing employee home locations and
    their morning routes, with routes colored by transport score.
    
    Parameters
    ----------
    summary_gdf : gpd.GeoDataFrame
        GeoDataFrame containing employee_id and transport_score columns.
    route_gdf : gpd.GeoDataFrame | None, optional
        GeoDataFrame containing employee_id and morning_q25_route columns.
        If omitted or empty, only homes and workplace are plotted.
    workplace : PointLike
        Point location of the workplace.
    title : str | None, optional
        Title for the plot.
    figsize : tuple[float, float], default (12, 10)
        Figure size in inches.
    ax : Axes | None, optional
        Existing matplotlib Axes to plot on.
    add_basemap : bool, default True
        Whether to add a basemap layer.
    show_axis : bool, default False
        Whether to show the plot axes.
    fit_bounds : bool, default True
        Whether to fit the plot bounds to the data.
    fit_bounds_padding_fraction : float, default 0.02
        Fractional padding around the bounds.
    home_markersize : float, default 28
        Marker size for employee homes.
    workplace_markersize : float, default 140
        Marker size for workplace.
    route_linewidth : float, default 4
        Line width for routes.
    route_alpha : float, default 0.95
        Transparency for routes.
    home_alpha : float, default 0.95
        Transparency for employee homes.
    score_cmap : str, default "viridis"
        Colormap for transport score.
    show_colorbar : bool, default True
        Whether to show the colorbar.
    colorbar_label : str, default "Transport score"
        Label for the colorbar.
    basemap_alpha : float, default 0.5
        Transparency for the basemap.
    show_home_component_labels : bool, default True
        Whether to annotate every Nth home with concise component scores.
    label_every_nth_home : int, default 10
        Label every Nth home point (10 means 10th, 20th, 30th, ...).
    label_fontsize : float, default 7.5
        Font size used for home component score labels.
    label_alpha : float, default 0.9
        Alpha used for home component score labels.
    
    Returns
    -------
    Axes
        The matplotlib Axes object with the plot.
    
    Raises
    ------
    ValueError
        If ``summary_gdf`` is empty or missing required columns.
    """
    if summary_gdf.empty:
        raise ValueError("summary_gdf is empty.")

    for required in ["employee_id", "transport_score"]:
        if required not in summary_gdf.columns:
            raise ValueError(f"summary_gdf must contain '{required}'.")

    _require_crs(summary_gdf, "summary_gdf")

    merged_plot: gpd.GeoDataFrame | None = None
    if route_gdf is not None and not route_gdf.empty:
        for required in ["employee_id", "morning_q25_route"]:
            if required not in route_gdf.columns:
                raise ValueError(f"route_gdf must contain '{required}'.")

        _require_crs(route_gdf, "route_gdf")
        merged = route_gdf.merge(
            summary_gdf[["employee_id", "transport_score"]],
            on="employee_id",
            how="inner",
        )
        if not merged.empty:
            merged_plot = gpd.GeoDataFrame(merged, geometry="morning_q25_route", crs=route_gdf.crs)

    first_crs = summary_gdf.crs if summary_gdf.crs is not None else (route_gdf.crs if route_gdf is not None else None)
    target_crs = _get_target_crs("matplotlib", add_basemap, first_crs)
    summary_plot = summary_gdf.to_crs(target_crs).copy()
    if merged_plot is not None:
        merged_plot = merged_plot.to_crs(target_crs)
    workplace_plot = _point_to_gdf(workplace, target_crs, "workplace")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if add_basemap:
        contextily = _require_optional_dependency(
            "contextily",
            "contextily is required when add_basemap=True for matplotlib backend.",
        )

    score_values = pd.to_numeric(summary_plot["transport_score"], errors="coerce")
    valid_scores = score_values.dropna()
    if valid_scores.empty:
        raise ValueError("summary_gdf transport_score contains no numeric values.")

    score_min = float(valid_scores.min())
    score_max = float(valid_scores.max())
    if score_max > score_min:
        norm = mcolors.Normalize(vmin=score_min, vmax=score_max)
    else:
        # Avoid a zero-width normalization range when all scores are identical.
        pad = max(abs(score_min) * 0.01, 1e-9)
        norm = mcolors.Normalize(vmin=score_min - pad, vmax=score_max + pad)

    summary_plot = summary_plot.assign(_plot_transport_score=score_values)
    cmap = plt.get_cmap(score_cmap)
    perfect_color = cmap(1.0)

    component_cols = [
        "relative_time_score",
        "absolute_time_score",
        "consistency_score",
        "walking_score",
        "transfers_score",
    ]

    if merged_plot is not None:
        for _, row in merged_plot.iterrows():
            geom = row.get("morning_q25_route")
            score = pd.to_numeric(pd.Series([row.get("transport_score")]), errors="coerce").iloc[0]
            if geom is None or getattr(geom, "is_empty", True):
                continue
            if pd.isna(score):
                continue
            route_color = cmap(norm(float(score)))
            gpd.GeoSeries([geom], crs=merged_plot.crs).plot(
                ax=ax,
                color=[route_color],
                linewidth=route_linewidth,
                alpha=route_alpha,
                zorder=4,
            )

    summary_plot.plot(
        ax=ax,
        column="_plot_transport_score",
        cmap=score_cmap,
        norm=norm,
        markersize=home_markersize,
        alpha=home_alpha,
        zorder=5,
    )

    if show_home_component_labels and label_every_nth_home > 0:
        available_cols = [col for col in component_cols if col in summary_plot.columns]
        if len(available_cols) == len(component_cols):
            initials = {
                "absolute_time_score": "A",
                "relative_time_score": "R",
                "consistency_score": "C",
                "walking_score": "W",
                "transfers_score": "T",
            }
            for idx, (_, row) in enumerate(summary_plot.iterrows(), start=1):
                if idx % label_every_nth_home != 0:
                    continue
                geom = row.get("geometry")
                if geom is None or getattr(geom, "is_empty", True):
                    continue

                parts: list[str] = []
                for col in component_cols:
                    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                    if pd.isna(value):
                        parts.append(f"{initials[col]}:na")
                    else:
                        parts.append(f"{initials[col]}:{float(value):.2f}")

                ax.annotate(
                    " ".join(parts),
                    xy=(geom.x, geom.y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    alpha=label_alpha,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "alpha": 0.7,
                        "edgecolor": "none",
                    },
                    zorder=8,
                )

    workplace_plot.plot(
        ax=ax,
        marker="*",
        color="red",
        markersize=workplace_markersize,
        zorder=6,
    )

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=mcolors.to_hex(perfect_color), markersize=12),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markersize=8),
    ]
    ax.legend(legend_handles, ["Workplace", "Employee homes"], loc="best")

    if show_colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=score_cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label=colorbar_label)

    bounds_layers = [summary_plot, workplace_plot]
    if merged_plot is not None:
        bounds_layers.append(merged_plot)
    bounds = _series_bounds(bounds_layers)
    if fit_bounds and bounds is not None:
        minx, miny, maxx, maxy = bounds
        dx = max(maxx - minx, 1e-9)
        dy = max(maxy - miny, 1e-9)
        pad_x = dx * max(float(fit_bounds_padding_fraction), 0.0)
        pad_y = dy * max(float(fit_bounds_padding_fraction), 0.0)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    if add_basemap:
        contextily.add_basemap(
            ax,
            source=contextily.providers.OpenStreetMap.Mapnik,
            headers=_OSM_TILE_HEADERS,
            crs=target_crs,
            reset_extent=False,
            alpha=basemap_alpha,
        )

    if title is not None:
        ax.set_title(title)
    if not show_axis:
        ax.set_axis_off()
    return ax


def plot_commute_map(
    *,
    probability_grid: gpd.GeoDataFrame | None = None,
    probability_column: str = "sampling_probability",
    employees: gpd.GeoDataFrame | None = None,
    target_area: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    workplace: PointLike | None = None,
    routes: RouteInput | None = None,
    route_origins: PointLike | Iterable[PointLike] | None = None,
    route_destinations: PointLike | Iterable[PointLike] | None = None,
    route_options: int | Iterable[int | None] | None = None,
    backend: Backend = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 10),
    ax: Axes | None = None,
    folium_map: folium.Map | None = None,
    add_basemap: bool = True,
    show_probability_grid: bool = True,
    show_employees: bool = True,
    show_target_area: bool = True,
    show_workplace: bool = True,
    show_routes: bool = True,
    show_route_origins: bool = True,
    show_route_destinations: bool = True,
    show_route_segment_labels: bool = True,
    show_route_times: bool = True,
    show_route_modes: bool = True,
    show_route_ids: bool = True,
    show_route_option_labels: bool = True,
    show_legend: bool = True,
    show_colorbar: bool = True,
    show_axis: bool = False,
    fit_bounds: bool = True,
    fit_bounds_padding_fraction: float = 0.02,
    probability_norm: mcolors.Normalize | None = None,
    probability_cmap: str = "viridis",
    probability_alpha: float = 0.65,
    probability_edgecolor: str = "none",
    probability_linewidth: float = 0.0,
    probability_colorbar_label: str | None = None,
    employee_alpha: float = 0.85,
    route_alpha: float = 0.9,
    employee_markersize: float = 18,
    origin_markersize: float | None = None,
    destination_markersize: float | None = None,
    workplace_markersize: float | None = None,
    route_linewidth: float = 3,
    route_color_strategy: RouteColorStrategy = "mode",
    route_cmap: str = "hsv",
    cluster_employees: bool = False,
    employee_tooltip_column: str | None = None,
    employee_label: str = "Employees",
    workplace_label: str = "Workplace",
    target_area_label: str = "Target area",
    origin_label: str = "Start",
    destination_label: str = "Destination",
    deduplicate_route_points: bool = True,
    target_area_linewidth: float = 2,
    target_area_alpha: float = 0.9,
	reachable: gpd.GeoDataFrame | None = None,
	show_isochrones: bool = True,
	isochrone_bins: Sequence[float] = (0, 15, 30, 45, 60),
	isochrone_alpha: float = 0.35,
    **kwargs: Any,
) -> Axes | folium.Map:
    """Create a composable commute map using Matplotlib or Folium.
    
    This function provides a flexible interface for visualizing commute data,
    including probability grids, employee locations, routes, and target areas.
    It supports both static (Matplotlib) and interactive (Folium) rendering backends.
    
    Parameters
    ----------
    probability_grid : geopandas.GeoDataFrame, optional
        Polygon/point grid containing a probability column.
    probability_column : str, default "sampling_probability"
        Column to use for coloring the probability layer.
    employees : geopandas.GeoDataFrame, optional
        Synthetic employee point locations.
    target_area : geopandas.GeoDataFrame | geopandas.GeoSeries, optional
        Target-area polygon or boundary.
    workplace : Point | tuple | GeoSeries | GeoDataFrame, optional
        Workplace point location.
    routes : geopandas.GeoDataFrame | iterable of GeoDataFrame, optional
        One or more route segment GeoDataFrames.
    route_options : int | iterable[int | None], optional
        Itinerary option selector when route data includes multiple options.
    backend : {"matplotlib", "folium"}, default "matplotlib"
        Rendering backend.
    title : str | None, optional
        Title for the plot/map.
    figsize : tuple[float, float], default (12, 10)
        Figure size for matplotlib backend.
    ax : Axes | None, optional
        Existing matplotlib Axes to plot on.
    folium_map : folium.Map | None, optional
        Existing Folium map for interactive backend.
    add_basemap : bool, default True
        Whether to add a basemap layer.
    show_probability_grid : bool, default True
        Whether to show the probability grid layer.
    show_employees : bool, default True
        Whether to show employee points.
    show_target_area : bool, default True
        Whether to show the target area.
    show_workplace : bool, default True
        Whether to show the workplace point.
    show_routes : bool, default True
        Whether to show route segments.
    show_route_origins : bool, default True
        Whether to show route origin points.
    show_route_destinations : bool, default True
        Whether to show route destination points.
    show_route_segment_labels : bool, default True
        Whether to show labels on route segments.
    show_route_times : bool, default True
        Whether to include time information in labels.
    show_route_modes : bool, default True
        Whether to include mode information in labels.
    show_route_ids : bool, default True
        Whether to include route IDs in labels.
    show_route_option_labels : bool, default True
        Whether to include option numbers in labels.
    show_legend : bool, default True
        Whether to show the legend.
    show_colorbar : bool, default True
        Whether to show the colorbar.
    show_axis : bool, default False
        Whether to show the plot axes.
    fit_bounds : bool, default True
        Whether to fit plot bounds to the data.
    fit_bounds_padding_fraction : float, default 0.02
        Fractional padding around the bounds.
    probability_norm : mcolors.Normalize | None, optional
        Normalization for probability grid.
    probability_cmap : str, default "viridis"
        Colormap for probability grid.
    probability_alpha : float, default 0.65
        Transparency for probability grid.
    probability_edgecolor : str, default "none"
        Edge color for probability grid polygons.
    probability_linewidth : float, default 0.0
        Line width for probability grid polygons.
    probability_colorbar_label : str | None, optional
        Label for the probability colorbar.
    employee_alpha : float, default 0.85
        Transparency for employee points.
    route_alpha : float, default 0.9
        Transparency for routes.
    employee_markersize : float, default 18
        Marker size for employee points.
    origin_markersize : float | None, optional
        Marker size for origin points.
    destination_markersize : float | None, optional
        Marker size for destination points.
    workplace_markersize : float | None, optional
        Marker size for workplace point.
    route_linewidth : float, default 3
        Line width for routes.
    route_color_strategy : RouteColorStrategy, default "mode"
        Strategy for coloring routes.
    route_cmap : str, default "hsv"
        Colormap for route coloring.
    cluster_employees : bool, default False
        Whether to cluster employee points (Folium only).
    employee_tooltip_column : str | None, optional
        Column for employee tooltip text.
    employee_label : str, default "Employees"
        Label for employee points in legend.
    workplace_label : str, default "Workplace"
        Label for workplace in legend.
    target_area_label : str, default "Target area"
        Label for target area in legend.
    origin_label : str, default "Start"
        Label for origin points in legend.
    destination_label : str, default "Destination"
        Label for destination points in legend.
    deduplicate_route_points : bool, default True
        Whether to deduplicate route origin/destination points.
    target_area_linewidth : float, default 2
        Line width for target area boundary.
    target_area_alpha : float, default 0.9
        Transparency for target area.
    **kwargs : Any
        Additional arguments passed to the plotting function.
    
    Returns
    -------
    Axes | folium.Map
        Rendered map object for the selected backend.
    
    Raises
    ------
    ValueError
        If no enabled layers with data were provided.
    """
    normalized_backend = _normalize_backend(backend)
    route_frames = _normalize_routes(routes)

    if not _has_data_for_enabled_layers(
        show_probability_grid=show_probability_grid,
        probability_grid=probability_grid,
        show_employees=show_employees,
        employees=employees,
        show_target_area=show_target_area,
        target_area=target_area,
        show_workplace=show_workplace,
        workplace=workplace,
        show_routes=show_routes,
        route_count=len(route_frames),
        show_route_origins=show_route_origins,
        route_origins=route_origins,
        show_route_destinations=show_route_destinations,
        route_destinations=route_destinations,
    ):
        raise ValueError("No enabled layers with data were provided.")

    if show_probability_grid and probability_grid is not None:
        _validate_probability_grid(probability_grid, probability_column)
    if show_employees and employees is not None:
        _validate_employee_points(employees)

    target_area_gdf: gpd.GeoDataFrame | None = None
    if target_area is not None:
        target_area_gdf = _to_geodataframe(target_area, "target_area")

    isochrones = _create_isochrones(reachable, bins=isochrone_bins) if show_isochrones and reachable is not None else None

    first_crs = None
    for layer in [probability_grid, employees, target_area_gdf, isochrones, *route_frames]:
        if layer is not None and hasattr(layer, "crs") and layer.crs is not None:
            first_crs = layer.crs
            break

    if first_crs is None:
        first_crs = config.CRS_WGS84

    target_crs = _get_target_crs(normalized_backend, add_basemap, first_crs)

    probability_grid_plot = probability_grid.to_crs(target_crs).copy() if probability_grid is not None else None
    employees_plot = employees.to_crs(target_crs).copy() if employees is not None else None
    target_area_plot = target_area_gdf.to_crs(target_crs).copy() if target_area_gdf is not None else None
    isochrones_plot = isochrones.to_crs(target_crs) if isochrones is not None else None

    selected_routes: list[gpd.GeoDataFrame] = []
    selected_options = _expand_route_options(route_options, len(route_frames))
    for idx, route in enumerate(route_frames):
        chosen = _select_route_option(route, selected_options[idx], idx)
        selected_routes.append(_sorted_route_segments(chosen.to_crs(target_crs)))

    origins_expanded = _expand_point_inputs(route_origins, len(selected_routes), "route_origins")
    destinations_expanded = _expand_point_inputs(route_destinations, len(selected_routes), "route_destinations")

    origins_plot: list[gpd.GeoDataFrame | None] = []
    destinations_plot: list[gpd.GeoDataFrame | None] = []
    for idx in range(len(selected_routes)):
        origin_item = origins_expanded[idx]
        destination_item = destinations_expanded[idx]
        origins_plot.append(_point_to_gdf(origin_item, target_crs, f"route_origins[{idx}]") if origin_item is not None else None)
        destinations_plot.append(
            _point_to_gdf(destination_item, target_crs, f"route_destinations[{idx}]") if destination_item is not None else None
        )

    workplace_plot = _point_to_gdf(workplace, target_crs, "workplace") if workplace is not None else None

    if normalized_backend == "matplotlib":
        had_data_before = ax.has_data() if ax is not None else False
        if ax is None:
            subplot_figsize = kwargs.pop("figsize", figsize)
            _, ax = plt.subplots(figsize=subplot_figsize)

        legend_handles: list[Line2D] = []
        legend_labels: list[str] = []

        if add_basemap:
            contextily = _require_optional_dependency(
                "contextily",
                "contextily is required when add_basemap=True for matplotlib backend.",
            )

        if show_probability_grid and probability_grid_plot is not None:
            probability_grid_plot.plot(
                ax=ax,
                column=probability_column,
                cmap=probability_cmap,
                alpha=probability_alpha,
                edgecolor=probability_edgecolor,
                linewidth=probability_linewidth,
                norm=probability_norm,
                legend=False,
            )
            if show_colorbar:
                valid_values = pd.Series(probability_grid_plot[probability_column]).dropna()
                if not valid_values.empty:
                    norm = probability_norm or mcolors.Normalize(vmin=float(valid_values.min()), vmax=float(valid_values.max()))
                    sm = cm.ScalarMappable(norm=norm, cmap=probability_cmap)
                    sm.set_array([])
                    colorbar_label = probability_colorbar_label or probability_column
                    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label=colorbar_label)

        if show_target_area and target_area_plot is not None:
            target_area_plot.boundary.plot(
                ax=ax,
                linewidth=target_area_linewidth,
                alpha=target_area_alpha,
                color="black",
                zorder=3,
            )
            legend_handles.append(Line2D([0], [0], color="black", lw=target_area_linewidth))
            legend_labels.append(target_area_label)

        route_bounds_layers: list[gpd.GeoDataFrame] = []
        if show_routes:
            for route_idx, route_df in enumerate(selected_routes):
                if route_df.empty:
                    continue
                route_bounds_layers.append(route_df)
                for seg_idx, (_, row) in enumerate(route_df.iterrows()):
                    mode_name = _segment_mode(row)
                    color = _color_for_segment(
                        route_index=route_idx,
                        segment_index=seg_idx,
                        mode_name=mode_name,
                        strategy=route_color_strategy,
                    )
                    gpd.GeoSeries([row.geometry], crs=route_df.crs).plot(
                        ax=ax,
                        color=[color],
                        linewidth=route_linewidth,
                        alpha=route_alpha,
                        zorder=4,
                    )

                    if show_route_segment_labels:
                        label = _segment_label(
                            row,
                            route_index=route_idx,
                            show_route_ids=show_route_ids,
                            show_route_modes=show_route_modes,
                            show_route_times=show_route_times,
                        )
                        if show_legend and label is not None:
                            legend_handles.append(Line2D([0], [0], color=color, lw=route_linewidth))
                            if "option" in route_df.columns and show_route_option_labels:
                                option_value = row.get("option")
                                if pd.notna(option_value):
                                    label = f"{label} (Option {option_value})"
                            legend_labels.append(label)

                        midpoint = _line_midpoint(row.geometry)
                        if midpoint is not None and label is not None:
                            ax.text(
                                midpoint.x,
                                midpoint.y,
                                label,
                                fontsize=8,
                                color=mcolors.to_hex(color),
                                zorder=9,
                            )

        if show_employees and employees_plot is not None:
            employees_plot.plot(
                ax=ax,
                markersize=employee_markersize,
                alpha=employee_alpha,
                color="#1f77b4",
                zorder=5,
            )
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8)
            )
            legend_labels.append(employee_label)

        plotted_points: set[tuple[str, float, float]] = set()

        def _plot_points(
            points: gpd.GeoDataFrame | None,
            marker: str,
            color: str,
            label: str,
            z: int,
            size: float,
        ) -> None:
            """Helper function to plot points with deduplication support."""
            if points is None or points.empty:
                return
            for geom in points.geometry:
                if geom is None or geom.is_empty:
                    continue
                key = (label, round(float(geom.x), 7), round(float(geom.y), 7))
                if deduplicate_route_points and key in plotted_points:
                    continue
                plotted_points.add(key)
                ax.scatter([geom.x], [geom.y], marker=marker, s=size, color=color, zorder=z)
            legend_handles.append(Line2D([0], [0], marker=marker, color="w", markerfacecolor=color, markersize=8))
            legend_labels.append(label)

        origin_size = origin_markersize if origin_markersize is not None else employee_markersize * 2
        destination_size = destination_markersize if destination_markersize is not None else employee_markersize * 2
        workplace_size = workplace_markersize if workplace_markersize is not None else employee_markersize * 2

        if show_route_origins:
            for origin_gdf in origins_plot:
                _plot_points(origin_gdf, "^", "#2ca02c", origin_label, 6, origin_size)

        if show_route_destinations:
            for destination_gdf in destinations_plot:
                _plot_points(destination_gdf, "s", "#ff7f0e", destination_label, 7, destination_size)

        if show_workplace and workplace_plot is not None:
            _plot_points(workplace_plot, "*", "#d62728", workplace_label, 8, workplace_size)

        if show_legend and legend_labels:
            dedup: dict[str, Line2D] = {}
            for handle, label in zip(legend_handles, legend_labels):
                if label and label not in dedup:
                    dedup[label] = handle
            if dedup:
                ax.legend(dedup.values(), dedup.keys(), loc="best")

        bounds_layers = [
            layer
            for layer in [probability_grid_plot, employees_plot, target_area_plot, isochrones_plot, *route_bounds_layers, workplace_plot]
            if layer is not None
        ]
        for layer in origins_plot + destinations_plot:
            if layer is not None:
                bounds_layers.append(layer)

        bounds = _series_bounds(bounds_layers)
        if fit_bounds and bounds is not None and not had_data_before:
            minx, miny, maxx, maxy = bounds
            dx = max(maxx - minx, 1e-9)
            dy = max(maxy - miny, 1e-9)
            padding_fraction = max(float(fit_bounds_padding_fraction), 0.0)
            pad_x = dx * padding_fraction
            pad_y = dy * padding_fraction
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

        if add_basemap:
            contextily.add_basemap(
                ax,
                source=contextily.providers.OpenStreetMap.Mapnik,
                headers=_OSM_TILE_HEADERS,
                crs=target_crs,
                reset_extent=False,
            )

        if title is not None:
            ax.set_title(title)

        if not show_axis:
            ax.set_axis_off()

        if show_isochrones and isochrones_plot is not None:
            colors = {
                "0-15 min": "#2E7D32",
                "15-30 min": "#66BB6A",
                "30-45 min": "#FFCA28",
                "45-60 min": "#EF6C00",
            }

            for band in reversed(list(colors)):
                band_gdf = isochrones_plot[
                    isochrones_plot["time_band"] == band
                ]

                if band_gdf.empty:
                    continue

                band_gdf.plot(
                    ax=ax,
                    color=colors[band],
                    alpha=isochrone_alpha,
                    edgecolor="black",
                    linewidth=1,
                    zorder=2,
                )

                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=colors[band],
                        linewidth=6,
                        alpha=isochrone_alpha,
                    )
                )
                legend_labels.append(band)

        return ax

    folium_mod = _require_optional_dependency(
        "folium",
        "folium is required when backend='folium'. Install it with `pip install folium`.",
    )
    from branca.colormap import LinearColormap

    plugins = None
    if cluster_employees:
        try:
            from folium import plugins as folium_plugins

            plugins = folium_plugins
        except ImportError:
            plugins = None

    visible_layers_for_bounds: list[gpd.GeoDataFrame] = []
    for layer in [probability_grid_plot, employees_plot, target_area_plot, *selected_routes, workplace_plot]:
        if layer is not None:
            visible_layers_for_bounds.append(layer)
    for layer in origins_plot + destinations_plot:
        if layer is not None:
            visible_layers_for_bounds.append(layer)
    bounds = _series_bounds(visible_layers_for_bounds)

    if folium_map is None:
        center = [0.0, 0.0]
        if bounds is not None:
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        folium_map = folium_mod.Map(location=center, zoom_start=12, tiles=None)

    def _has_layer_control(map_obj: folium.Map) -> bool:
        """Check if the Folium map has a LayerControl."""
        return any(child.__class__.__name__ == "LayerControl" for child in map_obj._children.values())

    def _has_tile_layer(map_obj: folium.Map) -> bool:
        """Check if the Folium map has a TileLayer."""
        return any(child.__class__.__name__ == "TileLayer" for child in map_obj._children.values())

    if add_basemap and not _has_tile_layer(folium_map):
        folium_mod.TileLayer("OpenStreetMap", name="Basemap", control=False).add_to(folium_map)

    overlay_count = 0

    if show_probability_grid and probability_grid_plot is not None:
        thinning_factor = 100
        probability_grid_thinned = probability_grid_plot[
            probability_grid_plot.index.isin(probability_grid_plot.index[::thinning_factor])
        ].copy()
        feature_group = folium_mod.FeatureGroup(name="Probability grid", show=True)
        values = probability_grid_thinned[probability_column].dropna()
        colormap = None
        if not values.empty:
            cmap = plt.get_cmap(probability_cmap)
            colors = [mcolors.to_hex(cmap(index / 8)) for index in range(9)]
            colormap = LinearColormap(colors, vmin=float(values.min()), vmax=float(values.max()))
            if show_colorbar:
                colormap.caption = probability_column
                colormap.add_to(folium_map)

        def _style_fn(feature: dict[str, Any]) -> dict[str, Any]:
            value = feature["properties"].get(probability_column)
            if value is None or pd.isna(value):
                color = "#00000000"
            elif colormap is not None:
                color = colormap(float(value))
            else:
                color = "#1f77b4"
            return {
                "fillColor": color,
                "color": "#666666",
                "weight": 0.5,
                "fillOpacity": probability_alpha,
            }

        folium_mod.GeoJson(
            probability_grid_thinned[[probability_column, "geometry"]].to_json(),
            style_function=_style_fn,
            tooltip=folium_mod.GeoJsonTooltip(
                fields=[probability_column], aliases=[probability_column]
            ),
            name="Probability grid",
        ).add_to(feature_group)
        feature_group.add_to(folium_map)
        overlay_count += 1

    if show_target_area and target_area_plot is not None:
        feature_group = folium_mod.FeatureGroup(name=target_area_label, show=True)
        folium_mod.GeoJson(
            target_area_plot.to_json(),
            style_function=lambda _: {
                "fillOpacity": 0.0,
                "color": "black",
                "weight": target_area_linewidth,
                "opacity": target_area_alpha,
            },
            name=target_area_label,
        ).add_to(feature_group)
        feature_group.add_to(folium_map)
        overlay_count += 1

    if show_isochrones and isochrones_plot is not None:
        isochrone_feature_group = folium_mod.FeatureGroup(name="Isochrones", show=True)
        colors = {
            "0-15 min": "#2E7D32",
            "15-30 min": "#66BB6A",
            "30-45 min": "#FFCA28",
            "45-60 min": "#EF6C00",
        }
        
        for band in list(colors):
            band_gdf = isochrones_plot[
                isochrones_plot["time_band"] == band
            ]
            
            if band_gdf.empty:
                continue
            
            style_fn = lambda f, band_name=band, color=colors[band], alpha=isochrone_alpha: {
                "fillColor": color,
                "color": "black",
                "weight": 1,
                "fillOpacity": alpha,
            }
            
            folium_mod.GeoJson(
                band_gdf.to_json(),
                style_function=style_fn,
                name=f"{band}",
            ).add_to(isochrone_feature_group)
        
        isochrone_feature_group.add_to(folium_map)
        overlay_count += 1

    if show_routes and selected_routes:
        routes_group = folium_mod.FeatureGroup(name="Routes", show=True)
        for route_idx, route_df in enumerate(selected_routes):
            if route_df.empty:
                continue
            for seg_idx, (_, row) in enumerate(route_df.iterrows()):
                mode_name = _segment_mode(row)
                color = mcolors.to_hex(
                    _color_for_segment(
                        route_index=route_idx,
                        segment_index=seg_idx,
                        mode_name=mode_name,
                        strategy=route_color_strategy,
                    )
                )
                label = (
                    _segment_label(
                        row,
                        route_index=route_idx,
                        show_route_ids=show_route_ids,
                        show_route_modes=show_route_modes,
                        show_route_times=show_route_times,
                    )
                    if show_route_segment_labels
                    else None
                )
                if "option" in route_df.columns and show_route_option_labels and label is not None:
                    option_value = row.get("option")
                    if pd.notna(option_value):
                        label = f"{label} (Option {option_value})"

                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                if isinstance(geom, LineString):
                    coords = [(y, x) for x, y in geom.coords]
                    folium_mod.PolyLine(
                        locations=coords,
                        color=color,
                        weight=route_linewidth,
                        opacity=route_alpha,
                        tooltip=label,
                    ).add_to(routes_group)
                elif isinstance(geom, MultiLineString):
                    for line in geom.geoms:
                        coords = [(y, x) for x, y in line.coords]
                        folium_mod.PolyLine(
                            locations=coords,
                            color=color,
                            weight=route_linewidth,
                            opacity=route_alpha,
                            tooltip=label,
                        ).add_to(routes_group)
        routes_group.add_to(folium_map)
        overlay_count += 1

    if show_employees and employees_plot is not None:
        employees_group = folium_mod.FeatureGroup(name=employee_label, show=True)
        tooltip_col = employee_tooltip_column
        if tooltip_col is None and "employee_id" in employees_plot.columns:
            tooltip_col = "employee_id"

        marker_parent: Any = employees_group
        if cluster_employees and plugins is not None:
            marker_parent = plugins.MarkerCluster(name=f"{employee_label} cluster")
            marker_parent.add_to(employees_group)

        for _, row in employees_plot.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            tooltip = str(row[tooltip_col]) if tooltip_col is not None and tooltip_col in row and pd.notna(row[tooltip_col]) else None
            folium_mod.CircleMarker(
                location=(geom.y, geom.x),
                radius=max(2, int(employee_markersize / 4)),
                color="#1f77b4",
                fill=True,
                fill_opacity=employee_alpha,
                tooltip=tooltip,
            ).add_to(marker_parent)

        employees_group.add_to(folium_map)
        overlay_count += 1

    plotted_points_folium: set[tuple[str, float, float]] = set()

    def _add_point_markers(points: gpd.GeoDataFrame | None, label: str, color: str, icon: str) -> None:
        """Helper function to add point markers to Folium map with deduplication."""
        if points is None or points.empty:
            return
        group = folium_mod.FeatureGroup(name=label, show=True)
        for geom in points.geometry:
            if geom is None or geom.is_empty:
                continue
            key = (label, round(float(geom.x), 7), round(float(geom.y), 7))
            if deduplicate_route_points and key in plotted_points_folium:
                continue
            plotted_points_folium.add(key)
            folium_mod.Marker(
                location=(geom.y, geom.x),
                tooltip=label,
                icon=folium_mod.Icon(color=color, icon=icon, prefix="fa"),
            ).add_to(group)
        group.add_to(folium_map)

    if show_route_origins:
        for points in origins_plot:
            _add_point_markers(points, origin_label, "green", "play")
            overlay_count += 1

    if show_route_destinations:
        for points in destinations_plot:
            _add_point_markers(points, destination_label, "orange", "stop")
            overlay_count += 1

    if show_workplace and workplace_plot is not None:
        _add_point_markers(workplace_plot, workplace_label, "red", "briefcase")
        overlay_count += 1

    if fit_bounds and bounds is not None:
        folium_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    if show_legend and overlay_count > 1 and not _has_layer_control(folium_map):
        folium_mod.LayerControl(collapsed=False).add_to(folium_map)

    return folium_map


def plot_sampling_probability_map(
    probability_grid: gpd.GeoDataFrame,
    *,
    target_area: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    workplace: PointLike | None = None,
    employees: gpd.GeoDataFrame | None = None,
    backend: Backend = "matplotlib",
    **kwargs: Any,
) -> Axes | folium.Map:
    """Plot a probability grid with optional overlays.
    
    This function creates a visualization of a sampling probability grid,
    optionally showing target areas, workplace, and employee points.
    It supports both Matplotlib and Folium backends.
    
    Parameters
    ----------
    probability_grid : gpd.GeoDataFrame
        GeoDataFrame containing probability values.
    target_area : gpd.GeoDataFrame | gpd.GeoSeries | None, optional
        Target-area polygon or boundary.
    workplace : PointLike | None, optional
        Workplace point location.
    employees : gpd.GeoDataFrame | None, optional
        Employee point locations.
    backend : Backend, default "matplotlib"
        Rendering backend.
    **kwargs : Any
        Additional keyword arguments passed to plot_commute_map.
    
    Returns
    -------
    Axes | folium.Map
        The rendered map object.
    """
    legacy_aliases = {
        "cmap": "probability_cmap",
        "norm": "probability_norm",
        "alpha": "probability_alpha",
        "edgecolor": "probability_edgecolor",
    }
    for old_name, new_name in legacy_aliases.items():
        if old_name in kwargs and new_name not in kwargs:
            kwargs[new_name] = kwargs.pop(old_name)

    if backend == "matplotlib":
        if "probability_column" not in kwargs:
            if "probability_density_km2" in probability_grid.columns:
                kwargs["probability_column"] = "probability_density_km2"
            else:
                kwargs["probability_column"] = "sampling_probability"

        if kwargs["probability_column"] == "probability_density_km2":
            if "probability_norm" not in kwargs:
                values = pd.Series(probability_grid["probability_density_km2"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
                values = values[values > 0]
                if not values.empty:
                    vmin = float(values.quantile(0.05))
                    vmax = float(values.quantile(0.995))

                    if vmax <= vmin:
                        vmin = float(values.min())
                        vmax = float(values.max())

                    if vmax > vmin and vmin > 0:
                        spread_ratio = vmax / max(vmin, 1e-15)
                        if spread_ratio >= 50:
                            kwargs["probability_norm"] = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
                        else:
                            kwargs["probability_norm"] = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)

            kwargs.setdefault("probability_alpha", 0.02)
            kwargs.setdefault("probability_edgecolor", "none")
            kwargs.setdefault("probability_linewidth", 0.0)
            kwargs.setdefault("probability_colorbar_label", "Sampling probability density per km²")
        else:
            kwargs.setdefault("probability_alpha", 0.65)

        kwargs.setdefault("target_area_linewidth", 2)
        kwargs.setdefault("workplace_markersize", 250)
        kwargs.setdefault("show_colorbar", True)
        kwargs.setdefault("show_axis", False)

    return plot_commute_map(
        probability_grid=probability_grid,
        target_area=target_area,
        workplace=workplace,
        employees=employees,
        backend=backend,
        **kwargs,
    )


def plot_synthetic_employees(
    employees: gpd.GeoDataFrame,
    *,
    target_area: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    workplace: PointLike | None = None,
    backend: Backend = "matplotlib",
    **kwargs: Any,
) -> Axes | folium.Map:
    """Plot synthetic employee points with optional overlays.
    
    This function creates a visualization of employee point locations
    with optional target areas and workplace markers.
    
    Parameters
    ----------
    employees : gpd.GeoDataFrame
        GeoDataFrame containing employee point locations.
    target_area : gpd.GeoDataFrame | gpd.GeoSeries | None, optional
        Target-area polygon or boundary.
    workplace : PointLike | None, optional
        Workplace point location.
    backend : Backend, default "matplotlib"
        Rendering backend.
    **kwargs : Any
        Additional keyword arguments passed to plot_commute_map.
    
    Returns
    -------
    Axes | folium.Map
        The rendered map object.
    """
    return plot_commute_map(
        employees=employees,
        target_area=target_area,
        workplace=workplace,
        backend=backend,
        **kwargs,
    )


def plot_routes(
    routes: RouteInput,
    *,
    origins: PointLike | Iterable[PointLike] | None = None,
    destinations: PointLike | Iterable[PointLike] | None = None,
    route_options: int | Iterable[int | None] | None = None,
    backend: Backend = "matplotlib",
    **kwargs: Any,
) -> Axes | folium.Map:
    """Plot one or more routes with optional origin and destination points.
    
    This function visualizes route segments with optional start and end points.
    It supports both Matplotlib and Folium backends.
    
    Parameters
    ----------
    routes : RouteInput
        GeoDataFrame or iterable of GeoDataFrames containing route segments.
    origins : PointLike | Iterable[PointLike] | None, optional
        Origin point(s) for the routes.
    destinations : PointLike | Iterable[PointLike] | None, optional
        Destination point(s) for the routes.
    route_options : int | Iterable[int | None] | None, optional
        Itinerary option selector when routes have multiple options.
    backend : Backend, default "matplotlib"
        Rendering backend.
    **kwargs : Any
        Additional keyword arguments passed to plot_commute_map.
    
    Returns
    -------
    Axes | folium.Map
        The rendered map object.
    """
    return plot_commute_map(
        routes=routes,
        route_origins=origins,
        route_destinations=destinations,
        route_options=route_options,
        backend=backend,
        **kwargs,
    )


def plot_histogram(
    data: pd.DataFrame | pd.Series,
    column: str | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "",
    figsize: tuple[float, float] = (6, 4),
    ax: Axes | None = None,
    bins: int | Sequence[float] = 15,
    density: bool = True,
    show_kde: bool = True,
    kde_linewidth: float = 2.0,
    kde_bw_method: str | float | None = 0.3,
    hist_alpha: float = 0.65,
    hist_color: str = "#4C78A8",
    kde_color: str = "#1f2a44",
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    show_quantile_labels: bool = True,
    quantile_linestyle: str = ":",
    quantile_linewidth: float = 1.5,
    quantile_text_size: float = 10,
    xlim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    x_ticks: Sequence[float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_grid: bool = False,
    hide_y_ticks: bool = True,
    remove_spines: bool = True,
    tick_left: bool = False,
    tick_bottom: bool = False,
    hist_kwargs: dict[str, Any] | None = None,
    kde_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Plot a styled histogram from a DataFrame column or Series.

    The plot includes optional KDE overlay and percentile guide lines to
    provide a compact distribution summary suitable for reports/notebooks.

    Parameters
    ----------
    data : pandas.DataFrame | pandas.Series
        Input values as a DataFrame or Series.
    column : str | None, optional
        Column name when ``data`` is a DataFrame. If ``data`` is a Series,
        this is ignored.
    title : str | None, optional
        Plot title.
    xlabel : str | None, optional
        X-axis label. Defaults to ``column`` (for DataFrame) or Series name.
    ylabel : str, default ""
        Y-axis label.
    figsize : tuple[float, float], default (6, 4)
        Figure size when creating a new axes.
    ax : Axes | None, optional
        Existing matplotlib axes to draw on.
    bins : int | Sequence[float], default 15
        Histogram bin specification.
    density : bool, default True
        Whether to normalize histogram as a density.
    show_kde : bool, default True
        Whether to overlay a KDE curve.
    kde_linewidth : float, default 2.0
        KDE line width.
    kde_bw_method : str | float | None, default 0.3
        Bandwidth method passed to KDE. Smaller numeric values produce a KDE
        that follows local data variation more closely.
    hist_alpha : float, default 0.65
        Histogram bar transparency.
    hist_color : str, default "#4C78A8"
        Histogram color.
    kde_color : str, default "#1f2a44"
        KDE line color.
    quantiles : Sequence[float], default (0.05, 0.25, 0.5, 0.75, 0.95)
        Quantiles to draw as vertical guide lines.
    show_quantile_labels : bool, default True
        Whether to annotate quantile labels.
    quantile_linestyle : str, default ":"
        Line style for quantile guide lines.
    quantile_linewidth : float, default 1.5
        Line width for quantile guide lines.
    quantile_text_size : float, default 10
        Font size for quantile annotations.
    xlim : tuple[float, float] | None, optional
        X-axis limits.
    xticks : Sequence[float] | None, optional
        Explicit x-axis tick locations.
    x_ticks : Sequence[float] | None, optional
        Alias for ``xticks``.
    ylim : tuple[float, float] | None, optional
        Y-axis limits.
    show_grid : bool, default False
        Whether to show plot grid.
    hide_y_ticks : bool, default True
        Whether to hide Y ticks and labels.
    remove_spines : bool, default True
        Whether to remove all axis spines.
    tick_left : bool, default False
        Whether to show left ticks.
    tick_bottom : bool, default False
        Whether to show bottom ticks.
    hist_kwargs : dict[str, Any] | None, optional
        Extra keyword args passed to ``Series.plot(kind='hist')``.
    kde_kwargs : dict[str, Any] | None, optional
        Extra keyword args passed to ``Series.plot(kind='kde')``.

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.

    Raises
    ------
    ValueError
        If input is empty, column is missing, or no numeric values remain.
    TypeError
        If ``data`` is not a DataFrame or Series.
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column must be provided when data is a DataFrame.")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        series = data[column]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise TypeError("data must be a pandas DataFrame or Series.")

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError("No numeric values available to plot.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    local_hist_kwargs = dict(hist_kwargs or {})
    local_kde_kwargs = dict(kde_kwargs or {})

    numeric_series.plot(
        kind="hist",
        ax=ax,
        bins=bins,
        density=density,
        alpha=hist_alpha,
        color=hist_color,
        **local_hist_kwargs,
    )

    histogram_peak = max(
        (float(patch.get_height()) for patch in ax.patches if pd.notna(patch.get_height())),
        default=0.0,
    )
    if histogram_peak <= 0.0:
        histogram_peak = float(ax.get_ylim()[1])

    if show_kde:
        if "bw_method" not in local_kde_kwargs:
            local_kde_kwargs["bw_method"] = kde_bw_method
        numeric_series.plot(
            kind="kde",
            ax=ax,
            color=kde_color,
            linewidth=kde_linewidth,
            **local_kde_kwargs,
        )

    if quantiles:
        q_values = numeric_series.quantile(list(quantiles))
        q_pairs = sorted(zip(quantiles, q_values.to_list()), key=lambda item: item[0])
        line_count = len(q_pairs)

        if line_count == 1:
            y_max_fractions = [0.35]
        else:
            y_start = 0.16
            y_end = 0.56
            step = (y_end - y_start) / max(line_count - 1, 1)
            y_max_fractions = [y_start + idx * step for idx in range(line_count)]

        center_index = (line_count - 1) / 2
        alpha_span = max(center_index, 1.0)

        x_min = float(numeric_series.min())
        x_max = float(numeric_series.max())
        x_offset = (x_max - x_min) * 0.015 if x_max > x_min else 0.05
        axis_top = float(ax.get_ylim()[1])
        axis_top = axis_top if axis_top > 0 else max(histogram_peak, 1.0)
        label_gap = histogram_peak * 0.02

        for idx, (q, value) in enumerate(q_pairs):
            alpha = 0.6 + 0.4 * (1.0 - abs(idx - center_index) / alpha_span)
            y_fraction = y_max_fractions[idx]
            line_top_data = min(histogram_peak * y_fraction, histogram_peak * 0.98)
            line_top_fraction = min(max(line_top_data / axis_top, 0.0), 1.0)
            ax.axvline(
                value,
                alpha=alpha,
                ymax=line_top_fraction,
                linestyle=quantile_linestyle,
                linewidth=quantile_linewidth,
                color=kde_color,
            )

            if show_quantile_labels:
                percentile = int(round(q * 100))
                label = f"{percentile}th"
                if percentile == 95:
                    label = "95th percentile"
                label_y = min(line_top_data + label_gap, histogram_peak * 0.995)
                ax.text(
                    value - x_offset,
                    label_y,
                    label,
                    size=quantile_text_size,
                    alpha=min(alpha + 0.05, 1.0),
                    color=kde_color,
                )

    final_xlabel = xlabel
    if final_xlabel is None:
        if isinstance(data, pd.DataFrame):
            final_xlabel = column
        else:
            final_xlabel = series.name if series.name is not None else "Value"

    ax.set_xlabel(str(final_xlabel))
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        data_min = float(numeric_series.min())
        data_max = float(numeric_series.max())

        # Estimate bin width and add two bins of padding on both sides.
        bin_edges = np.histogram_bin_edges(numeric_series.to_numpy(), bins=bins)
        diffs = np.diff(bin_edges)
        positive_diffs = diffs[diffs > 0]

        if positive_diffs.size > 0:
            bin_width = float(np.median(positive_diffs))
        else:
            data_span = data_max - data_min
            bin_width = data_span / max(1, int(bins)) if isinstance(bins, int) and data_span > 0 else 1.0

        pad = 2.0 * bin_width
        if data_max > data_min:
            ax.set_xlim(data_min - pad, data_max + pad)
        else:
            # Keep a visible window for constant-value data.
            fallback_pad = max(pad, 1.0)
            ax.set_xlim(data_min - fallback_pad, data_max + fallback_pad)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if xticks is not None and x_ticks is not None:
        raise ValueError("Use only one of xticks or x_ticks.")
    resolved_xticks = xticks if xticks is not None else x_ticks
    if resolved_xticks is not None:
        ax.set_xticks(list(resolved_xticks))

    if hide_y_ticks:
        ax.set_yticklabels([])
        ax.set_yticks([])

    ax.grid(show_grid)

    if title is not None:
        ax.set_title(title, pad=10)

    ax.tick_params(left=tick_left, bottom=tick_bottom)

    if remove_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    return ax

def _create_isochrones(
    reachable: gpd.GeoDataFrame,
    *,
    bins: Sequence[float] = (0, 15, 30, 45, 60),
) -> gpd.GeoDataFrame:
    """Create dissolved isochrone polygons from reachable grid points."""
    labels = [
        f"{bins[i]:g}-{bins[i + 1]:g} min"
        for i in range(len(bins) - 1)
    ]

    points = reachable.to_crs(config.CRS_LOCAL_METRIC).copy()
    points["time_band"] = pd.cut(
        points["travel_time_min"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    points = points.dropna(subset=["time_band"])

    grid_step_m = float(points["grid_step_m"].iloc[0])

    parts = gpd.GeoDataFrame(
        points[["time_band"]],
        geometry=points.buffer(grid_step_m * 0.72),
        crs=points.crs,
    )

    return parts.dissolve(by="time_band", as_index=False)


__all__ = [
    "plot_commute_map",
    "plot_histogram",
    "plot_sampling_probability_map",
    "plot_synthetic_employees",
    "plot_routes",
    "plot_transport_score_routes",
]
