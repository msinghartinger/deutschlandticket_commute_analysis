from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString, Point

if TYPE_CHECKING:
	import folium
	from matplotlib.axes import Axes


PointLike = Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame
RouteInput = gpd.GeoDataFrame | Iterable[gpd.GeoDataFrame]
RouteColorStrategy = Literal["segment", "mode", "route"]
Backend = Literal["matplotlib", "folium"]


def _normalize_backend(backend: str) -> Backend:
	normalized = backend.lower().strip()
	if normalized not in {"matplotlib", "folium"}:
		raise ValueError("Unsupported backend. Use 'matplotlib' or 'folium'.")
	return normalized  # type: ignore[return-value]


def _require_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
	if gdf.crs is None:
		raise ValueError(f"{name} must have a CRS.")


def _to_geodataframe(value: gpd.GeoDataFrame | gpd.GeoSeries, name: str) -> gpd.GeoDataFrame:
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
	return isinstance(value, (Point, tuple, gpd.GeoSeries, gpd.GeoDataFrame))


def _point_to_gdf(value: PointLike, target_crs: Any, name: str) -> gpd.GeoDataFrame:
	if isinstance(value, Point):
		gdf = gpd.GeoDataFrame(geometry=[value], crs="EPSG:4326")
	elif isinstance(value, tuple):
		if len(value) != 2:
			raise ValueError(f"{name} tuple must be (latitude, longitude).")
		lat, lon = value
		gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
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
	sort_cols: list[str] = []
	for col in ["segment", "leg_index"]:
		if col in route.columns:
			sort_cols.append(col)
	if sort_cols:
		return route.sort_values(sort_cols).copy()
	return route.copy()


def _normalize_mode_name(mode_value: object) -> str:
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
	_require_crs(probability_grid, "probability_grid")
	if probability_grid.empty:
		raise ValueError("probability_grid is empty.")
	if probability_column not in probability_grid.columns:
		raise ValueError(
			f"probability_grid must contain '{probability_column}'."
		)


def _validate_employee_points(employees: gpd.GeoDataFrame) -> None:
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
	if backend == "folium":
		return "EPSG:4326"
	if backend == "matplotlib" and add_basemap:
		return "EPSG:3857"
	return first_crs


def _series_bounds(geometries: Sequence[gpd.GeoDataFrame]) -> tuple[float, float, float, float] | None:
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
	return any(
		[
			show_probability_grid and probability_grid is not None,
			show_employees and employees is not None,
			show_target_area and target_area is not None,
			show_workplace and workplace is not None,
			show_routes and route_count > 0,
			show_route_origins and route_origins is not None and route_count > 0,
			show_route_destinations and route_destinations is not None and route_count > 0,
		]
	)


def _require_optional_dependency(module_name: str, install_hint: str) -> Any:
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
	cmap = cm.get_cmap(cmap_name)
	if strategy == "route":
		value = (route_index % 20) / 20
	elif strategy == "mode":
		mode_key = hash(mode_name) % 20
		value = mode_key / 20
	else:
		value = (segment_index % 20) / 20
	return cmap(value)


def _line_midpoint(geom: object) -> Point | None:
	if isinstance(geom, LineString):
		return geom.interpolate(0.5, normalized=True)
	if isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
		return geom.geoms[0].interpolate(0.5, normalized=True)
	return None


def _iter_line_strings(geom: object) -> list[LineString]:
	if isinstance(geom, LineString):
		return [geom]
	if isinstance(geom, MultiLineString):
		return [line for line in geom.geoms if isinstance(line, LineString) and len(line.coords) >= 2]
	return []


def plot_transport_score_routes(
	*,
	summary_gdf: gpd.GeoDataFrame,
	route_gdf: gpd.GeoDataFrame,
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
) -> Axes:
	"""Plot employee homes and morning q25 routes colored by transport score."""
	if summary_gdf.empty:
		raise ValueError("summary_gdf is empty.")
	if route_gdf.empty:
		raise ValueError("route_gdf is empty.")

	for required in ["employee_id", "transport_score"]:
		if required not in summary_gdf.columns:
			raise ValueError(f"summary_gdf must contain '{required}'.")

	for required in ["employee_id", "morning_q25_route"]:
		if required not in route_gdf.columns:
			raise ValueError(f"route_gdf must contain '{required}'.")

	_require_crs(summary_gdf, "summary_gdf")
	_require_crs(route_gdf, "route_gdf")

	merged = route_gdf.merge(
		summary_gdf[["employee_id", "transport_score"]],
		on="employee_id",
		how="inner",
	)
	if merged.empty:
		raise ValueError("No overlapping employee_id values between summary_gdf and route_gdf.")

	first_crs = summary_gdf.crs if summary_gdf.crs is not None else route_gdf.crs
	target_crs = _get_target_crs("matplotlib", add_basemap, first_crs)
	summary_plot = summary_gdf.to_crs(target_crs).copy()
	merged_plot = gpd.GeoDataFrame(merged, geometry="morning_q25_route", crs=route_gdf.crs).to_crs(target_crs)
	workplace_plot = _point_to_gdf(workplace, target_crs, "workplace")

	if ax is None:
		_, ax = plt.subplots(figsize=figsize)

	if add_basemap:
		contextily = _require_optional_dependency(
			"contextily",
			"contextily is required when add_basemap=True for matplotlib backend.",
		)

	norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
	cmap = cm.get_cmap(score_cmap)
	perfect_color = cmap(norm(1.0))

	for _, row in merged_plot.iterrows():
		geom = row.get("morning_q25_route")
		score = float(pd.to_numeric(pd.Series([row.get("transport_score")]), errors="coerce").iloc[0])
		if geom is None or getattr(geom, "is_empty", True):
			continue
		route_color = cmap(norm(max(0.0, min(1.0, score))))
		gpd.GeoSeries([geom], crs=merged_plot.crs).plot(
			ax=ax,
			color=[route_color],
			linewidth=route_linewidth,
			alpha=route_alpha,
			zorder=4,
		)

	summary_plot.plot(
		ax=ax,
		column="transport_score",
		cmap=score_cmap,
		norm=norm,
		markersize=home_markersize,
		alpha=home_alpha,
		zorder=5,
	)

	workplace_plot.plot(
		ax=ax,
		marker="*",
		color=[perfect_color],
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

	bounds = _series_bounds([summary_plot, merged_plot, workplace_plot])
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
	route_color_strategy: RouteColorStrategy = "segment",
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
	**kwargs: Any,
) -> Axes | folium.Map:
	"""Create a composable commute map using Matplotlib or Folium.

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

	Returns
	-------
	matplotlib.axes.Axes | folium.Map
		Rendered map object for the selected backend.
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

	first_crs = None
	for layer in [probability_grid, employees, target_area_gdf, *route_frames]:
		if layer is not None and hasattr(layer, "crs") and layer.crs is not None:
			first_crs = layer.crs
			break

	if first_crs is None:
		first_crs = "EPSG:4326"

	target_crs = _get_target_crs(normalized_backend, add_basemap, first_crs)

	probability_grid_plot = probability_grid.to_crs(target_crs).copy() if probability_grid is not None else None
	employees_plot = employees.to_crs(target_crs).copy() if employees is not None else None
	target_area_plot = target_area_gdf.to_crs(target_crs).copy() if target_area_gdf is not None else None

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
			for layer in [probability_grid_plot, employees_plot, target_area_plot, *route_bounds_layers, workplace_plot]
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
				crs=target_crs,
				reset_extent=False,
			)

		if title is not None:
			ax.set_title(title)

		if not show_axis:
			ax.set_axis_off()

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
		return any(child.__class__.__name__ == "LayerControl" for child in map_obj._children.values())

	def _has_tile_layer(map_obj: folium.Map) -> bool:
		return any(child.__class__.__name__ == "TileLayer" for child in map_obj._children.values())

	if add_basemap and not _has_tile_layer(folium_map):
		folium_mod.TileLayer("OpenStreetMap", name="Basemap", control=False).add_to(folium_map)

	overlay_count = 0

	if show_probability_grid and probability_grid_plot is not None:
		feature_group = folium_mod.FeatureGroup(name="Probability grid", show=True)
		values = pd.Series(probability_grid_plot[probability_column])
		valid = values.dropna()
		colormap = None
		if not valid.empty:
			cmap_obj = cm.get_cmap(probability_cmap)
			sampled_colors = [mcolors.to_hex(cmap_obj(i / 8)) for i in range(9)]
			colormap = LinearColormap(sampled_colors, vmin=float(valid.min()), vmax=float(valid.max()))
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

		tooltip_fields = [probability_column]
		folium_mod.GeoJson(
			probability_grid_plot[[probability_column, "geometry"]].to_json(),
			style_function=_style_fn,
			tooltip=folium_mod.GeoJsonTooltip(fields=tooltip_fields, aliases=[probability_column]),
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
	"""Plot a probability grid with optional overlays."""
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
	"""Plot synthetic employee points with optional overlays."""
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
	"""Plot one or more routes with optional origin and destination points."""
	return plot_commute_map(
		routes=routes,
		route_origins=origins,
		route_destinations=destinations,
		route_options=route_options,
		backend=backend,
		**kwargs,
	)


__all__ = [
	"plot_commute_map",
	"plot_sampling_probability_map",
	"plot_synthetic_employees",
	"plot_routes",
	"plot_transport_score_routes",
]

