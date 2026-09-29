from __future__ import annotations

import subprocess
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import numpy as np

import geopandas as gpd
import pandas as pd
import r5py
from r5py import TransportMode, TransportNetwork
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union

from . import config


def _resolve_osm_files(data_dir: str | Path | None = None) -> list[Path]:
    """
    Resolve the paths to the local OSM PBF files used to build the transport network.

    This function locates the required OpenStreetMap PBF files in the specified
    or default data directory, ensuring all necessary files are present before
    returning the list of paths.

    Parameters
    ----------
    data_dir : str | Path | None, optional
        The directory path containing the OSM PBF files. If None, defaults
        to 'data/raw/osm' relative to the project root.

    Returns
    -------
    list[Path]
        A list of Path objects representing the valid locations of the OSM files.

    Raises
    ------
    FileNotFoundError
        If any of the required OSM PBF files are missing from the directory.
    """
    # Determine the directory to search for OSM files
    if data_dir is None:
        # Default to 'data/raw/osm' relative to the script's location (parents[2])
        data_dir = config.OSM_DATA_DIR
    else:
        # Convert input to Path object if string is provided
        data_dir = Path(data_dir)

    # Define the list of required OSM PBF filenames
    required_files = config.OSM_PBF_FILENAMES

    # Construct full paths for each required file
    paths = [data_dir / name for name in required_files]

    # Identify any paths that do not exist on the filesystem
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        # Raise an error if required files are not found
        raise FileNotFoundError(
            "Missing OSM PBF files: " + ", ".join(missing)
        )

    return paths


def _resolve_gtfs_file(data_dir: str | Path | None = None) -> Path:
    """Resolve the GTFS ZIP file used for public transport routing."""
    if data_dir is None:
        data_dir = config.RAW_DATA_DIR
    else:
        data_dir = Path(data_dir)

    gtfs_name = config.GTFS_FILENAME
    candidate_dirs = [
        data_dir,
        data_dir.parent,
        data_dir.parent.parent,
        config.RAW_DATA_DIR,
    ]

    seen_dirs: set[Path] = set()
    for candidate_dir in candidate_dirs:
        if candidate_dir in seen_dirs:
            continue
        seen_dirs.add(candidate_dir)
        gtfs_path = candidate_dir / gtfs_name
        if gtfs_path.exists():
            return gtfs_path

    raise FileNotFoundError(f"Missing GTFS file: {gtfs_name}")


@lru_cache(maxsize=1)
def _build_transport_network_cached(combined_path_str: str, gtfs_path_str: str, allow_errors: bool) -> TransportNetwork:
    """
    Build and cache a transport network for a given merged OSM file.

    Args:
        combined_path_str (str): The file path string to the merged OSM file.
        gtfs_path_str (str): The file path string to the GTFS ZIP file.
        allow_errors (bool): Boolean flag indicating whether to allow errors during processing.

    Returns:
        TransportNetwork: A TransportNetwork object instantiated from the provided path.
    """
    return TransportNetwork(
        Path(combined_path_str),
        gtfs=[Path(gtfs_path_str)],
        allow_errors=allow_errors,
    )


def build_transport_network(
    data_dir: str | Path | None = None,
    allow_errors: bool = config.ALLOW_ROUTING_DATA_ERRORS,
) -> TransportNetwork:
    """
    Build an r5py transport network from the local OSM PBF files.

    This function resolves the input OSM file paths, merges them into a single
    combined OSM PBF file if it does not already exist, and initializes a
    TransportNetwork instance from the resulting data. The resulting network is
    cached so subsequent route calculations reuse it instead of rebuilding it.

    Args:
        data_dir (str | Path | None): Directory containing source OSM PBF files.
            If None, defaults to the expected input location.
        allow_errors (bool): If True, allows loading errors during network
            initialization. If False (default), raises an error.

    Returns:
        TransportNetwork: An r5py transport network object initialized from
            the merged OSM data.

    Notes:
        The merged output file is expected to be named "northern-germany.osm.pbf"
        and is stored in the "data/raw/osm/merged" directory relative to the
        script location.
    """
    # Resolve the list of OSM file paths based on the provided data_dir
    osm_files = _resolve_osm_files(data_dir)
    gtfs_file = _resolve_gtfs_file(data_dir)

    # Define the output directory for the merged OSM file relative to the script
    combined_dir = config.MERGED_OSM_DIR
    # Create the directory structure if it does not already exist
    combined_dir.mkdir(parents=True, exist_ok=True)
    # Define the full path for the merged OSM PBF file
    combined_path = config.MERGED_OSM_PATH

    # Check if the merged file already exists on the filesystem
    if not combined_path.exists():
        # Execute osmium merge command to combine all OSM files into a single file
        command = [
            "osmium",
            "merge",
            *[str(path) for path in osm_files],
            "-o",
            str(combined_path),
        ]
        # Run the subprocess command and raise an exception if it fails
        subprocess.run(command, check=True, capture_output=True)

    return _build_transport_network_cached(
        str(combined_path.resolve()),
        str(gtfs_file.resolve()),
        allow_errors,
    )


def _normalize_departure_time(departure_time: datetime | date | str | None) -> datetime | None:
    """Coerce supported departure time inputs into a datetime object."""
    if departure_time is None:
        return None
    if isinstance(departure_time, datetime):
        return departure_time
    if isinstance(departure_time, date):
        return datetime.combine(departure_time, datetime.min.time())
    if isinstance(departure_time, str):
        try:
            return datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("departure_time must be a datetime, date, or ISO-8601 string") from exc
    raise TypeError("departure_time must be a datetime, date, or ISO-8601 string")


def _normalize_leg_mode(value: object) -> str | None:
    """Translate r5py transport mode values into a simple leg-level mode label."""
    if value is None:
        return None

    if isinstance(value, str):
        return value.lower()

    if hasattr(value, "name"):
        name = str(value.name).lower()
        if name == "bicycle":
            return "bike"
        if name == "transit":
            return "public_transport"
        return name

    return str(value).lower()


def _compute_option_duration_min(itinerary_rows: gpd.GeoDataFrame) -> float:
    """Compute end-to-end duration for one itinerary option in minutes.

    Duration is defined as the elapsed time from the first leg departure to the
    final arrival at destination, where final arrival is computed as
    departure_time + travel_time on each leg.
    """
    if itinerary_rows.empty:
        return 0.0

    has_time_columns = {"departure_time", "travel_time"}.issubset(itinerary_rows.columns)
    if has_time_columns:
        valid_rows = itinerary_rows[
            itinerary_rows["departure_time"].notna() & itinerary_rows["travel_time"].notna()
        ].copy()
        if not valid_rows.empty:
            valid_rows["arrival_time"] = valid_rows["departure_time"] + valid_rows["travel_time"]
            first_departure = valid_rows["departure_time"].min()
            final_arrival = valid_rows["arrival_time"].max()
            if pd.notna(first_departure) and pd.notna(final_arrival):
                return float((final_arrival - first_departure).total_seconds() / 60.0)

    # Fallback for unexpected schemas: sum leg travel times.
    if "travel_time" in itinerary_rows.columns:
        return float(
            itinerary_rows["travel_time"].apply(
                lambda value: value.total_seconds() / 60.0 if pd.notna(value) else 0.0
            ).sum()
        )
    return 0.0


def route_between_points(
    origin: Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame,
    destination: Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame,
    transport_modes: Iterable[str] | None = None,
    data_dir: str | Path | None = None,
    crs: str = config.CRS_WGS84,
    departure_time: datetime | date | str | None = None,
) -> tuple[float | None, float | None, gpd.GeoDataFrame | None]:
    """Calculate the best route between two locations using specified transport modes.

    This function computes routes using R5 routing engine for car and/or bike modes.
    It handles various input formats for locations (Points, tuples, GeoDataFrames) and
    ensures coordinate reference systems are aligned before projection.
    The routing is performed in a metric CRS (EPSG:25832) for accurate distance calculations.

    Parameters
    ----------
    origin, destination :
        Start and end locations. Accepts Shapely Point, (lon, lat) tuple, or
        GeoSeries/GeoDataFrame. If GeoSeries/GeoDataFrame is provided, only the first
        geometry is used.
    transport_modes :
        List of transport modes to evaluate. Defaults to ["car", "bike"]. Supported:
        "car", "bike", "walk", and "public_transit". To request a combination of
        modes, pass an iterable containing the desired mode names.
    data_dir :
        Path to the directory containing OpenStreetMap PBF files for the routing network.
    crs :
        Coordinate Reference System of the input coordinates. Defaults to EPSG:4326 (WGS84).
    departure_time :
        Departure time for time-dependent routing. Required for public transport and multimodal
        routing when transit availability should be considered.

    Returns
    -------
    tuple[float | None, float | None, gpd.GeoDataFrame | None]
        A tuple containing:
        - distance_km: Total distance in kilometers (float or None).
        - duration_min: Duration in minutes (float or None).
        - route_gdf: GeoDataFrame with route geometry (may be empty if routing failed).
        Returns (None, None, route_gdf) if valid routes exist but no best match was found
        based on the selection heuristic. Raises RuntimeError if routing fails for all modes.

    Raises
    ------
    ValueError :
        If transport modes are unsupported or if the CRS of origin/destination is missing.
    RuntimeError :
        If no valid routes are found for any requested transport mode.
    TypeError :
        If an unsupported input type is provided for origin or destination.
    """
    # Set default transport modes if not provided
    if transport_modes is None:
        transport_modes = config.DEFAULT_ROUTE_MODES
    if isinstance(transport_modes, str):
        transport_modes = [transport_modes]

    # Normalize transport mode strings to lowercase for consistent comparison
    transport_modes = [mode.lower() for mode in transport_modes]
    # Identify and flag any unsupported transport modes
    unsupported = [mode for mode in transport_modes if mode not in {"car", "bike", "walk", "public_transit"}]
    if unsupported:
        raise ValueError(f"Unsupported transport modes: {unsupported}")

    # Helper function to standardize input locations into Shapely Point objects
    def _to_point(value: Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame) -> Point:
        """Convert various location input types to a Shapely Point."""
        if isinstance(value, Point):
            # Already a Point object
            return value
        if isinstance(value, tuple):
            # Treat tuple inputs as (latitude, longitude), which is the common
            # convention for geographic coordinates in notebooks and geospatial data.
            lat, lon = value
            return Point(lon, lat)
        if isinstance(value, gpd.GeoSeries):
            # Extract the first geometry from the series
            return value.iloc[0]
        if isinstance(value, gpd.GeoDataFrame):
            # Extract the first geometry from the dataframe
            return value.geometry.iloc[0]
        raise TypeError("Unsupported point input type")

    # Convert origin and destination inputs to Point objects.
    # Tuple inputs are treated as latitude/longitude pairs in WGS84, regardless
    # of any CRS passed in by the caller.
    origin_point = _to_point(origin)
    destination_point = _to_point(destination)

    origin_crs = config.CRS_WGS84 if isinstance(origin, tuple) else crs
    destination_crs = config.CRS_WGS84 if isinstance(destination, tuple) else crs

    # Create temporary GeoDataFrames for origin and destination for CRS handling
    origin_gdf = gpd.GeoDataFrame({"id": [0]}, geometry=[origin_point], crs=origin_crs)
    destination_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[destination_point], crs=destination_crs)

    # Validate that CRS information is present for both points
    if origin_gdf.crs is None:
        raise ValueError("origin CRS is required")
    if destination_gdf.crs is None:
        raise ValueError("destination CRS is required")

    # Ensure both GeoDataFrames share the same CRS by reprojecting destination if needed
    if origin_gdf.crs != destination_gdf.crs:
        destination_gdf = destination_gdf.to_crs(origin_gdf.crs)

    # Project coordinates to EPSG:4326 for routing calculations
    projected_crs = config.CRS_WGS84
    origin_projected = origin_gdf.to_crs(projected_crs)
    destination_projected = destination_gdf.to_crs(projected_crs)

    # Build the transport network from local OSM data, tolerating GTFS warnings
    transport_network = build_transport_network(
        data_dir=data_dir,
        allow_errors=config.ALLOW_ROUTING_DATA_ERRORS,
    )

    normalized_departure_time = _normalize_departure_time(departure_time)
    if normalized_departure_time is None and any(mode == "public_transit" for mode in transport_modes):
        normalized_departure_time = datetime.now()

    # List to store routing results for each transport mode
    candidate_results: list[dict] = []
    for mode in transport_modes:
        # Map string mode to one or more TransportMode enums expected by r5py
        if mode == "car":
            mode_enums = [TransportMode.CAR]
        elif mode == "bike":
            mode_enums = [TransportMode.BICYCLE]
        elif mode == "walk":
            mode_enums = [TransportMode.WALK]
        elif mode == "public_transit":
            mode_enums = [TransportMode.TRANSIT, TransportMode.WALK]
        else:
            mode_enums = []

        try:
            request_kwargs = {
                "transport_network": transport_network,
                "origins": origin_projected,
                "destinations": destination_projected,
                "transport_modes": mode_enums,
                "snap_to_network": config.SNAP_TO_NETWORK,
            }
            if normalized_departure_time is not None:
                request_kwargs["departure"] = normalized_departure_time

            # Request detailed itineraries from the routing engine
            result = r5py.DetailedItineraries(**request_kwargs)

            # Skip if no route was found for this mode
            if result.empty:
                continue

            itinerary_gdf = result.copy()
            if "option" in itinerary_gdf.columns:
                option_groups = itinerary_gdf.groupby("option", dropna=False)
                best_option = None
                best_key = None
                for option_value, option_rows in option_groups:
                    option_duration_min = _compute_option_duration_min(option_rows)
                    option_distance_km = float(option_rows["distance"].sum() / 1000.0)
                    key = (option_duration_min, option_distance_km)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_option = option_value

                if best_option is not None:
                    itinerary_gdf = itinerary_gdf[itinerary_gdf["option"] == best_option].copy()

            itinerary_gdf["mode"] = mode
            itinerary_gdf["leg_mode"] = itinerary_gdf["transport_mode"].apply(_normalize_leg_mode)
            itinerary_gdf["distance_km"] = itinerary_gdf["distance"] / 1000.0
            itinerary_gdf["duration_min"] = itinerary_gdf["travel_time"].apply(
                lambda value: value.total_seconds() / 60.0 if pd.notna(value) else None
            )
            itinerary_gdf["leg_index"] = range(len(itinerary_gdf))

            total_distance_km = float(itinerary_gdf["distance_km"].sum())
            total_duration_min = _compute_option_duration_min(itinerary_gdf)

            candidate_results.append(
                {
                    "mode": mode,
                    "distance_km": total_distance_km,
                    "duration_min": total_duration_min,
                    "itinerary_gdf": itinerary_gdf,
                }
            )
        except Exception as exc:
            # Store error details if routing fails for this mode
            candidate_results.append(
                {
                    "mode": mode,
                    "distance_km": None,
                    "duration_min": None,
                    "itinerary_gdf": None,
                    "error": str(exc),
                }
            )

    # Raise an error if routing failed for all requested transport modes
    if not candidate_results:
        raise RuntimeError("Routing failed for all requested transport modes")

    # Identify the best route based on duration and distance (Pareto optimization)
    best_row = None
    best_key = None
    for row in candidate_results:
        # Skip rows where distance or duration are missing
        if row.get("distance_km") is None or row.get("duration_min") is None:
            continue
        # Create a tuple key: (duration, distance) for comparison
        key = (row["duration_min"], row["distance_km"])
        # Select the first row with the smallest key (duration first, then distance)
        if best_key is None or key < best_key:
            best_key = key
            best_row = row

    # Return the metrics and itinerary GDF. If no valid route found (best_row is None),
    # return None values for metrics along with the GDF (which may be empty).
    if best_row is None:
        return None, None, None

    return best_row["distance_km"], best_row["duration_min"], best_row["itinerary_gdf"]

def calculate_reachable_grid(
    transport_network: TransportNetwork,
    origin: gpd.GeoDataFrame,
    departure_time: datetime,
    *,
    departure_time_window: timedelta = config.REACHABILITY_TIME_WINDOW,
    analysis_radius_m: float = config.REACHABILITY_RADIUS_M,
    grid_step_m: float = config.REACHABILITY_GRID_STEP_M,
    max_travel_time_min: float = config.MAX_REACHABILITY_TIME_MIN,
) -> gpd.GeoDataFrame:
    """Calculate public-transport travel times to a regular grid around origin."""
    origin_wgs84 = origin.to_crs(config.CRS_WGS84)
    origin_projected = origin.to_crs(config.CRS_LOCAL_METRIC)
    center = origin_projected.geometry.iloc[0]

    # Ensure origin has a unique 'id' column for r5py.TravelTimeMatrix
    if "id" not in origin_wgs84.columns or not origin_wgs84["id"].is_unique:
        origin_wgs84 = origin_wgs84.copy()
        origin_wgs84["id"] = range(len(origin_wgs84))

    xs = np.arange(
        center.x - analysis_radius_m,
        center.x + analysis_radius_m + grid_step_m,
        grid_step_m,
    )
    ys = np.arange(
        center.y - analysis_radius_m,
        center.y + analysis_radius_m + grid_step_m,
        grid_step_m,
    )

    points = [
        Point(x, y)
        for x in xs
        for y in ys
        if Point(x, y).distance(center) <= analysis_radius_m
    ]

    destinations = gpd.GeoDataFrame(
        {"id": range(1, len(points) + 1)},
        geometry=points,
        crs=config.CRS_LOCAL_METRIC,
    ).to_crs(config.CRS_WGS84)

    ttm = r5py.TravelTimeMatrix(
        transport_network,
        origins=origin_wgs84,
        destinations=destinations,
        departure=departure_time,
        departure_time_window=departure_time_window,
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        snap_to_network=config.SNAP_TO_NETWORK,
    )

    if ttm.empty:
        return destinations.iloc[0:0].assign(travel_time_min=np.nan)

    travel_time = ttm["travel_time"]

    if pd.api.types.is_timedelta64_dtype(travel_time):
        ttm["travel_time_min"] = travel_time.dt.total_seconds() / 60
    else:
        ttm["travel_time_min"] = pd.to_numeric(
            travel_time,
            errors="coerce",
        )

    reachable = (
        ttm.loc[
            ttm["travel_time_min"].notna()
            & ttm["travel_time_min"].between(
                0,
                max_travel_time_min,
                inclusive="right",
            )
        ]
        .groupby("to_id", as_index=False)["travel_time_min"]
        .min()
        .rename(columns={"to_id": "id"})
    )

    result = destinations.merge(reachable, on="id", how="inner")
    result["grid_step_m"] = grid_step_m
    result["departure_time"] = departure_time
    return result