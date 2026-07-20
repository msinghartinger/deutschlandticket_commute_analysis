from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import r5py
from r5py import TransportMode, TransportNetwork
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union


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
        data_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "osm"
    else:
        # Convert input to Path object if string is provided
        data_dir = Path(data_dir)

    # Define the list of required OSM PBF filenames
    required_files = [
        "hamburg-260718.osm.pbf",
        "niedersachsen-260718.osm.pbf",
        "schleswig-holstein-260718.osm.pbf",
    ]

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


@lru_cache(maxsize=1)
def _build_transport_network_cached(combined_path_str: str, allow_errors: bool) -> TransportNetwork:
    """
    Build and cache a transport network for a given merged OSM file.

    Args:
        combined_path_str (str): The file path string to the merged OSM file.
        allow_errors (bool): Boolean flag indicating whether to allow errors during processing.

    Returns:
        TransportNetwork: A TransportNetwork object instantiated from the provided path.
    """
    # Instantiate the TransportNetwork using the provided path and error handling flag
    # This function is cached to avoid rebuilding the network for the same path multiple times
    return TransportNetwork(Path(combined_path_str), allow_errors=allow_errors)


def build_transport_network(
    data_dir: str | Path | None = None,
    allow_errors: bool = False,
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

    # Define the output directory for the merged OSM file relative to the script
    combined_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "osm" / "merged"
    # Create the directory structure if it does not already exist
    combined_dir.mkdir(parents=True, exist_ok=True)
    # Define the full path for the merged OSM PBF file
    combined_path = combined_dir / "northern-germany.osm.pbf"

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

    return _build_transport_network_cached(str(combined_path.resolve()), allow_errors)


def route_between_points(
    origin: Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame,
    destination: Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame,
    transport_modes: Iterable[str] | None = None,
    data_dir: str | Path | None = None,
    crs: str = "EPSG:4326",
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
        List of transport modes to evaluate. Defaults to ["car", "bike"]. Supported: "car", "bike".
    data_dir :
        Path to the directory containing OpenStreetMap PBF files for the routing network.
    crs :
        Coordinate Reference System of the input coordinates. Defaults to EPSG:4326 (WGS84).

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
        transport_modes = ["car", "bike"]

    # Normalize transport mode strings to lowercase for consistent comparison
    transport_modes = [mode.lower() for mode in transport_modes]
    # Identify and flag any unsupported transport modes
    unsupported = [mode for mode in transport_modes if mode not in {"car", "bike"}]
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

    origin_crs = "EPSG:4326" if isinstance(origin, tuple) else crs
    destination_crs = "EPSG:4326" if isinstance(destination, tuple) else crs

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
    projected_crs = "EPSG:4326"
    origin_projected = origin_gdf.to_crs(projected_crs)
    destination_projected = destination_gdf.to_crs(projected_crs)

    # Build the transport network from local OSM data
    transport_network = build_transport_network(data_dir=data_dir)

    # List to store routing results for each transport mode
    route_rows: list[dict] = []
    for mode in transport_modes:
        # Map string mode to TransportMode enum expected by r5py
        mode_enum = TransportMode.CAR if mode == "car" else TransportMode.BICYCLE

        try:
            # Request detailed itineraries from the routing engine
            result = r5py.DetailedItineraries(
                transport_network=transport_network,
                origins=origin_projected,
                destinations=destination_projected,
                transport_modes=[mode_enum],
                snap_to_network=True,
            )

            # Skip if no route was found for this mode
            if result.empty:
                continue

            # Extract metrics and geometry from the first result
            row = result.iloc[0]
            distance_m = float(row["distance"])
            duration_s = row["travel_time"].total_seconds()
            route_geometry = row["geometry"]

            # Convert units from meters to kilometers and seconds to minutes
            distance_km = distance_m / 1000.0 if distance_m else None
            duration_min = duration_s / 60.0 if duration_s else None

            # Append result to the collection
            route_rows.append(
                {
                    "mode": mode,
                    "distance_km": distance_km,
                    "duration_min": duration_min,
                    "geometry": route_geometry,
                }
            )
        except Exception as exc:
            # Store error details if routing fails for this mode
            route_rows.append(
                {
                    "mode": mode,
                    "distance_km": None,
                    "duration_min": None,
                    "geometry": None,
                    "error": str(exc),
                }
            )

    # Raise an error if routing failed for all requested transport modes
    if not route_rows:
        raise RuntimeError("Routing failed for all requested transport modes")

    # Create a GeoDataFrame containing all evaluated routes
    route_gdf = gpd.GeoDataFrame(route_rows, geometry="geometry", crs=projected_crs)

    # Identify the best route based on duration and distance (Pareto optimization)
    best_row = None
    best_key = None
    for row in route_rows:
        # Skip rows where distance or duration are missing
        if row.get("distance_km") is None or row.get("duration_min") is None:
            continue
        # Create a tuple key: (duration, distance) for comparison
        key = (row["duration_min"], row["distance_km"])
        # Select the first row with the smallest key (duration first, then distance)
        if best_key is None or key < best_key:
            best_key = key
            best_row = row

    # Return the metrics and route GDF. If no valid route found (best_row is None),
    # return None values for metrics along with the GDF (which may be empty).
    if best_row is None:
        return None, None, route_gdf

    return best_row["distance_km"], best_row["duration_min"], route_gdf
