from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import exp, log1p, sqrt
from pathlib import Path
from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import r5py
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union

from .routing import build_transport_network, route_between_points

# Constants for scoring and routing
SERVICE_DATE = date(2026, 9, 15)

MORNING_START = time(7, 0)
MORNING_END = time(9, 0)

EVENING_START = time(16, 0)
EVENING_END = time(18, 0)

DEPARTURE_INTERVAL = timedelta(minutes=10)

WORKPLACE = (53.686439, 10.046120)
WORKPLACE_CRS = "EPSG:4326"


RELATIVE_TIME_WEIGHT = 0.25
ABSOLUTE_TIME_WEIGHT = 0.25
CONSISTENCY_WEIGHT = 0.30
WALKING_WEIGHT = 0.10
TRANSFERS_WEIGHT = 0.10

assert abs(RELATIVE_TIME_WEIGHT + ABSOLUTE_TIME_WEIGHT + CONSISTENCY_WEIGHT + WALKING_WEIGHT + TRANSFERS_WEIGHT - 1.0) < 1e-9, "Weights must sum to 1.0"

# # Per trip convenience score weight
# TIME_CONVENIENCE_WEIGHT = 0.5
# TRANSFERS_CONVENIENCE_WEIGHT = 0.25
# WALKING_CONVENIENCE_WEIGHT = 0.25

# assert abs(TIME_CONVENIENCE_WEIGHT + TRANSFERS_CONVENIENCE_WEIGHT + WALKING_CONVENIENCE_WEIGHT - 1.0) < 1e-9, "Convenience weights must sum to 1.0"

PointLike = Point | tuple[float, float] | gpd.GeoSeries | gpd.GeoDataFrame

def _coerce_point(value: PointLike, *, crs: Any, name: str, input_crs: Any | None = None) -> Point:
    """Convert various input types to a single Point geometry.

    Handles Point, tuple (lon, lat), GeoSeries, and GeoDataFrame inputs.
    Converts to EPSG:4326 coordinates if necessary.

    Args:
        value: Input point in any supported format.
        crs: Coordinate reference system for the output.
        name: Human-readable name for error messages.

    Returns:
        A Shapely Point in the specified CRS.

    Raises:
        ValueError: If input is empty or doesn't contain valid point geometry.
        TypeError: If input type is not supported.
    """
    if isinstance(value, Point):
        geom = value
        source_crs = input_crs
        if source_crs is not None and source_crs != crs:
            geom = gpd.GeoSeries([geom], crs=source_crs).to_crs(crs).iloc[0]
        return geom
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} tuple must be (latitude, longitude).")
        lat, lon = value
        geom = Point(lon, lat)
        source_crs = input_crs
        if source_crs is not None and source_crs != crs:
            geom = gpd.GeoSeries([geom], crs=source_crs).to_crs(crs).iloc[0]
        return geom
    if isinstance(value, gpd.GeoSeries):
        if value.empty:
            raise ValueError(f"{name} is empty.")
        geom = value.iloc[0]
        if not isinstance(geom, Point):
            raise ValueError(f"{name} must contain point geometry.")
        if value.crs != crs:
            geom = gpd.GeoSeries([geom], crs=value.crs).to_crs(crs).iloc[0]
        return geom
    if isinstance(value, gpd.GeoDataFrame):
        if value.empty:
            raise ValueError(f"{name} is empty.")
        geom = value.geometry.iloc[0]
        if not isinstance(geom, Point):
            raise ValueError(f"{name} must contain point geometry.")
        if value.crs != crs:
            geom = gpd.GeoSeries([geom], crs=value.crs).to_crs(crs).iloc[0]
        return geom
    raise TypeError(f"Unsupported point input for {name}.")


@dataclass
class EmployeeRoutingBatchResult:
    transit_routes: pd.DataFrame
    car_route: pd.DataFrame


@dataclass
class EmployeeTransportScoreResult:
    employee_id: int | str
    transport_score: float
    relative_time_score: float
    absolute_time_score: float
    consistency_score: float
    walking_score: float
    transfers_score: float
    pt_travel_time_p25_total: float
    morning_q25_route_geometry: Any
    car_route_record: pd.DataFrame
    pt_route_record: pd.DataFrame


@dataclass
class TransportScoreResult:
    employee_id: int | str
    transport_score: float
    relative_time_score: float
    absolute_time_score: float
    consistency_score: float
    walking_score: float
    transfers_score: float


def _mode_to_text(value: object) -> str:
    """Normalize transport mode values to lowercase text labels."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if hasattr(value, "name"):
        return str(value.name).lower()
    return str(value).lower()


def _extract_number_of_transfers(route_gdf: gpd.GeoDataFrame) -> int:
    """Extract the number of transfers from a public transit route.
    
    Counts the number of distinct public transit legs (excluding walking segments).
    The number of transfers is the count of transit legs minus 1.
    
    Args:
        route_gdf: GeoDataFrame with leg information from route_between_points
        
    Returns:
        Number of transfers (0 if only one or no transit vehicles)
    """
    if route_gdf is None or route_gdf.empty:
        return 0

    # Prefer the normalized `leg_mode` column produced by routing; otherwise,
    # derive a comparable lowercase mode label from `transport_mode`.
    mode_source = "leg_mode" if "leg_mode" in route_gdf.columns else "transport_mode"
    modes = route_gdf[mode_source].apply(_mode_to_text)

    # Filter for non-walking legs (public transit vehicles).
    transit_legs = route_gdf[modes != "walk"]
    if transit_legs.empty:
        return 0

    # Count distinct transit legs when an explicit leg identifier exists.
    if "segment" in transit_legs.columns:
        num_vehicles = int(transit_legs["segment"].nunique())
    elif "leg_index" in transit_legs.columns:
        num_vehicles = int(transit_legs["leg_index"].nunique())
    else:
        num_vehicles = len(transit_legs)
    
    # Number of transfers = number of vehicles - 1
    return max(0, num_vehicles - 1)


def _extract_total_walking_time(route_gdf: gpd.GeoDataFrame) -> float:
    """Extract total walking time from a public transit route.
    
    Sums up the travel time for all walking segments in the route.
    
    Args:
        route_gdf: GeoDataFrame with leg information from route_between_points
        
    Returns:
        Total walking time in minutes
    """
    if route_gdf is None or route_gdf.empty:
        return 0.0
    
    # Filter for walking legs only using normalized mode labels.
    mode_source = "leg_mode" if "leg_mode" in route_gdf.columns else "transport_mode"
    modes = route_gdf[mode_source].apply(_mode_to_text)
    walking_legs = route_gdf[modes == "walk"]
    
    # Sum travel_time and convert to minutes
    total_walking_time = 0.0
    for travel_time in walking_legs["travel_time"]:
        if pd.notna(travel_time):
            # travel_time is a timedelta, convert to minutes
            total_walking_time += travel_time.total_seconds() / 60.0
    
    return total_walking_time


def calculate_car_record(
    *,
    employee_id: int | str,
    direction: Literal["morning", "evening"],
    home: PointLike,
    home_crs: Any = "EPSG:4326",
) -> pd.DataFrame:
    """Calculate travel metrics for a single employee using car transport.
    
    Args:
        employee_id: Unique employee identifier
        direction: Either 'morning' or 'evening' to determine departure time and route
        home: Home location (Point, tuple, GeoSeries, or GeoDataFrame)
        home_crs: CRS of home input when home is a Point or tuple
        
    Returns:
        DataFrame with single row containing employee_id, direction, departure_time, 
        travel_time, and distance
    """
    home_point = _coerce_point(home, crs="EPSG:4326", name="home", input_crs=home_crs)
    workplace_point = _coerce_point(
        WORKPLACE,
        crs="EPSG:4326",
        name="WORKPLACE",
        input_crs=WORKPLACE_CRS,
    )

    if direction == "morning":
        departure_time = MORNING_START
        origin = home_point
        destination = workplace_point
    elif direction == "evening":
        departure_time = EVENING_START
        origin = workplace_point
        destination = home_point
    else:
        raise ValueError(f"Direction must be 'morning' or 'evening', got '{direction}'")
    
    # Combine date and time into a datetime
    departure_datetime = datetime.combine(SERVICE_DATE, departure_time)
    
    # Route between points using car transport mode
    distance_km, duration_min, _ = route_between_points(
        origin=origin,
        destination=destination,
        departure_time=departure_datetime,
        transport_modes=["car"],
    )
    
    # Create result DataFrame
    return pd.DataFrame({
        "employee_id": [employee_id],
        "direction": [direction],
        "departure_time": [departure_time],
        "travel_time": [duration_min],
        "distance": [distance_km],
    }) 

def calculate_public_transport_record(
    *,
    employee_id: int | str,
    direction: Literal["morning", "evening"],
    home: PointLike,
    home_crs: Any = "EPSG:4326",
    include_route_gdf: bool = False,
) -> pd.DataFrame:
    """Calculate travel metrics for a single employee using public transport.
    
    Args:
        employee_id: Unique employee identifier
        direction: Either 'morning' or 'evening' to determine departure time range and route
        home: Home location (Point, tuple, GeoSeries, or GeoDataFrame)
        home_crs: CRS of home input when home is a Point or tuple
        include_route_gdf: Whether to include raw route leg GeoDataFrame per row
        
    Returns:
        DataFrame with rows for each departure time containing employee_id, direction, 
        departure_time, travel_time, number_of_transfers, and total_walking_time
    """
    home_point = _coerce_point(home, crs="EPSG:4326", name="home", input_crs=home_crs)
    workplace_point = _coerce_point(
        WORKPLACE,
        crs="EPSG:4326",
        name="WORKPLACE",
        input_crs=WORKPLACE_CRS,
    )

    if direction == "morning":
        start_time = MORNING_START
        end_time = MORNING_END
        origin = home_point
        destination = workplace_point
    elif direction == "evening":
        start_time = EVENING_START
        end_time = EVENING_END
        origin = workplace_point
        destination = home_point
    else:
        raise ValueError(f"Direction must be 'morning' or 'evening', got '{direction}'")
    
    # Generate all departure times from start to end in intervals
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    interval_minutes = int(DEPARTURE_INTERVAL.total_seconds() // 60)
    
    departure_times = []
    current_minutes = start_minutes
    while current_minutes <= end_minutes:
        hours = current_minutes // 60
        minutes = current_minutes % 60
        departure_times.append(time(hours, minutes))
        current_minutes += interval_minutes
    
    results = []
    for departure_time in departure_times:
        # Combine date and time into a datetime
        departure_datetime = datetime.combine(SERVICE_DATE, departure_time)
        
        # Route between points using public transit transport mode
        distance_km, duration_min, route_gdf = route_between_points(
            origin=origin,
            destination=destination,
            departure_time=departure_datetime,
            transport_modes=["public_transit"],
        )
        
        # Extract metrics from route result
        travel_time = duration_min
        number_of_transfers = _extract_number_of_transfers(route_gdf)
        total_walking_time = _extract_total_walking_time(route_gdf)
        
        row = {
            "employee_id": employee_id,
            "direction": direction,
            "departure_time": departure_time,
            "travel_time": travel_time,
            "number_of_transfers": number_of_transfers,
            "total_walking_time": total_walking_time,
        }
        if include_route_gdf:
            row["route_gdf"] = route_gdf
        results.append(row)
    
    return pd.DataFrame(results)


def _subset_direction(record: pd.DataFrame, direction: Literal["morning", "evening"]) -> pd.DataFrame:
    """Return rows for a single commute direction."""
    subset = record[record["direction"] == direction]
    if subset.empty:
        raise ValueError(f"Missing '{direction}' records.")
    return subset


def _percentile_value(record: pd.DataFrame, column: str, quantile: float) -> float:
    """Return a numeric percentile from a route record column."""
    if column not in record.columns:
        raise ValueError(f"Missing required column '{column}'.")
    values = pd.to_numeric(record[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Column '{column}' does not contain any numeric values.")
    return float(values.quantile(quantile))


def _pt_travel_time_p25_total(*, pt_route_record: pd.DataFrame) -> float:
    """Return the summed public-transport 25th percentile travel time across both directions."""
    return (
        _percentile_value(_subset_direction(pt_route_record, "morning"), "travel_time", 0.25)
        + _percentile_value(_subset_direction(pt_route_record, "evening"), "travel_time", 0.25)
    )


def _morning_q25_route_geometry(*, pt_route_record: pd.DataFrame) -> Any:
    """Return geometry for the morning route nearest the 25th-percentile travel time."""
    if "route_gdf" not in pt_route_record.columns:
        raise ValueError("pt_route_record must include 'route_gdf' to extract q25 route geometry.")

    morning = _subset_direction(pt_route_record, "morning").copy()
    q25 = _percentile_value(morning, "travel_time", 0.25)
    travel = pd.to_numeric(morning["travel_time"], errors="coerce")
    if travel.dropna().empty:
        return None

    index = (travel - q25).abs().idxmin()
    route_gdf = morning.loc[index, "route_gdf"]
    if route_gdf is None or getattr(route_gdf, "empty", True):
        return None

    merged = unary_union(route_gdf.geometry.values)
    try:
        return linemerge(merged)
    except Exception:
        return merged


def _car_travel_time_p25_total(*, car_route_record: pd.DataFrame) -> float:
    """Return the summed car 25th percentile travel time across both directions."""
    return (
        _percentile_value(_subset_direction(car_route_record, "morning"), "travel_time", 0.25)
        + _percentile_value(_subset_direction(car_route_record, "evening"), "travel_time", 0.25)
    )


def _score_relative_time(*, car_route_record: pd.DataFrame, pt_route_record: pd.DataFrame) -> float:
    """Return the normalized relative-time score before weighting."""
    pt_total = _pt_travel_time_p25_total(pt_route_record=pt_route_record)
    car_total = _car_travel_time_p25_total(car_route_record=car_route_record)
    if car_total <= 0:
        raise ValueError("Car travel time must be positive.")
    value = pt_total / car_total
    score = (3.5 - value) / (3.5 - 0.5)
    return max(0.0, min(1.0, score))


def _score_absolute_time(*, pt_route_record: pd.DataFrame) -> float:
    """Return the normalized absolute-time score before weighting."""
    value = _pt_travel_time_p25_total(pt_route_record=pt_route_record)
    score = (180.0 - value) / 180.0
    return max(0.0, min(1.0, score))


def _score_consistency(*, pt_route_record: pd.DataFrame) -> float:
    """Return the normalized consistency score before weighting."""
    direction_values: list[float] = []
    for direction in ("morning", "evening"):
        subset = _subset_direction(pt_route_record, direction)
        q25 = _percentile_value(subset, "travel_time", 0.25)
        q75 = _percentile_value(subset, "travel_time", 0.75)
        if q75 <= 0:
            raise ValueError(f"{direction} public transport 75th percentile must be positive.")
        direction_values.append(q75 / q25)

    value = float(sum(direction_values) / len(direction_values))
    score = (1.33 - value) / (1.33 - 1.0)
    return max(0.0, min(1.0, score))


def _score_walking(*, pt_route_record: pd.DataFrame) -> float:
    """Return the normalized walking-burden score before weighting."""
    value = (
        _percentile_value(_subset_direction(pt_route_record, "morning"), "total_walking_time", 0.25)
        + _percentile_value(_subset_direction(pt_route_record, "evening"), "total_walking_time", 0.25)
    )
    score = (60.0 - value) / 60.0
    return max(0.0, min(1.0, score))


def _score_transfers(*, pt_route_record: pd.DataFrame) -> float:
    """Return the normalized transfer-burden score before weighting."""
    value = (
        _percentile_value(_subset_direction(pt_route_record, "morning"), "number_of_transfers", 0.25)
        + _percentile_value(_subset_direction(pt_route_record, "evening"), "number_of_transfers", 0.25)
    )
    score = (8.0 - value) / 8.0
    return max(0.0, min(1.0, score))

def calculate_transport_score(
    *,
    employee_id: int | str,
    car_route_record: pd.DataFrame,
    pt_route_record: pd.DataFrame,
) -> TransportScoreResult:
    """Calculate a transport score based on travel time, transfers, and walking time.
    
    Args:
        car_route_record: DataFrame containing car travel metrics
        pt_route_record: DataFrame containing public transport travel metrics

    Returns:
        TransportScoreResult containing the total weighted score and the five
        unweighted component scores.
    """
    if "employee_id" in car_route_record.columns:
        car_ids = car_route_record["employee_id"].dropna().unique()
        if len(car_ids) > 0 and not all(str(value) == str(employee_id) for value in car_ids):
            raise ValueError("car_route_record contains rows for a different employee_id.")

    if "employee_id" in pt_route_record.columns:
        pt_ids = pt_route_record["employee_id"].dropna().unique()
        if len(pt_ids) > 0 and not all(str(value) == str(employee_id) for value in pt_ids):
            raise ValueError("pt_route_record contains rows for a different employee_id.")

    relative_time_score = _score_relative_time(
        car_route_record=car_route_record,
        pt_route_record=pt_route_record,
    )
    absolute_time_score = _score_absolute_time(pt_route_record=pt_route_record)
    consistency_score = _score_consistency(pt_route_record=pt_route_record)
    walking_score = _score_walking(pt_route_record=pt_route_record)
    transfers_score = _score_transfers(pt_route_record=pt_route_record)

    transport_score = (
        relative_time_score * RELATIVE_TIME_WEIGHT
        + absolute_time_score * ABSOLUTE_TIME_WEIGHT
        + consistency_score * CONSISTENCY_WEIGHT
        + walking_score * WALKING_WEIGHT
        + transfers_score * TRANSFERS_WEIGHT
    )

    return TransportScoreResult(
        employee_id=employee_id,
        transport_score=transport_score,
        relative_time_score=relative_time_score,
        absolute_time_score=absolute_time_score,
        consistency_score=consistency_score,
        walking_score=walking_score,
        transfers_score=transfers_score,
    )


def calculate_employee_transport_score(
    *,
    employee_id: int | str,
    home: PointLike,
    home_crs: Any = "EPSG:4326",
) -> EmployeeTransportScoreResult:
    """
    Calculate route records, transport score, and summed PT 25th-percentile travel time.

    This function aggregates commute data for both car and public transport modes for a specific employee.
    It combines morning and evening route records to compute a comprehensive transport score and
    the 25th percentile of total public transport travel time.

    Args:
        employee_id: The unique identifier for the employee (int or str).
        home: The location object representing the employee's home address.
        home_crs: The coordinate reference system for the home location (default: "EPSG:4326").

    Returns:
        EmployeeTransportScoreResult: A structured result object containing the transport score,
        public transport travel time statistics (p25 total), and the concatenated route records
        for both car and public transport.
    """
    # Aggregate morning and evening car route records
    car_route_record = pd.concat(
        [
            calculate_car_record(
                employee_id=employee_id,
                direction="morning",
                home=home,
                home_crs=home_crs,
            ),
            calculate_car_record(
                employee_id=employee_id,
                direction="evening",
                home=home,
                home_crs=home_crs,
            ),
        ],
        ignore_index=True,
    )

    # Aggregate morning and evening public transport route records
    pt_route_record = pd.concat(
        [
            calculate_public_transport_record(
                employee_id=employee_id,
                direction="morning",
                home=home,
                home_crs=home_crs,
                include_route_gdf=True,
            ),
            calculate_public_transport_record(
                employee_id=employee_id,
                direction="evening",
                home=home,
                home_crs=home_crs,
                include_route_gdf=True,
            ),
        ],
        ignore_index=True,
    )

    # Construct and return the final result object with calculated metrics and records
    transport_score_result = calculate_transport_score(
        employee_id=employee_id,
        car_route_record=car_route_record,
        pt_route_record=pt_route_record,
    )

    return EmployeeTransportScoreResult(
        employee_id=employee_id,
        transport_score=transport_score_result.transport_score,
        relative_time_score=transport_score_result.relative_time_score,
        absolute_time_score=transport_score_result.absolute_time_score,
        consistency_score=transport_score_result.consistency_score,
        walking_score=transport_score_result.walking_score,
        transfers_score=transport_score_result.transfers_score,
        pt_travel_time_p25_total=_pt_travel_time_p25_total(pt_route_record=pt_route_record),
        morning_q25_route_geometry=_morning_q25_route_geometry(pt_route_record=pt_route_record),
        car_route_record=car_route_record,
        pt_route_record=pt_route_record,
    )


def _calculate_employee_transport_score_row(
    *,
    employee_id: int | str,
    home: Any,
    home_crs: Any,
) -> dict[str, Any]:
    """Return transport-score summary fields for one employee row."""
    result = calculate_employee_transport_score(
        employee_id=employee_id,
        home=home,
        home_crs=home_crs,
    )
    return {
        "total_travel_time": result.pt_travel_time_p25_total,
        "transport_score": result.transport_score,
        "relative_time_score": result.relative_time_score,
        "absolute_time_score": result.absolute_time_score,
        "consistency_score": result.consistency_score,
        "walking_score": result.walking_score,
        "transfers_score": result.transfers_score,
        "morning_q25_route": result.morning_q25_route_geometry,
    }


def calculate_transport_scores_for_geodataframe(
    *,
    employee_gdf: gpd.GeoDataFrame,
    employee_id_column: str,
    home_column: str,
    home_crs: Any | None = None,
    max_workers: int | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Score all employees in a GeoDataFrame in parallel and return a reduced copy.

    The returned frame keeps only the requested employee-id and home columns,
    renames them to ``employee_id`` and ``home``, and appends the transport
    summary metrics for each employee.
    """
    if employee_id_column not in employee_gdf.columns:
        raise ValueError(f"Missing employee id column '{employee_id_column}'.")
    if home_column not in employee_gdf.columns:
        raise ValueError(f"Missing home column '{home_column}'.")

    selected = employee_gdf[[employee_id_column, home_column]].copy()
    selected = selected.rename(columns={employee_id_column: "employee_id", home_column: "home"})

    source_home_crs = home_crs if home_crs is not None else employee_gdf.crs
    records = selected.to_dict("records")
    total = len(records)
    if total == 0:
        empty_summary = pd.DataFrame(
            columns=[
                "employee_id",
                "home",
                "total_travel_time",
                "transport_score",
                "relative_time_score",
                "absolute_time_score",
                "consistency_score",
                "walking_score",
                "transfers_score",
            ]
        )
        empty_routes = pd.DataFrame(columns=["employee_id", "home", "morning_q25_route"])
        try:
            summary_gdf = gpd.GeoDataFrame(empty_summary, geometry="home", crs=source_home_crs)
        except Exception:
            summary_gdf = gpd.GeoDataFrame(empty_summary)
        try:
            route_gdf = gpd.GeoDataFrame(empty_routes, geometry="morning_q25_route", crs="EPSG:4326")
        except Exception:
            route_gdf = gpd.GeoDataFrame(empty_routes)
        return summary_gdf, route_gdf

    score_rows: list[dict[str, Any] | None] = [None] * total
    future_to_index: dict[Any, int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in enumerate(records):
            future = executor.submit(
                _calculate_employee_transport_score_row,
                employee_id=row["employee_id"],
                home=row["home"],
                home_crs=source_home_crs,
            )
            future_to_index[future] = index

        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            score_rows[index] = future.result()
            completed += 1
            print(f"Processed {completed}/{total} employees", flush=True)

    score_frame = pd.DataFrame(score_rows)
    summary_columns = [
        "total_travel_time",
        "transport_score",
        "relative_time_score",
        "absolute_time_score",
        "consistency_score",
        "walking_score",
        "transfers_score",
    ]
    summary_result = pd.concat(
        [selected.reset_index(drop=True), score_frame[summary_columns].reset_index(drop=True)],
        axis=1,
    )

    route_result = selected[["employee_id", "home"]].reset_index(drop=True).copy()
    route_result["morning_q25_route"] = score_frame["morning_q25_route"].reset_index(drop=True)

    try:
        summary_gdf = gpd.GeoDataFrame(summary_result, geometry="home", crs=source_home_crs)
    except Exception:
        summary_gdf = gpd.GeoDataFrame(summary_result)

    try:
        route_gdf = gpd.GeoDataFrame(route_result, geometry="morning_q25_route", crs="EPSG:4326")
    except Exception:
        route_gdf = gpd.GeoDataFrame(route_result)

    return summary_gdf, route_gdf

    