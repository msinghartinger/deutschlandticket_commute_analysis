from datetime import datetime
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from commute_analysis.routing import route_between_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "osm"
ORIGIN = (53.56407423036692, 9.992173763673405)
DESTINATION = (53.68624090313588, 10.046480574496965)
WORKPLACE = (53.686439, 10.046120)


def test_route_between_points_returns_a_valid_car_route_for_sample_coordinates():
    distance_km, duration_min, route_gdf = route_between_points(
        ORIGIN,
        DESTINATION,
        transport_modes=["car"],
        data_dir=DATA_DIR,
    )

    assert distance_km is not None
    assert duration_min is not None
    assert route_gdf is not None
    assert len(route_gdf) >= 1
    assert route_gdf.iloc[0].geometry is not None
    assert distance_km > 0
    assert duration_min > 0


def test_route_between_points_treats_tuple_inputs_as_lon_lat_even_if_crs_is_provided():
    distance_km, duration_min, route_gdf = route_between_points(
        ORIGIN,
        DESTINATION,
        transport_modes=["car"],
        data_dir=DATA_DIR,
        crs="EPSG:3035",
    )

    assert distance_km is not None
    assert duration_min is not None
    assert route_gdf is not None
    assert len(route_gdf) >= 1


def test_route_between_points_returns_routes_for_each_synthetic_employee_to_the_workplace():
    test_employees_path = PROJECT_ROOT / "data" / "processed" / "synthetic_employees_test.gpkg"
    synthetic_employees = gpd.read_file(test_employees_path, layer="synthetic_employees")

    assert not synthetic_employees.empty

    route_results = []
    for employee_geometry in synthetic_employees.geometry:
        if employee_geometry is None or employee_geometry.is_empty:
            continue

        distance_km, duration_min, route_gdf = route_between_points(
            employee_geometry,
            WORKPLACE,
            transport_modes=["car"],
            data_dir=DATA_DIR,
            crs=synthetic_employees.crs,
        )

        assert distance_km is not None
        assert duration_min is not None
        assert route_gdf is not None
        assert len(route_gdf) >= 1

        route_results.append((distance_km, duration_min))

    assert route_results
    assert min(distance for distance, _ in route_results) >= 0


def test_route_between_points_supports_public_transit_with_departure_time():
    departure_time = datetime(2026, 7, 20, 8, 0, 0)

    distance_km, duration_min, route_gdf = route_between_points(
        ORIGIN,
        DESTINATION,
        transport_modes=["walk", "public_transit"],
        data_dir=DATA_DIR,
        departure_time=departure_time,
    )

    assert distance_km is not None
    assert duration_min is not None
    assert route_gdf is not None
    assert len(route_gdf) >= 1
    assert route_gdf.iloc[0].geometry is not None
    assert route_gdf["mode"].notna().all()
    assert "leg_mode" in route_gdf.columns
    assert route_gdf["leg_mode"].notna().all()
    assert route_gdf["option"].nunique() == 1

    origin_point = Point(ORIGIN[1], ORIGIN[0])
    destination_point = Point(DESTINATION[1], DESTINATION[0])

    first_segment = route_gdf.iloc[0].geometry
    last_segment = route_gdf.iloc[-1].geometry

    assert first_segment is not None
    assert last_segment is not None

    first_segment_coords = list(first_segment.coords) if not hasattr(first_segment, "geoms") else list(first_segment.geoms[0].coords)
    last_segment_coords = list(last_segment.coords) if not hasattr(last_segment, "geoms") else list(last_segment.geoms[-1].coords)

    assert Point(first_segment_coords[0]).distance(origin_point) < 0.01
    assert Point(last_segment_coords[-1]).distance(destination_point) < 0.01


def test_route_between_points_returns_routes_for_each_synthetic_employee_to_the_workplace_using_walk_and_public_transit():
    test_employees_path = PROJECT_ROOT / "data" / "processed" / "synthetic_employees_test.gpkg"
    synthetic_employees = gpd.read_file(test_employees_path, layer="synthetic_employees")

    assert not synthetic_employees.empty

    route_results = []
    for employee_geometry in synthetic_employees.geometry:
        if employee_geometry is None or employee_geometry.is_empty:
            continue

        distance_km, duration_min, route_gdf = route_between_points(
            employee_geometry,
            WORKPLACE,
            transport_modes=["walk", "public_transit"],
            data_dir=DATA_DIR,
            crs=synthetic_employees.crs,
        )

        assert distance_km is not None
        assert duration_min is not None
        assert route_gdf is not None
        assert len(route_gdf) >= 1

        route_results.append((distance_km, duration_min))

    assert route_results
    assert min(distance for distance, _ in route_results) >= 0
