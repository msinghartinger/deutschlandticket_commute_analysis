from pathlib import Path

import geopandas as gpd

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
