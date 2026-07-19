import geopandas as gpd
from shapely.geometry import box

from commute_analysis import synthetic_population


def test_sample_population_weighted_locations_supports_uniform_filter():
    grid = gpd.GeoDataFrame(
        {"population": [100, 1]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )

    result = synthetic_population.sample_population_weighted_locations(
        grid,
        "population",
        n=1000,
        filter_mode="uniform",
        seed=0,
    )

    counts = result["source_population"].value_counts().sort_index()
    assert len(result) == 1000
    assert result.crs == grid.crs
    assert list(result.columns)[:2] == ["employee_id", "source_population"]
    assert counts.loc[100] > 900


def test_sample_population_weighted_locations_supports_gaussian_filter():
    grid = gpd.GeoDataFrame(
        {"population": [1, 1]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    target_area = gpd.GeoDataFrame(
        {"name": ["target"]},
        geometry=[box(0, 0, 20, 20)],
        crs="EPSG:4326",
    )

    result = synthetic_population.sample_population_weighted_locations(
        grid,
        "population",
        n=50,
        target_area=target_area,
        filter_mode="gaussian",
        seed=0,
    )

    assert len(result) == 50
    assert result.crs == grid.crs


def test_sample_population_weighted_locations_rejects_unknown_filter_mode():
    grid = gpd.GeoDataFrame(
        {"population": [1]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    try:
        synthetic_population.sample_population_weighted_locations(
            grid,
            "population",
            n=3,
            filter_mode="unknown",
            seed=0,
        )
    except ValueError as exc:
        assert "filter_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown filter_mode")
