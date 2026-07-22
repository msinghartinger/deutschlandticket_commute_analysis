"""Tests for scoring module."""
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import geopandas as gpd
import pytest
from datetime import time
from shapely.geometry import Point

from commute_analysis import routing, scoring


@pytest.fixture
def base_dir():
    """Get the project base directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def transport_network(base_dir):
    """Build transport network for testing."""
    data_dir = base_dir / "data" / "raw" / "osm"
    return routing.build_transport_network(data_dir=data_dir, allow_errors=True)


@pytest.fixture
def employee_data(base_dir):
    """Load synthetic employees test data."""
    gdf = gpd.read_file(base_dir / "data" / "processed" / "synthetic_employees_test.gpkg")
    # Convert from EPSG:3035 to EPSG:4326 for routing
    return gdf.to_crs("EPSG:4326")


def test_calculate_car_record_morning(employee_data):
    """Test calculate_car_record for morning commute."""
    employee = employee_data.iloc[0]
    home_point = employee.geometry
    
    result = scoring.calculate_car_record(
        employee_id=employee["employee_id"],
        direction="morning",
        home=(home_point.y, home_point.x),
    )
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert list(result.columns) == [
        "employee_id",
        "direction",
        "departure_time",
        "travel_time",
        "distance",
    ]
    
    # Check data types and values
    assert result["employee_id"].iloc[0] == employee["employee_id"]
    assert result["direction"].iloc[0] == "morning"
    assert result["departure_time"].iloc[0] == scoring.MORNING_START
    
    # Check sensible values
    assert result["travel_time"].iloc[0] > 0, "Travel time should be positive"
    assert result["distance"].iloc[0] > 0, "Distance should be positive"


def test_calculate_car_record_evening(employee_data):
    """Test calculate_car_record for evening commute."""
    employee = employee_data.iloc[0]
    home_point = employee.geometry
    
    result = scoring.calculate_car_record(
        employee_id=employee["employee_id"],
        direction="evening",
        home=(home_point.y, home_point.x),
    )
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result["direction"].iloc[0] == "evening"
    assert result["departure_time"].iloc[0] == scoring.EVENING_START
    
    # Check sensible values
    assert result["travel_time"].iloc[0] > 0, "Travel time should be positive"
    assert result["distance"].iloc[0] > 0, "Distance should be positive"


def test_calculate_car_record_invalid_direction(employee_data):
    """Test calculate_car_record with invalid direction."""
    employee = employee_data.iloc[0]
    home_point = employee.geometry
    
    with pytest.raises(ValueError, match="Direction must be 'morning' or 'evening'"):
        scoring.calculate_car_record(
            employee_id=employee["employee_id"],
            direction="invalid",
            home=(home_point.y, home_point.x),
        )


def test_calculate_public_transport_record_morning(employee_data):
    """Test calculate_public_transport_record for morning commute."""
    employee = employee_data.iloc[0]
    home_point = employee.geometry
    
    result = scoring.calculate_public_transport_record(
        employee_id=employee["employee_id"],
        direction="morning",
        home=(home_point.y, home_point.x),
    )
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0, "Should have at least one departure time"
    assert list(result.columns) == [
        "employee_id",
        "direction",
        "departure_time",
        "travel_time",
        "number_of_transfers",
        "total_walking_time",
    ]
    
    # Check all rows have the same employee_id and direction
    assert (result["employee_id"] == employee["employee_id"]).all()
    assert (result["direction"] == "morning").all()
    
    # Check departure times are within expected range
    assert (result["departure_time"] >= scoring.MORNING_START).all()
    assert (result["departure_time"] <= scoring.MORNING_END).all()
    
    # Check departure times are in intervals
    expected_intervals = _generate_departure_times(
        scoring.MORNING_START, scoring.MORNING_END, scoring.DEPARTURE_INTERVAL
    )
    assert set(result["departure_time"].tolist()) == set(expected_intervals)
    
    # Check sensible values
    assert (result["travel_time"] > 0).all(), "Travel time should be positive"
    assert (result["number_of_transfers"] >= 0).all(), "Transfers should be non-negative"
    assert (result["total_walking_time"] >= 0).all(), "Walking time should be non-negative"
    
    # Check that walking time is less than total travel time
    assert (result["total_walking_time"] <= result["travel_time"]).all(), \
        "Walking time should not exceed total travel time"


def test_helper_extract_number_of_transfers():
    """Test _extract_number_of_transfers helper function."""
    # Test with None
    assert scoring._extract_number_of_transfers(None) == 0
    
    # Test with empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame()
    assert scoring._extract_number_of_transfers(empty_gdf) == 0


def test_helper_extract_total_walking_time():
    """Test _extract_total_walking_time helper function."""
    # Test with None
    assert scoring._extract_total_walking_time(None) == 0.0
    
    # Test with empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame()
    assert scoring._extract_total_walking_time(empty_gdf) == 0.0


def _generate_departure_times(start_time, end_time, interval):
    """Generate departure times in intervals (helper for tests)."""
    from datetime import time as dt_time
    
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    interval_minutes = int(interval.total_seconds() // 60)
    
    times = []
    current_minutes = start_minutes
    while current_minutes <= end_minutes:
        hours = current_minutes // 60
        minutes = current_minutes % 60
        times.append(dt_time(hours, minutes))
        current_minutes += interval_minutes
    
    return times


def test_calculate_transport_scores_for_geodataframe_saves_incrementally(tmp_path, monkeypatch):
    """Test that each employee is written to disk as soon as it finishes processing."""
    employee_gdf = gpd.GeoDataFrame(
        {
            "employee_id": [1, 2],
            "home": [Point(10.0, 53.0), Point(10.1, 53.1)],
        },
        geometry="home",
        crs="EPSG:4326",
    )

    def fake_row(employee_id, home, home_crs):
        return {
            "total_travel_time": float(employee_id) * 10.0,
            "transport_score": float(employee_id) * 0.1,
            "relative_time_score": 0.1,
            "absolute_time_score": 0.2,
            "consistency_score": 0.3,
            "walking_score": 0.4,
            "transfers_score": 0.5,
            "morning_q25_route": Point(home.x, home.y),
        }

    saved_frames = []

    def fake_to_file(self, output_path, driver=None, mode=None, layer=None, **kwargs):
        if mode == "w":
            Path(output_path).touch()
        geometry_column_count = len(self.select_dtypes(include=["geometry"]).columns)
        saved_frames.append(
            {
                "path": Path(output_path).name,
                "mode": mode,
                "layer": layer,
                "rows": len(self),
                "geometry": self.geometry.name,
                "geometry_column_count": geometry_column_count,
            }
        )

    monkeypatch.setattr(scoring, "_calculate_employee_transport_score_row", fake_row)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file, raising=False)

    summary_gdf, route_gdf = scoring.calculate_transport_scores_for_geodataframe(
        employee_gdf=employee_gdf,
        employee_id_column="employee_id",
        home_column="home",
        max_workers=1,
        save_df=True,
        save_dir=tmp_path,
    )

    assert len(summary_gdf) == 2
    assert len(route_gdf) == 2
    assert [item["path"] for item in saved_frames] == [
        "summary_gdf.gpkg",
        "route_gdf.gpkg",
        "summary_gdf.gpkg",
        "route_gdf.gpkg",
    ]
    assert [item["mode"] for item in saved_frames] == ["w", "w", "a", "a"]
    assert [item["rows"] for item in saved_frames] == [1, 1, 1, 1]
    assert [item["geometry_column_count"] for item in saved_frames] == [1, 1, 1, 1]


def test_calculate_transport_scores_for_geodataframe_skips_existing_saved_rows(tmp_path, monkeypatch):
    """Test that preexisting saved rows are skipped when overwrite is False."""
    employee_gdf = gpd.GeoDataFrame(
        {
            "employee_id": [1, 2, 3],
            "home": [Point(10.0, 53.0), Point(10.1, 53.1), Point(10.2, 53.2)],
        },
        geometry="home",
        crs="EPSG:4326",
    )

    summary_path = tmp_path / "summary_gdf.gpkg"
    route_path = tmp_path / "route_gdf.gpkg"
    summary_path.touch()
    route_path.touch()

    saved_outputs = {
        "summary_gdf.gpkg": gpd.GeoDataFrame(
            {
                "employee_id": [1],
                "home": [Point(10.0, 53.0)],
                "total_travel_time": [11.0],
                "transport_score": [0.11],
                "relative_time_score": [0.1],
                "absolute_time_score": [0.2],
                "consistency_score": [0.3],
                "walking_score": [0.4],
                "transfers_score": [0.5],
            },
            geometry="home",
            crs="EPSG:4326",
        ),
        "route_gdf.gpkg": gpd.GeoDataFrame(
            {
                "employee_id": [1],
                "home": [Point(10.0, 53.0).wkt],
                "morning_q25_route": [Point(10.0, 53.0)],
            },
            geometry="morning_q25_route",
            crs="EPSG:4326",
        ),
    }
    processed_ids = []

    def fake_read_file(path, layer=None, **kwargs):
        return saved_outputs[Path(path).name].copy()

    def fake_row(employee_id, home, home_crs):
        processed_ids.append(employee_id)
        return {
            "total_travel_time": float(employee_id) * 10.0,
            "transport_score": float(employee_id) * 0.1,
            "relative_time_score": 0.1,
            "absolute_time_score": 0.2,
            "consistency_score": 0.3,
            "walking_score": 0.4,
            "transfers_score": 0.5,
            "morning_q25_route": Point(home.x, home.y),
        }

    def fake_to_file(self, output_path, driver=None, mode=None, layer=None, **kwargs):
        name = Path(output_path).name
        frame = self.copy()
        geometry_columns = list(frame.select_dtypes(include=["geometry"]).columns)
        for column in geometry_columns:
            if column == frame.geometry.name:
                continue
            frame[column] = frame[column].to_wkt()
        if mode == "w" or name not in saved_outputs:
            saved_outputs[name] = frame
        else:
            saved_outputs[name] = pd.concat([saved_outputs[name], frame], ignore_index=True)
        Path(output_path).touch()

    monkeypatch.setattr(gpd, "read_file", fake_read_file)
    monkeypatch.setattr(scoring, "_calculate_employee_transport_score_row", fake_row)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fake_to_file, raising=False)

    summary_gdf, route_gdf = scoring.calculate_transport_scores_for_geodataframe(
        employee_gdf=employee_gdf,
        employee_id_column="employee_id",
        home_column="home",
        max_workers=1,
        save_df=True,
        save_dir=tmp_path,
        overwrite=False,
    )

    assert processed_ids == [2, 3]
    assert sorted(summary_gdf["employee_id"].tolist()) == [1, 2, 3]
    assert sorted(route_gdf["employee_id"].tolist()) == [1, 2, 3]


def test_load_transport_scores_from_gpkg(tmp_path, monkeypatch):
    """Test loading saved score outputs from summary and route GeoPackages."""
    summary_path = tmp_path / "summary_gdf.gpkg"
    route_path = tmp_path / "route_gdf.gpkg"
    summary_path.touch()
    route_path.touch()

    summary_source = gpd.GeoDataFrame(
        {
            "employee_id": [1],
            "home": [Point(10.0, 53.0)],
            "transport_score": [0.85],
        },
        geometry="home",
        crs="EPSG:4326",
    )
    route_source = gpd.GeoDataFrame(
        {
            "employee_id": [1],
            "home": [Point(10.0, 53.0).wkt],
            "morning_q25_route": [Point(10.1, 53.1)],
        },
        geometry="morning_q25_route",
        crs="EPSG:4326",
    )

    def fake_read_file(path, layer=None, **kwargs):
        if Path(path).name == "summary_gdf.gpkg":
            return summary_source.copy()
        if Path(path).name == "route_gdf.gpkg":
            return route_source.copy()
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(gpd, "read_file", fake_read_file)

    summary_gdf, route_gdf = scoring.load_transport_scores_from_gpkg(save_dir=tmp_path)

    assert len(summary_gdf) == 1
    assert len(route_gdf) == 1
    assert summary_gdf.geometry.name == "home"
    assert route_gdf.geometry.name == "morning_q25_route"
    assert isinstance(route_gdf.loc[0, "home"], Point)
