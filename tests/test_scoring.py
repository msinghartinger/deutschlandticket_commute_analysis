"""Tests for scoring module."""
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import geopandas as gpd
import pytest
from datetime import time

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
