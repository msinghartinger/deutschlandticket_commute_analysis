import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commute_analysis import synthetic_population


def test_load_population_grid_csv_and_returns_dataframe():
   
    BASE_DIR = "/home/msing/proj/jjdatascience/deutschlandticket-commute-analysis"
    df = pd.read_csv(Path(BASE_DIR) / "data" / "raw" / "population_grid.csv")

    # transform this dataframe into a geopandas dataframe with a geometry column that contains the centroid of the grid cell
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_mp_100m"], df["y_mp_100m"]),
        crs="EPSG:3035",
    )

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "geometry" in gdf.columns
    assert 'x_mp_100m' in gdf.columns
    assert 'y_mp_100m' in gdf.columns
    assert len(gdf) > 100
