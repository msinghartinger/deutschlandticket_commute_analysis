import importlib
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon

from commute_analysis.synthetic_population import (
    calculate_sampling_probability_grid,
    sample_population_weighted_locations,
)
import commute_analysis.visualization as viz
import commute_analysis.routing as routing
import commute_analysis.geometry as geometry
import commute_analysis.scoring as scoring

# define constants
BASE_DIR = Path(__file__).resolve().parents[2]

RADIUS_KM = 50
WORKPLACE = (53.686439, 10.046120)

GAUSSIAN_SCALE = 10_000

NUM_EMPLOYEES = 100

# ------------------------------------------------------------
# Load and prepare sampling probability grid
# ------------------------------------------------------------

def prepare_probability_grid(target_area):
    population_df = pd.read_csv(Path(BASE_DIR) / "data" / "raw" / "population_grid.csv")

    # transform this dataframe into a geopandas dataframe with a geometry column that contains the centroid of the grid cell
    population_gdf = gpd.GeoDataFrame(
        population_df,
        geometry=gpd.points_from_xy(population_df["x_mp_100m"], population_df["y_mp_100m"]),
        crs="EPSG:3035",
    )
    
    probability_grid = calculate_sampling_probability_grid(
        population_grid=population_gdf,
        population_column="Einwohner",
        target_area=target_area,
        filter_mode="gaussian",
        gaussian_scale=GAUSSIAN_SCALE,
    ).copy()

    return probability_grid

def to_gpd_point(input):
    return gpd.GeoDataFrame({"name": ["Workplace"]}, geometry=[Point(input[1], input[0])], crs="EPSG:4326")






