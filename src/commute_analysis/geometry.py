
import geopandas as gpd
from shapely.geometry import Point

from . import config

# create a geopandas.GeoDataFrame target_area that filters a circular area
def create_target_area(
    center_lat,
    center_lon,
    radius_km: float = config.ANALYSIS_RADIUS_KM,
):
    center = Point(center_lon, center_lat)

    target_area = gpd.GeoDataFrame(
        geometry=[center],
        crs=config.CRS_WGS84,
    )

    # Accurate metric projection for Hamburg / northern Germany
    target_area = target_area.to_crs(config.CRS_LOCAL_METRIC)

    target_area["geometry"] = target_area.geometry.buffer(
        radius_km * 1000
    )

    return target_area