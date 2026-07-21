
import geopandas as gpd
from shapely.geometry import Point

# create a geopandas.GeoDataFrame target_area that filters a circular area of 30 km radius around the center point (53.686439, 10.046120)
def create_target_area(center_lat, center_lon, radius_km):
    center = Point(center_lon, center_lat)

    target_area = gpd.GeoDataFrame(
        geometry=[center],
        crs="EPSG:4326",
    )

    # Accurate metric projection for Hamburg / northern Germany
    target_area = target_area.to_crs("EPSG:25832")

    target_area["geometry"] = target_area.geometry.buffer(
        radius_km * 1000
    )

    return target_area