# from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

import requests

def download_population_grid() -> pd.DataFrame:
    """
    Download the population grid from the official source, save a CSV copy into
    the project's data/raw folder, and return a DataFrame.

    The function downloads the published ZIP, extracts the CSV, reads it into
    a DataFrame, writes a copy to data/raw/population_grid.csv inside the
    deutschlandticket-commute-analysis project root (if that ancestor exists),
    and returns the DataFrame.
    """
    url = (
        "https://www.destatis.de/static/DE/zensus/gitterdaten/"
        "Zensus2022_Bevoelkerungszahl.zip"
    )

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        # list contents for debugging purposes
        # print(archive.namelist())

        filename = "Zensus2022_Bevoelkerungszahl_100m-Gitter.csv"

        with archive.open(filename) as csv_file:
            population = pd.read_csv(csv_file, sep=";")

    # Determine where to save the CSV: prefer an ancestor named
    # 'deutschlandticket-commute-analysis' if present, else use the repository
    # root two levels above this file.
    repo_dir_name = "deutschlandticket-commute-analysis"
    current = Path(__file__).resolve()
    candidate_root = current.parents[2]

    repo_root = None
    if candidate_root.name == repo_dir_name:
        repo_root = candidate_root
    else:
        for p in candidate_root.parents:
            if p.name == repo_dir_name:
                repo_root = p
                break

    if repo_root is None:
        repo_root = candidate_root

    output_path = repo_root / "data" / "raw" / "population_grid.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV (use semicolon as in source if desired, here save as standard CSV)
    population.to_csv(output_path, index=False)

    return population




def random_point_in_polygon(
    polygon: BaseGeometry,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> Point:
    """
    Generate a random point within the bounds of a given polygon using rejection sampling.

    This function attempts to find a point that falls inside the specified polygon
    by generating random coordinates within the polygon's bounding box and checking
    if the point is covered by the polygon.

    Args:
        polygon (BaseGeometry): The geometry polygon to sample from.
        rng (np.random.Generator): A NumPy random number generator for reproducibility.
        max_attempts (int): The maximum number of random points to try before giving up.
            Defaults to 10,000.

    Returns:
        Point: A Point object located inside the polygon.

    Raises:
        RuntimeError: If a valid point is not found within the maximum number of attempts.
    """
    # Extract the bounding box coordinates from the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Iterate to attempt generating valid points within the bounding box
    for _ in range(max_attempts):
        # Generate a candidate point with random coordinates within bounds
        point = Point(
            rng.uniform(min_x, max_x),
            rng.uniform(min_y, max_y),
        )

        # Check if the generated point lies within the polygon
        if polygon.covers(point):
            return point

    # If no valid point is found after all attempts, raise an error
    raise RuntimeError(
        "Random-point generation failed for one polygon."
    )

def calculate_sampling_probability_grid(
    population_grid: gpd.GeoDataFrame,
    population_column: str,
    target_area: gpd.GeoDataFrame | None = None,
    filter_mode: str = "uniform",
    gaussian_scale: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Prepare a population grid and calculate the sampling probability
    assigned to each grid cell.

    Parameters
    ----------
    population_grid : geopandas.GeoDataFrame
        Polygon grid containing population counts.
    population_column : str
        Column containing population counts.
    target_area : geopandas.GeoDataFrame, optional
        Area to which the grid should be clipped.
    filter_mode : {"uniform", "gaussian"}, default "uniform"
        "uniform" uses population alone.
        "gaussian" multiplies population by a distance-decay factor.
    gaussian_scale : float, optional
        Gaussian scale in the units of the grid CRS, normally metres.
        If omitted, the 95th percentile of cell-centroid distances is used.

    Returns
    -------
    geopandas.GeoDataFrame
        Processed grid with the following additional columns:

        - base_population
        - distance
        - distance_weight
        - sampling_weight
        - sampling_probability
    """
    if population_grid.crs is None:
        raise ValueError("population_grid must have a CRS.")

    if population_column not in population_grid.columns:
        raise KeyError(
            f"Column {population_column!r} is missing."
        )

    if filter_mode not in {"uniform", "gaussian"}:
        raise ValueError(
            "filter_mode must be one of {'uniform', 'gaussian'}."
        )

    grid = population_grid.copy()

    grid[population_column] = pd.to_numeric(
        grid[population_column],
        errors="coerce",
    ).fillna(0)

    if target_area is not None:
        if target_area.crs is None:
            raise ValueError("target_area must have a CRS.")

        target_area = target_area.to_crs(grid.crs)
        grid = gpd.clip(grid, target_area)

    grid = grid[
        (grid[population_column] > 0)
        & grid.geometry.notna()
        & ~grid.geometry.is_empty
    ].copy()

    if grid.empty:
        raise ValueError(
            "No populated grid cells remain after filtering."
        )

    grid["base_population"] = grid[
        population_column
    ].to_numpy(dtype=float)

    # Defaults for population-only weighting
    grid["distance"] = np.nan
    grid["distance_weight"] = 1.0

    if filter_mode == "gaussian":
        if target_area is not None:
            center = target_area.geometry.union_all().centroid
        else:
            center = grid.geometry.union_all().centroid

        cell_centroids = grid.geometry.centroid

        distances = cell_centroids.distance(center).to_numpy(
            dtype=float
        )

        if gaussian_scale is None:
            scale = max(
                float(np.quantile(distances, 0.95)),
                1.0,
            )
        else:
            scale = float(gaussian_scale)

            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    "gaussian_scale must be a positive finite number."
                )

        distance_weights = np.exp(
            -0.5 * (distances / scale) ** 2
        )

        grid["distance"] = distances
        grid["distance_weight"] = distance_weights

        # Store the actual scale used for inspection
        grid.attrs["gaussian_scale"] = scale

    grid["sampling_weight"] = (
        grid["base_population"]
        * grid["distance_weight"]
    )

    valid_weights = np.where(
        np.isfinite(grid["sampling_weight"])
        & (grid["sampling_weight"] > 0),
        grid["sampling_weight"],
        0.0,
    )

    total_weight = valid_weights.sum()

    if total_weight <= 0:
        raise ValueError(
            "No positive sampling weights remain after filtering."
        )

    grid["sampling_weight"] = valid_weights
    grid["sampling_probability"] = (
        valid_weights / total_weight
    )

    return grid

def sample_population_weighted_locations(
    population_grid: gpd.GeoDataFrame,
    population_column: str,
    n: int,
    target_area: gpd.GeoDataFrame | None = None,
    seed: int | None = None,
    filter_mode: str = "uniform",
    gaussian_scale: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Sample synthetic locations from a population-weighted polygon grid.
    """
    if n <= 0:
        raise ValueError("n must be greater than zero.")

    grid = calculate_sampling_probability_grid(
        population_grid=population_grid,
        population_column=population_column,
        target_area=target_area,
        filter_mode=filter_mode,
        gaussian_scale=gaussian_scale,
    )

    rng = np.random.default_rng(seed)

    selected_positions = rng.choice(
        len(grid),
        size=n,
        replace=True,
        p=grid["sampling_probability"].to_numpy(),
    )

    selected_cells = grid.iloc[
        selected_positions
    ].reset_index(drop=True)

    points = [
        random_point_in_polygon(geometry, rng)
        for geometry in selected_cells.geometry
    ]

    result = gpd.GeoDataFrame(
        {
            "employee_id": [
                f"EMP-{i:05d}"
                for i in range(1, n + 1)
            ],
            "source_population":
                selected_cells["base_population"].to_numpy(),
            "source_probability":
                selected_cells["sampling_probability"].to_numpy(),
            "distance_weight":
                selected_cells["distance_weight"].to_numpy(),
        },
        geometry=points,
        crs=grid.crs,
    )

    return result

