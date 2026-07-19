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


def sample_population_weighted_locations(
    population_grid: gpd.GeoDataFrame,
    population_column: str,
    n: int,
    target_area: gpd.GeoDataFrame | None = None,
    seed: int | None = None,
    filter_mode: str = "uniform",
) -> gpd.GeoDataFrame:
    """
    Sample synthetic locations from a polygon grid based on population weights.

    This function selects grid cells with probability proportional to their
    population values and generates a synthetic point within each selected
    cell. The resulting GeoDataFrame contains the generated points along
    with the employee ID and the population count of the source cell.

    Parameters
    ----------
    population_grid : geopandas.GeoDataFrame
        Input grid where each row represents a polygon cell. Must have a valid CRS.
    population_column : str
        Column name in `population_grid` containing the population counts to use for weighting.
    n : int
        Number of synthetic locations to generate. Must be greater than zero.
    target_area : geopandas.GeoDataFrame, optional
        Optional GeoDataFrame to clip the input grid to before sampling. Must have a valid CRS.
    seed : int, optional
        Random seed for reproducibility.
    filter_mode : str, default "uniform"
        Weighting strategy to apply before sampling. Supported values are
        "uniform" (use the raw population values), and "gaussian" (multiply the population values
        by a Gaussian decay centered on the target area centroid).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with 'employee_id', 'source_population', and 'geometry'
        columns. The geometry column contains the generated points in the same CRS as the input grid.

    Raises
    ------
    ValueError
        If `population_grid` lacks a CRS, `n` is non-positive, `target_area` (if provided) lacks a CRS,
        or no valid population cells remain after filtering.
    KeyError
        If `population_column` does not exist in `population_grid`.
    """
    # Validate that the input grid has a defined coordinate reference system
    if population_grid.crs is None:
        raise ValueError("population_grid must have a CRS.")

    # Verify that the specified population column exists in the grid
    if population_column not in population_grid.columns:
        raise KeyError(
            f"Column {population_column!r} is missing."
        )

    # Ensure the requested sample size is valid
    if n <= 0:
        raise ValueError("n must be greater than zero.")

    grid = population_grid.copy()

    # Convert population values to numeric, replacing errors with 0
    grid[population_column] = pd.to_numeric(
        grid[population_column],
        errors="coerce",
    ).fillna(0)

    # If a target area is provided, clip the grid to that area
    if target_area is not None:
        if target_area.crs is None:
            raise ValueError("target_area must have a CRS.")

        # Reproject target area to match the grid's CRS before clipping
        target_area = target_area.to_crs(grid.crs)
        grid = gpd.clip(grid, target_area)

    # Filter grid to keep only valid cells with positive population
    grid = grid[
        (grid[population_column] > 0)
        & grid.geometry.notna()
        & ~grid.geometry.is_empty
    ].copy()

    # Raise an error if no valid population cells remain after filtering
    if grid.empty:
        raise ValueError(
            "No populated grid cells remain after filtering."
        )

    if filter_mode not in {"uniform", "gaussian"}:
        raise ValueError(
            "filter_mode must be one of {'uniform', 'gaussian'}."
        )

    # Initialize a random number generator with the provided seed
    rng = np.random.default_rng(seed)

    # Calculate sampling weights normalized by the selected weighting strategy.
    base_population = grid[population_column].to_numpy(dtype=float)
    if filter_mode == "uniform":
        weights = base_population
    elif filter_mode == "gaussian":
        if target_area is not None:
            center = unary_union(target_area.geometry.tolist())
        else:
            center = unary_union(grid.geometry.tolist())

        center = center.centroid
        cell_centroids = grid.geometry.centroid
        distances = np.array(
            [center.distance(centroid) for centroid in cell_centroids],
            dtype=float,
        )
        scale = max(float(np.quantile(distances, 0.95)), 1.0)
        gaussian = np.exp(-0.5 * (distances / scale) ** 2)
        weights = base_population * gaussian
    else:
        weights = base_population

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        raise ValueError("No positive sampling weights remain after filtering.")

    weights /= weights.sum()

    # Select n grid indices based on the calculated population weights
    selected_positions = rng.choice(
        len(grid),
        size=n,
        replace=True,
        p=weights,
    )

    # Retrieve the geometries corresponding to the selected positions
    selected_cells = grid.iloc[selected_positions].reset_index(
        drop=True
    )

    # Generate a random point within each selected cell's polygon
    points = [
        random_point_in_polygon(geometry, rng)
        for geometry in selected_cells.geometry
    ]

    # Construct and return the result GeoDataFrame
    result = gpd.GeoDataFrame(
        {
            "employee_id": [
                f"EMP-{i:05d}"
                for i in range(1, n + 1)
            ],
            "source_population":
                selected_cells[population_column].to_numpy(),
        },
        geometry=points,
        crs=grid.crs,
    )

    return result