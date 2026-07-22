
# Deutschlandticket Commute Analysis

This project evaluates how attractive public transport is for a set of synthetic employees commuting to a workplace in northern Germany. It combines population-based home sampling, public-transport routing, commute scoring, and map-based visualization to estimate commute times and a derived transport adoption score.

## Main Deliverables

- [notebooks/deutschlandticket_analysis.ipynb](notebooks/deutschlandticket_analysis.ipynb): the full analysis notebook.
- [notebooks/deutschlandticket_analysis.html](notebooks/deutschlandticket_analysis.html): HTML export of the notebook.
- [notebooks/deutschlandticket_analysis.pdf](notebooks/deutschlandticket_analysis.pdf): PDF export of the notebook.
- [notebooks/download_data.ipynb](notebooks/download_data.ipynb): helper notebook to download and prepare raw input data.

## Project Components

- `src/commute_analysis/notebook_utils.py`: helper utilities used by the notebook workflow.
- `src/commute_analysis/geometry.py`: target-area and geometry helpers.
- `src/commute_analysis/routing.py`: transport-network construction and route calculations.
- `src/commute_analysis/scoring.py`: commute-time and transport-score calculations.
- `src/commute_analysis/visualization.py`: plotting functions for maps, routes, scores, and histograms.

## Dependencies

The project targets Python 3.10+ and uses the core packages listed in `pyproject.toml`:

- `geopandas`
- `numpy`
- `pandas`
- `requests`
- `shapely`

For the full analysis workflow, you will also need:

- `matplotlib` for static plots
- `folium` for interactive maps
- `contextily` for basemaps in Matplotlib plots
- `r5py` for public-transport routing
- Java runtime support for `r5py` (for example `openjdk-21-jdk`)

For development and testing:

- `pytest`

## Data Sources

- Population grid data: https://www.destatis.de/static/DE/zensus/gitterdaten/
- OpenStreetMap data: https://download.geofabrik.de/europe/germany.html
- HVV route and schedule data: https://suche.transparenz.hamburg.de/dataset/hvv-fahrplandaten-gtfs-april-2026-bis-dezember-2026
- Reference on public transport attractiveness: https://link.springer.com/article/10.1186/s12544-023-00609-x

## Quick Start

1. Install the Python dependencies, Java runtime (for `r5py`), and `osmium` CLI (for OSM merge).
2. Open and run [notebooks/download_data.ipynb](notebooks/download_data.ipynb) from top to bottom.
3. Confirm these files exist after the download notebook finishes:
	- `data/raw/population_grid.csv`
	- `data/raw/hvv_Rohdaten_GTFS_Fpl_26.ZIP`
	- `data/raw/osm/hamburg-260718.osm.pbf`
	- `data/raw/osm/niedersachsen-260718.osm.pbf`
	- `data/raw/osm/schleswig-holstein-260718.osm.pbf`
	- `data/raw/osm/merged/northern-germany.osm.pbf`
4. Open and run [notebooks/deutschlandticket_analysis.ipynb](notebooks/deutschlandticket_analysis.ipynb).

## Data Handling

- Raw data under `data/raw/` is intentionally not tracked in Git.
- Recreate raw inputs at any time by running [notebooks/download_data.ipynb](notebooks/download_data.ipynb).

## Repository Layout

- `data/`: raw and processed inputs used by the analysis.
- `notebooks/`: the analysis notebook and exported deliverables.
- `outputs/`: generated maps, figures, and summary files.
- `src/commute_analysis/`: reusable analysis code.
- `tests/`: automated checks for core functionality.