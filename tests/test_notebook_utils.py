import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from commute_analysis import notebook_utils


@pytest.mark.parametrize("separator", [",", ";"])
def test_prepare_probability_grid_reads_csv_delimiters(
    tmp_path, monkeypatch, separator
):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "x_mp_100m": [4337050],
            "y_mp_100m": [2689150],
            "Einwohner": [4],
        }
    ).to_csv(data_dir / "population_grid.csv", index=False, sep=separator)
    monkeypatch.setattr(notebook_utils, "BASE_DIR", tmp_path)

    def check_grid(*, population_grid, **kwargs):
        assert population_grid.crs.to_epsg() == 3035
        assert list(population_grid.geometry.iloc[0].coords) == [(4337050.0, 2689150.0)]
        assert population_grid["Einwohner"].tolist() == [4]
        return population_grid

    monkeypatch.setattr(notebook_utils, "calculate_sampling_probability_grid", check_grid)
    area = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")

    result = notebook_utils.prepare_probability_grid(area)

    assert len(result) == 1