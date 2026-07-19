import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commute_analysis import synthetic_population


class DummyResponse:
    def __init__(self, payload: bytes):
        self.content = payload

    def raise_for_status(self):
        return None


def test_download_population_grid_saves_csv_and_returns_dataframe(monkeypatch, tmp_path):
    expected = pd.DataFrame({"gitter_id": [1, 2], "population": [10, 20]})

    csv_bytes = expected.to_csv(index=False).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", csv_bytes)

    payload = buffer.getvalue()

    def fake_get(url, timeout=120):
        assert url.startswith("https://")
        return DummyResponse(payload)

    monkeypatch.setattr(synthetic_population.requests, "get", fake_get)

    output_file = Path(synthetic_population.__file__).resolve().parents[2] / "data" / "raw" / "population_grid.csv"
    if output_file.exists():
        output_file.unlink()

    df = synthetic_population.download_population_grid()

    assert isinstance(df, pd.DataFrame)
    pd.testing.assert_frame_equal(df, expected)
    assert output_file.exists()
    saved = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(saved, expected)

    output_file.unlink()
