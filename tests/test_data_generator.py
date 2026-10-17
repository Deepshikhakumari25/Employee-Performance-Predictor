import pandas as pd

from src.config import REQUIRED_COLUMNS
from src.data_generator import generate_synthetic_employee_data


def test_generate_synthetic_employee_data(tmp_path):
    output_file = tmp_path / "employee_data.csv"
    df = generate_synthetic_employee_data(rows=250, output_path=output_file, random_seed=12)
    repeat = generate_synthetic_employee_data(
        rows=250, output_path=tmp_path / "repeat.csv", random_seed=12
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 250
    assert list(df.columns) == REQUIRED_COLUMNS
    pd.testing.assert_frame_equal(df, repeat)
    assert set(df["perf_band_next"].unique()).issubset({"High", "Medium", "Low"})
    assert output_file.exists()
