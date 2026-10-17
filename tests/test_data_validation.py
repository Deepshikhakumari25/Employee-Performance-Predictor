import pandas as pd
import pytest

from src.data_generator import generate_synthetic_employee_data
from src.data_validation import assert_valid_dataset, load_and_validate_csv, validate_dataset


def make_valid_frame(tmp_path):
    return generate_synthetic_employee_data(rows=30, output_path=tmp_path / "valid_fixture.csv")


def test_valid_dataset_passes(tmp_path):
    assert validate_dataset(make_valid_frame(tmp_path))["is_valid"]


def test_missing_required_column_is_reported(tmp_path):
    df = make_valid_frame(tmp_path).drop(columns=["department"])
    result = validate_dataset(df)
    assert not result["is_valid"]
    assert any("Missing required columns" in issue for issue in result["issues"])


def test_invalid_target_is_reported_and_raises(tmp_path):
    df = make_valid_frame(tmp_path)
    df.loc[0, "perf_band_next"] = "Outstanding"
    result = validate_dataset(df)
    assert any("Invalid target labels" in issue for issue in result["issues"])
    with pytest.raises(ValueError, match="validation failed"):
        assert_valid_dataset(df)


def test_invalid_numeric_values_and_ranges_are_reported(tmp_path):
    df = make_valid_frame(tmp_path)
    df["on_time_delivery_rate"] = df["on_time_delivery_rate"].astype(object)
    df.loc[0, "on_time_delivery_rate"] = "not-a-number"
    df.loc[1, "manager_score"] = 101
    result = validate_dataset(df)
    assert any("numeric data type" in issue for issue in result["issues"])
    assert any("Out-of-range values" in issue for issue in result["issues"])


def test_empty_dataset_is_invalid():
    assert "Dataset is empty." in validate_dataset(pd.DataFrame())["issues"]


def test_malformed_csv_is_reported(tmp_path):
    malformed_path = tmp_path / "malformed.csv"
    malformed_path.write_text('"unterminated,header\nvalue', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed CSV"):
        load_and_validate_csv(malformed_path)


def test_missing_csv_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        load_and_validate_csv(tmp_path / "missing.csv")
