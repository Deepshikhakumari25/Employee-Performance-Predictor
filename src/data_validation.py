from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def validate_dataset(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> dict:
    """Return a summary of data issues without discarding any rows."""
    issues: list[str] = []
    target_required = list(required_columns) if required_columns is not None else REQUIRED_COLUMNS

    if df is None or df.empty:
        issues.append("Dataset is empty.")
        return {"is_valid": False, "issues": issues}

    missing_columns = [column for column in target_required if column not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}.")

    if target_column not in df.columns:
        issues.append(f"Target column '{target_column}' is missing.")

    unexpected_columns = [column for column in df.columns if column not in target_required]
    if unexpected_columns:
        issues.append(f"Unexpected columns found: {unexpected_columns}.")

    if df.duplicated().any():
        issues.append(f"Duplicate records found: {int(df.duplicated().sum())} rows.")

    if "employee_id" in df.columns and df["employee_id"].duplicated().any():
        issues.append("Duplicate employee_id values were found.")
    if "employee_id" in df.columns and not pd.api.types.is_numeric_dtype(df["employee_id"]):
        issues.append("Identifier column 'employee_id' must have a numeric data type.")

    missing_value_columns = df.columns[df.isna().any()].tolist()
    if missing_value_columns:
        issues.append(f"Missing values detected in: {missing_value_columns}.")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                issues.append(f"Numeric column '{column}' must have a numeric data type.")
            else:
                series = pd.to_numeric(df[column], errors="coerce")
                if series.isna().any():
                    issues.append(f"Numeric column '{column}' contains invalid non-numeric values.")

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            if df[column].isnull().any():
                issues.append(f"Categorical column '{column}' contains null values.")

    valid_categories = {
        "department": {"Engineering", "HR", "Sales", "Finance", "Marketing", "Operations", "Product"},
        "job_level": {"Associate", "Senior", "Lead", "Manager"},
        "salary_percentile_band": {"Low", "Mid", "High", "Top"},
        "gender": {"Woman", "Man", "Non-binary"},
    }
    for column, allowed_values in valid_categories.items():
        if column in df.columns:
            invalid_values = sorted(
                set(df[column].dropna().astype(str).unique()) - allowed_values
            )
            if invalid_values:
                issues.append(f"Invalid values in '{column}': {invalid_values}.")

    if target_column in df.columns:
        invalid_labels = sorted(df[target_column].dropna().unique().tolist())
        allowed = ["High", "Medium", "Low"]
        invalid = [label for label in invalid_labels if label not in allowed]
        if invalid:
            issues.append(f"Invalid target labels found: {invalid}.")

    numeric_ranges = {
        "experience_years": (0, 60),
        "manager_tenure": (0, 50),
        "projects_count": (0, 100),
        "avg_task_delay_days": (0, 180),
        "on_time_delivery_rate": (0, 100),
        "bug_count": (0, 1000),
        "code_review_score": (0, 100),
        "qa_defect_density": (0, 20),
        "story_points_completed": (0, 500),
        "billable_hours_ratio": (0, 1.5),
        "training_hours": (0, 500),
        "certifications_count": (0, 50),
        "internal_hackathons_participated": (0, 50),
        "sick_days": (0, 180),
        "unplanned_absences": (0, 180),
        "avg_login_hours": (0, 24),
        "peer_feedback_score": (0, 100),
        "manager_score": (0, 100),
        "kudos_count": (0, 500),
        "promotions_in_2y": (0, 20),
    }

    for column, (lower, upper) in numeric_ranges.items():
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            out_of_range = series[(series < lower) | (series > upper)]
            if not out_of_range.empty:
                issues.append(
                    f"Out-of-range values found in '{column}' with {len(out_of_range)} record(s)."
                )

    if not np_values_allowed(df):
        issues.append("Infinite or non-finite numeric values were found.")

    return {"is_valid": len(issues) == 0, "issues": issues}


def np_values_allowed(df: pd.DataFrame) -> bool:
    """Check the DataFrame for NaN/Inf values that are invalid in a numeric dataset."""
    numeric_df = df.select_dtypes(include="number")
    return not np.isinf(numeric_df.to_numpy()).any() and not np.isnan(numeric_df.to_numpy()).any()


def assert_valid_dataset(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Raise a clear error if the dataset does not pass validation."""
    result = validate_dataset(df, required_columns=required_columns, target_column=target_column)
    if not result["is_valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(result["issues"]))


def load_and_validate_csv(path: str | Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV and validate it before returning it."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed CSV file: {file_path}") from exc

    assert_valid_dataset(df, required_columns=required_columns)
    return df
