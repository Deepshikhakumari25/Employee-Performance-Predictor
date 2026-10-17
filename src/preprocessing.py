from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, DEFAULT_MODEL_FEATURES, NUMERIC_COLUMNS, TARGET_COLUMN


def get_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[list[str], list[str]]:
    """Return the default model features while excluding target, employee ID, and protected/salary fields."""
    candidate_columns = [
        column
        for column in DEFAULT_MODEL_FEATURES
        if column in df.columns and column != target_column
    ]
    numeric_columns = [column for column in NUMERIC_COLUMNS if column in candidate_columns]
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in candidate_columns and column not in {"gender", "salary_percentile_band"}]
    remaining = [column for column in candidate_columns if column not in numeric_columns and column not in categorical_columns]
    if remaining:
        categorical_columns.extend(remaining)
    return numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Create a transformer for numeric and categorical features."""
    num_features = list(numeric_columns) if numeric_columns is not None else list(NUMERIC_COLUMNS)
    cat_features = list(categorical_columns) if categorical_columns is not None else list(CATEGORICAL_COLUMNS)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(model, numeric_columns: Sequence[str], categorical_columns: Sequence[str]):
    """Wrap preprocessing and the estimator into a single scikit-learn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            ("model", model),
        ]
    )
