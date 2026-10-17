from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_METADATA_PATH, MODEL_PATH, RANDOM_SEED, TARGET_COLUMN
from src.data_validation import assert_valid_dataset
from src.preprocessing import build_model_pipeline, get_feature_columns


def get_model_candidates() -> dict[str, object]:
    """Return the baseline and candidate classifiers for comparison."""
    return {
        "Baseline": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
    }


def load_training_frame(data_path: str | Path = "data/raw/synthetic_employee_data.csv") -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load the dataset and separate features from the target label."""
    df = pd.read_csv(data_path)
    assert_valid_dataset(df)

    numeric_columns, categorical_columns = get_feature_columns(df, target_column=TARGET_COLUMN)
    feature_columns = numeric_columns + categorical_columns

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    return X, y, numeric_columns, categorical_columns


def train_model_candidates(
    data_path: str | Path = "data/raw/synthetic_employee_data.csv",
) -> tuple[dict[str, object], dict[str, float], list[str], list[str]]:
    """Train all candidate models and select the best one using cross-validation."""
    X, y, numeric_columns, categorical_columns = load_training_frame(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    candidates = get_model_candidates()
    cv_scores: dict[str, float] = {}
    trained_models: dict[str, object] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, estimator in candidates.items():
        pipeline = build_model_pipeline(estimator, numeric_columns, categorical_columns)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_scores[name] = float(scores.mean())
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    best_name = max(cv_scores, key=cv_scores.get)
    best_pipeline = trained_models[best_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    return trained_models, cv_scores, numeric_columns, categorical_columns


def save_selected_model(best_pipeline, metadata: dict) -> None:
    """Persist the best pipeline and metadata."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(best_pipeline, MODEL_PATH)
    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_and_save_model(
    data_path: str | Path = "data/raw/synthetic_employee_data.csv",
) -> dict:
    """Train and save the final pipeline using macro-F1 model selection."""
    X, y, numeric_columns, categorical_columns = load_training_frame(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    candidates = get_model_candidates()
    cv_scores: dict[str, float] = {}
    trained_models: dict[str, object] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, estimator in candidates.items():
        pipeline = build_model_pipeline(estimator, numeric_columns, categorical_columns)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_scores[name] = float(scores.mean())
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    best_name = max(cv_scores, key=cv_scores.get)
    best_pipeline = trained_models[best_name]

    import joblib

    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "selected_model": best_name,
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "cv_macro_f1": cv_scores,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "target_classes": sorted(y.unique().tolist()),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(train_and_save_model())
