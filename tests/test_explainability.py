import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_generator import generate_synthetic_employee_data
from src.explainability import (
    calculate_permutation_importance,
    run_explainability,
    save_explanation_report,
    save_importance_chart,
)
from src.preprocessing import build_model_pipeline, get_feature_columns


def fit_small_model(tmp_path):
    df = generate_synthetic_employee_data(rows=180, output_path=tmp_path / "synthetic.csv")
    numeric, categorical = get_feature_columns(df)
    X = df[numeric + categorical]
    y = df["perf_band_next"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model_pipeline(
        LogisticRegression(max_iter=1000),
        numeric,
        categorical,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_permutation_importance_and_outputs(tmp_path):
    model, X_test, y_test = fit_small_model(tmp_path)
    importance = calculate_permutation_importance(model, X_test, y_test, n_repeats=2)
    assert list(importance.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(importance["feature"]) == set(X_test.columns)
    csv_path, text_path = save_explanation_report(importance, tmp_path / "reports")
    chart_path = save_importance_chart(importance, tmp_path / "figures")
    assert pd.read_csv(csv_path).shape[0] == X_test.shape[1]
    assert "does not establish causation" in open(text_path, encoding="utf-8").read()
    assert (tmp_path / "figures" / "permutation_feature_importance.png").is_file()
    assert chart_path.endswith("permutation_feature_importance.png")


def test_permutation_importance_rejects_invalid_input(tmp_path):
    model, X_test, y_test = fit_small_model(tmp_path)
    with pytest.raises(ValueError, match="same number of rows"):
        calculate_permutation_importance(model, X_test, y_test.iloc[:-1])
    with pytest.raises(ValueError, match="feature names or order"):
        calculate_permutation_importance(model, X_test.iloc[:, ::-1], y_test)


def test_run_explainability_handles_missing_artifact(tmp_path):
    data_path = tmp_path / "data.csv"
    generate_synthetic_employee_data(rows=100, output_path=data_path)
    with pytest.raises(FileNotFoundError, match="Trained model file not found"):
        run_explainability(data_path, tmp_path / "missing.joblib")
