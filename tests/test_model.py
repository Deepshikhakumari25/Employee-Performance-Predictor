import joblib
from pathlib import Path
from sklearn.dummy import DummyClassifier

from src.config import MODEL_PATH
from src.evaluate_model import evaluate_model, save_model_evaluation_report
from src.train_model import get_model_candidates, load_training_frame, train_and_save_model


def test_baseline_is_included_in_model_candidates():
    assert isinstance(get_model_candidates()["Baseline"], DummyClassifier)


def test_train_and_save_model_creates_joblib_model():
    metadata = train_and_save_model("data/raw/synthetic_employee_data.csv")

    assert "selected_model" in metadata
    assert metadata["selected_model"] in {"Baseline", "Logistic Regression", "Random Forest"}
    assert "gender" not in metadata["feature_columns"]
    assert "salary_percentile_band" not in metadata["feature_columns"]
    saved_model = joblib.load(MODEL_PATH)
    assert saved_model.predict_proba is not None
    X, y, _, _ = load_training_frame()
    predictions = saved_model.predict(X.iloc[:12])
    assert set(predictions).issubset({"High", "Medium", "Low"})
    metrics = evaluate_model(saved_model, X.iloc[:50], y.iloc[:50])
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 3


def test_saved_classification_report_uses_calculated_per_class_f1(tmp_path):
    results = {
        "Logistic Regression": {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "macro_f1": 0.5,
            "per_class": {
                "High": {
                    "precision": 0.6,
                    "recall": 0.5,
                    "f1-score": 0.5454,
                    "support": 10.0,
                }
            },
        }
    }

    paths = save_model_evaluation_report(results, tmp_path)
    report = Path(paths["text_path"]).read_text(encoding="utf-8")

    assert "f1=0.5454" in report
