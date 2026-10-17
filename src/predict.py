from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, TARGET_COLUMN


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the trained model pipeline if it exists."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def validate_prediction_input(data: dict | pd.DataFrame | list[dict], expected_features: list[str] | None = None) -> pd.DataFrame:
    """Validate a prediction payload using the same feature order from training."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame([data])
    elif isinstance(data, list):
        frame = pd.DataFrame(data)
    else:
        raise TypeError("Prediction data must be a dict, list of dicts, or DataFrame.")

    if frame.empty:
        raise ValueError("Prediction input cannot be empty.")

    if expected_features is not None:
        missing = [feature for feature in expected_features if feature not in frame.columns]
        if missing:
            raise ValueError(f"Missing required prediction features: {missing}")
        extra = [feature for feature in frame.columns if feature not in expected_features]
        if extra:
            frame = frame[expected_features]
    return frame


def predict_performance(input_data: dict | pd.DataFrame | list[dict], model_path: str | Path = MODEL_PATH) -> dict:
    """Return a prediction and probability estimates when supported by the model."""
    model = load_model(model_path)
    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        raise ValueError("The trained model does not expose expected feature names.")

    frame = validate_prediction_input(input_data, expected_features)
    prediction = model.predict(frame)
    result = {"prediction": prediction[0]}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        labels = model.classes_
        result["probabilities"] = {str(label): round(float(prob), 4) for label, prob in zip(labels, probabilities)}
    return result


if __name__ == "__main__":
    sample = {
        "department": "Engineering",
        "job_level": "Senior",
        "experience_years": 5,
        "manager_tenure": 2,
        "projects_count": 6,
        "avg_task_delay_days": 4,
        "on_time_delivery_rate": 92,
        "bug_count": 2,
        "code_review_score": 88,
        "qa_defect_density": 1.4,
        "story_points_completed": 120,
        "billable_hours_ratio": 0.8,
        "training_hours": 38,
        "certifications_count": 3,
        "internal_hackathons_participated": 2,
        "sick_days": 1,
        "unplanned_absences": 0,
        "avg_login_hours": 9.5,
        "peer_feedback_score": 87,
        "manager_score": 88,
        "kudos_count": 9,
        "promotions_in_2y": 1,
        "salary_percentile_band": "High",
        "gender": "Woman",
    }
    print(predict_performance(sample))
