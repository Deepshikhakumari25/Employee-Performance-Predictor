import pytest

from src.predict import predict_performance, validate_prediction_input


def sample_prediction():
    return {
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
    }


def test_predict_performance_returns_valid_result():
    result = predict_performance(sample_prediction())
    assert result["prediction"] in {"High", "Medium", "Low"}
    assert "probabilities" in result
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


def test_prediction_rejects_missing_expected_feature():
    sample = sample_prediction()
    del sample["training_hours"]
    with pytest.raises(ValueError, match="Missing required prediction features"):
        validate_prediction_input(sample, list(sample_prediction()))


def test_prediction_missing_model_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        predict_performance(sample_prediction(), tmp_path / "missing.joblib")
