from datetime import date

import pytest

from src.recommendations import build_development_summary, generate_recommendations


def test_recommendations_are_supported_non_punitive_and_limited():
    suggestions = generate_recommendations(
        {
            "on_time_delivery_rate": 72,
            "bug_count": 15,
            "training_hours": 5,
        }
    )
    assert len(suggestions) == 2
    assert suggestions[0]["signal"]["feature"] == "on_time_delivery_rate"
    assert "reason" in suggestions[0]
    text = " ".join(item["action"] for item in suggestions).lower()
    assert not any(word in text for word in ("fire", "discipline", "lazy", "incompetent"))


def test_recommendations_skip_missing_or_nonnumeric_signals():
    suggestions = generate_recommendations(
        {
            "training_hours": "unknown",
            "on_time_delivery_rate": 90,
            "bug_count": 2,
            "code_review_score": 90,
            "peer_feedback_score": 90,
        }
    )
    assert len(suggestions) == 2
    assert all(item["basis"] == "general_human_led_prompt" for item in suggestions)
    assert all(item["signal"] is None for item in suggestions)


def test_development_summary_has_required_fields_and_global_driver_scope():
    summary = build_development_summary(
        "Medium",
        {"training_hours": 10, "on_time_delivery_rate": 75},
        [
            {"feature": "training_hours", "importance_mean": 0.12},
            {"feature": "on_time_delivery_rate", "importance_mean": 0.09},
            {"feature": "manager_score", "importance_mean": 0.05},
            {"feature": "bug_count", "importance_mean": 0.02},
        ],
        {"High": 0.2, "Medium": 0.6, "Low": 0.2},
        review_date=date(2026, 11, 1),
    )
    assert summary["predicted_simulated_category"] == "Medium"
    assert len(summary["relevant_input_signals"]) == 2
    assert len(summary["top_global_model_related_features"]) == 3
    assert "not an individual explanation" in summary["top_global_model_related_features"][0]["scope"]
    assert len(summary["development_actions"]) == 2
    assert summary["suggested_review_date"] == "2026-11-01"
    assert summary["uncertainty_and_disclaimer"]["human_review_required"]
    assert "synthetic labels" in summary["uncertainty_and_disclaimer"]["notice"]


def test_summary_rejects_unknown_prediction():
    with pytest.raises(ValueError, match="prediction"):
        build_development_summary("Unknown", {})
