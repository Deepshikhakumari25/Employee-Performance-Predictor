from __future__ import annotations

from datetime import date, timedelta
from numbers import Real
from typing import Mapping, Sequence


def generate_recommendations(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Suggest at most two non-punitive development actions from supported signals."""
    if not isinstance(row, Mapping):
        raise TypeError("Recommendation input must be a mapping of feature names to values.")

    suggestions: list[dict[str, object]] = []
    rules = [
        (
            "on_time_delivery_rate",
            lambda value: value < 80,
            "Review workload, task estimates, blockers, and sprint planning together.",
            "The provided simulated on-time delivery rate is below 80%.",
        ),
        (
            "bug_count",
            lambda value: value > 8,
            "Consider quality-focused learning or a peer pairing session on recent work.",
            "The provided simulated bug count is above 8.",
        ),
        (
            "code_review_score",
            lambda value: value < 75,
            "Offer a collaborative code-review practice session with specific examples.",
            "The provided simulated code-review score is below 75.",
        ),
        (
            "training_hours",
            lambda value: value < 20,
            "Explore relevant learning opportunities during agreed work time.",
            "The provided simulated training hours are below 20.",
        ),
        (
            "peer_feedback_score",
            lambda value: value < 75,
            "Offer a structured, human-led feedback conversation focused on support and goals.",
            "The provided simulated peer-feedback score is below 75.",
        ),
    ]
    for feature, condition, action, reason in rules:
        value = row.get(feature)
        if isinstance(value, Real) and condition(float(value)):
            suggestions.append(
                {
                    "action": action,
                    "reason": reason,
                    "signal": {"feature": feature, "value": float(value)},
                    "basis": "supported_input_signal",
                }
            )
            if len(suggestions) == 2:
                break
    if not suggestions:
        suggestions = [
            {
                "action": "Ask which skills or learning opportunities the person would like to explore.",
                "reason": (
                    "No rule-based input signal crossed a threshold; this is a general, "
                    "choice-based development conversation, not a model finding."
                ),
                "signal": None,
                "basis": "general_human_led_prompt",
            },
            {
                "action": "Review current work priorities together and invite discussion of blockers or support.",
                "reason": (
                    "No rule-based input signal crossed a threshold; this is a general, "
                    "collaborative check-in, not a model finding."
                ),
                "signal": None,
                "basis": "general_human_led_prompt",
            },
        ]
    return suggestions


def build_development_summary(
    prediction: str,
    row: Mapping[str, object],
    feature_importance: Sequence[Mapping[str, object]] | None = None,
    probabilities: Mapping[str, float] | None = None,
    *,
    review_date: date | None = None,
) -> dict[str, object]:
    """Create a human-review summary without claiming individual causal explanations."""
    if prediction not in {"High", "Medium", "Low"}:
        raise ValueError("prediction must be one of High, Medium, or Low.")
    if not isinstance(row, Mapping):
        raise TypeError("row must map feature names to supplied input values.")

    signal_names = (
        "on_time_delivery_rate",
        "bug_count",
        "code_review_score",
        "training_hours",
        "peer_feedback_score",
    )
    relevant_signals = [
        {"feature": name, "value": row[name]}
        for name in signal_names
        if name in row and isinstance(row[name], Real)
    ]
    drivers: list[dict[str, object]] = []
    if feature_importance is not None:
        for item in feature_importance:
            name = item.get("feature")
            value = item.get("importance_mean")
            if isinstance(name, str) and isinstance(value, Real):
                drivers.append(
                    {
                        "feature": name,
                        "global_importance": float(value),
                        "scope": "global model association; not an individual explanation",
                    }
                )
                if len(drivers) == 3:
                    break

    uncertainty: dict[str, object] = {
        "notice": (
            "This category and any probabilities are outputs of a model trained on synthetic labels. "
            "They are not validated appraisal outcomes and do not predict an individual with certainty."
        ),
        "human_review_required": True,
    }
    if probabilities is not None:
        uncertainty["model_probabilities"] = {
            str(label): float(probability) for label, probability in probabilities.items()
        }

    return {
        "predicted_simulated_category": prediction,
        "relevant_input_signals": relevant_signals,
        "top_global_model_related_features": drivers,
        "development_actions": generate_recommendations(row),
        "suggested_review_date": (review_date or (date.today() + timedelta(days=30))).isoformat(),
        "uncertainty_and_disclaimer": uncertainty,
    }
