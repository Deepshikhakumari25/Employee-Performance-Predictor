import pandas as pd
from sklearn.dummy import DummyClassifier

from src.preprocessing import build_model_pipeline, build_preprocessor, get_feature_columns


def test_preprocessor_handles_numeric_and_categorical_features():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "department": ["Engineering", "Sales", "Engineering"],
            "job_level": ["Senior", "Associate", "Lead"],
            "experience_years": [5, 2, 7],
            "manager_tenure": [3, 1, 4],
            "projects_count": [6, 4, 8],
            "avg_task_delay_days": [3, 7, 5],
            "on_time_delivery_rate": [90, 78, 88],
            "bug_count": [2, 5, 1],
            "code_review_score": [85, 68, 90],
            "qa_defect_density": [1.2, 2.0, 1.1],
            "story_points_completed": [120, 95, 130],
            "billable_hours_ratio": [0.8, 0.7, 0.9],
            "training_hours": [40, 25, 55],
            "certifications_count": [3, 1, 4],
            "internal_hackathons_participated": [2, 1, 3],
            "sick_days": [1, 2, 0],
            "unplanned_absences": [0, 2, 1],
            "avg_login_hours": [9.0, 8.5, 9.5],
            "peer_feedback_score": [88, 70, 92],
            "manager_score": [90, 75, 94],
            "kudos_count": [10, 4, 8],
            "promotions_in_2y": [1, 0, 1],
            "salary_percentile_band": ["High", "Mid", "Top"],
            "gender": ["Woman", "Man", "Non-binary"],
            "perf_band_next": ["High", "Medium", "High"],
        }
    )

    num_cols, cat_cols = get_feature_columns(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(df[num_cols + cat_cols])

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > 0
    assert len(num_cols) > 0
    assert len(cat_cols) > 0
    assert "gender" not in cat_cols
    assert "salary_percentile_band" not in cat_cols


def test_missing_values_and_unseen_categories_work_without_test_fit():
    X_train = pd.DataFrame(
        {
            "experience_years": [1.0, None, 5.0, 3.0],
            "department": ["Engineering", "Sales", "Sales", "HR"],
        }
    )
    y_train = pd.Series(["Low", "High", "Medium", "Low"])
    X_test = pd.DataFrame({"experience_years": [2.0], "department": ["New department"]})
    pipeline = build_model_pipeline(
        DummyClassifier(strategy="most_frequent"),
        ["experience_years"],
        ["department"],
    )
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    assert prediction[0] in {"High", "Medium", "Low"}
    assert pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].statistics_[0] == 3.0
