from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "synthetic_employee_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "synthetic_employee_data_processed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "employee_performance_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
RANDOM_SEED = 42
TARGET_COLUMN = "perf_band_next"

DEFAULT_MODEL_FEATURES = [
    "department",
    "job_level",
    "experience_years",
    "manager_tenure",
    "projects_count",
    "avg_task_delay_days",
    "on_time_delivery_rate",
    "bug_count",
    "code_review_score",
    "qa_defect_density",
    "story_points_completed",
    "billable_hours_ratio",
    "training_hours",
    "certifications_count",
    "internal_hackathons_participated",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    "peer_feedback_score",
    "manager_score",
    "kudos_count",
    "promotions_in_2y",
]

REQUIRED_COLUMNS = [
    "employee_id",
    "department",
    "job_level",
    "experience_years",
    "manager_tenure",
    "projects_count",
    "avg_task_delay_days",
    "on_time_delivery_rate",
    "bug_count",
    "code_review_score",
    "qa_defect_density",
    "story_points_completed",
    "billable_hours_ratio",
    "training_hours",
    "certifications_count",
    "internal_hackathons_participated",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    "peer_feedback_score",
    "manager_score",
    "kudos_count",
    "promotions_in_2y",
    "salary_percentile_band",
    "gender",
    TARGET_COLUMN,
]

NUMERIC_COLUMNS = [
    "experience_years",
    "manager_tenure",
    "projects_count",
    "avg_task_delay_days",
    "on_time_delivery_rate",
    "bug_count",
    "code_review_score",
    "qa_defect_density",
    "story_points_completed",
    "billable_hours_ratio",
    "training_hours",
    "certifications_count",
    "internal_hackathons_participated",
    "sick_days",
    "unplanned_absences",
    "avg_login_hours",
    "peer_feedback_score",
    "manager_score",
    "kudos_count",
    "promotions_in_2y",
]

CATEGORICAL_COLUMNS = [
    "department",
    "job_level",
    "salary_percentile_band",
    "gender",
]

TARGET_CLASSES = ["High", "Medium", "Low"]
