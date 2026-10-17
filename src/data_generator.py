from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, RAW_DATA_PATH, TARGET_COLUMN


def generate_synthetic_employee_data(
    rows: int = 2200,
    output_path: str | Path = RAW_DATA_PATH,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic employee dataset with a rule-based target label."""
    rng = np.random.default_rng(random_seed)

    department_options = [
        "Engineering",
        "HR",
        "Sales",
        "Finance",
        "Marketing",
        "Operations",
        "Product",
    ]
    job_levels = ["Associate", "Senior", "Lead", "Manager"]
    salary_bands = ["Low", "Mid", "High", "Top"]
    gender_options = ["Woman", "Man", "Non-binary"]

    employee_id = np.arange(1, rows + 1)
    department = rng.choice(department_options, size=rows)
    job_level = rng.choice(job_levels, size=rows)
    experience_years = rng.integers(1, 16, size=rows)
    manager_tenure = rng.integers(0, 10, size=rows)
    projects_count = rng.integers(1, 12, size=rows)
    avg_task_delay_days = rng.integers(1, 22, size=rows)
    on_time_delivery_rate = np.clip(rng.normal(82, 12, size=rows), 35, 100)
    bug_count = np.clip(rng.normal(6, 4, size=rows), 0, 30).round().astype(int)
    code_review_score = np.clip(rng.normal(80, 12, size=rows), 30, 100)
    qa_defect_density = np.clip(rng.normal(1.8, 1.1, size=rows), 0.1, 8)
    story_points_completed = np.clip(rng.normal(90, 30, size=rows), 15, 220).round().astype(int)
    billable_hours_ratio = np.clip(rng.normal(0.68, 0.18, size=rows), 0.0, 1.0)
    training_hours = np.clip(rng.normal(28, 18, size=rows), 0, 120).round().astype(int)
    certifications_count = rng.integers(0, 8, size=rows)
    internal_hackathons_participated = rng.integers(0, 6, size=rows)
    sick_days = rng.integers(0, 12, size=rows)
    unplanned_absences = rng.integers(0, 9, size=rows)
    avg_login_hours = np.clip(rng.normal(8.5, 1.8, size=rows), 4, 14)
    peer_feedback_score = np.clip(rng.normal(79, 12, size=rows), 25, 100)
    manager_score = np.clip(rng.normal(81, 10, size=rows), 30, 100)
    kudos_count = rng.integers(0, 16, size=rows)
    promotions_in_2y = rng.integers(0, 3, size=rows)
    salary_percentile_band = rng.choice(salary_bands, size=rows)
    gender = rng.choice(gender_options, size=rows)

    performance_score = (
        0.30 * on_time_delivery_rate
        + 0.18 * peer_feedback_score
        + 0.18 * manager_score
        + 0.12 * code_review_score
        + 0.08 * np.clip(story_points_completed / 2.0, 0, 100)
        + 0.06 * training_hours
        + 0.08 * promotions_in_2y * 25
        + 0.05 * np.clip(experience_years * 3, 0, 100)
        - 0.12 * avg_task_delay_days
        - 0.12 * bug_count
        - 0.10 * qa_defect_density
        - 0.08 * sick_days
        - 0.08 * unplanned_absences
    )
    performance_score = performance_score + rng.normal(0, 8, size=rows)
    performance_score = np.clip(performance_score, 0, 100)

    score_cut_1 = np.quantile(performance_score, 0.33)
    score_cut_2 = np.quantile(performance_score, 0.66)

    perf_band = np.where(
        performance_score >= score_cut_2,
        "High",
        np.where(performance_score >= score_cut_1, "Medium", "Low"),
    )

    df = pd.DataFrame(
        {
            "employee_id": employee_id,
            "department": department,
            "job_level": job_level,
            "experience_years": experience_years,
            "manager_tenure": manager_tenure,
            "projects_count": projects_count,
            "avg_task_delay_days": avg_task_delay_days,
            "on_time_delivery_rate": on_time_delivery_rate.round(2),
            "bug_count": bug_count,
            "code_review_score": code_review_score.round(2),
            "qa_defect_density": qa_defect_density.round(2),
            "story_points_completed": story_points_completed,
            "billable_hours_ratio": billable_hours_ratio.round(2),
            "training_hours": training_hours,
            "certifications_count": certifications_count,
            "internal_hackathons_participated": internal_hackathons_participated,
            "sick_days": sick_days,
            "unplanned_absences": unplanned_absences,
            "avg_login_hours": avg_login_hours.round(2),
            "peer_feedback_score": peer_feedback_score.round(2),
            "manager_score": manager_score.round(2),
            "kudos_count": kudos_count,
            "promotions_in_2y": promotions_in_2y,
            "salary_percentile_band": salary_percentile_band,
            "gender": gender,
            TARGET_COLUMN: perf_band,
        }
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    generate_synthetic_employee_data()
