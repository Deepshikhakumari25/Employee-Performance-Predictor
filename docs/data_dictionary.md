# Data dictionary

`data/raw/synthetic_employee_data.csv` is generated locally by `src/data_generator.py`, with 2,200 fictional records and a fixed default random seed of 42. It contains no real employee information.

| Column | Type | Meaning |
|---|---|---|
| `employee_id` | integer | Unique synthetic row ID; excluded from the model. |
| `department` | category | One of Engineering, HR, Sales, Finance, Marketing, Operations, Product. |
| `job_level` | category | Associate, Senior, Lead, or Manager. |
| `experience_years` | integer | Simulated years of experience. |
| `manager_tenure` | integer | Simulated manager tenure value in years. |
| `projects_count` | integer | Simulated project count. |
| `avg_task_delay_days` | integer | Simulated average task-delay duration. |
| `on_time_delivery_rate` | numeric | Generated on-time delivery percentage, 0–100. |
| `bug_count` | integer | Simulated bug count. |
| `code_review_score` | numeric | Simulated review score, 0–100. |
| `qa_defect_density` | numeric | Simulated quality-assurance defect density. |
| `story_points_completed` | integer | Simulated completed story points. |
| `billable_hours_ratio` | numeric | Simulated billable-hours share, 0–1. |
| `training_hours` | integer | Simulated learning hours. |
| `certifications_count` | integer | Simulated certification count. |
| `internal_hackathons_participated` | integer | Simulated internal event participation count. |
| `sick_days` | integer | Synthetic sick-day count; not a real health record. |
| `unplanned_absences` | integer | Synthetic absence count. |
| `avg_login_hours` | numeric | Simulated average login hours. |
| `peer_feedback_score` | numeric | Simulated feedback score, 0–100. |
| `manager_score` | numeric | Simulated manager score, 0–100. |
| `kudos_count` | integer | Simulated recognition count. |
| `promotions_in_2y` | integer | Simulated prior promotion count used in the synthetic rule. |
| `salary_percentile_band` | category | Synthetic Low, Mid, High, or Top band; excluded from the default model. |
| `gender` | category | Synthetic Woman, Man, or Non-binary audit-only attribute; excluded from prediction. |
| `perf_band_next` | category | Generated target: High, Medium, or Low. |

The model's default feature set uses 22 columns: all of the simulated operational measures plus `department` and `job_level`, excluding `employee_id`, `gender`, `salary_percentile_band`, and `perf_band_next`. Not every field is equally justified for real HR analysis; this feature set is a teaching example only.

The target score inputs, clipping, normal noise, and data-specific quantile thresholds are specified in [methodology.md](methodology.md). Generated labels are not genuine appraisal outcomes.
