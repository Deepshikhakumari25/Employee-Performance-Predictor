# AI-Powered Employee Performance Predictor

An educational, local machine-learning portfolio project that predicts a **simulated** next-cycle performance category (High, Medium, or Low) from reproducible synthetic records.

> **Important:** This is not a validated HR decision system. The labels and records are synthetic; the model learns patterns from the label generator. Do not use it for hiring, firing, promotion, compensation, discipline, employee ranking, or other consequential employment decisions.

## Project overview

### Problem statement

This project demonstrates how to build and evaluate a multi-class classification workflow around a fictional organization's generated records. The example asks whether a model can learn the rules used to create a simulated label; it does not establish how real employees perform.

### Objectives and key features

- Reproducible synthetic data generation and reusable data validation.
- Exploratory charts and summary report.
- A scikit-learn preprocessing/model pipeline.
- Dummy baseline, Logistic Regression, and Random Forest comparison.
- Stratified train/test split, training-only cross-validation, and macro-F1 selection.
- Saved-model prediction with optional probability estimates.
- Holdout permutation importance and group fairness metrics.
- Non-punitive, optional development suggestions requiring human review.
- Streamlit dashboard, automated tests, and interview-oriented documentation.

### Technology

Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn, Streamlit, Joblib, and pytest. SHAP and MLflow are not required.

## System architecture

```mermaid
flowchart LR
  A[Fixed-seed synthetic generator] --> B[CSV and data validation]
  B --> C[EDA charts and report]
  B --> D[Stratified train/test split]
  D --> E[Training-only CV: baseline, Logistic Regression, Random Forest]
  E --> F[Selected fitted pipeline and metadata]
  D --> G[Held-out evaluation]
  F --> H[Validated prediction]
  G --> I[Permutation importance]
  G --> J[Gender, department, and job-level audit]
  H --> K[Optional development suggestions]
  F --> L[Streamlit dashboard]
  G --> L
  I --> L
  J --> L
```

The dashboard loads existing artifacts; refreshing it does **not** retrain the model.

## Dataset and provenance

No public dataset is used. `src/data_generator.py` creates 2,200 fictional records locally with seed 42 and saves `data/raw/synthetic_employee_data.csv`. There are no real employee records or external usage terms. Data are generated, not scraped or sourced from a company.

### Target definition

The target column is `perf_band_next`. A hidden synthetic score combines:

```text
0.30 * on_time_delivery_rate
+ 0.18 * peer_feedback_score
+ 0.18 * manager_score
+ 0.12 * code_review_score
+ 0.08 * clip(story_points_completed / 2, 0, 100)
+ 0.06 * training_hours
+ 0.08 * promotions_in_2y * 25
+ 0.05 * clip(experience_years * 3, 0, 100)
- 0.12 * avg_task_delay_days
- 0.12 * bug_count
- 0.10 * qa_defect_density
- 0.08 * sick_days
- 0.08 * unplanned_absences
+ Normal(0, 8) noise
```

The resulting score is clipped to 0–100. The sample's 33rd and 66th percentiles are computed, then values at/above the upper threshold become High, values at/above the lower threshold become Medium, and the remainder become Low. Because several model inputs contribute to this score, the exercise deliberately teaches a model to recover generator patterns. It does not simulate genuine appraisal outcomes or prove causation.

### Data dictionary summary

The CSV includes a unique `employee_id`; department and job level; experience and manager tenure; project, delivery, delay, bug, code-review, QA, story-point, billable-hour, training, certification, hackathon, absence, login-hour, feedback, manager-score, kudos, and promotion measures; synthetic salary band and gender; and `perf_band_next`. See [docs/data_dictionary.md](docs/data_dictionary.md) for the complete field list.

`employee_id` is excluded from training. `gender` is audit-only and `salary_percentile_band` is not a default predictive feature. The pipeline uses 22 operational or role-related features; no target column is used as an input.

## Preprocessing and machine-learning methodology

Numerical inputs are median-imputed and standardized. Categorical inputs are most-frequent-imputed and one-hot encoded with unseen categories ignored. These steps are inside a scikit-learn `Pipeline`/`ColumnTransformer`, so cross-validation fits preprocessing within each training fold. There is one generated record per fictional employee, so employee-group splitting is unnecessary for this dataset.

An 80/20 stratified split uses seed 42. Candidate selection uses five-fold stratified CV on the training portion and macro-F1. The held-out test set is used once for the reported candidate comparison, global permutation importance, and group audit. The random holdout is not temporal validation.

## Actual model comparison

Results below were generated locally from the fixed-seed 2,200-row dataset and the saved evaluation JSON. Metrics are from the 440-row held-out test set; values are rounded to three decimals.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Dummy baseline | 0.336 | 0.336 | 0.336 | 0.336 |
| Logistic Regression | 0.502 | 0.485 | 0.501 | 0.488 |
| Random Forest | 0.477 | 0.465 | 0.476 | 0.469 |

Logistic Regression had the highest training-CV macro-F1 (0.477) and was selected. The seeded DummyClassifier CV macro-F1 was 0.334 and Random Forest's was 0.467. These are actual results for this synthetic dataset only, not evidence of real-world performance. Macro-F1 averages class-wise F1 equally; accuracy alone may hide uneven class errors. Precision describes correctness among predicted members of a class, recall describes how many actual class members were found, and the confusion matrix shows which categories are mixed up. False positives and false negatives have different implications in real use, which is another reason not to use this system for employment decisions.

## Explainability

Global permutation importance shuffles each feature in held-out data and measures the change in macro-F1. It indicates how much this particular fitted model relies on each feature in this test sample. It does not establish causation and is not an explanation of any individual prediction. The dashboard explicitly keeps global model associations separate from provided individual input signals.

## Fairness auditing

The held-out predictions are summarized by gender, department, and job level. The JSON includes group counts, predicted-High rates, macro precision/recall/F1, High-class precision/recall, High false-positive/false-negative rates, confusion matrices, and max-minus-min error-rate differences where denominators are available. Metrics are withheld below the configured group-size minimum of 20. Missing audit values are represented as a `Missing` group.

These descriptive synthetic-data metrics are not proof of fairness. Small samples, label quality, group definitions, and proxy variables limit interpretation. No group-specific thresholding or automatic mitigation is applied.

## Optional development recommendations

Recommendations are generated when supported numeric input patterns cross documented example thresholds. They are non-punitive suggestions (for example, a collaborative review of workload, blockers, or learning opportunities), include the signal that prompted them, and are capped at two actions in the summary. If no rule fires, the summary offers two clearly labeled general, human-led development conversation prompts and explicitly states that they are not model findings. A human must review them. The code does not infer personality or personal circumstances, and does not recommend discipline, firing, salary, or promotion actions.

## Dashboard

Run the app to explore generated data, EDA charts, actual evaluation reports, prediction probabilities, global importance, audit metrics, and responsible-use limitations. Optional CSV uploads are read as data only and must pass the complete project schema validation; incompatible data is not used for prediction.

## Installation and execution

Windows PowerShell:

```powershell
cd "C:\projects\AI-Employee-Performance-Predictor"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is blocked, use Command Prompt:

```cmd
cd /d "C:\projects\AI-Employee-Performance-Predictor"
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete workflow from the project root:

```powershell
python src\data_generator.py
python src\eda.py
python src\train_model.py
python src\evaluate_model.py
python -m pytest -q
python -m streamlit run app.py
```

Evaluation produces the actual metrics, confusion matrices, permutation-importance report/chart, and fairness JSON. Prediction can also be tested with `python src\predict.py`. Model files are generated locally and excluded from version control.

## Screenshots

These are genuine captures of the running Streamlit app and charts generated by this project's code.

![Project structure on the public GitHub repository](images/screenshots/01_project_structure.png)
![Dataset preview](images/screenshots/02_dataset_preview.png)
![Dataset validation](images/screenshots/03_data_validation.png)
![Target class distribution dashboard](images/screenshots/04_class_distribution.png)
![Department and training-hours analysis](images/screenshots/05_department_analysis.png)
![Synthetic work-quality analysis](images/screenshots/06_feature_analysis.png)
![Numeric feature correlation dashboard](images/screenshots/07_correlation_heatmap.png)
![Target class distribution](outputs/figures/target_distribution.png)
![Department-wise category counts](outputs/figures/department_target_distribution.png)
![Model comparison](outputs/figures/model_comparison_macro_f1.png)
![Model evaluation dashboard](images/screenshots/08_model_comparison.png)
![Logistic Regression confusion matrix dashboard](images/screenshots/09_confusion_matrix.png)
![Saved classification report dashboard](images/screenshots/10_classification_report.png)
![Permutation feature importance](outputs/figures/permutation_feature_importance.png)
![Global feature importance dashboard](images/screenshots/11_feature_importance.png)
![Fairness audit dashboard](images/screenshots/12_fairness_audit.png)
![Fictional prediction output](images/screenshots/13_prediction_output.png)
![Streamlit overview](images/screenshots/14_dashboard_overview.png)
![Streamlit analytics](images/screenshots/15_dashboard_analytics.png)
![Public GitHub repository](images/screenshots/17_github_repository.png)

All included application and repository captures are genuine. The test-terminal capture (`16_test_results.png`) has not been taken; see [images/README.md](images/README.md) for exact manual capture steps.

## Project structure

```text
app.py
requirements.txt
src/                 generator, validation, preprocessing, EDA, training,
                     evaluation, prediction, explainability, fairness, advice
data/raw/            generated synthetic CSV
models/              local fitted model and metadata (not committed)
outputs/figures/     generated EDA, evaluation, and importance charts
outputs/reports/     actual evaluation, audit, and importance reports
tests/               pytest suite
docs/                architecture, methodology, safety, and learning guides
images/screenshots/  manually captured genuine app screenshots
```

## Tests and verification

Verified with Python 3.14.6 and the project virtual environment:

```text
28 passed in 18.85s
```

The complete dataset-generation → EDA → model-training → evaluation/artifact workflow completed before this test run. The Streamlit app is also launched and checked separately during project verification. Refer to the current `outputs/reports/` files for the generated detailed results.

## Limitations and responsible AI

This is a small, synthetic, static dataset with generator-created labels. The model may learn the formula rather than workplace relationships. Group-level metrics do not remove bias or establish fairness, and a protected attribute's exclusion does not eliminate proxy effects. Probabilities have not been calibrated as real-world uncertainty. Privacy-safe demonstration data do not authorize using actual employee data. Use human oversight for any discussion; never use these outputs for consequential employment decisions. Read [docs/limitations.md](docs/limitations.md) and [docs/responsible_ai.md](docs/responsible_ai.md).

## Future improvements

| Improvement | Status |
|---|---|
| Verified public dataset integration | Planned |
| Temporal modeling and temporal holdout evaluation | Planned |
| Probability calibration | Planned |
| Additional fairness measures and stronger sample-size analysis | Planned |
| Data-drift monitoring | Planned |
| FastAPI prediction endpoint | Planned |
| MLflow tracking | Planned |
| Docker packaging | Planned |
| GitHub Actions | Planned |
| Further dashboard analytics and tests | Planned |

## License

MIT License; see [LICENSE](LICENSE). Dependencies retain their own licenses. No third-party dataset is included.

## Author

Author name and contact were not provided for this workspace. Add your own details before publishing if desired; do not imply organizational affiliation.

## GitHub publication status

The project is published in the [public Employee-Performance-Predictor repository](https://github.com/Deepshikhakumari25/Employee-Performance-Predictor) on the `main` branch. Its public status, default branch, source tree, README rendering, screenshots, MIT license, and setup/test instructions were verified after upload. Never commit credentials, `.venv/`, or confidential data.
