# Methodology

## Data provenance

The project uses a locally generated fictional dataset (2,200 rows by default, fixed seed 42). There are no public or company records. Its values are not observations of actual employees.

## Simulated target

For each generated row, a hidden score is computed as:

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

The score is clipped to 0–100. The dataset-specific 33rd and 66th score percentiles are then used as the class boundaries: score >= upper boundary is High; score >= lower boundary and below the upper boundary is Medium; otherwise Low. Thresholds are calculated from each generated dataset rather than hard-coded score values.

`employee_id` does not influence the score or model. Gender and salary band are independently generated and do not enter the score. `manager_score`, peer feedback, learning, delivery, quality, experience, promotions, and absence/delay indicators do influence the synthetic score. Those are artificial assumptions. Because many inputs are directly related to label construction, strong test results would still only indicate the model approximates its synthetic generator.

## Validation and preprocessing

Validation checks the required columns, unexpected columns, duplicates, duplicate IDs, missing values, numeric data types, known categories, target labels, finite values, and documented numeric ranges. Invalid rows are reported, not silently removed.

The model pipeline median-imputes and standardizes numeric fields. It imputes categorical nulls using the most frequent training category and one-hot encodes with `handle_unknown="ignore"`. The pipeline is fitted inside CV folds during model selection, avoiding preprocessing fit leakage from validation folds.

## Training and evaluation

The one-row-per-fictional-employee table uses a stratified 80/20 random split with seed 42. Five-fold stratified CV is applied only to the training portion. Macro-F1 is used for model selection so each class contributes equally to the average. The test set is not used for repeated tuning. Baseline, Logistic Regression, and Random Forest are reported with accuracy, macro precision/recall/F1, class report, and confusion matrix.

The current saved evaluation is one seed-specific synthetic holdout. It is not an estimate of deployment performance, temporal generalization, causal effect, or validity on people.

## Explainability

Permutation importance uses the fitted pipeline and held-out feature frame. Each column is shuffled in turn; the drop in macro-F1 is averaged across repeated shuffles. This is a global model-reliance measure. Correlated features can share or mask apparent importance, and the score does not mean that changing a feature will cause a change in employee performance. The dashboard does not label this as an individual prediction explanation.

## Group disparity audit

The saved model predicts on the same deterministic held-out records. Metrics are calculated for gender, department, and job level, with a minimum group size of 20 for metric reporting. Macro metrics cover classes observed in that group; High-specific false-positive and false-negative rates are reported only when denominators exist. Confusion matrices retain all three fixed class positions.

These are descriptive comparisons for synthetic labels. Similar rates cannot establish fairness, and different rates do not reveal causes. Group composition, missingness, synthetic label assumptions, sample size, proxies, and metric choice all matter. No automated threshold adjustment or mitigation is used.

## Recommendations

Recommendation examples are triggered only by supplied numeric values, such as delivery below 80%, training below 20 hours, review/peer-feedback below 75, or bug count above 8. Each triggered output includes its input signal, basis, and reason. Up to two signal-supported suggestions appear. If no rule fires, two general, choice-based human-led conversation prompts are returned and explicitly labeled as not model findings. A development summary includes human review and synthetic-data caveats. Suggestions are not performance findings or disciplinary advice.
