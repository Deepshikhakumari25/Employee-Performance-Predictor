# Interview preparation

Use these answers to explain the implementation honestly. This is an educational project on synthetic data; no result here demonstrates real employee-appraisal validity.

## Common questions

### 1. What is the project objective?
- **Simple:** It demonstrates predicting a simulated next-cycle category from generated data.
- **Technical:** It is a reproducible three-class scikit-learn classification pipeline with validation, preprocessing, training-only cross-validation, holdout evaluation, permutation importance, group audits, prediction, and a Streamlit UI.
- **Example:** It predicts one of High, Medium, or Low for a fictional input row.

### 2. What is classification?
- **Simple:** Classification selects a category rather than predicting a continuous number.
- **Technical:** A supervised classifier learns a mapping from feature vectors to discrete target labels.
- **Example:** The target classes here are High, Medium, and Low.

### 3. Why use synthetic data?
- **Simple:** It supports a privacy-safe practice project without real employee information.
- **Technical:** A seeded NumPy generator makes the feature distributions and labels reproducible, but the generated assumptions can be unrealistic.
- **Example:** Every row is fictional and generated locally with seed 42.

### 4. How is the target generated?
- **Simple:** A score is calculated from selected simulated work measures, then split into three bands.
- **Technical:** A weighted score includes delivery, feedback, manager/review scores, training, output, promotions, experience, delays, bugs, QA defects, and absences; Gaussian noise is added, scores are clipped, and sample-specific 33rd/66th percentile thresholds create the labels.
- **Example:** It uses `perf_band_next`; that label is not an actual appraisal outcome.

### 5. What is preprocessing?
- **Simple:** Preprocessing prepares different input types for a model.
- **Technical:** The numeric pipeline median-imputes then standardizes; the categorical pipeline mode-imputes then one-hot encodes with unknown values ignored.
- **Example:** “Engineering” becomes encoded columns, while scores are scaled.

### 6. Why split train and test data?
- **Simple:** Training data teaches the model; test data provides a separate evaluation.
- **Technical:** A stratified 80/20 split with a fixed seed preserves class proportions while keeping a holdout out of model fitting and cross-validation.
- **Example:** The test set contains 440 rows in the generated 2,200-row run.

### 7. What is data leakage?
- **Simple:** Leakage lets training use information that would not be available when making a real prediction.
- **Technical:** It includes using the target/future outcomes as features or fitting transforms using test rows. Pipeline preprocessing is fitted within training folds; `employee_id` and the target are excluded.
- **Example:** Including `perf_band_next` among X would trivially reveal y and is prohibited.

### 8. What is Logistic Regression?
- **Simple:** It estimates how input patterns relate to the probability of each class.
- **Technical:** Multinomial logistic regression models class probabilities from a linear combination of transformed inputs; this version uses balanced class weights.
- **Example:** It was the selected model for the current synthetic dataset based on training CV macro-F1.

### 9. What is Random Forest?
- **Simple:** It combines many decision trees to make a prediction.
- **Technical:** Bootstrap-trained trees with feature subsampling are aggregated; the project uses 300 trees and balanced class weights.
- **Example:** It can model nonlinear feature interactions, but it did not outperform Logistic Regression on this held-out run.

### 10. What is DummyClassifier?
- **Simple:** It is a simple baseline with no meaningful learned relationship.
- **Technical:** The project uses the stratified strategy to sample predictions using observed class frequencies.
- **Example:** Its held-out macro-F1 was approximately 0.323 in the recorded run.

### 11. What is cross-validation?
- **Simple:** It repeats training and validation on different folds of the training data.
- **Technical:** Five-fold stratified CV estimates macro-F1 for candidate selection, only on the training partition; preprocessing is part of each fold's pipeline.
- **Example:** The mean CV macro-F1 for Logistic Regression was about 0.477.

### 12. What is accuracy?
- **Simple:** The fraction of all predictions that match the true class.
- **Technical:** Correct predictions divided by total predictions; it can obscure poor results on less common classes.
- **Example:** Logistic Regression accuracy was 0.502 on this holdout.

### 13. What is precision?
- **Simple:** Among predictions for a class, precision asks how many were correct.
- **Technical:** For a class, TP/(TP+FP); the report also provides macro precision across classes.
- **Example:** A High precision issue means many rows predicted High were actually another category.

### 14. What is recall?
- **Simple:** Among the actual examples of a class, recall asks how many were found.
- **Technical:** For a class, TP/(TP+FN); macro recall averages class recalls.
- **Example:** Low recall for Low means some actual Low rows were predicted as Medium or High.

### 15. What is F1-score?
- **Simple:** F1 balances precision and recall.
- **Technical:** The harmonic mean 2PR/(P+R); it is low if either precision or recall is low.
- **Example:** The evaluation report has separate F1 scores for each class.

### 16. What is macro-F1?
- **Simple:** It averages F1 equally across all classes.
- **Technical:** Calculate each class's F1 independently and take the unweighted mean, giving smaller classes equal importance.
- **Example:** The candidates were selected using five-fold training macro-F1, rather than test accuracy.

### 17. What is a confusion matrix?
- **Simple:** It shows which actual classes were predicted as each class.
- **Technical:** Rows are actual labels and columns predicted labels, in the fixed order Low, Medium, High.
- **Example:** A value in actual Medium/predicted High counts a false-positive High and false-negative Medium.

### 18. What is class imbalance?
- **Simple:** Some categories have many more examples than others.
- **Technical:** A majority-heavy distribution may inflate accuracy and make minority-class estimates unstable; macro metrics and stratification help expose/limit some effects.
- **Example:** Check the target distribution and per-class support before interpreting aggregate metrics.

### 19. What is permutation importance?
- **Simple:** It measures how much model performance changes when one feature is scrambled.
- **Technical:** On held-out rows, each column is independently permuted repeatedly, then macro-F1 decrease mean and standard deviation are recorded.
- **Example:** The saved CSV ranks global features for this model and holdout.

### 20. What does explainable AI mean here?
- **Simple:** It means making model behavior easier to inspect.
- **Technical:** This project uses global permutation importance; it does not produce causal findings or per-person explanations.
- **Example:** A top feature is associated with the model's overall holdout performance, not proof that it caused a prediction.

### 21. What is algorithmic bias?
- **Simple:** A model may systematically work differently for different groups.
- **Technical:** Unequal error rates can arise from data, labels, measurement, sampling, features, or system design; removing sensitive inputs alone does not eliminate proxies.
- **Example:** The audit compares groups in this synthetic holdout but cannot establish discrimination or fairness.

### 22. What is fairness auditing?
- **Simple:** It compares performance and prediction patterns among defined groups.
- **Technical:** The report includes group counts, predicted-High rate, macro precision/recall/F1, High-class precision/recall, High FPR/FNR, and confusion matrices; metrics are withheld below n=20.
- **Example:** Similar error rates in a synthetic sample are not proof the system is fair.

### 23. Why use Streamlit?
- **Simple:** It provides a quick interactive Python dashboard.
- **Technical:** The app loads the saved pipeline and generated reports, displays data and metrics, validates uploads, and gathers features for prediction without retraining on refresh.
- **Example:** Run `python -m streamlit run app.py`.

### 24. What is model persistence?
- **Simple:** It saves a trained model so the app does not retrain every time.
- **Technical:** Joblib stores the complete sklearn preprocessing/classifier pipeline; JSON stores selection and feature metadata.
- **Example:** `models/employee_performance_model.joblib` is loaded for inference.

### 25. Why automated testing?
- **Simple:** Tests catch regressions in data, models, and behavior.
- **Technical:** Pytest covers seeded data generation, validation, preprocessing, training, inference, importance outputs, fairness metrics, and recommendation rules.
- **Example:** An unseen category test confirms safe one-hot inference.

### 26. What are the project's limitations?
- **Simple:** It is a synthetic exercise and not suitable for real appraisal.
- **Technical:** Labels derive from a known weighted score; fixed random holdout results do not establish external, temporal, causal, calibration, or fairness validity.
- **Example:** The saved metric table is evidence only about the generated rows and split.

### 27. What are reasonable future improvements?
- **Simple:** Improve temporal evaluation, calibration, monitoring, and packaging.
- **Technical:** Planned items include a verified public dataset, temporal modeling, calibrated probabilities, additional fairness measures, drift monitoring, API, MLflow, Docker, CI, and broader dashboard/tests.
- **Example:** Every listed future item is Planned unless code and tests are later added.

## Project pitches

### 30-second explanation

“I built a local Streamlit portfolio project that demonstrates a multi-class employee-performance prediction workflow using only synthetic data. It generates and validates fictional records, compares a baseline with Logistic Regression and Random Forest, evaluates on a fixed holdout, and produces global permutation-importance and group-disparity reports. The app loads a saved pipeline instead of retraining. I explicitly frame the results as synthetic and not appropriate for employment decisions.”

### One-minute explanation

“The project simulates a fictional organization with 2,200 generated records and a documented weighted-score target called `perf_band_next`. I implemented validation, EDA, numeric/categorical preprocessing in a scikit-learn pipeline, and a stratified 80/20 split. Candidate selection uses five-fold macro-F1 on training data; in the current run Logistic Regression was selected, with held-out macro-F1 around 0.488, compared with 0.469 for Random Forest and 0.323 for the DummyClassifier. I also calculate holdout permutation importance and group metrics for gender, department, and job level, withholding performance measures for small groups. Streamlit displays saved reports and produces an optional prediction with non-punitive development prompts. Because the target is rule-generated and the model has not been validated on real people, the whole system is educational only.”

### Resume-ready project entry

**AI-Powered Employee Performance Predictor — Python, scikit-learn, Streamlit**

Built a reproducible synthetic-data classification demonstration with data validation, EDA, baseline/model comparison, saved inference pipeline, holdout evaluation, global permutation importance, group disparity audit, recommendation rules, Streamlit dashboard, and pytest coverage. Documented synthetic-label provenance, limitations, and prohibited employment uses.

### Verified implementation bullets

- Generated and validated 2,200 fixed-seed synthetic records with a documented three-class target.
- Compared DummyClassifier, Logistic Regression, and Random Forest using training-only stratified CV and held-out classification metrics.
- Implemented holdout permutation importance and gender/department/job-level disparity reports with small-group metric withholding.
- Delivered saved-pipeline inference, a Streamlit dashboard, and automated tests for core data/model/audit/recommendation behavior.

Do not represent the held-out score as a real-world performance improvement.

### GitHub summary

“An educational employee-performance classification demo using generated synthetic data, scikit-learn, global permutation importance, group audits, non-punitive development suggestions, and Streamlit. Not validated for real HR decisions.”

### Technical follow-up questions to prepare for

- Why macro-F1 rather than accuracy for candidate selection?
- Why does the model's score not establish real-world validity?
- What would change if multiple dated rows existed for each employee?
- How could temporal leakage arise in a future dataset?
- What do permutation importance and correlated features do to each other's ranks?
- Which group metric should be prioritized, and why is there no single universal answer?
- How should an app handle missing, drifted, or untrusted inputs?
- What evidence would be needed before any model could be considered for a high-impact domain?

The responsible answer to the last question is that this portfolio model is not appropriate for consequential employment use.
