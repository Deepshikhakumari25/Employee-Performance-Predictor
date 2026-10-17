from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    FIGURE_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    REPORT_DIR,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from src.data_validation import validate_dataset
from src.predict import predict_performance
from src.recommendations import build_development_summary


st.set_page_config(page_title="AI Employee Performance Predictor", page_icon="📊", layout="wide")
st.title("AI-Powered Employee Performance Predictor")
st.caption(
    "A local educational simulation using generated fictional records. "
    "It is not validated for employee appraisal or employment decisions."
)


@st.cache_data
def load_csv(path: str, modified_time: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: str, modified_time: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource
def load_pipeline(path: str, modified_time: int):
    return joblib.load(path)


def show_image(filename: str, caption: str) -> None:
    image_path = FIGURE_DIR / filename
    if image_path.is_file():
        st.image(str(image_path), caption=caption, width="stretch")
    else:
        st.info(f"Chart is not available yet: {filename}. Run EDA/evaluation to generate it.")


def report_or_message(path: Path, label: str) -> dict | None:
    if not path.is_file():
        st.info(f"{label} is not available yet. Run the training and evaluation workflow first.")
        return None
    try:
        return load_json(str(path), path.stat().st_mtime_ns)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read {label}: {exc}")
        return None


try:
    dataset = (
        load_csv(str(RAW_DATA_PATH), RAW_DATA_PATH.stat().st_mtime_ns)
        if RAW_DATA_PATH.is_file()
        else None
    )
except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    dataset = None
    st.error(f"Could not load the project dataset: {exc}")

metadata: dict = {}
if MODEL_METADATA_PATH.is_file():
    metadata = report_or_message(MODEL_METADATA_PATH, "Model metadata") or {}

evaluation = report_or_message(REPORT_DIR / "model_evaluation.json", "Evaluation report")
fairness = report_or_message(REPORT_DIR / "fairness_audit.json", "Fairness audit")

model = None
if MODEL_PATH.is_file():
    try:
        model = load_pipeline(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, EOFError, ImportError) as exc:
        st.error(f"Could not load the saved model pipeline: {exc}")

overview_tab, data_tab, eda_tab, evaluation_tab, prediction_tab, explainability_tab, fairness_tab, responsible_tab = st.tabs(
    [
        "Project overview",
        "Data explorer",
        "Exploratory analytics",
        "Model evaluation",
        "Performance prediction",
        "Explainability",
        "Fairness audit",
        "Responsible AI",
    ]
)

with overview_tab:
    st.subheader("Project overview")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Objective:** demonstrate an end-to-end, reproducible multi-class
            classification workflow for a simulated next-cycle performance band.

            **Dataset provenance:** locally generated synthetic records; no
            public employee dataset or real employee information is used.
            """
        )
        st.markdown(f"**Target classes:** {', '.join(TARGET_CLASSES)}")
        st.markdown(
            f"**Selected model:** {metadata.get('selected_model', 'Unavailable — train the model first')}"
        )
        st.markdown(f"**Default model features:** {len(metadata.get('feature_columns', [])) or 'Unavailable'}")
    with right:
        if dataset is not None:
            st.metric("Synthetic records", f"{len(dataset):,}")
            st.metric("Dataset columns", len(dataset.columns))
        else:
            st.warning("The generated dataset is missing. Generate it before using the explorer or prediction form.")
        st.warning(
            "Synthetic-data notice: this model learns patterns in a rule-based generator, "
            "not validated workplace relationships. Never use its output for hiring, firing, "
            "promotion, compensation, discipline, or other consequential decisions."
        )
    st.markdown("**Important limitations:** synthetic target labels, a single random holdout, possible proxy effects, and no validation on real organizations.")
    if evaluation:
        st.markdown("#### Actual held-out evaluation")
        overview_rows = [
            {
                "Model": name,
                "Accuracy": values["accuracy"],
                "Macro precision": values["precision"],
                "Macro recall": values["recall"],
                "Macro-F1": values["macro_f1"],
            }
            for name, values in evaluation.items()
        ]
        st.dataframe(pd.DataFrame(overview_rows).set_index("Model"), width="stretch")

with data_tab:
    st.subheader("Data explorer")
    uploaded = st.file_uploader("Optional CSV upload (read as data only; uploaded content is never executed)", type=["csv"])
    selected_data = dataset
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            validation = validate_dataset(uploaded_df)
            if validation["is_valid"]:
                selected_data = uploaded_df
                st.success("Uploaded CSV passed the project's complete dataset schema checks.")
            else:
                st.error("Uploaded CSV is not compatible with this project dataset schema.")
                st.write("Correct the reported issues, or use the generated project dataset:")
                for issue in validation["issues"]:
                    st.write(f"- {issue}")
                st.info(
                    "The upload was not used. Prediction requires the exact trained feature schema; "
                    "do not remove or rename columns to force compatibility."
                )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not parse the uploaded CSV: {exc}")

    if selected_data is not None:
        st.write(f"Dimensions: **{selected_data.shape[0]:,} rows × {selected_data.shape[1]} columns**")
        st.dataframe(selected_data.head(20), width="stretch")
        st.markdown("#### Column descriptions")
        descriptions = {
            "employee_id": "Synthetic row identifier; excluded from model inputs.",
            "department": "Fictional functional department.",
            "job_level": "Fictional role level.",
            "perf_band_next": "Rule-generated simulated category: High, Medium, or Low.",
            "gender": "Synthetic audit-only field; excluded from default prediction features.",
            "salary_percentile_band": "Synthetic salary band; excluded from default prediction features.",
        }
        st.dataframe(
            pd.DataFrame(
                {
                    "column": selected_data.columns,
                    "description": [descriptions.get(column, "Synthetic measure; see docs/data_dictionary.md.") for column in selected_data.columns],
                    "dtype": [str(dtype) for dtype in selected_data.dtypes],
                }
            ),
            width="stretch",
        )
        missing = selected_data.isna().sum().loc[lambda counts: counts > 0].rename("missing_count")
        st.markdown("#### Data quality")
        quality_left, quality_right = st.columns(2)
        with quality_left:
            st.write("Missing-value counts")
            st.dataframe(missing.to_frame() if not missing.empty else pd.DataFrame({"status": ["No missing values"]}))
        with quality_right:
            st.metric("Duplicate records", int(selected_data.duplicated().sum()))
        validation = validate_dataset(selected_data)
        if validation["is_valid"]:
            st.success("Dataset validation passed.")
        else:
            st.error("Dataset validation found issues:")
            for issue in validation["issues"]:
                st.write(f"- {issue}")

with eda_tab:
    st.subheader("Exploratory data analytics")
    if dataset is not None:
        st.plotly_chart(
            px.histogram(
                dataset,
                x=TARGET_COLUMN,
                category_orders={TARGET_COLUMN: TARGET_CLASSES},
                title="Synthetic target class distribution",
            ),
            width="stretch",
        )
        show_image("department_target_distribution.png", "Target-category counts by department")
        show_image("training_hours_vs_target.png", "Training hours by generated category")
        show_image("on_time_delivery_vs_target.png", "Delivery rate by generated category")
        show_image("work_quality_features.png", "Synthetic quality-related measures by category")
        show_image("correlation_heatmap.png", "Numeric feature correlations; correlation does not imply causation")
        with st.expander("More generated EDA charts"):
            for filename, caption in [
                ("missing_values.png", "Missing-value counts"),
                ("duplicate_records.png", "Duplicate-record summary"),
                ("numeric_feature_distributions.png", "Numeric feature distributions"),
                ("feature_association.png", "Feature association chart"),
                ("job_level_analysis.png", "Job level distribution by target"),
            ]:
                show_image(filename, caption)
    else:
        st.info("Generate the synthetic dataset and run EDA to display charts.")

with evaluation_tab:
    st.subheader("Model evaluation")
    if evaluation:
        comparison_rows = []
        for name, values in evaluation.items():
            comparison_rows.append(
                {
                    "Model": name,
                    "Accuracy": values["accuracy"],
                    "Macro precision": values["precision"],
                    "Macro recall": values["recall"],
                    "Macro-F1": values["macro_f1"],
                }
            )
        st.dataframe(pd.DataFrame(comparison_rows).set_index("Model"), width="stretch")
        show_image("model_comparison_macro_f1.png", "Actual model comparison on the fixed held-out test split")
        model_names = list(evaluation)
        selected_eval_model = st.selectbox("Confusion matrix and class report", model_names, key="evaluation_model")
        model_metrics = evaluation[selected_eval_model]
        st.write(
            f"Accuracy: **{model_metrics['accuracy']:.3f}** · "
            f"Macro precision: **{model_metrics['precision']:.3f}** · "
            f"Macro recall: **{model_metrics['recall']:.3f}** · "
            f"Macro-F1: **{model_metrics['macro_f1']:.3f}**"
        )
        st.dataframe(
            pd.DataFrame(
                model_metrics["confusion_matrix"],
                index=["Actual Low", "Actual Medium", "Actual High"],
                columns=["Predicted Low", "Predicted Medium", "Predicted High"],
            ),
            width="stretch",
        )
        class_rows = [
            {"Class": label, **values}
            for label, values in model_metrics["per_class"].items()
            if label in TARGET_CLASSES
        ]
        st.dataframe(pd.DataFrame(class_rows).set_index("Class"), width="stretch")
        classification_path = REPORT_DIR / "classification_report.txt"
        if classification_path.is_file():
            with st.expander("Saved classification report"):
                st.code(classification_path.read_text(encoding="utf-8"))
        st.caption(
            "Metrics describe one stratified random holdout from synthetic data. "
            "Macro averages weight each class equally; small classes and generated labels limit interpretation."
        )
    else:
        st.info("Evaluation results are not available. Run model training and evaluation.")

with prediction_tab:
    st.subheader("Performance prediction")
    st.warning(
        "Use fictional example inputs only. Do not enter confidential employee information. "
        "The output is a synthetic simulation and requires human review."
    )
    if model is None:
        st.error("Saved model pipeline is unavailable. Train it with `python src\\train_model.py`.")
    elif dataset is None:
        st.error("The generated dataset is required to prepare example input ranges.")
    else:
        expected_features = list(getattr(model, "feature_names_in_", []))
        if not expected_features:
            st.error("The saved model does not expose the expected feature schema.")
        else:
            form_row: dict[str, object] = {}
            input_left, input_right = st.columns(2)
            for index, feature in enumerate(expected_features):
                source = dataset[feature]
                with (input_left if index % 2 == 0 else input_right):
                    if pd.api.types.is_numeric_dtype(source):
                        value = float(source.median())
                        minimum = float(source.min())
                        maximum = float(source.max())
                        if minimum == maximum:
                            minimum, maximum = minimum - 1.0, maximum + 1.0
                        if pd.api.types.is_integer_dtype(source):
                            form_row[feature] = st.number_input(
                                feature.replace("_", " ").title(),
                                min_value=int(minimum),
                                max_value=int(maximum),
                                value=int(round(value)),
                                step=1,
                            )
                        else:
                            form_row[feature] = st.number_input(
                                feature.replace("_", " ").title(),
                                min_value=minimum,
                                max_value=maximum,
                                value=min(max(value, minimum), maximum),
                                step=0.1,
                                format="%.2f",
                            )
                    else:
                        choices = sorted(source.dropna().astype(str).unique().tolist())
                        if choices:
                            form_row[feature] = st.selectbox(
                                feature.replace("_", " ").title(), choices
                            )
                        else:
                            st.error(f"No valid example categories are available for {feature}.")

            with st.form("prediction_form"):
                request_prediction = st.form_submit_button("Generate simulated prediction")
            with st.expander("Recommendation and explanation options"):
                include_development_actions = st.checkbox(
                    "Show optional development suggestions", value=True
                )

            if request_prediction:
                try:
                    result = predict_performance(form_row, MODEL_PATH)
                    importance_path = REPORT_DIR / "permutation_importance.csv"
                    importance_rows = (
                        pd.read_csv(importance_path).to_dict(orient="records")
                        if importance_path.is_file()
                        else []
                    )
                    summary = build_development_summary(
                        str(result["prediction"]),
                        form_row,
                        importance_rows,
                        result.get("probabilities"),
                        review_date=date.today() + timedelta(days=30),
                    )
                    st.success(f"Predicted simulated category: **{summary['predicted_simulated_category']}**")
                    if result.get("probabilities"):
                        st.markdown("**Model probability estimates (not certainty):**")
                        st.bar_chart(pd.Series(result["probabilities"], name="Model probability"))
                    st.markdown("**Relevant provided input signals**")
                    st.dataframe(pd.DataFrame(summary["relevant_input_signals"]), width="stretch")
                    if summary["top_global_model_related_features"]:
                        st.markdown("**Top global model-related features**")
                        st.caption("These are dataset-level associations, not explanations of this individual result.")
                        st.dataframe(pd.DataFrame(summary["top_global_model_related_features"]), width="stretch")
                    if include_development_actions:
                        st.markdown("**Possible development actions for human discussion**")
                        for action in summary["development_actions"]:
                            st.markdown(f"- **{action['action']}**  \n  Why suggested: {action['reason']}")
                    st.write(f"Suggested review date: **{summary['suggested_review_date']}**")
                    st.info(summary["uncertainty_and_disclaimer"]["notice"])
                except (ValueError, TypeError, FileNotFoundError, OSError) as exc:
                    st.error(f"Prediction could not be generated: {exc}")

with explainability_tab:
    st.subheader("Global explainability")
    importance_path = REPORT_DIR / "permutation_importance.csv"
    if importance_path.is_file():
        try:
            importance = pd.read_csv(importance_path)
            show_image(
                "permutation_feature_importance.png",
                "Features whose shuffling changes held-out macro-F1 the most",
            )
            st.dataframe(importance, width="stretch")
            st.markdown(
                """
                **How to read this:** permutation importance measures how much the fitted model's
                held-out macro-F1 changes when one input column is shuffled. A larger positive
                value indicates greater reliance by this model on that feature in this evaluation
                sample. It is a **global** summary, not an explanation for an individual prediction.
                Feature importance does not prove causation; correlated inputs, synthetic labels,
                and the selected test sample can affect the ranking.
                """
            )
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Could not read permutation-importance results: {exc}")
    else:
        st.info("Global feature importance is unavailable. Run `python src\\evaluate_model.py` after training.")

with fairness_tab:
    st.subheader("Fairness audit")
    if fairness:
        st.warning(fairness.get("data_notice", "Descriptive group audit; not proof of fairness."))
        st.caption(
            f"Metrics are withheld for groups smaller than {fairness.get('minimum_group_size', 'the configured minimum')} records."
        )
        for field, audit in fairness.get("audits", {}).items():
            st.markdown(f"#### {field.replace('_', ' ').title()}")
            if audit.get("status") == "unavailable":
                st.info(audit.get("message", "Audit data is unavailable."))
                continue
            groups = audit.get("groups", [])
            count_rows = [{"Group": group["group"], "Records": group["count"], "Status": group["status"]} for group in groups]
            st.dataframe(pd.DataFrame(count_rows), width="stretch")
            metric_rows = [
                {
                    "Group": group["group"],
                    **{
                        key: group[key]
                        for key in (
                            "predicted_high_rate",
                            "macro_precision",
                            "macro_recall",
                            "macro_f1",
                            "high_precision",
                            "high_recall",
                            "high_false_positive_rate",
                            "high_false_negative_rate",
                        )
                        if key in group
                    },
                }
                for group in groups
                if group.get("status") == "reported"
            ]
            if metric_rows:
                st.dataframe(pd.DataFrame(metric_rows).set_index("Group"), width="stretch")
            for group in groups:
                if group.get("warning"):
                    st.warning(f"{group['group']}: {group['warning']}")
                elif group.get("status") == "reported":
                    with st.expander(f"{group['group']} confusion matrix"):
                        st.dataframe(
                            pd.DataFrame(
                                group["confusion_matrix"],
                                index=[f"Actual {label}" for label in TARGET_CLASSES],
                                columns=[f"Predicted {label}" for label in TARGET_CLASSES],
                            ),
                            width="stretch",
                        )
            st.write("High-category error-rate differences (max minus min):", audit.get("error_rate_differences", {}))
        st.markdown(
            "Group rates are descriptive and depend on synthetic labels, sample size, and group definitions. "
            "Similar values do not prove fairness, and differences do not identify causes. "
            "No group-specific thresholds or mitigation are applied."
        )
    else:
        st.info("Fairness results are unavailable. Run the model evaluation workflow.")

with responsible_tab:
    st.subheader("Responsible AI and limitations")
    st.markdown(
        """
        - **Synthetic data:** all records and labels are generated; the model learns generator patterns,
          not validated workplace relationships.
        - **Uncertainty:** probabilities are model estimates, not guarantees or calibrated appraisal outcomes.
        - **Bias:** synthetic groups can still show unequal error rates; excluding sensitive inputs does not
          remove proxy effects or prove fairness.
        - **Leakage:** use only the defined input features; never include future outcomes or the target itself.
        - **Privacy:** do not enter actual employee data, personal information, or confidential company records.
        - **Human oversight:** development suggestions are optional prompts for respectful, human-led discussion.
        - **Prohibited uses:** do not use this educational project for hiring, firing, promotion, compensation,
          disciplinary action, employee ranking, or any other consequential employment decision.
        """
    )
    st.caption("A model's prediction is neither a diagnosis nor a measure of a person's worth.")

st.divider()
st.caption("Local portfolio demonstration · synthetic data only · model training is not run by dashboard refresh.")
