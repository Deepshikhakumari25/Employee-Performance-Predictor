from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, MODEL_PATH, REPORT_DIR, TARGET_COLUMN
from src.data_validation import assert_valid_dataset
from src.preprocessing import get_feature_columns
from src.train_model import get_model_candidates, load_training_frame


def prepare_test_set(data_path: str | Path = "data/raw/synthetic_employee_data.csv"):
    """Load and split the data for evaluation."""
    from sklearn.model_selection import train_test_split

    X, y, numeric_columns, categorical_columns = load_training_frame(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_test, y_test, numeric_columns, categorical_columns


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate classification metrics for one model."""
    predictions = model.predict(X_test)
    per_class = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["Low", "Medium", "High"]).tolist(),
    }
    return metrics


def save_model_evaluation_report(results: dict, output_dir: str | Path = REPORT_DIR) -> dict:
    """Save the metric summary as JSON and text."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "model_evaluation.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report_lines = []
    for model_name, metrics in results.items():
        report_lines.append(f"Model: {model_name}")
        report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision (macro): {metrics['precision']:.4f}")
        report_lines.append(f"Recall (macro): {metrics['recall']:.4f}")
        report_lines.append(f"F1 (macro): {metrics['macro_f1']:.4f}")
        report_lines.append("Per-class metrics:")
        for label, values in metrics["per_class"].items():
            if label in {"Low", "Medium", "High"}:
                report_lines.append(
                    f"  {label}: precision={values.get('precision', 0):.4f}, "
                    f"recall={values.get('recall', 0):.4f}, "
                    f"f1={values.get('f1-score', 0):.4f}, support={values.get('support', 0)}"
                )
        report_lines.append("")

    text_path = output_dir / "classification_report.txt"
    text_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"json_path": str(json_path), "text_path": str(text_path)}


def generate_comparison_charts(results: dict, output_dir: str | Path = FIGURE_DIR) -> list[str]:
    """Produce confusion matrix and model comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for model_name, metrics in results.items():
        cm = metrics["confusion_matrix"]
        labels = ["Low", "Medium", "High"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion matrix for {model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        path = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        saved_paths.append(str(path))

    comparison = {
        model_name: metrics["macro_f1"] for model_name, metrics in results.items()
    }
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(comparison.keys()), y=list(comparison.values()), color="#3977a8")
    plt.title("Model comparison by macro-F1")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = output_dir / "model_comparison_macro_f1.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))
    return saved_paths


def run_evaluation_pipeline(data_path: str | Path = "data/raw/synthetic_employee_data.csv") -> dict:
    """Evaluate the baseline, logistic regression, and random forest models."""
    X, y, numeric_columns, categorical_columns = load_training_frame(data_path)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidate_models = get_model_candidates()
    results = {}
    for name, estimator in candidate_models.items():
        from src.preprocessing import build_model_pipeline

        pipeline = build_model_pipeline(estimator, numeric_columns, categorical_columns)
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)

    save_model_evaluation_report(results)
    generate_comparison_charts(results)

    from src.explainability import run_explainability
    from src.fairness_audit import run_fairness_audit

    run_explainability(data_path, MODEL_PATH)
    run_fairness_audit(data_path, MODEL_PATH)
    return results


if __name__ == "__main__":
    run_evaluation_pipeline()
