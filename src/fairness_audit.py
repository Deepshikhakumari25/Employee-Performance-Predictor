from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, RANDOM_SEED, REPORT_DIR, TARGET_COLUMN, TARGET_CLASSES
from src.data_validation import assert_valid_dataset
from src.preprocessing import get_feature_columns


def fairness_audit(
    df: pd.DataFrame,
    predictions: Sequence[str] | pd.Series | np.ndarray,
    protected_columns: Sequence[str] = ("gender", "department", "job_level"),
    target_col: str = TARGET_COLUMN,
    *,
    min_group_size: int = 20,
    synthetic_data: bool = False,
) -> dict:
    """Calculate descriptive group metrics; small groups are explicitly withheld."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Fairness audit requires a non-empty DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Fairness audit target column '{target_col}' is missing.")
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1.")

    prediction_values = np.asarray(predictions)
    if prediction_values.ndim != 1 or len(prediction_values) != len(df):
        raise ValueError("Predictions must be one-dimensional and match the audit row count.")
    if pd.isna(prediction_values).any():
        raise ValueError("Predictions cannot contain missing values.")
    observed = set(df[target_col].dropna().astype(str).unique())
    unknown_targets = observed - set(TARGET_CLASSES)
    unknown_predictions = set(map(str, np.unique(prediction_values))) - set(TARGET_CLASSES)
    if df[target_col].isna().any() or unknown_targets:
        raise ValueError(f"Audit target contains missing or invalid labels: {sorted(unknown_targets)}.")
    if unknown_predictions:
        raise ValueError(f"Predictions contain invalid labels: {sorted(unknown_predictions)}.")

    output = {
        "status": "completed",
        "data_notice": (
            "Synthetic-data demonstration; results do not establish real-world fairness."
            if synthetic_data
            else "Descriptive group metrics; results do not establish real-world fairness."
        ),
        "minimum_group_size": min_group_size,
        "classes": TARGET_CLASSES,
        "metrics_explanation": {
            "predicted_high_rate": "Share of group records predicted High; rate parity alone is not a fairness guarantee.",
            "precision_recall_f1": "Macro averages over the classes observed in that group; small groups are withheld.",
            "high_false_positive_rate": "Share of actual non-High records predicted High.",
            "high_false_negative_rate": "Share of actual High records predicted non-High.",
            "limitations": (
                "These metrics are descriptive, sensitive to group size and label quality, and do not capture "
                "all fairness concerns. Similar values do not prove the system is fair; differences do not "
                "explain their cause. Do not use them alone for consequential decisions."
            ),
        },
        "audits": {},
    }

    target_values = df[target_col].astype(str).to_numpy()
    prediction_values = prediction_values.astype(str)
    for column in protected_columns:
        if column not in df.columns:
            output["audits"][column] = {
                "status": "unavailable",
                "message": f"Audit field '{column}' is absent from the supplied data.",
                "groups": [],
                "error_rate_differences": {},
            }
            continue

        group_values = df[column].astype("string").fillna("Missing").to_numpy()
        groups = []
        for group_name in sorted(np.unique(group_values).tolist()):
            mask = group_values == group_name
            y_true = target_values[mask]
            y_pred = prediction_values[mask]
            count = int(mask.sum())
            group_result = {"group": group_name, "count": count}
            if count < min_group_size:
                group_result.update(
                    {
                        "status": "insufficient_sample",
                        "warning": (
                            f"Metrics withheld because n={count} is below the configured "
                            f"minimum of {min_group_size}; this is not evidence of equal performance."
                        ),
                    }
                )
            else:
                class_labels = [label for label in TARGET_CLASSES if label in set(y_true) | set(y_pred)]
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=class_labels,
                    output_dict=True,
                    zero_division=0,
                )
                high_actual = y_true == "High"
                non_high_actual = ~high_actual
                high_predicted = y_pred == "High"
                high_tp = int(np.sum(high_actual & high_predicted))
                high_fp = int(np.sum(non_high_actual & high_predicted))
                high_fn = int(np.sum(high_actual & ~high_predicted))
                high_tn = int(np.sum(non_high_actual & ~high_predicted))
                group_result.update(
                    {
                        "status": "reported",
                        "predicted_high_rate": float(np.mean(high_predicted)),
                        "macro_precision": float(report["macro avg"]["precision"]),
                        "macro_recall": float(report["macro avg"]["recall"]),
                        "macro_f1": float(report["macro avg"]["f1-score"]),
                        "high_precision": float(high_tp / (high_tp + high_fp)) if high_tp + high_fp else None,
                        "high_recall": float(high_tp / (high_tp + high_fn)) if high_tp + high_fn else None,
                        "high_false_positive_rate": float(high_fp / (high_fp + high_tn)) if high_fp + high_tn else None,
                        "high_false_negative_rate": float(high_fn / (high_fn + high_tp)) if high_fn + high_tp else None,
                        "actual_class_counts": {
                            label: int(np.sum(y_true == label)) for label in TARGET_CLASSES
                        },
                        "confusion_matrix": confusion_matrix(
                            y_true, y_pred, labels=TARGET_CLASSES
                        ).tolist(),
                    }
                )
            groups.append(group_result)

        def spread(metric: str) -> float | None:
            values = [
                group[metric]
                for group in groups
                if group.get("status") == "reported" and group.get(metric) is not None
            ]
            return float(max(values) - min(values)) if len(values) >= 2 else None

        output["audits"][column] = {
            "status": "completed",
            "groups": groups,
            "error_rate_differences": {
                "high_false_positive_rate_max_minus_min": spread("high_false_positive_rate"),
                "high_false_negative_rate_max_minus_min": spread("high_false_negative_rate"),
            },
        }
    return output


def save_fairness_report(
    summary: dict,
    output_dir: str | Path = REPORT_DIR,
) -> str:
    """Write fairness metrics as machine-readable JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fairness_audit.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def run_fairness_audit(
    data_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    *,
    min_group_size: int = 20,
) -> dict:
    """Audit the saved model against the deterministic held-out test rows."""
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Trained model file not found: {model_file}")
    df = pd.read_csv(data_path)
    assert_valid_dataset(df)
    numeric_columns, categorical_columns = get_feature_columns(df, TARGET_COLUMN)
    feature_columns = numeric_columns + categorical_columns
    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    model = joblib.load(model_file)
    predictions = model.predict(X_test)
    audit_rows = df.loc[X_test.index].reset_index(drop=True)
    audit_rows[TARGET_COLUMN] = y_test.reset_index(drop=True)
    summary = fairness_audit(
        audit_rows,
        predictions,
        min_group_size=min_group_size,
        synthetic_data=True,
    )
    save_fairness_report(summary)
    return summary


if __name__ == "__main__":
    run_fairness_audit(PROJECT_ROOT / "data" / "raw" / "synthetic_employee_data.csv")
