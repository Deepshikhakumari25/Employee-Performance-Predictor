from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, MODEL_PATH, RANDOM_SEED, REPORT_DIR, TARGET_COLUMN
from src.data_validation import assert_valid_dataset
from src.preprocessing import get_feature_columns


def calculate_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_repeats: int = 10,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Measure holdout macro-F1 change when each input feature is shuffled."""
    if not isinstance(X_test, pd.DataFrame) or X_test.empty:
        raise ValueError("Explainability requires a non-empty feature DataFrame.")
    if not isinstance(y_test, pd.Series) or len(X_test) != len(y_test):
        raise ValueError("Feature and target inputs must have the same number of rows.")
    if X_test.isna().all(axis=None):
        raise ValueError("Explainability input contains no usable feature values.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if list(X_test.columns) != list(getattr(model, "feature_names_in_", X_test.columns)):
        raise ValueError("Input feature names or order do not match the fitted model.")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_explanation_report(
    importance: pd.DataFrame,
    output_dir: str | Path = REPORT_DIR,
) -> tuple[str, str]:
    """Save the importance data and a plain-language interpretation."""
    required = {"feature", "importance_mean", "importance_std"}
    if not required.issubset(importance.columns) or importance.empty:
        raise ValueError(f"Importance report must include non-empty columns: {sorted(required)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "permutation_importance.csv"
    text_path = output_dir / "explainability_summary.txt"
    importance.to_csv(csv_path, index=False)

    strongest = importance.iloc[0]
    text_path.write_text(
        "\n".join(
            [
                "Global permutation importance (held-out data)",
                "",
                "Features are shuffled one at a time and the change in macro-F1 is measured. "
                "A larger positive value means the fitted model relied more on that input for "
                "this evaluation set.",
                f"Top feature by measured importance: {strongest['feature']} "
                f"(macro-F1 decrease {strongest['importance_mean']:.4f} "
                f"+/- {strongest['importance_std']:.4f} across shuffles).",
                "This is a global model-behavior summary, not an explanation of an individual "
                "prediction. Importance does not establish causation and may vary with the data.",
            ]
        ),
        encoding="utf-8",
    )
    return str(csv_path), str(text_path)


def save_importance_chart(
    importance: pd.DataFrame,
    output_dir: str | Path = FIGURE_DIR,
    top_n: int = 15,
) -> str:
    """Save a readable horizontal chart of the strongest global features."""
    if importance.empty or top_n < 1:
        raise ValueError("A non-empty importance table and positive top_n are required.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    displayed = importance.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(displayed))))
    ax.barh(
        displayed["feature"],
        displayed["importance_mean"],
        xerr=displayed["importance_std"],
        color="#3977a8",
        alpha=0.9,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Global permutation importance (macro-F1)")
    ax.set_xlabel("Decrease in held-out macro-F1 when feature is shuffled")
    ax.set_ylabel("Input feature")
    fig.tight_layout()
    path = output_dir / "permutation_feature_importance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def run_explainability(
    data_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    *,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Load the fitted model and calculate global importance on its holdout split."""
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
    importance = calculate_permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats
    )
    save_explanation_report(importance)
    save_importance_chart(importance)
    return importance


if __name__ == "__main__":
    run_explainability(PROJECT_ROOT / "data" / "raw" / "synthetic_employee_data.csv")
