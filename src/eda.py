from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, TARGET_COLUMN
from src.data_validation import assert_valid_dataset


def generate_eda_report(data_path: str | Path, output_dir: str | Path = FIGURE_DIR) -> list[str]:
    """Generate a full EDA report and save images in the figure directory."""
    df = pd.read_csv(data_path)
    assert_valid_dataset(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    plt.figure(figsize=(6, 4))
    plt.title("Dataset overview")
    plt.text(0.02, 0.5, f"Rows: {len(df)}\nColumns: {len(df.columns)}\nTarget: {TARGET_COLUMN}", fontsize=12)
    plt.axis("off")
    path = output_dir / "dataset_overview.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.index, y=missing.values, color="#3977a8")
    plt.title("Missing-value analysis")
    plt.xticks(rotation=45)
    plt.ylabel("Missing values")
    plt.tight_layout()
    path = output_dir / "missing_values.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    dup_count = df.duplicated().sum()
    plt.figure(figsize=(4, 4))
    plt.bar(["Duplicate rows", "Unique rows"], [dup_count, len(df) - dup_count], color=["#d62728", "#2ca02c"])
    plt.title("Duplicate-record analysis")
    plt.ylabel("Row count")
    plt.tight_layout()
    path = output_dir / "duplicate_records.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        order=["High", "Medium", "Low"],
        hue_order=["High", "Medium", "Low"],
        palette="Set2",
        legend=False,
    )
    plt.title("Target class distribution")
    plt.xlabel("Performance band")
    plt.ylabel("Count")
    plt.tight_layout()
    path = output_dir / "target_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "employee_id"]
    numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
    if numeric_cols:
        num_cols = min(4, len(numeric_cols))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, col in enumerate(numeric_cols[:4]):
            ax = axes[idx // 2, idx % 2]
            sns.histplot(data=df, x=col, bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        path = output_dir / "numeric_feature_distributions.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths.append(str(path))

    plt.figure(figsize=(9, 6))
    sns.countplot(data=df, x="department", hue=TARGET_COLUMN, palette="viridis")
    plt.title("Department-wise class distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = output_dir / "department_target_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="training_hours",
        hue=TARGET_COLUMN,
        order=["High", "Medium", "Low"],
        hue_order=["High", "Medium", "Low"],
        palette="Set2",
        dodge=False,
        legend=False,
    )
    plt.title("Training hours vs target category")
    plt.tight_layout()
    path = output_dir / "training_hours_vs_target.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="on_time_delivery_rate",
        hue=TARGET_COLUMN,
        order=["High", "Medium", "Low"],
        hue_order=["High", "Medium", "Low"],
        palette="Set2",
        dodge=False,
        legend=False,
    )
    plt.title("On-time delivery vs target category")
    plt.tight_layout()
    path = output_dir / "on_time_delivery_vs_target.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    quality_cols = ["code_review_score", "peer_feedback_score", "manager_score", "qa_defect_density"]
    selected = df[quality_cols + [TARGET_COLUMN]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(quality_cols):
        ax = axes[i // 2, i % 2]
        sns.boxplot(data=selected, x=TARGET_COLUMN, y=col, order=["High", "Medium", "Low"], ax=ax)
        ax.set_title(f"{col} by target")
    fig.tight_layout()
    path = output_dir / "work_quality_features.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved_paths.append(str(path))

    corr_cols = [
        "experience_years",
        "projects_count",
        "on_time_delivery_rate",
        "code_review_score",
        "peer_feedback_score",
        "manager_score",
        "training_hours",
        "story_points_completed",
        "kudos_count",
    ]
    corr_df = df[corr_cols].copy()
    corr = corr_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation heatmap for numeric features")
    plt.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    cat_cols = ["department", "job_level", "salary_percentile_band", "gender"]
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col, hue=TARGET_COLUMN, palette="pastel")
            plt.title(f"{col} feature analysis")
            plt.xticks(rotation=20)
            plt.tight_layout()
            path = output_dir / f"{col}_analysis.png"
            plt.savefig(path, dpi=200)
            plt.close()
            saved_paths.append(str(path))

    # A simple association metric suitable for beginner-friendly EDA.
    mi_df = df[["on_time_delivery_rate", "manager_score", "peer_feedback_score", "training_hours", TARGET_COLUMN]].copy()
    mi_df[TARGET_COLUMN] = mi_df[TARGET_COLUMN].map({"Low": 0, "Medium": 1, "High": 2})
    plt.figure(figsize=(7, 5))
    sns.heatmap(mi_df.corr(), annot=True, cmap="YlGnBu")
    plt.title("Feature association analysis")
    plt.tight_layout()
    path = output_dir / "feature_association.png"
    plt.savefig(path, dpi=200)
    plt.close()
    saved_paths.append(str(path))

    summary = [
        "AI-Powered Employee Performance Predictor EDA Summary",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing values: {df.isna().sum().sum()}",
        f"Duplicate rows: {df.duplicated().sum()}",
        "Target labels are synthetic and intended for educational classification, not real HR decisions.",
        "The synthetic generator uses measurable work-quality signals such as delivery rate, manager feedback, and training hours.",
        "Correlation does not imply causation; these plots highlight patterns in the generated data.",
    ]
    report_path = output_dir.parent / "eda_summary.txt"
    report_path.write_text("\n".join(summary), encoding="utf-8")
    return saved_paths


if __name__ == "__main__":
    generate_eda_report("data/raw/synthetic_employee_data.csv")
