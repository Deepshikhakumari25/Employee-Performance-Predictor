import json

import numpy as np
import pandas as pd
import pytest

from src.fairness_audit import fairness_audit, save_fairness_report


def audit_frame():
    return pd.DataFrame(
        {
            "gender": ["Woman", "Woman", "Man", "Man", None, "Non-binary"] * 5,
            "department": ["Engineering", "Sales", "Engineering", "Sales", "HR", "HR"] * 5,
            "job_level": ["Senior", "Associate", "Lead", "Manager", "Associate", "Lead"] * 5,
            "perf_band_next": ["High", "Low", "High", "Medium", "Low", "Medium"] * 5,
        }
    )


def test_fairness_audit_reports_group_metrics_and_missing_values():
    df = audit_frame()
    predictions = df["perf_band_next"].copy()
    result = fairness_audit(df, predictions, min_group_size=2, synthetic_data=True)

    assert result["status"] == "completed"
    assert result["data_notice"].startswith("Synthetic-data demonstration")
    gender_groups = result["audits"]["gender"]["groups"]
    assert any(group["group"] == "Missing" for group in gender_groups)
    reported = next(group for group in gender_groups if group["group"] == "Woman")
    assert reported["count"] == 10
    assert reported["macro_f1"] == 1.0
    assert len(reported["confusion_matrix"]) == 3
    assert "high_false_positive_rate_max_minus_min" in result["audits"]["gender"]["error_rate_differences"]


def test_small_groups_withhold_metrics_and_missing_audit_column_is_handled():
    df = audit_frame().iloc[:4]
    result = fairness_audit(
        df,
        df["perf_band_next"],
        protected_columns=("gender", "missing_attribute"),
        min_group_size=3,
        synthetic_data=True,
    )
    assert any(group["status"] == "insufficient_sample" for group in result["audits"]["gender"]["groups"])
    assert result["audits"]["missing_attribute"]["status"] == "unavailable"


def test_missing_target_classes_and_zero_error_denominators_are_safe():
    df = pd.DataFrame(
        {
            "gender": ["Woman"] * 25,
            "perf_band_next": ["High"] * 25,
        }
    )
    result = fairness_audit(df, np.array(["High"] * 25), synthetic_data=True)
    group = result["audits"]["gender"]["groups"][0]
    assert group["status"] == "reported"
    assert group["high_false_positive_rate"] is None
    assert group["high_false_negative_rate"] == 0.0
    assert group["actual_class_counts"] == {"High": 25, "Medium": 0, "Low": 0}


def test_invalid_dimensions_and_labels_are_rejected():
    df = audit_frame()
    with pytest.raises(ValueError, match="match the audit row count"):
        fairness_audit(df, ["High"])
    invalid = df["perf_band_next"].copy()
    invalid.iloc[0] = "Unknown"
    with pytest.raises(ValueError, match="invalid labels"):
        fairness_audit(df, invalid)


def test_fairness_report_is_valid_json(tmp_path):
    df = audit_frame()
    report = fairness_audit(
        df, df["perf_band_next"], min_group_size=2, synthetic_data=True
    )
    path = save_fairness_report(report, tmp_path)
    assert path.endswith("fairness_audit.json")
    assert json.loads(open(path, encoding="utf-8").read())["status"] == "completed"
