import json
from pathlib import Path

import pytest

from verify_projection_report_v1 import check_comparison, check_links


def fixture():
    score = {"mse": .25, "mae": .4, "active_channels": 2,
             "observed_counts": [2, 3], "per_channel_mse": [.1, .4], "per_channel_mae": [.2, .6]}
    result = {"datasets": {"synthetic": {"raw_scores": {"tune": {"normalized": score}}}}}
    row = {"dataset": "synthetic", "method": "RAW_BOLT", "seed": "", "split": "tune",
           "metric_scale": "normalized", **{key: str(value) if not isinstance(value, list) else json.dumps(value)
                                            for key, value in score.items()}}
    return result, row


def test_primary_csv_scalar_mutation_is_detected():
    result, row = fixture()
    assert check_comparison(result, [row]) == 9
    row["mse"] = ".125"
    with pytest.raises(ValueError, match="CSV_SCORE_MISMATCH"):
        check_comparison(result, [row])


def test_duplicate_and_changed_channel_vector_are_detected():
    result, row = fixture()
    with pytest.raises(ValueError, match="DUPLICATE_COMPARISON_ROW"):
        check_comparison(result, [row, row])
    row["observed_counts"] = "[3, 2]"
    with pytest.raises(ValueError, match="CSV_CHANNEL_VECTOR_MISMATCH"):
        check_comparison(result, [row])


def test_image_links_require_real_files(tmp_path):
    report = tmp_path / "REPORT_KO.md"
    report.write_text("![실제 그림](figure.png)\n[외부](https://example.com)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MISSING_LOCAL_REPORT_LINK"):
        check_links(report)
    (tmp_path / "figure.png").write_bytes(b"fixture")
    assert check_links(report) == ["figure.png"]
