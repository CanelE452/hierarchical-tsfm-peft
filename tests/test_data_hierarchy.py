import numpy as np
import pandas as pd
import pytest

from hier_peft.data import DataFailure, materialize_hierarchical_dataset
from hier_peft.hierarchy import build_support_inclusion_graph


def grouped_summing_matrix():
    ids = ("Total", "Region_A", "Region_B", "Purpose_X", "Purpose_Y", "A_X", "A_Y", "B_X", "B_Y")
    bottom = ("A_X", "A_Y", "B_X", "B_Y")
    return pd.DataFrame(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        index=ids,
        columns=bottom,
    )


def synthetic_frames(n_times=180):
    S_df = grouped_summing_matrix()
    dates = pd.date_range("2000-01-01", periods=n_times, freq="MS")
    bottom_values = np.vstack([
        np.arange(n_times),
        np.arange(n_times) + 10,
        np.arange(n_times) + 20,
        np.arange(n_times) + 30,
    ]).astype(float)
    values = S_df.to_numpy(dtype=float) @ bottom_values
    rows = []
    for row_idx, unique_id in enumerate(S_df.index):
        for date_idx, date in enumerate(dates):
            rows.append({"unique_id": unique_id, "ds": date, "y": values[row_idx, date_idx]})
    tags = {
        "Total": np.array(["Total"]),
        "Region": np.array(["Region_A", "Region_B"]),
        "Purpose": np.array(["Purpose_X", "Purpose_Y"]),
        "Bottom": np.array(["A_X", "A_Y", "B_X", "B_Y"]),
    }
    return pd.DataFrame(rows), S_df, tags


def test_support_inclusion_graph_uses_multiple_direct_parents_without_name_parsing():
    graph = build_support_inclusion_graph(grouped_summing_matrix())
    idx = {series_id: pos for pos, series_id in enumerate(graph.ids)}
    assert set(graph.parents[idx["A_X"]]) == {idx["Region_A"], idx["Purpose_X"]}
    assert idx["Total"] not in graph.parents[idx["A_X"]]
    assert idx["A_X"] in graph.children[idx["Region_A"]]
    assert idx["A_X"] in graph.children[idx["Purpose_X"]]
    assert graph.root_indices == (idx["Total"],)
    assert len(graph.leaf_indices) == 4


def test_materialize_dataset_returns_integer_level_tags_and_valid_split():
    Y_df, S_df, tags = synthetic_frames()
    dataset = materialize_hierarchical_dataset("Synthetic", Y_df, S_df, tags)
    assert dataset.values.shape == (9, 180)
    assert dataset.S.shape == (9, 4)
    np.testing.assert_array_equal(dataset.bottom_indices, np.array([5, 6, 7, 8]))
    assert all(np.issubdtype(indices.dtype, np.integer) for indices in dataset.tags.values())
    assert dataset.split_manifest["origin_counts"]["CALIBRATION"] == 13
    assert dataset.split_manifest["origin_counts"]["VALIDATION"] == 13
    assert dataset.split_manifest["origin_counts"]["TEST"] == 37


def test_hierarchy_sum_mismatch_fails_closed_with_audit_payload():
    Y_df, S_df, tags = synthetic_frames()
    Y_df.loc[(Y_df["unique_id"] == "Total") & (Y_df["ds"] == pd.Timestamp("2000-01-01")), "y"] += 1
    with pytest.raises(DataFailure) as error:
        materialize_hierarchical_dataset("Synthetic", Y_df, S_df, tags)
    assert error.value.code == "BLOCKED_HIERARCHY_INCONSISTENT"
    audit = error.value.artifacts["data_audit"]
    assert audit["status"] == "FAIL"
    assert audit["summing_consistency"]["status"] == "FAIL"


def test_missing_month_in_one_series_fails_before_training_split():
    Y_df, S_df, tags = synthetic_frames()
    mask = (Y_df["unique_id"] == "B_Y") & (Y_df["ds"] == pd.Timestamp("2001-01-01"))
    Y_df = Y_df.loc[~mask].copy()
    with pytest.raises(DataFailure) as error:
        materialize_hierarchical_dataset("Synthetic", Y_df, S_df, tags)
    assert error.value.code == "Y_INCOMPLETE_COMMON_TIMELINE"
