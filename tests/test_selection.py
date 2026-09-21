import pytest
import json
from hier_peft.screen import select_candidate, continuation


def test_validation_ties_prefer_lower_lr_then_earlier_checkpoint():
    def item(value, lr, step):
        return dict(validation_primary=value, lr=lr, checkpoint=step, selection_role="VALIDATION")
    candidates = [item(1, 3e-4, 0), item(1, 1e-4, 128), item(1, 1e-4, 0), item(2, 1e-4, 0)]
    assert select_candidate(candidates) == candidates[2]
    candidates[0]["selection_role"] = "TEST"
    with pytest.raises(ValueError, match="VALIDATION"):
        select_candidate(candidates)


def test_gate_requires_both_repeats_and_pre_reconciliation_signal():
    rows = [dict(comparator=arm, seed=seed, reconciliation=rec, primary_difference=-0.01, bottom_relative_difference=0.0)
            for arm in ["LORA_SELF", "LORA_POOL"] for seed in [92110, 92111, 92112]
            for rec in ["mint_shrink", "unreconciled"]]
    assert continuation(rows)["decision"] == "PROCEED_TO_TOURISM_CONFIRMATION"
    bad = [dict(row) for row in rows]
    next(r for r in bad if r["seed"] == 92112 and r["reconciliation"] == "mint_shrink")["primary_difference"] = 0.001
    assert continuation(bad)["decision"] == "STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN"
    for row in rows:
        if row["reconciliation"] == "unreconciled":
            row["primary_difference"] = 0.01
    assert continuation(rows)["decision"] == "STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN"


def test_gate_round_trips_through_json_without_numpy_scalar_types():
    rows = [dict(comparator=arm, seed=seed, reconciliation=rec, primary_difference=-0.01, bottom_relative_difference=0.0)
            for arm in ["LORA_SELF", "LORA_POOL"] for seed in [92110, 92111, 92112]
            for rec in ["mint_shrink", "unreconciled"]]
    result = continuation(rows)
    assert all(type(value) is bool for value in result["checks"].values())
    assert json.loads(json.dumps(result)) == result
