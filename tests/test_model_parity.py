from hier_peft.smoke import run_smoke


def test_actual_labour_model_parity_and_all_arm_smoke():
    result = run_smoke()
    assert result["status"] == "PASS"
    assert len(result["initialization_checks"]) == 12
    assert len(result["optimizer_checks"]) == 4
