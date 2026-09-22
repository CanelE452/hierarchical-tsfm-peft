# HOLD_NO_AUTO_RESCUE

{
  "status": "HOLD_NO_AUTO_RESCUE",
  "base_outcome": {
    "status": "GO_STANDARD_ONLY",
    "relative_quality_loss_pct": 0.4291868409387334,
    "total_time_saved_pct": -63.146444823943604,
    "response_gain_pct": -0.015852187083378603,
    "quality_ok": true,
    "total_time_ok": false,
    "response_added_value": false,
    "dominating_controls": [
      "F_TOP3",
      "VALUE_DELTA"
    ],
    "scope": "one-development-panel, two seeds, fixed budget; not inferential equivalence"
  },
  "reasons": [
    "dominating control meets mean criteria but not the per-seed test"
  ],
  "standard_controls_meeting_goal": [],
  "timing_hold": false,
  "noisy_fits": [],
  "seed_throughput_gap": {
    "F_FULL": 0.015784482402920755,
    "F_TOP3": 0.023999057834023146,
    "E_EMLOC": 0.0152790169050469,
    "E_DELTA": 0.027495587469598393,
    "E_VALUE_DELTA": 0.003049688445697773,
    "E_RESPONSE_DELTA": 0.019575634317722745
  },
  "quality_failure": false,
  "emloc_baseline": "LIMITED_BASELINE",
  "limited_budget": {
    "F_FULL_92711": false,
    "F_FULL_92712": true,
    "F_TOP3_92711": true,
    "F_TOP3_92712": true,
    "E_EMLOC_92711": false,
    "E_EMLOC_92712": true,
    "E_DELTA_92711": false,
    "E_DELTA_92712": true,
    "E_VALUE_DELTA_92711": false,
    "E_VALUE_DELTA_92712": true,
    "E_RESPONSE_DELTA_92711": false,
    "E_RESPONSE_DELTA_92712": true
  },
  "pareto": [
    {
      "arm": "F_FULL",
      "mean_test_primary": 0.17577917523128148,
      "mean_total_s": 193.4713630999613,
      "pareto_optimal": true,
      "dominated_by": [],
      "test_by_seed": [
        0.17519769816205782,
        0.17636065230050513
      ],
      "total_by_seed": [
        195.02660029975232,
        191.9161259001703
      ],
      "meets_standard_criteria": null
    },
    {
      "arm": "F_TOP3",
      "mean_test_primary": 0.1765057012199715,
      "mean_total_s": 87.3881500994612,
      "pareto_optimal": true,
      "dominated_by": [],
      "test_by_seed": [
        0.1761593021061516,
        0.17685210033379142
      ],
      "total_by_seed": [
        88.68541909992928,
        86.09088109899312
      ],
      "meets_standard_criteria": false
    },
    {
      "arm": "E_EMLOC",
      "mean_test_primary": 0.17677985437342558,
      "mean_total_s": 233.54165740017197,
      "pareto_optimal": false,
      "dominated_by": [
        "F_FULL",
        "F_TOP3"
      ],
      "test_by_seed": [
        0.17626310955364302,
        0.17729659919320817
      ],
      "total_by_seed": [
        232.2488605004619,
        234.83445429988205
      ],
      "meets_standard_criteria": false
    },
    {
      "arm": "E_DELTA",
      "mean_test_primary": 0.1765588884959493,
      "mean_total_s": 303.6201813995722,
      "pareto_optimal": false,
      "dominated_by": [
        "F_FULL",
        "F_TOP3"
      ],
      "test_by_seed": [
        0.17578175298139995,
        0.17733602401049864
      ],
      "total_by_seed": [
        300.46366059948923,
        306.7767021996551
      ],
      "meets_standard_criteria": false
    },
    {
      "arm": "E_VALUE_DELTA",
      "mean_test_primary": 0.17650561631997305,
      "mean_total_s": 314.04403545038076,
      "pareto_optimal": false,
      "dominated_by": [
        "F_FULL"
      ],
      "test_by_seed": [
        0.17568137312409707,
        0.17732985951584904
      ],
      "total_by_seed": [
        312.936262400588,
        315.1518085001735
      ],
      "meets_standard_criteria": false
    },
    {
      "arm": "E_RESPONSE_DELTA",
      "mean_test_primary": 0.17653359632048476,
      "mean_total_s": 315.64165065001,
      "pareto_optimal": false,
      "dominated_by": [
        "F_FULL",
        "F_TOP3",
        "E_VALUE_DELTA"
      ],
      "test_by_seed": [
        0.1759661138997106,
        0.17710107874125894
      ],
      "total_by_seed": [
        313.9845670006471,
        317.2987342993729
      ],
      "meets_standard_criteria": false
    }
  ],
  "candidate": "E_RESPONSE_DELTA",
  "thresholds": {
    "quality_margin": 0.005,
    "min_total_time_saving": 0.2,
    "min_response_gain": 0.003,
    "timing_cv": 0.2,
    "seed_throughput": 0.2
  },
  "scope": "Jena only, two main seeds sharing one compression and one calibration per kind, fixed 128 updates; not significance/equivalence, not paper PASS",
  "automatic_follow_up": false,
  "created_utc": 1790087409.2001092
}
