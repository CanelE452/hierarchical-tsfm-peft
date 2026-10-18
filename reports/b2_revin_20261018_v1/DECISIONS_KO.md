# 선택·확장 결정과 범위

첫 비교는 기존과 같은 0/.25/1/3 epoch checkpoint다. 데이터의 OFF/ON×두 seed 중 하나라도 마지막 checkpoint가 strict best이면 해당 네 arm 모두 4/5/6 epoch까지 확인한다. 원 optimizer/sampler/CPU·CUDA RNG의 immutable continuation을 사용하고 마지막 선택 뒤 fresh A60을 수행한다.

공통 정규화는 데이터별 두 seed의 A_stop best MSE 평균으로 선택하며 동률 OFF다. 같은 정규화·장치·선택 규칙의 기존 B0가 없으면 Jena B0 두 seed를 추가한다.

실제 봉인 결정은 아래 입력과 DECISIONS.json의 source binding을 따른다.

```json
{
  "V_used_for_selection": false,
  "decisions": [
    {
      "V_used": false,
      "astop_two_seed_means": {
        "OFF": 0.40070097337786215,
        "ON": 0.43064743113484616
      },
      "cap": 3,
      "cohort": "jena",
      "common_revin": false,
      "extend_all_to6": false,
      "matched_arms": [
        "B2_jena_REVIN_OFF_s92601",
        "B2_jena_REVIN_OFF_s92602",
        "B2_jena_REVIN_ON_s92601",
        "B2_jena_REVIN_ON_s92602"
      ],
      "selection_split": "A_stop",
      "selections": {
        "B2_jena_REVIN_OFF_s92601": {
          "astop_mse": 0.4076374214888158,
          "checkpoint": {
            "reference": "repo://.cache/adapter_diagnostics_20261017_v2/runs/init_PCA_s92601/search/checkpoint_000003.pt",
            "sha256": "067a060b29a46b57de42cda664957bdf780233b6607eb3522d24a4633a036da2"
          },
          "epoch": 1.0
        },
        "B2_jena_REVIN_OFF_s92602": {
          "astop_mse": 0.3937645252669084,
          "checkpoint": {
            "reference": "repo://.cache/adapter_diagnostics_20261017_v2/runs/init_PCA_s92602/search/checkpoint_000003.pt",
            "sha256": "a40064d4be52a940e885536a3334ec56f3f286d1422506afb4f4c72fb3c07d8c"
          },
          "epoch": 1.0
        },
        "B2_jena_REVIN_ON_s92601": {
          "astop_mse": 0.442341770030015,
          "checkpoint": {
            "reference": "cache://runs/B2_jena_REVIN_ON_s92601/cap3/search/checkpoint_000003.pt",
            "sha256": "8dea780149f0783ad5a5c4ea39fc1b3d4248c92d6e055db157eff4940fefb6c4"
          },
          "epoch": 1.0
        },
        "B2_jena_REVIN_ON_s92602": {
          "astop_mse": 0.41895309223967725,
          "checkpoint": {
            "reference": "cache://runs/B2_jena_REVIN_ON_s92602/cap3/search/checkpoint_000003.pt",
            "sha256": "b699c6c3d994ac8b66730e0638ff19f5945f8f45cf8da707145c88a4ced22fdc"
          },
          "epoch": 1.0
        }
      },
      "status": "SEALED",
      "still_improving_at_cap": false
    },
    {
      "V_used": false,
      "astop_two_seed_means": {
        "OFF": 0.08843514437873742,
        "ON": 0.08898080131320138
      },
      "cap": 3,
      "cohort": "pems32",
      "common_revin": false,
      "extend_all_to6": true,
      "matched_arms": [
        "B2_pems32_REVIN_OFF_s92601",
        "B2_pems32_REVIN_OFF_s92602",
        "B2_pems32_REVIN_ON_s92601",
        "B2_pems32_REVIN_ON_s92602"
      ],
      "selection_split": "A_stop",
      "selections": {
        "B2_pems32_REVIN_OFF_s92601": {
          "astop_mse": 0.08840914691832405,
          "checkpoint": {
            "reference": "repo://.cache/adapter_improvements_20261018_v1/runs/B2_PCA_PEMS32_s92601/search/checkpoint_000003.pt",
            "sha256": "93eed90ab31c6c4bbe9006b95a2b10a074ba7da308fcc1869525d3f6c123943b"
          },
          "epoch": 3.0
        },
        "B2_pems32_REVIN_OFF_s92602": {
          "astop_mse": 0.0884611418391508,
          "checkpoint": {
            "reference": "repo://.cache/adapter_improvements_20261018_v1/runs/B2_PCA_PEMS32_s92602/search/checkpoint_000003.pt",
            "sha256": "fc155c3e9f1a876cffbeb2a6a5d6481b92901f778480288c8d6ce16e28dd765b"
          },
          "epoch": 3.0
        },
        "B2_pems32_REVIN_ON_s92601": {
          "astop_mse": 0.08868005251611974,
          "checkpoint": {
            "reference": "cache://runs/B2_pems32_REVIN_ON_s92601/cap3/search/checkpoint_000003.pt",
            "sha256": "39458bddbaa6279ff6272b3e597df3df655f08fcae947ee7d24b3d4932328ee3"
          },
          "epoch": 3.0
        },
        "B2_pems32_REVIN_ON_s92602": {
          "astop_mse": 0.08928155011028303,
          "checkpoint": {
            "reference": "cache://runs/B2_pems32_REVIN_ON_s92602/cap3/search/checkpoint_000003.pt",
            "sha256": "f381e704bdaf50bd6c7ff56fad38fa81367c04391cacfa3f00e0d0c49a3a04d0"
          },
          "epoch": 3.0
        }
      },
      "status": "SEALED",
      "still_improving_at_cap": true
    },
    {
      "V_used": false,
      "astop_two_seed_means": {
        "OFF": 0.08699562236068659,
        "ON": 0.08717517539067463
      },
      "cap": 6,
      "cohort": "pems32",
      "common_revin": false,
      "extend_all_to6": false,
      "matched_arms": [
        "B2_pems32_REVIN_OFF_s92601",
        "B2_pems32_REVIN_OFF_s92602",
        "B2_pems32_REVIN_ON_s92601",
        "B2_pems32_REVIN_ON_s92602"
      ],
      "selection_split": "A_stop",
      "selections": {
        "B2_pems32_REVIN_OFF_s92601": {
          "astop_mse": 0.08673066786541876,
          "checkpoint": {
            "reference": "cache://runs/B2_pems32_REVIN_OFF_s92601/cap6/search/checkpoint_000003.pt",
            "sha256": "102e732886cb7bbe2a8cc3111311595edc5ca654e06a5992b6655534f514672e"
          },
          "epoch": 6.0
        },
        "B2_pems32_REVIN_OFF_s92602": {
          "astop_mse": 0.0872605768559544,
          "checkpoint": {
            "reference": "cache://runs/B2_pems32_REVIN_OFF_s92602/cap6/search/checkpoint_000003.pt",
            "sha256": "272a9a070dbfe9ee5bed382a668e3791dff80baa06584f8a060d309486b97fdf"
          },
          "epoch": 6.0
        },
        "B2_pems32_REVIN_ON_s92601": {
          "astop_mse": 0.08732932114821022,
          "checkpoint": {
            "reference": "cache://runs/B2_pems32_REVIN_ON_s92601/cap6/search/checkpoint_000003.pt",
            "sha256": "61a8a5d88ec3fce01cffa1660bb86727bdcc35c28246fbbfe89fb907692988d5"
          },
          "epoch": 5.0
        },
        "B2_pems32_REVIN_ON_s92602": {
          "astop_mse": 0.08702102963313904,
          "checkpoint": {
            "reference": "cache://runs/B2_pems32_REVIN_ON_s92602/cap6/search/checkpoint_000003.pt",
            "sha256": "7100eca50796dfa3f971ea815e31bb39167fb5b538085fd08b507973f8899499"
          },
          "epoch": 6.0
        }
      },
      "status": "SEALED",
      "still_improving_at_cap": true
    }
  ],
  "status": "COMPLETE"
}
```

SOFTS, FreTS, LIFT, N-HiTS, CycleNet/TimesNet, ForeCA, 전체 AdaPTS 비교와 추가 K·초기화 탐색은 제외했다. PeMS32 D는 이번 최종 발표 근거로 쓰지 않아 seed2 보완 조건이 발동하지 않는다.

V는 반복 개발 평가로 이미 사용한 자료이며 독립 테스트 결과가 아니다. 새 선택에 M_cal/T를 사용하지 않았으며, 기존 T 노출과 pretraining overlap UNKNOWN의 범위는 원 계약을 따른다. 두 seed·고정 K의 결과를 모든 데이터·압축률의 일반화로 확대하지 않는다.
