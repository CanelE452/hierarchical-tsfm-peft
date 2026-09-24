"""Stage 1 — GIFT-Eval 리더보드에서 Chronos-2 zero-shot 대비 격차 채굴 (GPU 불필요, 재현용)

입력: GIFT-Eval 저장소 results/<model>/all_results.csv, config.json
      (git clone https://github.com/SalesforceAIResearch/gift-eval 후 RESULTS_DIR 지정)
출력: stage1_gift_eval_gap.csv (97개 설정 × 격차 지표)

격차 정의: g = (score_Chronos2 - score_ref) / score_Chronos2   (양수 = 참조가 더 좋음)
  gDL   : 데이터별 학습 딥러닝 10종 중 설정별 최솟값 (사후 최선 = 낙관적 참조)
          crossformer 제외 — 8개 설정에서 다른 모든 모델보다 50% 이상 낮은 비정상 값
          (예: solar/H/long CRPS 0.0049 vs 나머지 중앙값 0.415)
  gST   : 통계 기법 6종 중 설정별 최솟값
  gTTM  : TTM-R3 미세조정(FT) 대 사전학습(PT) — 같은 모델 계열의 미세조정 이득 (Chronos-2 기준 아님)
  gToto : Toto-2.0-2.5B 미세조정 대 기본 — 같은 모델 계열의 미세조정 이득
  제외: TurkForecast-FM-Chronos2-LoRA-v1 — 46개 설정에서 zero-shot과 완전히 같고,
        나머지에서 +97% ~ -309%의 비현실적 변동
  제외: testdata_leakage = "Yes" 인 제출 전부
지표: CRPS = eval_metrics/mean_weighted_sum_quantile_loss, MASE = eval_metrics/MASE[0.5]
"""
import os
import pandas as pd

RESULTS_DIR = os.environ.get("GIFT_RESULTS", "gift-eval/results")
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
MASE = "eval_metrics/MASE[0.5]"
DL = ["DLinear", "DLinear-t", "FFM", "N-BEATS", "PatchTST", "deepar",
      "iTransformer", "tft", "tide", "xLSTM-Mixer"]            # crossformer 제외
ST = ["auto_arima", "auto_ets", "auto_theta", "seasonal_naive", "naive", "FLAIR"]


def load(m):
    return pd.read_csv(os.path.join(RESULTS_DIR, m, "all_results.csv")).set_index("dataset")


def main():
    c2 = load("chronos-2")
    idx = c2.index
    t = pd.DataFrame(index=idx)
    t["domain"] = c2["domain"]
    t["num_variates"] = c2["num_variates"]
    parts = idx.to_series().str.split("/", expand=True)
    t["dataset"], t["freq"], t["term"] = parts[0], parts[1], parts[2]
    for metric, tag in [(CRPS, "crps"), (MASE, "mase")]:
        t[f"c2_{tag}"] = c2[metric]
        dl = pd.DataFrame({m: load(m)[metric].reindex(idx) for m in DL})
        st = pd.DataFrame({m: load(m)[metric].reindex(idx) for m in ST})
        t[f"gDL_{tag}"] = (c2[metric] - dl.min(axis=1)) / c2[metric]
        t[f"gDL_argmin_{tag}"] = dl.idxmin(axis=1)
        t[f"gST_{tag}"] = (c2[metric] - st.min(axis=1)) / c2[metric]
        for ft, base, name in [("TTM-R3-FT", "TTM-R3-PT", "gTTM"),
                               ("Toto-2.0-2.5B-FT", "Toto-2.0-2.5B", "gToto")]:
            a, b = load(ft)[metric].reindex(idx), load(base)[metric].reindex(idx)
            t[f"{name}_{tag}"] = (b - a) / b
    t.to_csv("stage1_gift_eval_gap.csv")
    print(t[[c for c in t.columns if c.startswith("g") and "argmin" not in c]]
          .describe().T[["mean", "50%", "max"]].round(3))
    print((t.gDL_crps >= 0.10).sum(), "configs with best-DL >= 10% better CRPS than Chronos-2")


if __name__ == "__main__":
    main()
