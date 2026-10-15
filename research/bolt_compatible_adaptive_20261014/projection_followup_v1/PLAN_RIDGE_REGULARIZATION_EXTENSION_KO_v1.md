# ridge 정규화 탐색 상단 선택의 후속 대조

새 방향 비교의 직접 ridge v1이 실제 완료됐다.12개 fit/84후보/168 FIT-TUNE/12DEV/실패0이며 [원 결과](runs/new_direction_direct_ridge_v1/RESULT.json) SHA256은 `b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed`다. 현재 독립 과학 감사는 준비 중이다. 원 실행/계수/계획/SOURCE는 수정하지 않는다.

두 Raw current-C 경로는 alpha .1을 골랐지만, Raw history2개와 encoded history8개는 모두 등록 grid의 최댓값10을 골랐다. 따라서 이 결과를 충분히 조정된 모든 단순 예측기의 부진으로 결론내리지 않는다. 이번 확장은 **TUNE 경계 선택에 대응하는 적응적 후속**이며 새로운 독립 연구나 사전 전체 고정 탐색이 아니다. 이미 본 DEV 성능에 따라 이득을 골라 최종 일반화 성능으로 주장하지 않는다. 최신 지속 연구 지시 안에서 실행하며 새 승인 대기는 없다.

## 범위 고정

history10경로에 한해서 기존 E/center/원 past/current-C anchor/feature32 빼기/전FIT RMS/true-observed FIT response/mask/비규제 intercept·fused FP32 CPU canonical32 산술을 그대로 쓴다. 새로운 E/D/native/F 학습·추론은0이다. 기존 Raw current-C2개의 결과는 원 잠금 기준 참고로만 재사용하고 새 fit으로 세지 않는다.

새 양수 alpha는 **100,1000,10000 세 개**다. 동일 thin SVD·SSE/n+alpha||standardized slope||² 규칙으로 10개 새 fit에서30후보를 만든다. 기존7alpha의70 FIT/TUNE 후보는 원 실제 예측·점수·계수hash를 그대로 재사용한다. 원 실행 source의 ALPHAS를 runtime monkeypatch하거나 수정하지 않고, 별도 새 버전 module/controller에 확장값을 명시한다.

추가로 각 history 경로에 다음 두 비학습 feature 대조를 둔다.

- **INTERCEPT_ONLY**: slope0, 각(h,c)의 true-observed FIT 미래 innovation mean을 intercept로 쓰고 currentC에 더한다. alpha가 무한히 큰 경우의 해석상 끝점이며 유한한 최적 alpha를 증명하지 않는다. JSON에 infinity/NaN을 쓰지 않는다.
- **CURRENT_C_ANCHOR**: slope0/intercept0으로 currentC를48번 반환한다. latent-last를 넣은 old L7 D의 FLAT과 다르다.

따라서50개 새 후보(30ridge+10mean+10anchor),100개 새 FIT/TUNE 예측을 실제 계산한다. 각 경로의 메뉴는 원7+새3+INTERCEPT_ONLY+CURRENT_C_ANCHOR=12개다. 주지표는 동일 all-observed channel-macro 정규화 MSE이며 자기 TUNE로 고른다. 정확 동률의 순서는 원 alpha0,1e-4,.001,.01,.1,1,10 →100→1000→10000→INTERCEPT_ONLY→CURRENT_C_ANCHOR다. 모든10개 성공/실패 잠금 뒤 DEV를 평가한다. 새 후보가 좋아졌다는 조건으로 실행 성공을 판정하지 않는다.

원 후보가 다시 선정되면 그것은 반드시 원 v1의 선정 후보여야 하며 계수·원 DEV actual-cache를 exact binding으로 재사용할 수 있다. 이때 새 DEV 추론을 했다고 세지 않는다. 새 후보가 선정되면 그 FP64/FP32 계수를 저장하고 실제 새 DEV forward를 수행한다. 두 개의 zero-slope 경로는 E/history가 예측에 필요하지 않으므로 compression/F 효과라고 부르지 않는다.

## 검증·해석

원 v1의 독립 과학 감사가 PASS한 뒤 실제 실행한다. source·합성 준비는 먼저 한다. 새 양수 ridge 해/비관측 placeholder 불변·intercept 비규제·0slope 끝점·정확동률·옛점수재사용·10잠금-beforeDEV·새/재사용 forward 수를 검산한다. 새로운 실제 양수 fit의 목표와 선택된 FP32 산술, 원 input/model cache·selection hash 보존, 자원 receipt를 남긴다.

정규화를 늘리면 좋아질 것이라고 예고하지 않는다. TUNE에서100이 나빠져10을 유지하는 결과도 그대론 보고한다. alpha10000도 상단을 고르면 그 사실을 남기고 더 촘촘한 최적화를 완료했다고 하지 않는다. 독립 일반화·새 TEST 접근·설치·큰 모델 학습·자동 commit/push는 하지 않는다.

기존76셀 whole-cost는 **v1 선정 모델**의 실제 측정으로 유지한다. 이번 확장 모델을 그 결과에 조용히 대체하지 않으며, 선정이 달라지면 새 비용 필요 여부를 별도로 명시한다. 먼저 원 native/ridge 감사·v1 전체 비용·이 경계 진단의 실제 결과까지 마쳐 교수님께 가져갈 근거를 완성한다.
