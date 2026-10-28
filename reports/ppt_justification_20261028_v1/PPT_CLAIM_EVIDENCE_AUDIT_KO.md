# PPT 주장과 기존 근거 감사

2026-10-28, 새 실험 결과를 보기 전 준비 감사다. 오늘 실험의 완료·성능·최종 PPT 검수 PASS를 뜻하지 않는다. 범위는 기존 공개 보고서·코드·사전 선택 기록과 허용된 V 예측의 동일 cohort 확인이다. 기존 파일은 수정하지 않았고 새 학습·추론·보호 T 수치 접근은 하지 않았다. 연결 파일의 SHA와 기계 판독용 상태는 같은 폴더의 `PPT_CLAIM_EVIDENCE_AUDIT.json`에 기록한다.

## 재사용 기준선과 오늘의 비교 계약

고정 seed는 92601–92605다. 기존 OURS의 선택 epoch는 Jena `[1,1,1,1,.25]`, Appliances `[.25,.25,1,1,.25]`, ETTh1 `[.25,.25,.25,.25,.25]`다. 모두 cap3 선택 0/5이므로 오늘의 `3/5 이상이면 전 5개 cap10` 규칙으로도 Full 기준선 연장은 필요 없다.

Jena No P는 기존 cap3에서 다섯 seed 모두 3을 선택했다. 따라서 오늘의 동일 연장 조건도 충족한다. 재사용할 기존 ext10의 탐색 checkpoint는 `0/.25/1/3/6/10`, 선택 epoch는 `[3,6,6,6,6]`이며 cap10 선택은 0/5다. 다섯 결과와 선택 기록은 COMPLETE다. Full 5개와 No P ext10 5개의 실제 V NPZ를 확인했으며, 정답·관측 mask·origin·채널 순서가 canonical OURS seed92603과 모두 정확히 같다. 43 origins × 48 horizon × 21 channels이며 각 NPZ의 SHA도 원 RUNS 행과 일치했다.

같은 적응형 선택 규칙은 같은 실제 epoch를 뜻하지 않는다. Jena 2×2는 Full/No P와 frozen F/shared Linear를 비교하되, 실제 cap·선택 epoch·용량을 함께 표시해야 한다. `NO_F_LINEAR_NO_P`는 P만 identity로 바꾸고 E·lag D·anchor와 채널 공유 Linear head를 유지해야 한다. 두 no-F arm의 head는 24,624개 파라미터로 같고, Jena 총 학습 파라미터는 Full no-F 60,341개, No P no-F 25,632개다. 이 대조는 선택한 모델·학습 규칙에서 P의 이득이 예측기 종류에 따라 달라지는지 묻는다. F 사전학습만의 인과 효과, 정보 보존 메커니즘, 모든 예측기에 대한 필요성을 단독 식별하지 않는다.

## 1. 채널 관계가 외부 적응의 동기인가

구조적으로 Raw의 각 F 호출은 한 시계열만 받고, E/P/D는 원채널 간 계산을 허용한다. 이는 외부 적응을 시도할 동기다. 다만 기존 own-channel 대 all-channel Ridge 진단은 Jena에서 all-channel이 더 나쁘고 PeMS32에서는 더 좋아 관계의 존재와 예측 이득이 같지 않음을 보여 준다. 현재 결과를 모든 데이터의 다변량 우월성으로 확대하지 않는다. 근거: `adapter_diagnostics_20261017_v2/QUESTIONS_AND_DECISIONS_KO.md` Q2.

## 2. P는 현재 구조의 성능에 기여하는가

현재 Jena 10-seed 비교에서 Full P 대비 No P의 중앙 V MSE는 11.12% 높고, 시간 분기 제거는 8.92% 높다. 채널 분기 제거의 중앙값 차이는 +0.17%로 등록 판정에서 불명확하며 동등성을 증명하지 않는다. 기존 A2 근접 용량 채널 대체도 10-seed 판정은 UNCLEAR다. 시간/채널 분기 용량과 전체 창 LayerNorm의 결합이 남으므로 시간축 자체가 보편적으로 우월하다고 할 수 없다. 근거: `jena_p_branch_20261027_v1/SUMMARY.csv`, `CONTRASTS.csv`, `RESULTS_KO.md`; 후속 `PAPER_ABLATION.csv`.

오늘 Jena의 추가 no-F Full/No P 대조는 P 기여의 예측기 의존성을 직접 보강한다. P 제거의 파라미터 감소는 두 예측기에서 동일하게 발생하지만, frozen F와 학습형 Linear의 함수 공간·사전학습·학습 파라미터 수·최적화는 여전히 다르다. 차이의 차이는 seed별로 먼저 계산한 뒤 분포를 보고하고, 중앙값들의 차이와 혼동하지 않는다.

## 3. P가 압축 전에 예측 정보를 더 잘 보존한다는 메커니즘인가

그 메커니즘은 아직 입증되지 않았다. 기존 Jena 두 seed에서 선택된 P/E를 고정한 전체 512×6 선형 reader의 MSE는 B2가 B0보다 각각 2.275%와 0.707% 높았다. MAE도 높았다. 이는 lambda=1, 3,097,584개 파라미터인 해당 reader 아래의 선형 접근성 개선을 지지하지 않는 반례다. 정보 자체의 부재나 모든 비선형 접근성까지 부정하지 않는다. 이미 존재하는 이 실험을 미실행으로 표시하거나 좋은 결과만 남기면 안 된다. 근거: 같은 Q2, `METHOD_CONTRACT.md`, `COMPLETION_AUDIT_KO.md`의 B01/G6.

오늘 2×2는 F와의 결합 여부를 더 직접적으로 묻지만, latent 정보량이나 F에 대한 정렬을 직접 측정하는 실험은 아니다. 새로운 reader나 진단을 시도한다면 과거 반례와 새 가설·선택 절차를 함께 보존해야 한다.

## 4. PCA 초기화의 근거는 없는가

기존 Jena K6, CUDA, seed92601/92602의 PCA 대 무작위 QR 초기화 비교가 이미 완료됐다. QR의 V MSE가 PCA보다 각각 2.863%와 6.205% 높았고 MAE도 높았다. 현재 10-seed 캠페인의 A3 미실행과 과거 두 seed 근거를 구분한다. 근거: diagnostics Q3, `INITIALIZATION_DIAGNOSTICS.csv`와 `research/adapter_diagnostics_20261017_v2/new_jobs.py`의 initialization.

소스상 QR arm은 E와 함께 D0도 `pinv(E)`로 다시 초기화한다. 따라서 정확한 표현은 PCA 기반 E/D 결합 초기화의 조건부 지지다. E만의 단독 인과 효과, PCA의 보편적 필요성, 예측정보 90% 보장은 아니다. 오늘 같은 두 seed를 다시 돌릴 필요는 없다.

## 5. E를 학습하는 것 자체가 필요한가

현재 P/E/F/lag D/anchor를 유지하면서 E만 고정한 직접 대조는 이번 감사에서 확인되지 않았다. E가 실제 변했다는 진단은 있지만, 변화가 필요했다는 성능 근거는 아니다. C1은 P·D·학습 방식도 달라 E 고정만의 대조가 아니다. 이 질문은 `same P/F/D + fixed PCA E` 대 공동학습 E로 별도 검증할 수 있으나 오늘 기본 실험이 답하지 않는다. 이미 확인한 과거 다른 구조의 E 고정 실험을 현행 B2로 대체 해석하지 않는다.

## 6. K와 tau=0.9는 정당화됐는가

A_fit 입력 분산 규칙은 재현 가능한 용량 선택 절차다. 40개 주 결과와 기존 K 민감도가 있으며, Jena K6는 비교한 K12보다 나빠 규칙의 한계가 이미 기록됐다. 이것은 근거가 전혀 없는 숫자 선택과 다르지만 최적 예측 차원의 증명도 아니다. tau=0.9, 선택한 lag 집합, MLP 폭 등은 모든 대안을 최적화한 값이 아니라 고정한 설계값이다. 근거: `k_tau_rule_20261023_v1/RESULTS_KO.md`, `K_TAU.csv`, `DECISION.json`.

## 7. 마지막 관측값 anchor가 필요한가

허용 과거의 마지막 값을 원채널 출력에 더하는 것은 미래 누수가 아니며 현재 수준을 K차원 경로 밖에 보존하는 구조적 역할이 있다. 기존 A4는 anchor·lag·차이 특징을 함께 바꾸므로 anchor 단독 효과는 분리하지 못했다. D0의 ETT 약90%는 고정 PCA 출력공간의 조건부 하한에서 평균 이동 항이 차지하는 비율이지, 전체 예측 오차나 개선량의 90%가 아니다.

코드에서 예측 origin을 o라고 하면 `FastWindows.get`의 입력 행은 `[o-L, o)`, 미래 정답은 `[o, o+H)`다. `x_last`는 입력 끝인 o−1 시점까지의 값으로부터 만들어지고 `predict(x,k)`에는 미래 정답을 넘기지 않는다. 입력 결측은 창 안의 과거 관측만으로 채우며, 창 안에 관측이 전혀 없으면 정규화 0을 사용한다. 따라서 정확한 용어는 마지막 입력값/마지막 context 값이며 항상 o−1에 실제 측정된 값이라는 뜻은 아니다. 근거: `research/informs_broad_validation_20261015_v1/training.py`의 FastWindows.get과 train_segment, `networks.py`의 ForecastAdapter.decode.

오늘 `NO_ANCHOR`는 같은 P/E/F/lag 특징/D를 유지하고 직접 `x_last` 가산만 빼서 재학습하므로 현재 설계에서 이 경로의 추가 기여를 직접 묻는다. ETTh1 기본 5seed와 Appliances 선택 5seed를 별도 상태로 표시한다. 결과가 좋더라도 모든 수준 이동에 대한 강건성이나 모든 압축 손실 복원을 입증하지 않는다. 근거: 후속 `PAPER_SUBSPACE.csv`, `D0_SUBSPACE_FLOOR.csv`, `PAPER_ABLATION.csv` 및 새 등록 모델.

NO_ANCHOR에서도 전체 과거 입력과 latent `z_last` 차분은 그대로 관측값에 의존한다. 그러므로 이 대조의 실패는 모든 observed-value 사용의 필요성, 미래 정답을 사용하는 것의 정당성, 관측값 없이 예측할 수 없음 같은 주장을 검증하지 않는다. anchor는 원채널 현재 수준의 직접 가산이고, lag는 잠재 과거·예측의 여러 시간 위치에서 변화 특징을 만드는 별도 요인이다.

## 8. lag 특징 자체의 근거는 있는가

원 비교와 후속 RUNS에서 A5는 모든 데이터 n=0이다. 미실행이므로 효과 없음·효과 있음 어느 쪽도 판정할 수 없다. 오늘 선택 `A5_LAG0` ETTh1 5seed는 anchor를 유지하면서 lag1–32만 제거하는 첫 직접 대조다. 결과는 현재 lag 묶음의 추가 기여를 다루며 개별 lag의 최적성·실제 물리 주기·인과 시차 식별은 아니다.

## 9. 고정 F의 추가 가치가 전혀 검증되지 않았는가

아니다. 기존 Jena CPU seed92601에서 B2 MSE .6834428964, 고정 최종 P/E/D에 Linear를 맞춘 R5a .7378513781, fresh 공동학습 R5b .7201979036이었다. PeMS08 CPU seed92601에서는 부모 .1042596968, 고정 P/E/D의 Linear head .1206228897이었다. PeMS에 joint no-F는 없으며 Jena 두 seed 반복은 B0/B2뿐이다. 근거: `priority_core_followup_20261016_v1/RESULTS_v2.csv`, `RESULTS_KO_v2.md`.

오늘 Jena/Appliances 각5seed는 현재 GPU·cohort·학습 계약으로 이 근거의 반복 수와 적용 범위를 넓힌다. 원 CPU R5b를 현재 GPU 5-seed 결과에 섞지 않는다. F가 없는 Linear 출력은 deterministic regression이며 quantile_0.5라고 표시하지 않는다. 기존 Ridge/Direct MLP가 더 정확한 사례도 유지한다.

## 10. 비교 방법 대비 우위가 확정됐는가

C2/C3는 원 AdaPTS 전체 재현이 아니라 동일 F·데이터 조건의 이식 어댑터다. 마지막 checkpoint 선택으로 연장이 필요한 여러 family가 미연장이라 비교 판정은 잠정적이다. Jena도 현재 등록 판정은 TIE_PROVISIONAL이다. 오늘 anchor/F/2×2 결과는 이 최적화 공백을 해결하지 않는다. 같은 표에서 MAE·MASE의 Raw 대비 악화, ETTh2 보조 데이터의 MSE 악화도 유지한다. 근거: 후속 `PAPER_COMPARISON.csv`, 원 `DEVIATIONS_FROM_ADAPTS.md`.

실증적으로 남은 큰 비교 질문은 C2/C3의 사전 규칙상 연장을 끝내는 것이다. 다만 오늘 기본 실험과 병행하면 마감 위험이 크므로 미완료를 숨기지 않고 별도 예산에서 수행해야 한다.

## 11. 비용과 효율의 근거는 어디까지인가

K<C로 F에 보내는 논리 시계열 수가 줄어드는 것은 구조적으로 확인 가능하다. 실제 속도·메모리는 P/E/D·행 배치·장치·batch를 포함한 전체 경로 실측이 필요하다. 기존 Jena/ETT의 GPU B8 지연·allocated 메모리 근거와 fixed48/default128 구분은 유지한다. 현재 PPT의 Appliances K10 비용 공백은 별도 측정해야 하며 새로운 정확도 학습의 완료만으로 채워지지 않는다. no-F 학습이 빠르다는 사실도 동일 프로토콜의 배포 지연·메모리 측정을 대신하지 않는다.

## 12. 일반화와 연구조건

오늘 결과도 반복해서 사용한 개발 V의 탐색 결과다. seed는 최적화 반복이며 새로운 데이터 표본·독립 미래 시험이 아니다. F를 고정한다는 것은 연구조건이고, 내부 튜닝이 불가능하거나 열등하다는 실증 결론이 아니다. 구현의 입력 gradient·동결·경계 검사는 코드가 의도대로 작동한다는 근거이며 가설의 성능 증명과 구분한다. 보호 T를 열거나 과거 수치를 재선택하지 않는다.

## 오늘 우선순위와 남는 질문

1. 승인된 기본20: no-F Full(Jena/Appliances), no-F NoP(Jena), no-anchor(ETTh1) 각5seed. 실행 후 같은 seed·cohort·선택 규칙·전체 완료된 연장 family의 주 결과 선택을 검산한다.
2. 필요한 cap10 연장: A_stop 조건으로 결정하고, 전5 완료 여부와 실제 선택 epoch를 공개한다. 시간 때문에 누락되면 잠정성 표시를 유지한다.
3. 선택 A5 ETTh1: anchor와 별개인 lag 질문을 닫는다. 다음 NO_ANCHOR Appliances는 다른 데이터에서의 재현 범위를 넓힌다.
4. 시간이 남으면 기존 선택 모델의 Appliances 비용 측정은 새 정확도 학습 없이 비용 공백을 줄일 수 있다. 별도 측정 계약과 parity 검증은 필요하다.
5. E 고정 대 학습, 충분 최적화 C2/C3, 새로운 독립 평가, P의 구체적 정보/정렬 메커니즘은 남는다. 모든 문구를 약하게 고쳤다고 이 질문들이 실증적으로 해결된 것은 아니다.
