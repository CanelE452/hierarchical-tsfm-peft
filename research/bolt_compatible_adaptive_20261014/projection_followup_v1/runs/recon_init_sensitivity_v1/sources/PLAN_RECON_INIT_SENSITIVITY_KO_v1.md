# Robin L7의 초기 D 민감도 — 같은 예측 목적 학습 두 조건

작성 시점의 새 실제 실행은0이다. 학습된 공간 진단의 별도 독립 감사·보고는 진행 중이다. SOURCE·합성 준비는 병행하되 이 진단의 실제 PASS와 보고서 검산 이후 새 모델 실행을 시작한다. 현재 검산·보고를 생략하거나 재학습으로 덮지 않는다.

[확인] 현재 full-gradient가 stop-gradient보다 L0 TUNE/DEV4쌍에서 모두 좋았으나 Raw 격차가 남았다. 학습된 full-gradient Robin L7 D의 DEV retained numeric minimum은 약.426~.428로 Raw.3404보다 높았다. 한편 이전 고정 PCA L7_RECON 실제 개발 MSE는.488093이며 그 D의 수치 공간 minimum은.306885였다. 이들은 서로 다른 D다. 낮은 과거 D의 minimum을 현재 D의 가능성으로 재사용하면 안 된다. [판단] 새로운 인코더나 더 긴 학습 전에 **forecast-LS D로 시작한 초기화가 현재 결과를 제한했는가**를 최소 대조한다.

## 새로 하는 것과 재사용

Robin·L7·seed92601/92602의2fit만 새로 실행한다. 기존 full-gradient L7 E갱신2fit은 FORECAST-LS 초기화 참고로 재사용한다. 새 조건은 같은 초기 PCA E·center에 기존 실제 PCA L7_RECON D32를 넣고, 이후 같은 actual-C 예측 목적 full-gradient E/D 학습을 수행한다. 이는 재구성 목적 학습과 예측 목적 학습 전체를 비교하는 실험도, 새 모델 계열도 아니다. 초기화가 다른 두 학습 절차의 민감도 대조다. 기존 D는 이전 FIT에서만 적합한 것을 SHA로 결속하고 다시 적합하지 않는다.

K4/C17, L512/H48, FIT256/TUNE64·노출 VAL30, 원 origin/scaler/mask/current-C anchor, lag0/1/2/4/8/16/32, AdamW.001/WD0/clip1/B4/600update·전체 FIT 고정 관측수 가중 macro MSE를 유지한다. seed별 이전600×4 순서를 그대로 쓴다. E와 D는 갱신하고 F 파라미터는 동결하되 실제 F 입력 gradient를 유지한다. 새 학습은2×600=1,200update, CP0/100/200/400/600의10CP다. 전체 TUNE MSE·이른 tie로 고르고2성공/실패 잠금·registry를 봉인한 뒤 DEV를 추론한다. 새 DEV로 선정 기준을 바꾸지 않는다.

## 검증과 판단

실제 CP0 forward는 기존 고정 PCA L7_RECON의 같은 canonical 입력과 원 등록 공차로 비교하고, E/center·새 초기 D32/배치 순서·원 F dual hash를 확인한다. CP0 검사는 optimizer0이다. 기존 train_condition을 직접 재사용하며 원 SOURCE를 수정하거나 optimizer state를 재개하지 않는다. 실패는 해당 seed에만 적용한다.

모든 CP·손실/gradient/clip 로그·선정 FIT/TUNE/DEV와 원 바인딩을 보존하고 저장 배열 산술·점수·선정을 검산한다. 기존 CP0_RECON과 FORECAST-LS CP0/선정 결과를 모두 보여준다. 학습 후 TUNE/DEV의 새 D/known-offset 공간 진단도 같은 complete-C mask·retained1e-15 정의로 계산한다. TUNE 점수 감소와 DEV 공간 악화를 구분한다. Raw/학습 전 RECON을 이기지 못해도 실제 값을 남긴다.

두 조건은 이미 노출된 DEV에서 문제를 관찰한 뒤의 적응적 탐색이다. 이 초기화 선택 자체가 DEV의 영향을 받았으므로 독립 일반화·모든 미래 표현의 한계/가능성을 검증한 것으로 말하지 않는다. 새 비용 측정은 현재 단계에 계획하지 않는다. 동일한 배포 구조여도 새 계수의 속도를 기존 cohort 수치로 대체하지 않고 미측정이라고 남긴다. 개선이 확인되면 그 상태의 실제 동일 cohort 비용을 다음에 측정한다. 결과가 나쁘면 같은 설정을 무조건 장기 학습으로 확대하지 않는다.

기존 상한 해제·계속 실행의 동일 목적 안에서 진행하는 최소 대조라는 작업 판단이며 개별 새 승인을 받았다고 쓰지 않는다. TEST·설치/다운로드·백본 파라미터 학습·기존 작업 종료·중단 학습 재개·자동 commit/push·논문/PPT는 포함하지 않는다. K 과거 buffer와 현재 C anchor를 쓰므로 strict K-only도 아니다.
