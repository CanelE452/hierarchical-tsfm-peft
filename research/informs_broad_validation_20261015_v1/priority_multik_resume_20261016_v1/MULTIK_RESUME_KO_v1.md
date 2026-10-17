# Beijing: 원 shared/max 부모 재개와 실제 여러 K 출력

[확인] 이 보고는 원 shared `[3,6,11]` 및 max-only `[11]` **두 부모**의 마지막 정상 refit CP0 재개와 K3/6/11 **여섯 출력**을 구분한다. 여섯 출력을 새 독립 학습6개로 세지 않는다. 아래는 같은 C11/L512/H48·seed92601·CPU2·반복 개발 V43/20539 관측 라벨의 normalized channel-macro MSE·MAE와 실제 warm whole batch 비용이다. 원 숫자는 다음 CSV에 보존한다.

[원정밀도 CSV](MULTIK_RESUME_RESULTS_v1.csv) · [입력/출력 SHA](MULTIK_RESUME_BINDINGS_v1.json) · [실제 독립 감사](INDEPENDENT_MULTIK_RESUME_AUDIT_ROOT_v1.json) · [원 재개 계약](README_PREPARATION_KO.md)

```text
role                       K MSE                 MAE                 B1 median sec/batch B8 median sec/batch accuracy-reused
SHARED_MULTI_K             3 0.87586273724465047 0.61153829039131014 0.029722349951043725 0.15991169982589781 False
SHARED_MULTI_K             6 0.77593374618427613 0.5833188274451897 0.047275499906390905 0.35027404990978539 False
SHARED_MULTI_K            11 0.73747267151816664 0.56770226809466584 0.076830150093883276 0.57159734982997179 False
MAX_ONLY_POSTHOC           3 0.99926187391394095 0.66169817877600345 0.030786049785092473 0.1591576999053359 False
MAX_ONLY_POSTHOC           6 0.89339392127061457 0.63330849407508838 0.049241999862715602 0.35081984987482429 False
MAX_ONLY_POSTHOC          11 0.78469638401417197 0.58154805595840253 0.077037150040268898 0.56367970001883805 False
NESTED_PCA_REFERENCE       3 0.79792503730478825 0.61396069087098837 0.030515099875628948 0.1594246500171721 True
NESTED_PCA_REFERENCE       6 0.77945659604838513 0.59563573063231001 0.046431950060650706 0.32212945003993809 True
NESTED_PCA_REFERENCE      11 0.75178995964171014 0.57654751167357154 0.077562900027260184 0.57407645019702613 True
INDEPENDENT_K6_REFERENCE   6 0.79796643709754878 0.59313931444669032 0.04792475001886487 0.33349604997783899 True
```

재사용 참고는 [고정 nested PCA3](../priority_nested_cost_20261016_v1/RESULTS_WITH_COST_KO_v1.md)와 [기존 독립 B0 K6](../priority_valid_minima_20261016_v1/CORE_CONTRAST_KO_v1.md)다. 새 감사의 rows6과 재사용 참고행을 분리하며, 이 builder가 참고의 모델/통계/비용을 다시 실행한 것은 아니다. 고정 PCA는 basis transpose 복원이고 학습 가능한 lag7 D·현재값 anchor와 다르므로 압축 학습만의 효과 대조가 아니다. 독립 K3/K11 새 학습·fullV·비용은 이번 완료에 포함되지 않는다.

첫 질문은 같은 한 부모의 실제 K 입력 전환에서 오차와 전체 비용이 어떻게 달라졌는지다. 둘째는 shared와 max-only 사후 prefix가 이 조건에서 어떻게 달랐는지다. 두 부모는 같은 A-only full-C PCA 초기화지만 K별 감독·노출·F 연산 배분이 다르다. 동일3 epoch·optimizer update가 동일 K별 노출·F 연산·총 학습 시간까지 보장하지 않는다. 총 예산을 맞춘 추가 비교와 K별 독립 모델의 총비용·전환 비용을 이번 표가 완료시키지는 않는다.

shared/max K11은 grid가 있어 PCA 초기화이고 원 independent K11은 grid 없는 identity/center0 초기화다. 해당 K11 간 차이를 공유 학습만의 순수 효과로 단정할 수 없다. K3/6의 PCA E·center는 같은 A-stage 전체 고유분해의 prefix 규칙이지만 최종 학습 가중치·D·용량과 노출은 다르다. 현재 비교는 같은 최종 표현에 predictor만 바꾼 실험이 아니다.

```text
parent   search-restore-call-sec historical-saved-search-segment-sec current-refit-sec search/refit-updates search/refit-exposures unsaved-progress
shared   0.0021976004354655743 4022.6722256001085 4788.9197912001982 1868/2262 29880/36189 UNKNOWN_NOT_LOGGED
max_only 0.0020553003996610641 6584.7130526001565 7342.5362463998608 1868/2262 29880/36189 UNKNOWN_NOT_LOGGED
```

search restore 호출 시간은 이번 재구축/복원 overhead이며 **과거 search 전체 wall**이 아니다. saved segment 시간은 과거 저장된 segment 기록 범위만 뜻한다. 원 search0/.25/1/3의1868 update·29880 exposure는 재사용 이력이며 새 search update/F call은0이다. 원 refit CP0에서3 epoch의2262 update·36189 exposure를 수행했다. 저장 전 손실 진행량은 UNKNOWN_NOT_LOGGED다. 원 stage별 session/total F counter는 입력 감사의 parent `stage_F_counters` 및 CSV JSON열에 있는 값만 보존하고, 누락값을 추정 합산하지 않는다. sampler·Torch CPU RNG 복원과 Python/NumPy 전역 RNG·중단 없는 bitwise parity는 구분한다.

부모 BASE의 `full_V_completed=false`는 그대로 보존한다. alias별 실제 predict/manifest·전체 저장 V FP64·fresh CPU6(4+2)로 완료를 판단하며 BASE를 true로 고치지 않는다. full V 검산은 저장 예측의 재채점이고 checkpoint 재추론은6 origin이다. 같은 V를 여러 K·방법에서 반복 사용했으며 독립 표본 수를 늘리지 않는다. search 마지막3 epoch 선택 상태는 UNDERTRAINED_OR_BUDGET_LIMIT이며 충분 수렴과 독립 일반화는 미확정이다.

각 B1/B8 비용은 물리 history/mask→채움·정규화→실제 K개 F 입력→lag D/현재값 anchor→원 C 미래·FP64 inverse/CPU 반환의 **한 batch 전체**다. B8을8로 나누거나 K/C를 속도비로 쓰지 않는다. warm10·ordinary timed20·별도 intrusive stage20을 분리하며 CSV에는 ordinary median/p90와 관측 guard tree RSS를 남긴다. stage 합=whole을 주장하지 않는다. 원 F pool cold load와 Route loading은 warm whole 밖이다. prefix K가 작아져도 부모 E/D는 최대 K11 구조이므로 모델 resident memory가 K배 줄었다고 하지 않는다. Route memory dict와 guard RSS는 다른 범위이며 F-only resident bytes는 미측정 NA다.

**원 N7 COST receipt에는 전체 F counter 합계·validation proof SHA 사전·physical output dtype 필드가 없다.** 해당 CSV열은 비워두고 NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT로 표시한다. 실제 argv/proof 결속과 공개50 trial에서 독립 합산한 counter는 별도 근거다. 그 합계를 원 receipt에 기록된 값으로 바꾸지 않는다. 비용 측정과 학습을 겹치지 않았다는 실제 quiet/guard 증명 및 종료는 원 감사 범위로 보존한다.

전체 E4/56/Master 완료가 아니다. Electricity 파일럿·독립 K3/K11·다른 자료/seed·충분 학습·동등 총 예산·저장량/전환 비용 등 원 잔여는 남는다. 고정 nested PCA와 독립 K6 재사용은 새 원리 학습/새 일반화가 아니다. 보호 T 새 수치 파싱·평가·선택·원 큰 큐 자동 재시작·새 모듈 추가를 수행하지 않았다. 부분 종료나 증거 누락이면 이 완성 builder는 거부하며 ROOT는 실패·미실행·저장점·재개 조건을 별도 정직한 부분 보고로 남겨야 한다.
