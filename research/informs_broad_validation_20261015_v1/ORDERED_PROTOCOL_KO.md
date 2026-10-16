# E4 공유 checkpoint와 실제 prefix K 비교

## API와 사전 가족 선택

`run_ordered_pilot(task, v_caches, cache_root, public_root, foundation, deadline, epochs=None, fold="primary", device="cuda")`가 전체 묶음을 실행한다. 데이터/F 로드, subprocess, registry 수정 또는 T_final 접근은 하지 않는다. 실제 실행·자원 lease는 root가 관리한다. 기본 epoch schedule은 `.25/1/3`이고 `.25` 제한 실행을 지원한다.

`lock_family`는 독립 E1 B0/B2의 K=`ceil(C/4), ceil(C/2), C`, screen seed92601의 전체 V 캐시를 요구한다. 원 Task의 V origin/정답/mask/채널 순서·A60 scaler와 대조하고 저장 예측에서 score를 재계산한다. 세 K의 V MSE를 동일 평균해 작은 가족을 선택하며 동점은 B0다. 다른 cohort나 누락·중복 K, 변조 score는 거부한다.

선택은 E4 학습 전에 `ordered/{fold}_{dataset}/FAMILY_LOCK.json`으로 고정한다. E4 run 폴더가 생긴 뒤 처음 lock을 만드는 것은 거부한다. 이후 같은 lock만 재사용할 수 있다. V를 사용한 가족 선택이므로 그 V 결과를 독립 일반화 검증으로 주장하지 않는다.

## 두 학습과 초기화 대조

`make_specs`는 등록된 두 parent ID를 만든다.

- `{fold}_{dataset}_E4_ORDERED_MULTI_K_kC_s92601`: k_grid=`[q,half,C]`.
- `{fold}_{dataset}_E4_MAXK_PCA_TRAINED_kC_s92601`: k_grid=`[C]`.

두 모델 모두 최대 C의 **같은 전체 PCA basis/center**에서 같은 seed로 시작한다. `k_grid=[C]`도 명시하여 독립 K=C의 identity 초기화로 바뀌는 것을 막는다. CP0의 model-state hash와 PCA 초기값이 같은지 CPU 검사로 확인했다. 독립 E1 K=C의 identity 초기화는 별도 비교 조건으로 보고한다.

실제 학습은 변경하지 않은 fit_base에 위임한다. A_fit stride1 순열과 A_stop checkpoint 선택, 선택 epoch 수를 고정한 fresh A60 재적합을 그대로 수행한다. Shared는 optimizer update마다 K를 순환하고 각 K의 원채널 미래 loss를 사용한다. Max-only는 C만 사용한다. K별 노출과 F training rows를 기록한다. 단순히 같은 epoch/optimizer update 수라는 이유로 같은 GPU 시간이나 각 K의 같은 노출을 주장하지 않는다. 선택 epoch가 다르면 refit 비용도 달라진다.

`.25`만 완료하면 LIMITED_SCREEN이다. 작은 합성 cohort처럼 update 수가 K 개수보다 적은 경우 일부 K는 직접 학습 노출이 0일 수 있다. 이를 충분한 다중 K 학습으로 해석하면 안 된다. 실제 receipt의 k_exposures를 확인해야 한다.

## 네 비교와 추론 좌표

`predict_prefix(alias_run_id, task, latent_k, cache_root, public_root, foundation, parent_receipt=None, variant="shared", fold="primary", deadline=None)`는 V-only 캐시를 생성한다.

1. 독립 K는 잠금 입력으로 전달한 E1 캐시를 그대로 연결한다. 새 fit이나 inference로 중복 계수하지 않는다.
2. Shared prefix는 같은 multi-K 부모 artifact 하나에서 E 앞 K column과 D의 `D[lag,:K,:]`를 사용한다.
3. Posthoc prefix는 C만 학습한 full-PCA control의 weights를 바꾸지 않고 자른다. Deployment-only 허용 K registry를 확장해 작은 prefix 추론을 열며, 학습 weights나 원 artifact는 수정하지 않는다. 작은 K를 직접 학습한 모델이라고 표현하지 않는다.
4. Nested PCA는 A60에서 적합한 전체 PCA artifact 하나를 저장하고 그 앞 K basis를 사용한다. Data/source/파일 SHA와 normalization을 검산해 재사용한다. 표준 고정 PCA inverse이며 학습 adapter의 현재값 anchor/lag7 D와 다르다.

학습 adapter의 D는 `[7,Kmax,C]` 상태에서 `[:, :K, :]`를 자른 후 flatten한다. F에 넘기는 latent도 실제 K개다. Fmax 출력을 먼저 만든 뒤 일부를 잘라 비용을 K로 표시하지 않는다. 각 route에서 실제 F logical counter 차이가 N×K와 같은지 확인한다. CPU brute-force lag/latent 합과 수치 parity를 검사했다.

## 출력·감사·registry 작업

Private V NPZ는 `ordered_predictions/{alias}/V_select.npz`에 다섯 고정 배열로 저장한다. 추론 전에 독립 Task 정답/mask/origin/channel fingerprint와 예상 F rows를 봉인한다. 공개 reference/manifest/SCORES/mtime0 gzip band 통계는 같은 alias 아래 보존한다. Source·부모 artifact·inference K를 분리해 기록하고, 같은 공유 weights 또는 nested PCA artifact를 사용한다는 사실을 SHA로 확인할 수 있다.

Nested PCA의 model identity에도 정확한 foundation SHA를 결속한다. PCA basis가 같더라도 F가 바뀐 캐시를 재사용할 수 없다. Wrong-F 캐시 재사용을 허용하던 RED 검사를 먼저 확인한 후 이 결속을 추가했고, 학습 prefix의 기존 F hash 검증과 함께 최종 suite를 통과했다.

공유 작은 K alias는 기존 parent ID에 `__prefix_kK`를 붙인다. 나머지 posthoc/nested/max ID는 등록 E4 ID를 사용한다. Module은 `registry_actions`만 반환하며 root가 검토해 derived shared prefix 행을 추가하고 기존 E4 행을 갱신해야 한다. 성공한 초기 score도 독립 auditor 실행 전에는 `PENDING_INDEPENDENT_AUDIT`이다.

각 예측 microbatch 전 운영 deadline을 확인한다. Deadline은 과학 config에 넣지 않는다. 중단 시 일부 완성 alias를 보존하며, 같은 계약으로 다시 호출하면 완성 fit/cache를 재사용한다. 전체 재호출 때 F 추가 호출이 0인 경우를 CPU 검사했다. 바뀐 계약이나 불완전 파일은 덮어쓰지 않는다.

## CPU 검증과 실제 실행 구분

최초 4 RED 실패 후 구현했다. 최종 suite에는 5 unittest가 있고, 합성 tiny CPU 실제 fit과 10개 V route를 포함한다.

- Tie B0/낮은 평균 B2와 immutable·late family lock.
- Shared/max-only 동일 full-PCA 초기 hash, 같은 search update/총 origin 노출 및 K별 F training rows 계산.
- D lag/latent 좌표 brute parity와 F 실제 입력 K.
- Shared/max/posthoc/nested 10 route 각각 독립 NPZ 재채점·공개 band 재집계 PASS.
- 같은 공유 artifact/nested artifact, 전체 반복 호출에서 추가 F0.

실제 자료나 Bolt/GPU/E4 성능을 실행·검증했다는 뜻이 아니다. Root가 승인된 캠페인 범위와 자원 안에서 실제 학습·추론·비용 측정·최종 감사를 실행해야 한다.
