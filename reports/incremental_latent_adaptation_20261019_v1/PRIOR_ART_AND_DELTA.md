# 선행 원리·기존 실행과 이번 검증의 차이

감사일: 2026-10-20 KST. 기준 저장소 HEAD: `451fb6c2b493f501776ecf945cc41913d764d299`. 이 문서는 결과를 보기 전의 범위 감사이며, 성능이나 신규성을 판정하는 최종 결과표가 아니다. 실제 실행 상태와 결과 판정은 이 캠페인의 `EXECUTION_STATUS.csv`, `VALIDATION_REPORT.json`, `CLAIM_STATUS.md`에서 따로 확인한다. 감사한 로컬 소스의 SHA와 재사용 판정은 [DUPLICATE_AUDIT.json](DUPLICATE_AUDIT.json)에 있다.

## 먼저 확인한 중복

[확인] 고정 부모 뒤의 잔차 보정, zero-init 출력, PCA 공간 제한, scalar 결합 및 앞부분 표현 사용은 이미 있는 원리다. 이 저장소 V9도 그중 일부를 실제로 구현했다. 따라서 새 이름이나 잔차·순차 연산 자체를 기여로 셀 수 없다.

[확인] 아래 기존 계약은 이번 필수 16구성의 결과를 대신하지 못한다. 이 결론의 범위는 아래에 명시한 캠페인·소스이며, 저장소 전체의 모든 실행을 검색해 부재를 증명했다는 뜻은 아니다.

```text
기존 묶음              이미 있는 내용                    이번 계약과 다른 점
V9 STAGED_P/RAW        고정 LEVEL + 새 시간선형 보정       부모·K·시기·선택구간·학습변수
smallk MAIN residual   MAIN + RAW33C/E33K/E512K Ridge      자료·부모·선택구간·λ·학습변수
RawContext alpha      G+α(S-G), M_fit/M_select, M_cal      새 P/E/F 경로를 학습하지 않음
RawContext head       고정 G/S 뒤 ConditionalHead         S lag 특징의 D-only 비교와 다름
```

V9의 `model_v9.py`에서 부모 전체를 동결하고 `parts()`의 `no_grad()` 안에서 부모와 고정 PCA 재구성 과거를 만든 뒤 새 `TemporalHead`만 학습하는 흐름을 읽었다. 출력은 `parent_output + alpha * correction`이며, 보정 head는 채널공유 bias 없는 `512→32→48` 선형 합성이고 마지막 층을 0으로 시작한다. Jena는 K6이고 Robin은 K5다. `protocol.json`은 full VAL checkpoint 선택, context normalization, effective batch4, 긴 epoch 상한과 scheduler를 사용한다. 이번 Jena K12/PeMS K8의 고정 G 뒤에서 M_fit으로 P/E를 바꾸고 F의 입력 기울기를 통과시키는 계약과 다르다. 기존 완료 상태는 `STATUS.md`의 공개 기록에서 확인했으며, 새 평가나 옛 TEST 배열 재개는 하지 않았다.

smallk MAIN residual의 계약·순수 수학 소스에서는 원 MAIN을 보존하고 FP32 `Y-MAIN`을 FP64 Ridge로 맞추며 원채널 과거 또는 고정 E 과거를 이용한다. 자료는 Robin/Beijing, 목표 H48, FIT/TUNE 및 과거 노출 평가 계약이다. `RAW33C`의 16-point mean 32개와 last 요약, 관측 mask별 thin SVD 공유, intercept 비규제, penalty `alpha*N_c`는 이번 A2와 겹치는 구현 원리다. 여기서 `N_c`는 모든 fitting origin×horizon의 채널별 관측 수이고, 개별 `(h,c)`의 관측 수가 아니다. 그러나 old wrapper의 H48 guard, 7개 λ와 ZERO/MEAN을 함께 고르는 first-tie 선택은 이번 H12 및 큰 λ exact-tie 규칙에 그대로 맞지 않는다. 같은 수학을 재사용하더라도 새 G 잔차·새 분할·새 λ 계약으로 실제 적합과 검산이 필요하다.

RawContext의 `calibration.py`는 dense M_cal을 60/40으로 나누고 경계에 걸치는 미래 정답 창을 제거한다. global alpha의 M_fit 계수/M_select 점수와 전체 M_cal fresh 계수를 구분한다. 이 분할과 고정 부모 검증은 이번에도 재사용 대상이다. 정확한 부모·scaler·origin·mask·원정밀도 예측의 동일성은 별도의 자산 감사가 확인해야 하며, 이 문헌 감사가 private 캐시를 검증한 것으로 간주하지 않는다. ConditionalHead의 추가 적합은 새 `READOUT_ONLY`의 동일 lag 특징+zero D 비교가 아니다.

## 실제로 다시 읽은 외부 출처

아래의 읽기 범위는 이번 감사에서 직접 열어 확인한 범위다. 저자 저장소의 `main` 링크는 접근일의 코드이며 특정 release 전체를 포팅·재현했다는 뜻이 아니다. 접근 실패는 독창성의 근거로 쓰지 않는다.

### CATS, ICML 2024

[공식 출판 기록·초록](https://proceedings.mlr.press/v235/lu24d.html), [저자 CATS 저장소 README](https://github.com/LJC-FVNR/CATS), 그 README가 지시한 [저자 모델 구현](https://github.com/LJC-FVNR/ARMA-Attention/blob/main/models/CATS.py)의 `Model` 생성과 `forward` 경로를 읽었다. 공식 PDF 두 경로는 본문 추출에 실패했고 OpenReview는 browser challenge였다. 검색기에 노출된 논문 일부를 최종 전문 독해로 기록하지 않는다.

[확인] 원 시계열에서 보조 시계열을 만들고 예측·출력 조합에 활용하는 것은 CATS의 선행이다. 확인한 코드도 encoder→predictor→decoder와 선택적인 연속성 손실·채널 선택을 가진다. 이번 검증은 고정된 작은 G 이후에 제한된 K를 동결 F로 보내는 경로의 추가 가치를 묻는다. CATS 전체를 같은 자료·부모·예산으로 재현하지 않으므로 CATS보다 우수하다는 판정은 미검증이다.

### AdaPTS, ICML 2025

[공식 출판 기록·초록](https://proceedings.mlr.press/v267/benechehab25a.html), [저자 adapters.py](https://github.com/abenechehab/AdaPTS/blob/main/src/adapts/adapters.py)의 `MultichannelProjector`와 torch transform/inverse 경로, [저자 adapts.py](https://github.com/abenechehab/AdaPTS/blob/main/src/adapts/adapts.py)의 `adapter_supervised_fine_tuning` 예측·손실·최적화·선택 경로를 읽었다. 최종 proceedings PDF는 접근 실패하여 최종판 전체 독해로 세지 않는다.

[확인] 잠재 공간으로 변환→단변량 foundation model→원채널 출력 변환, 미래 예측 목표로 adapter 학습은 이미 겹친다. 이번에는 별도의 고정 G를 보존하고 보정 출력을 0으로 시작하며 M_fit/M_select/full M_cal과 no-F/readout 대조를 명시한다. 이는 검증할 절차 차이이며, latent adapter나 forecast supervision 자체의 신규성 근거가 아니다. 저자 코드의 기본 scaler·random split·부가 손실을 이번 데이터 계약에 수입하지 않는다.

### Friedman과 공식 gradient boosting 설명

[2001 공식 DOI](https://doi.org/10.1214/aos/1013203451)는 Project Euclid iframe으로 연결되어 최종 방법 전문을 읽지 못했다. 실제 방법 원문은 [1999-02-24 저자 원고](https://www.cse.iitb.ac.in/~soumen/readings/papers/Friedman1999GreedyFuncApprox.pdf)의 §3/Algorithm 1, §4.1/LS Boost, §5/shrinkage를 읽었다. 또한 [scikit-learn 공식 mathematical formulation](https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation)의 가산 모델·이전 예측에 대한 손실 최소화·negative gradient 설명을 확인했다. 1999 원고를 2001 최종판 전문으로 표시하지 않는다.

[확인] 이전 예측에 함수를 더하며 제곱오차의 현재 잔차를 적합하는 원리는 표준 stagewise 학습이다. 이번 두 블록도 그 원리와 겹친다. 특이한 검증 대상은 고정 F를 통과하는 학습형 잠재 입력이 단순한 보정기보다 실제 추가 가치를 갖는지다. 같은 M_fit에서 앞 블록을 학습한 뒤 그 잔차를 맞추면 G에 대해서는 미학습 구간이어도 모든 앞 블록에 대해 out-of-sample인 잔차가 되지는 않는다.

### Nested Dropout, ICML 2014

[공식 출판 기록](https://proceedings.mlr.press/v32/rippel14.html)과 [공식 논문 PDF](https://proceedings.mlr.press/v32/rippel14.pdf)의 §2 nested truncation mixture objective와 §3의 제한된 semi-linear/PCA 관계 설명을 읽었다. 모든 증명을 재검증하거나 전체 실험을 재현하지 않았다.

[확인] 여러 길이의 앞부분 표현이 쓸모있도록 학습하는 원리는 선행이다. 이번 `JOINT_PREFIX`는 두 prefix 손실을 평균하는 대조이며 Nested Dropout의 확률적 학습을 그대로 재현한 모델은 아니다. 원 부모 E의 열 순서가 잔차 중요도 순서라는 보장이나 이 논문의 특수 PCA 결과를 현재 비선형 동결 F에 전이시키지 않는다. `JOINT_FULL`의 절반을 사후 절단한 성능만으로 순차 이점을 주장할 수 없다.

### Side-Tuning, ECCV 2020

[공식 기록](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1104_ECCV_2020_paper.php), [최종 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480698.pdf)의 잔차/boosting 관련 설명, §3과 §3.1의 base·side·alpha blending, §3.2의 incremental-learning 도입부를 읽었다. 전 과업·저자 코드 전체를 재검증하지 않았다.

[확인] 기반 모델을 고정하고 추가 네트워크를 결합하는 방식은 선행이다. 논문의 대표 결합은 학습되는 alpha blending이고, 이번 식은 G 계수1과 zero-init correction을 유지한다. 이 구체적 차이는 기존 V9에도 겹친다. 부모 보존은 선택한 새 과업의 오차가 나빠지지 않는다는 보장이 아니다.

### Cawley–Talbot, JMLR 2010

[공식 기록과 초록](https://jmlr.org/papers/v11/cawley10a.html)을 이번에 다시 읽었다. 전문은 이번 감사에서 읽지 않았다.

[확인] 유한 표본의 모델 선택 기준 자체에 과적합할 수 있다. 따라서 M_select의 checkpoint·진입·prefix 선택 노출과 반복 개발 V 사용을 기록해야 한다. 순서를 봉인해도 이미 노출된 V가 독립 TEST로 바뀌지 않는다. 두 seed 범위를 신뢰구간으로 해석하거나 겹치는 창을 독립 표본으로 세지 않는다.

### PyTorch autograd 공식 문서

[명세가 제시한 stable URL](https://docs.pytorch.org/docs/stable/notes/autograd.html)은 접근 시 2.14 문서로 redirect됐다. [2.10 버전 문서](https://docs.pytorch.org/docs/2.10/notes/autograd.html)의 `requires_grad`, grad modes, `no_grad`, eval 설명도 확인했다. 실제 사용 버전은 별도 환경 기록이 확정한다.

[확인] F parameter를 `requires_grad_(False)`로 고정하는 것과 F 연산 전체를 `no_grad()`로 감싸는 것은 다르다. 후자는 학습형 입력까지의 backward 경로를 기록하지 않는다. eval은 gradient 차단 기능이 아니다. 현재 새 경로의 실제 입력 기울기 통과는 문서 인용으로 입증되지 않으며, zero D 첫 step과 D 업데이트 뒤의 비퇴화 batch를 구분한 구현 검사로 확인해야 한다.

## 결과에서 남겨야 할 판단 범위

새로 검증하는 것은 G 뒤에 남은 예측 가능 정보를 어디에서 얼마나 경제적으로 얻는가다. `RESIDUAL_F`가 G와 기존 fitted alpha보다 좋아도 `READOUT_ONLY`가 같은 효과를 내면 P/E 추가학습의 필요성은 미확보다. Q·원채널 MLP·Ridge가 더 좋은 경우 F의 실용적 이점은 미확보다. Q와 F는 학습 파라미터 수가 다르고, alpha와 새 모델은 누적 학습량·구조도 다르므로 순수한 objective나 사전학습 유무의 인과효과로 표현하지 않는다.

C에 진입한다면 P를 A60에서 고정한 세 대조 내부에서 순차 순서와 prefix 목적을 비교해야 한다. B는 P도 학습하므로 B↔C 차이는 순차 순서만의 효과가 아니다. prefix 감소는 F 입력 행 수를 줄일 수 있으나 상주한 F 가중치 메모리를 줄이지는 않는다. K0의 G-only 메모리를 말하려면 실제 F 미로드 경로여야 한다.

이 감사만으로 확정되는 기여는 없다. 최종 기여 후보는 동일 부모·관측 집합·선택 절차·전체 경로 비용 아래에서 위 반론들을 통과하는 조건부 실증 결과다. 음의 결과, 상한까지 개선 중인 결과와 구현 실패를 구분하고, 두 개발 자료의 관측 차이를 인과 설명으로 바꾸지 않는다. 최초·유일·보편적 우위·성능 비저하 보장·논문화 가능성은 이 검토에서 승인하지 않는다.
