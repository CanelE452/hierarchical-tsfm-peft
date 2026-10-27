# G2 — AdaPTS식 어댑터 충실도

[확인] 비교 기준은 AdaPTS 고정 커밋 `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`이다. 2026-10-25에 해당 커밋의 공식 파일을 다시 읽었다. 결과는 **등록한 구조에 대해 PASS**다. 원 논문 전체 실행을 재현했다는 판정은 아니다. 학습·백본·전처리 차이는 `DEVIATIONS_FROM_ADAPTS.md`에 공개한다.

공식 소스: [adapters.py](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py), [adapts.py](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py).

```text
검사 항목          공식 구조                         이번 구현                            판정
C1 PCA             sklearn PCA 중심화·역변환          A60 중심화 PCA·E전치 역변환          PASS
C2 층              C→K 선형, K→C 선형                AdaPTSLinearPort 동일 차원           PASS
bias               두 선형층의 기본 bias              두 선형층 bias 유지                  PASS
초기화             nn.Linear 기본 초기화              기존 create_model의 기본 초기화      PASS
C3 dropout 위치    인코더 선형층 직후                  EncoderDropout, 인코더 출력에만 적용 PASS
C3 확률            기본 p=0.1                         p=0.1                                PASS
학습 표본          한 번의 stochastic transform       update당 한 표본                     PASS
예측 평균          stochastic latent forecast 평균    10개 latent forecast 평균 후 decode PASS
```

[확인] 공식 `LinearAutoEncoder`는 직접 만든 두 `nn.Linear`를 사용하며 별도 재초기화가 없다. `DropoutLinearAutoEncoder`는 인코더 선형층 뒤에만 dropout을 둔다. 현재 확인한 공식 파일의 위치는 각각 대략 331–391행과 520–583행이다. 입력 문서의 줄번호와 달라도 클래스·구조는 동일하다. 공식 `predict_multi_step`은 dropout을 활성화하고 기본 10개 잠재 예측을 평균한 뒤 역변환한다. [고정 어댑터 소스](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py#L331), [고정 예측 소스](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L160).

[확인] 로컬 구현 근거는 `research/informs_broad_validation_20261015_v1/networks.py:352`, `fit_base.py:55`, 새 `research/comparative_ablation_20261025_v1/models.py:50`이다. C3는 평가 전체를 감싸는 `evaluation_scope`에서 실행 seed+7000000으로 시작하고 RNG 상태를 복원한다. dropout-off V는 부가 진단 전용이다. `test_models.py`의 실제 13개 합성 시험 통과와 `G5_REVIEW.json`의 C3 표본·RNG 검토가 이 동작을 뒷받침한다.

## 중심화 PCA 수치 검사

[확인] 각 데이터의 `[0,A60_end)` 관측 통계로 정규화하고 인과적으로 결측을 채운 **동일한 행**에 sklearn `PCA(n_components=K, svd_solver='full')`를 적합했다. 부호 불변인 `E Eᵀ`와 중심화 재구성을 비교했다. V/T는 PCA 적합에 쓰지 않았다.

```text
데이터              투영행렬 최대 절대차    재구성 최대 절대차
jena                1.6903138000e-8          2.8583936107e-7
ETTh1               2.9112541045e-8          2.9468028373e-7
ETTh2_safe17400     4.1844809395e-8          3.7643758244e-7
appliances          4.5446194807e-8          2.6255415353e-7
```

[확인] 네 데이터 모두 기준 `1e-4` 이내다. 정확한 값과 적합 행수는 `PCA_SUBSPACE_VALIDATION.json`, 실행 코드는 `research/comparative_ablation_20261025_v1/smoke.py`에 있다. C1/C2/A1의 실제 F 포함 V 한 배치 순전파도 12/12 통과했으며 F 전후 고정 확인을 통과했다(`STEP0_SMOKE.json`). 이 검사는 정확도 결과나 학습 수렴을 판정하지 않는다.
