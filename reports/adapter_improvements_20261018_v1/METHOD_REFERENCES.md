# 방법론 근거와 공식 소스 결속

작성일: 2026-10-18. `[확인]`은 아래 명시한 본문·공식 코드 또는 실제 검증 범위다. 논문의 결과가 이번 데이터·동결 백본에서도 성립한다는 뜻은 아니다. 실제 캠페인 커버리지와 효과는 RUNS/RESULT_ANALYSIS를 따른다.

## DCA 초기화

[Clark, Livezey, Bouchard, NeurIPS 2019 본문](https://proceedings.neurips.cc/paper_files/paper/2019/file/704cddc91e28d1a5517518b2f12bc321-Paper.pdf)의 §2에서 Gaussian predictive information을 최대화하는 선형 부분공간이라는 근거를 확인했다. 이번 적용은 E의 초기화이며, 이후 P/E/D의 기존 미래 macro MSE 학습을 유지한다. 논문의 benchmark 재현 또는 미래 예측 개선 보장이 아니다.

[공식 DCA 저장소](https://github.com/BouchardLab/DynamicalComponentsAnalysis/tree/a140e6fa9524a3bd6d043ae9e96f6e149b9c3745)의 commit `a140e6fa9524a3bd6d043ae9e96f6e149b9c3745`를 고정했다. 읽고 직접 호출한 파일은 `src/dca/base.py`, `dca.py`, `cov_util.py`다. [OFFICIAL_SOURCE_MANIFEST.json](OFFICIAL_SOURCE_MANIFEST.json)은 전체 수신 파일 URL, 바이트 수, SHA256 및 tree 조회 receipt를 보관한다. 코드와 `LICENSE.txt` 원문은 새 private `.cache/adapter_improvements_20261018_v1/official/DCA/`에 보존한다. 라이선스는 UC/LBNL copyright, source/binary 고지 유지와 비보증 조건을 포함한다. 공개 wrapper는 upstream 코드를 복제하지 않고 검증한 private 모듈을 import한다.

[확인] 공식 생성자에 `d=K,T=5,init=PCA ndarray,n_init=1,stride=1,chunk_cov_estimate=None,tol=1e-6,ortho_lambda=10,verbose=False,block_toeplitz=False,method='toeplitzify',device='cpu',dtype=torch.float64,rng_or_seed=0`을 사용한다. `regularization=None`은 `estimate_data_statistics` 인자다. `maxiter/maxfun`은 생성자 인자가 아니다. `init_coef`는 문자열 `PCA`를 구현하지 않아 해당 scope의 실제 PCA 배열을 넘긴다.

[확인] 단일 시간 stream에서 공식 estimator는 `form_lag_matrix(X-mean,2*T)` → `np.cov(...,rowvar=False)` → `toeplitzify` → `rectify_spectrum` → lag covariance 추출 순으로 실행된다. 최소 eigenvalue가 0 이하일 때만 `(-minimum+1e-6)I`를 더한다. 새 wrapper는 이 함수를 직접 호출하며, rectification 전후 고유값·shift와 symmetry/Cholesky/slogdet sign을 추가 관측한다. 다른 covariance estimator·whitening·latent RMS 보정은 사용하지 않는다.

[확인] 공식 `build_loss`의 `-PI + 10*||VᵀV-I||²`를 그대로 호출한다. 공식 `_fit_projection`의 L-BFGS-B를 얇게 외부에서 실행해 maxiter200/maxfun1000/ftol1e-6과 optimizer status/iterations/nfev/objective를 노출한다. 이 상한과 fit당120초는 이번 운영 규칙이다. 별도 spawn CPU 프로세스의 deadline은 source import·covariance·solve를 포함하며, parent는 초과 시 프로세스를 종료한다. 실패·시간초과·비수렴은 COMPLETE가 아니며 PCA로 대체하지 않는다.

[확인] upstream `ObjectiveWrapper.grad`는 cached leaf에 `backward(retain_graph=True)`를 반복하고 `.grad`를 비우지 않는다. 작은 합성 사례에서 두 번째 gradient가 첫 번째의 2배가 됨을 재현했다. 새 `FreshObjective`는 공식 loss를 유지하면서 fresh leaf와 `autograd.grad`를 사용하고 detached 값/gradient만 cache한다. 같은 점 반복 gradient, 공식 첫 gradient, 중앙 유한차분을 대사한다. no-grad callback graph는 cache하지 않는다.

[확인] 기존 `data_contract.fit_statistics/causal_forward_fill/initialize_pca`의 FP32 fill → FP32 `(filled-mean)/std` → FP64 stream 순서, PCA FP32 center/basis를 맞춘다. 공식 covariance 계산이 stream을 내부에서 한 번 중심화한다. 모델의 PCA center는 그대로 유지하며, 모델 입력에서 mean을 추가로 빼지 않는다. A_fit과 A60은 별도 cache이며 fitting prefix 밖 lag pair를 만들지 않는다. data/mask/time/channel order/scaler/counts/center/PCA/설정/회전/producer SHA/의존성 버전을 cache key에 포함한다. 같은 scope의 deterministic 초기화만 downstream seed 사이에 공유한다. 로더는 hash와 scope를 검사하는 읽기 전용 함수다.

[SciPy orthogonal_procrustes 공식 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html)를 확인했다. 이번 규칙은 thin orthonormal Q 뒤 `QᵀE_PCA=USVᵀ`, `E_DCA=QUVᵀ`이며 scale은 곱하지 않는다. rank K, alignment rank/비유일성, 회전·E64 hash, subspace/orthogonality/PI 보존을 기록한다. E64를 E32로 cast한 뒤 실제 model.E의 같은 dtype/device에서 `torch.linalg.pinv`를 계산하고 D의 다른 lag를 0으로 둔다. private 저장 D0_32는 CPU 준비 진단값이며 실제 CUDA 설치 pinv와 bitwise 같다고 가정하지 않는다. PI 기저 회전 불변성은 단변량 F의 축별 예측 불변성을 뜻하지 않는다.

## LIFT 공식 출력 보정 준비

[Zhao, Shen, ICLR 2024 본문](https://proceedings.iclr.cc/paper_files/paper/2024/file/b52b07a239a7afa155ca25cf17a55074-Paper-Conference.pdf)의 lead-lag 정렬과 frequency-domain refiner 개념을 확인했다. [공식 소스 commit](https://github.com/SJTU-DMTai/LIFT/tree/79cf8f157bb9a616c9733daf1e9c2cfefdc98c94) `79cf8f157bb9a616c9733daf1e9c2cfefdc98c94`의 `models/LIFT.py`, `models/DLinear.py`, `util/lead_estimate.py`, `exp/exp_basic.py`, `run_longExp.py`, `settings.py`, `util/tools.py`, Weather recipe를 읽었다. 고정 tree에서 LICENSE 이름의 파일은 발견되지 않았다. 라이선스를 추측하지 않으며 공식 코드를 공개 패키지에 복제하지 않았다. private source cache와 파일 hash는 manifest에 명시한다.

[확인] `Model.forward(x,...,y_hat=...)`는 외부 y_hat가 제공되면 backbone 호출을 건너뛴다. X는 `[B,L,C]`, y_hat는 anchor까지 더한 최종 `[B,H,C]`이며 같은 scaler 좌표다. 내부에서 `[B,C,L/H]`로 전치한다. `LeadRefiner.forward`는 X의 instance mean/std로 X와 y_hat를 정규화하고, `shifted_leader_seq`는 context와 제공된 prediction만 연결해 leader를 정렬한다. 미래 target을 입력하지 않는다. 반면 `FilterFactory`의 state classifier에는 원 X가 들어간다. 따라서 refiner 전체의 채널별 offset 불변성을 주장하지 않는다. 원 C채널 context와 그 메모리 경로를 우회 제공하므로 모든 정보가 K차원 압축에서만 온다고 설명할 수 없다.

[확인] `scripts/LIFT/PatchTST/Weather.sh`의 H48 frozen recipe는 L336, n_leaders4, states20, temperature1, Adam lr0.001이다. `settings.get_hyperparams`가 Weather/PatchTST batch128을 적용하고, parser는 최대100epoch/patience5이며 freeze+pretrain에서 schedule type3을 사용한다. type3은 epoch3까지 base lr, 그 뒤 epoch당0.9 감소다. n_leaders는 압축 latent K와 별개다. 현재 캠페인의 L512/B2 및 짧은 endpoint 학습은 이 recipe의 이식이며 논문 표의 직접 재현이 아니다. README는 논문 제출 후 방법을 수정했고 모든 실험을 재실행하지 않았다고 명시한다.

[확인] preparation bridge는 private official 모듈을 그대로 import하며, 외부 최종 예측을 받는 경로에만 연결한다. 합성 forward/complex64 gradient, prediction 입력 gradient, RNG 복구, 원채널 축, 미래 tail canary 및 실제 backbone counter를 검증한다. 복소 parameter의 element 수·실수 자유도·bytes를 구분한다. 공식 초기화를 바꾸거나 CP0 identity를 강제하지 않는다. 이 준비는 refiner 훈련·V 성능·whole-path 비용 검증을 뜻하지 않는다. 실제 학습은 필수 P/E/D·PeMS 대응 묶음과 예약 예산을 확보하고 같은 B2 예측에 대한 output-gain 대조를 봉인한 뒤에만 입장한다.

## 다른 방법과 해석의 근거

- [Google TSMixer 원문/초록](https://arxiv.org/abs/2303.06053): 시간·feature 혼합 아이디어의 근거다. 동명 IBM 논문과 구분한다. 기존 `networks.py`가 기록한 Google source `5b09c22d73a9d35eb6c5d2a99b95677a45053466`의 부분 변형 B2를 재사용한다. 원 L→L 혼합을 폭 제한 MLP로 바꾼 기존 경로이며 이번 신규 논문 재현이 아니다. 이번 확인 수준은 초록과 기존 프로젝트의 공식 코드 매핑이다.
- [TimeMixer 원문/초록](https://arxiv.org/abs/2405.14616): decomposition/multiscale mixing 아이디어의 근거다. 기존 B4는 source `e24610583b36fdd8c76cc17a8df4e65759a5f460`의 일부 개념 변형이다. 새 causal local P의 kernels9/25·width32·반신호 대조는 이번 설계다. 원 PDM/FMM 전체 구현이나 논문 recipe를 재현하지 않는다. 이번 확인 수준은 초록과 기존 코드 주석 매핑이다.
- [AdaPTS, ICML2025 초록/출판 메타데이터](https://proceedings.mlr.press/v267/benechehab25a.html): 단변량 foundation model을 다변량 예측에 외부 어댑터로 연결하는 관련 연구다. 기존 port와 비교의 범위를 유지하며 이번 후보의 새 직접 재현으로 등록하지 않는다. 새 upstream 코드 다운로드 없음.
- [Matryoshka Representation Learning, NeurIPS2022 초록](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html): 중첩 표현의 개념적 근거다. 원 비전/언어 실험과 기존 시계열 multi-K 적용을 구분한다. 이번은 기존 receipt의 노출·저장·K전환 감사이며 신규 MRL 구현이 아니다. 새 upstream 코드 다운로드 없음.
- [Regularized Multi-Task Learning, KDD2004 본문](https://flora.insead.edu/fichiersti_wp/inseadwp2004/2004-11.pdf): 공유 계수와 task 편차를 규제하는 원리의 근거다. 이번 D_LINEAR/D_GAIN의 고정 horizon basis와 global observed-count 정규화, lambda/gamma grid는 이번에 등록한 설계다. 논문 구조를 그대로 재현했다고 하지 않는다. 새 upstream 코드 다운로드 없음.
- [An Analysis of Linear Time Series Forecasting Models, ICML2024 초록](https://proceedings.mlr.press/v235/toner24a.html): 강한 선형 기준선을 유지하는 관련 분석이다. 현재 Ridge의 우위/열위에 대한 데이터 오류·유일한 인과 설명을 제공하는 것으로 확대하지 않는다. 새 upstream 코드 다운로드 없음.

## 의존성과 수신량

DCA requirements는 numpy/scipy/h5py/pandas/scikit-learn/matplotlib/torch이며 버전 pin이 없다. LIFT requirements는 numpy1.23.5/pandas1.5.3/scikit-learn1.2.1/tqdm4.65.0/einops0.6.1/pytorch2.0.0 계열을 적는다. 이번 준비는 기존 프로젝트 환경을 사용하고 새 패키지를 설치하지 않았다. 실제 NumPy/SciPy/Torch/sklearn 버전은 DCA cache identity에 저장한다. paper 시점 환경과 동일하다고 주장하지 않는다.

공식 소스/tree/라이선스 수신량 합계는 manifest `sources[*].download_bytes`의 합으로 계산한다. 현재305,232bytes이다. 웹 본문 열람은 모델/데이터 asset 수집이 아니며 새 대형 backbone/data 다운로드는 없다. 자동 소스 다운로드는 runner import·no-argument 실행·deployment loader에서 발생하지 않는다.
