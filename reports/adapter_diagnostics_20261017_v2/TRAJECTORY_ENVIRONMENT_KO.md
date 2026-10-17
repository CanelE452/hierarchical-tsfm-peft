# B03 trajectory 재집계 실행 환경 확인

[확인] 같은 SHA의 V43 feature cache와 원 decoder D를 두 환경에서 읽어 평가했다. 새 모델·F 로드·optimizer·학습은 모두 0회이며 기존 점수·공차·과학 소스는 변경하지 않았다. 실행 시각과 원본 stdout SHA는 [검증 JSON](TRAJECTORY_ENVIRONMENT_CHECK.json)에 기록했다.

캠페인의 지정 venv는 Python 3.11.16 / NumPy 2.4.6 / OpenBLAS 0.3.31.188.0(Haswell, 2 threads)이다. 기본 `python`은 Python 3.13.9 / NumPy 2.3.5 / MKL 2025.0-Product(2 threads)였다. 별도 Python/NumPy/BLAS 변경의 인과 기여까지 분리한 검증은 아니다.

```text
seed   지정 venv native 최대 점수차   MKL native 최대 점수차   MKL strict 1e-12
92601  1.1102230246251565e-16          4.105846551638592e-9      FAIL
92602  0                              5.213079523613828e-9      FAIL

seed   두 환경에서 동일한 FP64 MSE   두 환경에서 동일한 FP64 MAE
92601  0.6834428892268574             0.4938849193026718
92602  0.6788139037779285             0.5018712131503099
```

[확인] 두 환경 모두 각 cache/bundle의 바이트 SHA와 scaler 결속을 확인했다. 실제 상대 `cache://`, `asset://`, `report://`, `repo://` reference와 SHA, 기존 producer 점수, native/FP64 점수 및 차이는 검증 JSON의 각 records/public_sources에 있다. 두 환경의 FP64 점수는 정확히 같았고 기존 producer와의 차이는 각각 최대 3.4878531174342697e-9, 1.2567556995080054e-9였다. 기존 cross-precision 공차 1e-6 안에 든다는 사실과 MKL native strict 실패는 별개다.

재현 시 저장소 루트에서 캠페인의 원래 venv Python 실행 파일을 `CAMPAIGN_PYTHON`으로, 승인된 자산 루트를 `DIAGNOSTICS_ASSET_ROOT`로 지정한다. JSON의 verification_python_source를 변경 없이 `VERIFICATION_PYTHON_SOURCE` 문자열에 넣고 실제 수행한 다음 두 호출을 사용한다. OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS는 각각 2였다.

```text
CAMPAIGN_PYTHON -c VERIFICATION_PYTHON_SOURCE
python -c VERIFICATION_PYTHON_SOURCE
```

원본 stdout은 JSON에 UTF-8 문자열로 포함했다. stdout_sha256은 subprocess가 받은 원바이트(플랫폼 줄바꿈 포함)의 SHA256이다. 공개 파일에는 개인 절대경로가 없다. native 1e-12 재현 확인 범위는 위 지정 venv이며, 다른 환경의 동일 재현을 보장하지 않는다. MKL의 수치 차이를 새로운 과학적 효과나 성능 변화로 해석해서는 안 된다.
