# R1 구현과 원문 대응

작성 2026-10-15. 본 문서는 공식 자료·새 코드 대응과 작은 합성 검사의 범위를 기록한다. 실제 자료의 성능·비용은 이 문서에서 측정하지 않았다.

## SPRINT의 확인 범위와 고정 revision

[확인] [PMLR 공식 서지·초록](https://proceedings.mlr.press/v306/xu26ac.html)을 열람했다. PMLR PDF 링크는 브라우징 도구에서 `Unsupported content-type: application/octet-stream`으로 실패했고 OpenReview는 검증 화면을 반환했다. 따라서 논문 본문·전체 실험을 읽었다고 주장하지 않는다. 공식 자료의 성능 수치를 이번 Bolt 포팅의 예상 개선율로 사용하지 않는다.

[확인] GitHub API에서 실행 전 `main`의 commit과 파일 tree를 확인했다. 계획에 적힌 두 blob과 일치했다.

```text
repository    NetManAIOps/SPRINT
commit        1cc1274719458f0cc820d922b974ae2b2e040306
SPRINT.py     623494941d55f1b537c8feafcb3757191c316de8
Chronos.py    cb89b8f54c0fda5804e25f781f082d65108f5464
run.py        be3e285e6aa32b2a721da09d0a78011c60d5b5b1
LICENSE       dd6c4466b13ee386b60eba90ded749d4ade85d09
```

[고정 SPRINT.py](https://github.com/NetManAIOps/SPRINT/blob/1cc1274719458f0cc820d922b974ae2b2e040306/models/SPRINT.py), [고정 Chronos.py](https://github.com/NetManAIOps/SPRINT/blob/1cc1274719458f0cc820d922b974ae2b2e040306/models/Chronos.py), [고정 run.py](https://github.com/NetManAIOps/SPRINT/blob/1cc1274719458f0cc820d922b974ae2b2e040306/run.py), [MIT 고지](https://github.com/NetManAIOps/SPRINT/blob/1cc1274719458f0cc820d922b974ae2b2e040306/LICENSE).

[확인·공식 코드 대응] `sprint_predict`는 다음 계산을 대응시킨다.

- `cycle//2` 이동평균: 양끝 반복 패딩의 비대칭 1행 차이까지 보존한다. R1의 trailing 평균과 다른 연산이다.
- 잔여 성분: 완전한 과거 주기들을 `0.9` 감쇠 가중 평균한 뒤 미래에 반복한다.
- trend: `cycle//4` 간격 추출 → F → 역사/미래 support의 자연 3차 spline → 원격자 복원이다.
- 표준화: 공식 `use_revin=1` 기본값과 sample variance(`correction=1`), `1e-5`를 보존한다.
- `label_len=0`, 미래 지원점 수 `H//stride+2`를 사용한다.

[변형] F는 공식 기본 `chronos-t5-small`의 sampling/집계 wrapper 대신 기존 frozen Chronos-Bolt의 q0.5다. 자연 spline은 설치가 필요한 `torchcubicspline` 대신 같은 natural 경계 선형식을 torch로 계산한다. 합성 입력에서 SciPy natural spline과 대조했다. 공식 전체 패키지나 모델을 설치·다운로드하지 않았다. 따라서 이름은 **SPRINT-on-Bolt**이고 공식 성능 재현이 아니다.

[계약] period는 원자료의 사전 sampling metadata 또는 A 구간만으로 정해야 한다. 구현은 period≥8 및 4의 배수를 요구하고, 512를 period 배수로 왼쪽 절단한다. period24면 유효504행·stride6·F 미래10행, period144면 유효432행·stride36·F 미래3행이다. 같은 origin/유효 lookback의 Raw가 필수 대조다. period 출처를 확인하지 못하면 A11만 BLOCKED다. 함수에 미래값·미래 mask·B/C 자료가 들어가는 경로는 없다. 과거 전체의 대칭 평균은 origin 기준 과거 전용이지만 각 역사 행의 prefix invariance를 주장하지 않는다.

## R1 시간 좌표와 운영적 분해

새 [transforms.py](transforms.py)의 A1–A6은 경계에서 실제 존재하는 과거로 나누는 trailing 평균과 `R=X-T`를 사용한다. 따라서 `T+R=X`이며 두 성분이 실제 계절/잡음의 정답이라는 가정은 없다. A1/2는 `F(T)+R_last`, A3/4는 `F(R)+T_last`, A5/6는 `F(T)+F(R)`다. A5/6는 두 배 streams·두 번 F 호출을 metadata에 남긴다.

A7/8의 coarse endpoint는 마지막 관측 `o-1`에 정렬한다. stride4에서 coarse 미래 지원점은 `o+3,o+7,...`이고 현재 관측과 선형 보간해 `o,...,o+47`을 반환한다. A9/10는 미래 block 평균을 2/4행 반복하며 endpoint라고 부르지 않는다. 모든 방법의 target은 원미래이고 평균 target으로 바꾸지 않는다. A12/13의 origin 공통집합은 controller가 보장해야 한다. 요청 horizon crop만큼 Bolt 고정 head 계산량이 줄었다고 해석하지 않는다.

## TimeMixer 원문과 이번 B4의 관계

[확인] [arXiv 2405.14616v1 HTML](https://arxiv.org/html/2405.14616v1)에서 §3.1, Figure1, §3.2, Figure2, AppendixF와 [공식 저장소](https://github.com/kwuking/TimeMixer)를 확인했다. 첨부 PDF의 파일 존재·페이지 픽셀을 열었다고 주장하지 않는다. HTML 원문 버전은 계획에 적힌 v1과 같다.

```text
원문 구성                  이번 실험에서 확인할 대응
§3.1 / Figure1              average pooling으로 여러 해상도의 과거 생성
§3.2 / Figure2, PDM         분해 후 잔여 fine→coarse, trend coarse→fine
§3.3, FMM                   B4 앞단에서는 사용하지 않고 frozen Bolt로 대체
AppendixF                   분해/혼합 가지 제거를 matched 재학습 대조로 검사
```

[해석] 원격자로 융합하는 과거→과거 앞단과 E/D를 붙인 B4는 Bolt용 변형이다. 전체 TimeMixer나 원문의 FMM 예측 성능을 재현한 것으로 쓰지 않는다. 모델 담당자의 실제 `models.py` 대응과 parameter 수는 별도 코드·실행 기록으로 확인해야 한다.

## 작은 검사 결과

[확인] `.venv/Scripts/python.exe -X utf8 research/temporal_frontend_campaign_20261015_v1/test_transforms.py`는 10개 합성 검사, 종료코드0이었다. trailing prefix·가법 identity·현재값 보존·endpoint 물리 좌표·block 평균 의미·가변/고정 horizon wrapper·자연 spline/SciPy 동등성·SPRINT 수동 계산/절단·상수·shape/streams·잘못된 period 거부를 검사한다. 실제 F·실제 자료·GPU·학습·보호 TEST를 사용하지 않았다. 이 PASS는 구현의 검사 범위이며 성능 개선·학습 수렴·독립 일반화의 증거가 아니다.
