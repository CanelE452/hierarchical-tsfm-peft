# 같은 frozen F의 점예측 요약과 MSE 목표를 분리

## 목적과 기존 근거

최신 방법론 완성·계속 실행 지시 안의 별도 원인 대조다. 보호 TEST·다운로드/설치·대형 모델 학습·기존 작업 종료·자동 Git 금지는 유지한다. 원 E/D·scaler·origin·관측 mask·MSE 지표를 바꾸지 않는다.

기존 실제 FrozenBolt는 모델이 반환한9분위 중0.5, 처음48시점을 점예측으로 쓴다. 실제 설치의 pipeline.predict_quantiles가 반환하는 mean도 median이라는 주석이 있어 그것을 조건부 평균으로 쓰면 안 된다. 기존 MSE/MAE/Huber 학습 대조는 decoder/codec 손실을 바꿨으며 full9 요약의 실제 비교는 SOURCE/metadata 조사에서 확인하지 못했다. 예전 quantile-projection 반례는 합성이며 실제 원인 증명이 아니다.

목적은 (1) 재구성/표현 부족 외에 점예측 함수의 목표 불일치가 있는지 (2) 같은 통계 요약을 Raw에도 적용했을 때 압축 경로의 상대 이득이 남는지 확인하는 것이다. MSE에서 평균을 쓰는 원리는 선행 통계 이론이며 새로운 방법론 기여라고 주장하지 않는다. [Gneiting(2011)](https://arxiv.org/abs/0912.0902), [AdaPTS](https://proceedings.mlr.press/v267/benechehab25a.html)의 범위를 함께 검토한다.

## 고정 입력과 출력 API

Robin C17/K4, L512/H48, 원 FIT256/TUNE64/기노출 DEV30을 유지한다. Raw-C와 원 RECON CP0/E-only92601/E-only92602 세 frozen E/D의4경로다. encoder·decoder·F·Q는 학습하지 않고 Q는 identity다. raw에 원 채널 F, 세 E/D에는 같은 latent history의 F를 적용한다. 현재 C anchor·원 old-past buffer·lag별 원점·원 L7 D를 유지한다.

기존 로드된 frozen bolt.model(context=rows).quantile_preds를 직접 호출한다. rows는 원 wrapper와 같은 z.transpose(1,2).reshape(N*K,L)이고 모델 내부 normalization/inverse를 그대로 사용한다. 추가 수동 정규화나 다른 pipeline은 사용하지 않는다. 모델 config의9수준 .1,.2,...,.9와 실제 [N*K,9,64]·FP32·finite를 검사한다. 로컬 바이너리·config·F dual hash와 normalization contract를 전후 검사한다. 새 모델은 로컬 검증된 동일 가중치1개이며 backbone 학습은0이다.

모든 경로에서 canonical CUDA B32로 full9×64를 실제 저장한다. 이 전체 텐서는 이번에 실제 생성된 자료이며 예전 cached median을 full quantile 결과처럼 쓰지 않는다. 첫48의 median/최종 C 출력은 예전 accepted actual cache와 등록된 atol1e-5/rtol2e-6으로 검산한다. Raw의 원 과거/mask/scaler/origin/미래와 E/D 경로의 source binding을 그대로 검사한다. 전체64 저장은 정보를 보존하는 캐시이며 이번 평가 horizon은48이다. 뒤16시점을 이용하는 새 방법이 실행됐다는 뜻은 아니다.

## 사전 고정한 세 점요약

```text
MEDIAN              weight = [0,0,0,0,1,0,0,0,0]
NINE_MEAN_PROXY     weight = [1/9,...,1/9]
TAIL_CLAMPED_PROXY  weight = [.15,.1,.1,.1,.1,.1,.1,.1,.15]
```

세 번째는 수준0.1~0.9 사이를 선형 보간하고 양 끝 tail을 각각 Q(.1)/Q(.9) 상수로 가정한 적분이다. 두 proxy를 실제 조건부 평균이라고 부르지 않는다. finite quantile9개와 tail 가정만으로 실제 평균을 식별할 수 없으며 crossing도 별도 보고한다. 분위 수준을 결과에 따라 정렬하거나 tail 외삽을 맞추지 않는다.

old decoder는 미래 latent에 선형이어서 같은 조건부 확률변수의 기대값과 교환 가능하지만, per-stream F는 각 latent의 과거만 조건으로 보고 전체 K 과거 조건부 기대값과 같다는 보장도 없다. median과 일반 선형 복원은 교환하지 않는다. 음의 D와 미정 공동 의존성 때문에 같은 τ끼리 decode한 C 값을 실제 C quantile/CRPS 분포로 보고하지 않는다. 이번에는 점예측 MSE/MAE만 비교한다.

## 공정한 선정과 실제 실행

네 경로 모두 같은3후보를 고정한다. FIT/TUNE24개 후보 점수를 모두 저장하고 각 경로의 TUNE observed-channel-macro normalized MSE가 가장 작은 summary를 고른다. 정확 동률이면 MEDIAN→NINE_MEAN_PROXY→TAIL_CLAMPED_PROXY 순서다. 네 선택 잠금·registry가 끝나기 전 DEV 원배열·full quantile을 열지 않는다.

DEV에서는 사전 등록된 세 요약을 모두 진단용으로 보고하되, 최종 방법은 TUNE에서 고른4개뿐이다. 나머지8개는 비선정 진단이고 DEV에서 승자를 바꾸지 않는다. 같은 자료의12행을12개 독립 검증으로 세지 않는다. 기존 Raw median뿐 아니라 같은 summary 메뉴로 선정한 Raw를 주 대조로 보존한다. compressed 메뉴만 바꾸고 Raw를 약한 median에 고정해 이득을 만들지 않는다.

fit이나 optimizer는0이다. full9 저장값에서 CPU NumPy의 독립 가중합과 old lag decode를 계산해 actual CUDA 점요약/C 결과를 검산한다. 미관측 target 변경은 forward·summary·선정에 들어가지 않아야 하며 평가 분모는 각 split 자신의 관측수다. source에서 성능 개선은 실행 성공 조건이 아니다. 경로 실패는 해당 artifact/이유를 보존하고 독립 경로를 계속한다.

모든 후보의 MSE/MAE·per-C·원단위 진단·selected flag·full quantile crossing/범위·실제 자원·원본 binding을 저장한다. 이 과학 실행은 whole-deployment latency 측정이 아니며 cached summary 산술 시간을 실제 F 전체 비용으로 주장하지 않는다. 실제 비용의 후속 측정 필요성과 이번의 미측정을 명시한다.

## 예측한 반례와 다음 판단

- proxy가 양쪽 모두 좋아져도 표현 호환성이 해결됐다는 뜻은 아니다. 같은 선택 Raw와 압축을 다시 비교한다.
- proxy가 FIT에서만 좋아지면 그 현상을 숨기지 않는다. tail 또는 보정 계수를 계속 늘리기 전 재평가한다.
- summary에 따라 압축 격차가 크게 달라져도 median 하나가 유일 원인이라고 단정하지 않는다. 고정 D 공간의 반대 근거와 기존 학습 실패도 유지한다.
- 세 summary가 비슷하거나 더 나쁘면 이 제한된 통계 요약의 우선순위를 낮춘다. true mean 추정·모든 probabilistic adapter가 불가능하다는 결론은 아니다.
