# 방법 근거와 주장 범위

2026-10-13. 아래는 구현·평가 결과와 분리한 선행 자료 확인 기록이다. 미래 성능 결과를 보고 방법이나 새 평가 자료를 바꾸기 위한 기록이 아니다.

- [NLinear 공식 구현](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py): 입력의 마지막 관측값을 빼고 선형 예측 뒤 다시 더한다. [판단] 채널별 현재 상태를 보존하고 작은 K로 미래 변화만 예측하는 출발점에 근거를 제공한다. 이번 `x_last + D(F(E(X))) - D(E(X)_last)`와 같은 구조의 재현이나 성능 보장은 아니다.
- [ExoTimer 공식 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869): 식13–16의 학습 가능한 hash projection, hash 기반 유사도, 연속 표현 결합 및 두 예측 branch를 확인했다. 별도 대조학습 손실을 본문에 없는 형태로 보충하지 않는다. 이번 FIT 미래 요약에 KMeans prototype을 적합하고 student가 soft partition을 맞추는 방법은 착안한 변형이다. 원 Bi-Hash의 구현·재현·우월성 증명이 아니다.
- [CPC 원저](https://arxiv.org/abs/1807.03748): 과거로부터 미래 잠재 표현을 예측하는 선행 학습 목적이다. 이번 고정 미래 변화 좌표·요약에 대한 MSE는 변환된 미래 정답을 맞추는 보조 회귀다. 미래 표현을 학습에 사용한다는 사실 자체를 새 원리나 비지도 입력 복원과 동일한 목적이라고 주장하지 않는다.
- [VICReg 원저](https://arxiv.org/abs/2105.04906): 표현 학습에서 collapse를 별도로 제어하는 근거다. 이번에는 basis·FIT 표준화·prototype을 고정하고 variance, effective rank, entropy, occupancy와 상수 분포 기준을 기록한다. 이 검사 통과가 모든 정보 붕괴를 배제하지 않는다.

새 자료의 공식 출처는 [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)과 [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)다. 두 자료는 CC BY 4.0이다. Appliances의 두 random variable을 제외하고, Beijing 관측소는 Aotizhongxin으로 성능 관찰 전에 고정했다. 실제 원본 SHA·채널·행 수·간격·결측·분할은 다운로드 후 생성하는 계약에 근거하며 사전 예상치를 실제 자산 확인으로 대체하지 않는다.

## 새 평가의 독립성

[확인] 기존 저장소의 직접 사용 흔적을 조사해 두 새 자료를 선정했고, 방법 선택 이후에만 원본을 내려받도록 실행 계약을 고정했다. 이는 우리 개발 비교와 선택에서의 독립성이다. 새 자료에서도 FIT/TUNE로 모델을 다시 적합하며 zero-shot 전이 실험이라고 부르지 않는다.

[Chronos-Bolt-small 공식 model card](https://huggingface.co/amazon/chronos-bolt-small)는 대규모 사전학습과 zero-shot benchmark를 설명한다. 함께 공개된 [Chronos 데이터 카드](https://huggingface.co/datasets/autogluon/chronos_datasets/raw/main/README.md)의 구성 이름에서 `appliance`, `beijing`, `air_quality`를 검색했지만 직접 일치는 찾지 못했다. 공개 [추가 자료 저장소](https://huggingface.co/datasets/autogluon/chronos_datasets_extra)는 학습과 평가용 자료 구성 코드를 함께 제공한다.

[판단] 이름 검색의 음성 결과만으로 정확한 로컬 Bolt snapshot의 전체 사전학습 corpus와 새 EVAL의 비중첩을 증명할 수 없다. 이 부분은 미확정으로 남기며, foundation 사전학습까지 완전히 독립된 일반화 검증이라고 주장하지 않는다. RAW Bolt와 동일한 동결 백본을 비교하는 압축군 사이의 통제와, 새로 적합한 ridge/native 기준군과의 차이를 각각 보고한다.
