# AdaPTS 원 실행과 이번 비교의 차이

이번 C1–C3는 같은 고정 Chronos-Bolt-small·같은 데이터·같은 선택 규칙에서 비교하는 **AdaPTS식 어댑터**다. 원 논문 재현이나 AdaPTS 일반 성능의 판정으로 표현하지 않는다. [AdaPTS 논문](https://proceedings.mlr.press/v267/benechehab25a.html).

공식 코드 기준 커밋은 `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`이다. 구조 동등성의 범위와 실측 PCA 오차는 `G2_FIDELITY.md`에 있다.

```text
항목              원 실험·공식 구현                   이번 캠페인
F                 논문 Moment 비교; 코드 다중 백본    고정 Chronos-Bolt-small, 같은 F_SHA
학습 일정         기본 최대 300, early stopping       C2/C3: 0/.25/1/3/6 checkpoint
선택·재학습       원 구현의 validation 절차           A_stop 최소, 정확한 동률은 이른 epoch,
                                                       선택 후 fresh A60 refit
연장              원 실행과 별도                       10 seed 중 5개 이상 상한 선택 시
                                                       해당 그룹 전체를 10 epoch까지 재실행
스케일러          MinMax+Standard 축별 pipeline        각 실행의 A_fit/A60 관측 통계
RevIN             선택 가능한 옵션                     전 방법 off
batch             기본 32, config에 따라 변경 가능     effective 16 / micro 4
optimizer·lr      Adam, 기본 1e-3                      AdamW, 1e-3, weight_decay=0
lr 일정           ReduceLROnPlateau                     고정 lr, clip 1
HPO               원 계획의 HEBO·설정 탐색             실행하지 않음
K                 원 실행의 구성·탐색                   Jena6 / ETTh1·ETTh2 4 / appliances10
표본 수           stochastic 예측 기본 10              C3 A_stop/V는 10; 학습은 1
seed              원 논문과 별도                       92601–92610; 학습 없는 C1은 92601
L/H·평가 구간     원 데이터·horizon 조건               L512/H48, 개발 V, 보호 T는 미평가
```

[확인] 공식 코드의 학습 기본값, Adam·학습률 스케줄러, 스케일러와 예측 표본 수를 확인했다. [adapts.py](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L345). 공식 실행 스크립트는 백본·RevIN·batch/lr 설정을 받으며 구성 파일을 사용할 수 있다. [run.py](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/scripts/run.py). HEBO 사용의 서술은 제공된 계획의 원 실험 설명을 따르며, 이번 감사는 HEBO를 실행하거나 전체 원 논문의 탐색 절차를 재검증하지 않았다.

[확인] 이번 C3는 평가마다 seed+7000000으로 시작하며 10개 F 잠재 예측을 평균하고 디코더에 넣는다. 동일 평가 재실행의 재현성을 위한 고정 규칙이다. dropout을 끈 V 수치는 주 판정에 쓰지 않는다.

[추정] 짧은 checkpoint 일정은 비교 대상의 수렴을 제한할 수 있다. 상한 선택률과 연장 결과를 함께 공개하고, 연장 대상이 예산 때문에 생략되면 해당 비교를 PROVISIONAL로 표시한다. 10 epoch에서도 5/10 이상이 상한을 고르면 UNDERTRAINED_AT_10으로 표시한다. 추가 연장·V 기반 K 변경·seed 선택은 허용하지 않는다.

[확인] C1/C2/C3 중 최선 대상은 데이터별 V 중앙값으로 정하므로 비교 대상에 유리한 선택이다. WIN/TIE/LOSS 및 절제 판정은 사전등록의 임시 spread 기준이고 통계적 유의성 판정이 아니다. Mann–Whitney p값은 기술 정보로만 기록한다.
