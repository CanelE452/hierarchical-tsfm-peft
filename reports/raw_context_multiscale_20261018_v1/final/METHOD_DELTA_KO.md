# 방법 차이와 문헌의 범위

S: Z=(P(X)-mu)E, U=F(Z), 기존 원격자 lag7 D와 원채널 마지막 값 연결.
A: E의 앞 절반은 (X-mu), 뒤 절반은 (P(X)-mu)에 적용한다. 총 P/E/D 용량과 PCA 열 순서를 유지한다.
B: A의 context 잠재축만 창 내부 인접2시점 평균 → 길이256의 실제 F 예측 → 과거·미래 block repeat → 원격자 lag7 decoder. P 자체는512시점에서 한 번 계산하고 anchor도 한 번만 더한다.

S→A는 원본/문맥 투영 분리, A→B는 평균화·낮은 격자 예측·반복 복원의 전체 차이다. A의 native 한 호출과 같은 weights의 비용 전용 두 호출을 나누어 제시한다. 같은 K·파라미터 수가 같은 유효 표현·학습 동역학·FLOPs를 뜻하지 않는다.

[TimeMixer, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)는 과거와 미래 예측의 여러 시간 해상도 활용 원리를 제공한다. 이 구현은 TimeMixer의 분해·mixing 전체 구조를 재현하지 않는다.
[NHITS, AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854)는 multi-rate sampling과 미래 보간이라는 인접 원리를 제공한다. 여기의 고정 block repeat는 NHITS의 학습된 계층적 보간 재현이 아니다.
[Bates와 Granger, The Combination of Forecasts,1969](https://doi.org/10.1057/jors.1969.103)는 고정 부모 오차를 이용한 예측 결합의 배경이다. 본 실험의 alpha는 M_fit에서 채널 동일 가중 MSE로 적합하고 M_select 진단 후 전체 M_cal에서 fresh 적합한다.

우리 Bolt용 원본/문맥 분리,512/256 두 호출, 원격자 lag 복원의 조합은 신규 변형이다. 위 문헌이 이 조합의 성공·효율성·신규성을 증명한다고 주장하지 않는다.

Conditional: g+H(past-current,g-current,b-g). Past control은 세 번째 항에 마지막 H개 실제 과거를 사용한다. hidden32 채널공유 head는 M_cal 내부 target-disjoint60/40에서 선택한 epoch로 전체 M_cal을 fresh fit한다. Past control은 B2/F를 적재하거나 호출하지 않는다.

원래 H×C 관측 target을 유지하며 normalized observed equal-channel macro MSE가 주지표다. MAE를 사후 주지표로 바꾸지 않는다. V는 개발 평가이며 새 보호 T 접근은 없다.
