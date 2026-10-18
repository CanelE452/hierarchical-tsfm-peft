# 슬라이드 22 뒤 추가할 수 있는 보충 소재 1장

제목: Row chunk를 허용한 동결 TSFM의 정확도–메모리–지연 선택지

이 파일은 소재이며 기존 24장 발표와 다른 13장 outline, PPT/PDF를 수정하지 않는다.

그림: [메모리–정확도](FIG1_memory_mse.png)와 [메모리–전체 지연](FIG2_memory_latency.png)을 함께 배치한다. 축은 allocated tensor MiB이며 전체 프로세스 hard cap으로 쓰지 않는다.

확인 예시: `B2_jena_K3_OFF_s92601_cap3_B1_R1`, MSE 0.835509, allocated 192.414 MiB, whole median 34.071 ms, 29.350 origin/s. 사후 설명용으로 TSFM 경계에서 고른 한 행이며 모델 선택 기준은 아니다.

- Raw·merged LoRA도 row chunk로 활성값 피크를 줄일 수 있으므로 B2의 메모리 절감을 독점적 이점으로 제시하지 않는다.
- B2의 선택 가치는 동일 데이터·B의 MSE·allocated 피크·whole latency 세 축 경계에 한정한다. 전체 GPU 대안에는 DirectG를 포함하고 CPU Ridge는 별도로 보인다.
- 반복 V, 두 seed, 과거 T/정규화 ancestry, PID GPU 미지원, 고정 타이밍 창과 20회 기술 통계의 한계를 각주에 둔다.
- K3/K12가 실행됐더라도 독립 네 모델 학습 비교이며 단일 checkpoint의 즉시 K 전환을 주장하지 않는다.

정확한 수치·전체 선택지는 [T1](T1_ORIGINAL_PATH.csv), [T2](T2_CHUNK_COMPARISONS.csv), [모든 범위의 Pareto](PARETO_TABLE.csv), [한국어 분석](RESULTS_KO.md)을 사용한다.
