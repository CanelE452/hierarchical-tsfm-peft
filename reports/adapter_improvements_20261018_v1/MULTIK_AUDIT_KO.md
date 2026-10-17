# 기존 다중 K 근거 감사

[확인] 기존 공개 근거에서 80개 stage/K 또는 예측 alias 행을 읽었다. 신규 학습과 F 호출은 0이다.

K별 origin 노출은 실제 k_exposures만 사용한다. 누락 로그, checkpoint bytes, 전환 시간은 UNKNOWN으로 남긴다. K×기록 노출로 얻은 행수와 직접 기록된 전체 F training rows는 구분한다. 같은 update 수는 같은 K별 노출을 뜻하지 않는다.

공유 parent, 독립 K, Kmax 전용 parent의 사후 prefix, nested PCA는 서로 다른 근거다. 과거 T 사용 및 반복 V 개발 평가 이력을 독립 holdout 검증으로 바꾸지 않는다.

D prefix는 reshape(7,Kmax,C)[:, :K, :]이며 flatten된 앞 7K행 절단을 사용하지 않는다. 이번 감사는 새로운 작은 K 비용 측정이나 신규 다중 K 정확도 실험이 아니다.
