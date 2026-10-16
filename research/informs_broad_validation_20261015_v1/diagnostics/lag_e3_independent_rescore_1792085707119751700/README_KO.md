# 새 지연 인코더 2조건 저장 V 독립 재채점

[확인] 원 REFERENCE_CONTRACT와 private NPZ의 target·mask·origin·channel fingerprint, 공개 SSE/SAE/count 및 F stream 장부를 독립 FP64 코드로 검산해 2조건 모두 통과했다. 원 수치는 RESULT.json에 있다. 전체 431개 등록 scientific identity를 보존하고 선택한 2개 저장 예측만 검사했다. 기존 393건과 중복하지 않지만 한 번의 전체 최종 감사가 아니다. 실제 checkpoint 재추론·학습 수렴·timer/T·일반화 검증은 별도다.

첫 driver는 별도 미실행 registry 행의 빈 reason을 canonical 검증용 fallback으로 처리하지 않아 exit1로 중단됐다. 0개의 bundle 감사가 반환됐으며 원자료·예측·registry·고정 source를 바꾸지 않았다. 해당 실패는 별도 DRIVER_FAILURE.json에 보존하고, 이후 검증용 사유 projection만 audit_campaign과 동일하게 적용했다. 원 registry 실행 상태를 완료로 바꾸지 않았다.
