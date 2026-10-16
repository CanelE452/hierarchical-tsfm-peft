# 기존 자원 제한 부분 실행의 무손실 보존

이 묶음은 이전 실행의 새 trial166개(자원 유효165개)와 전체 queue·부분 상태를 보존한다. 원8개 재사용1440은 외부 원 참조로 남아 있으며 여기서 다시 포장하거나 최종 K20의3600에 더하지 않는다. 부분 실행의 실패·자원 상태를 완료로 바꾸지 않았다. 새 학습·추론·비용 측정이 아니다.

- [원 경로·바이트·SHA](MANIFEST.json)
- [별도 임시 전수 복원·반복 복원](BUILD_RESULT.json)
- [중복 계산하지 않는 실행 범위](HISTORICAL_SCOPE.json)

원 bundle_multik.py restore --manifest <이 폴더/MANIFEST.json> --public-root <campaign>로 복원한다.
