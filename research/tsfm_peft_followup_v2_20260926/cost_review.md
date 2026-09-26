비용 측정 코드 검토 (2026-09-27)

`measure_cost.py`를 읽어 동일 CPU 표준화 입력, 전체 채널 출력, batch1/4, CUDA 동기화, 예열10회, 20회 전체24원점 처리와 3개 순서 블록을 확인했다. LoRA는 저장 adapter를 복원한 뒤 `merge_and_unload(safe_merge=True)`의 반환 backbone을 사용하며, 병합 전후 FP32 VAL 예측을 검사한다. 검사 자체를 독립 재현 또는 정확도 우월성 근거로 쓰지 않는다.

검토 중 확인한 구체적 문제는 완성 JSON만 존재 여부를 검사하여 이전 실패의 동명 partial을 덮어쓸 수 있다는 점이었다. root에 보고한 뒤 수정본 156–168행을 읽어 이름 검증, 최종/partial/attempt 존재 검사와 실패 traceback 보존을 확인했다. 해당 수정의 GPU 실행 여부·정합 결과는 최종 실행 기록을 따른다.

CPU 대조는 `measure_linear_cost.py`에서 기존 Electricity/Bull의 SHARED-LINEAR와 FACTOR-LINEAR 가중치를 재사용한다. Electricity 가중치 hash는 v1 residual-rank 진단의 weight receipt, Bull은 protected report의 weight receipt와 대조한다. 과거 Electricity report의 model.py hash는 해당 역사 commit에서 검증하며 현재 소스로 조용히 바꿔 인증하지 않는다.

동일 24개 VAL 원점·batch1/4·예열10 호출·20 전체 패스·3 순서 블록, CPU thread4로 측정한다. 행렬 계수는 FP32로 변환하고 NumPy float64 공식과 같은 입력에서 정합성을 먼저 확인한다. 새 학습이나 목표 성능 채점은 없다. 기존 정확도 수치와 FP32 배포 변환의 수치 차이를 구분한다.

CPU 메모리는 실행 프로세스 RSS/peak working set과 배포 tensor bytes를 따로 기록한다. 프로세스 메모리에는 인터프리터와 자료·네 선형 모델이 포함되므로 CUDA allocated peak와 같은 범위라고 비교하지 않는다. 선형 계수 수, 고정 PCA basis 수, 전체 tensor bytes를 구분한다. 실측은 GPU 진단/학습과 CPU 경합이 없도록 root의 실행 신호 후에만 한다.
