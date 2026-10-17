# 실행 교정 계보

이번 캠페인은 이전 STOPPED 실행의 원장·체크포인트를 변경하지 않고 별도 자원 원장을 사용한다. 계획 수정 전 기록은 `attempts/`에 보존했다. 수정은 새로운 V 성능을 보고 seed, split, epoch, lambda 후보 또는 공차를 바꾸기 위한 것이 아니다.

- 첫 계획의 Moirai smoke는 결측 비교 fixture의 aliasing 때문에 실패했다. 공식 imputer의 in-place NumPy 변경으로 비교 양쪽 mask가 달라졌다. 원본 복사로 교정하고 회귀 검증을 추가했다.
- 첫 Chronos2 smoke는 공식 DataLoader에 CUDA Tensor를 직접 넣어 실패했다. 공식 경로의 입력을 CPU로 준비하도록 교정했다. 이후 실제 공식 inference·mask parity와 origin 격리 검증을 통과했다.
- 두 번째 Moirai smoke에서는 공식 PackedStdScaler의 padding 대입이 입력 autograd의 saved sqrt tensor를 in-place 변경하는 문제가 실제 backward에서 확인됐다. 공식 통계·FP64·minimum scale·correction을 유지하고 padding 대입 두 곳만 `torch.where`로 옮겼다. 가중치·buffer는 추가하거나 변경하지 않는다. 원 scaler의 no-grad 출력과 새 scaler의 출력이 정확히 같은지, 실제 모델의 입력/E/P gradient와 동결 hash가 유효한지는 현재 D01 실행 결과로 별도 확인한다.
- PeMS의 원 CPU 경로는 실제 5-update profile에서 두 모델의 전체 완료 비용이 6시간을 초과했다. 원 CPU 실행은 예산 상태로 남기고 두 arm 모두 동일한 CPU→CUDA 수치적 continuation branch를 등록했다. 다음 update의 sampler 동일성·optimizer device·수치차를 측정했으며 CPU 경로와 bitwise 동일하다고 주장하지 않는다.
- common endpoint는 동일 CUDA backend로 72개 모두 다시 평가한다. 기존 CPU 48개는 계보로 보존하며 새 결과로 재표기하지 않는다. 각 새 평가의 첫 origin에서 CPU/CUDA parity를 사전 고정 공차로 확인한다.

`attempts/plan_v1/source_snapshot/`의 공개 코드는 새 namespace의 당시 구현이다. 변경 없는 기반 코드는 START_STATE의 기준 main과 당시 계획의 SHA 목록으로 식별한다. 전체 원본 snapshot과 오류 로그는 private cache에 보존하고 공개하지 않는다. 후속 실패 source의 hash와 상태도 이전 계획/receipt에 남지만 모든 중간 source의 완전한 독립 재실행을 보장하지 않는다.


## 독립 G5 검토 후 optimizer 복제 교정

C00 CUDA profile의 CPU 다음-update 복제는 `optimizer.load_state_dict`가 같은 CPU tensor storage를 참조할 수 있어 원 in-memory payload의 AdamW moments/step을 바꾸었다. 그 뒤 CUDA audit가 변경된 optimizer를 복제했으므로 앞선 C00의 CPU↔CUDA parameter 차이는 순수 장치 차이로 해석할 수 없다. 디스크 checkpoint와 실제 continuation은 매번 새로 로드하므로 이 audit의 aliasing 영향을 받지 않는다. 해당 과거 숫자는 attempts에 그대로 보존하되 교정 전 진단으로만 취급한다.

CPU audit에도 전체 payload deepcopy를 적용하고 실제 AdamW update가 원 payload를 변경하지 않는 회귀 검증을 추가했다. CPU/CUDA 공통 model/optimizer/sampler/CPU RNG의 초기 fingerprint가 같고 양쪽 audit 뒤 원 payload tensor/array fingerprint가 보존되는지 실행 중 검증한다. source 변경에 따른 새 plan에서는 관련 job을 다시 실행해 정확한 cache identity를 유지한다. 기존 실행 GPU/CPU 사용량은 누적하며 초기화하지 않는다.

독립 검토에서 D02의 CP0 선택 sampler `draw(0)` 경로와 E03 Ridge core dtype 표기도 교정했다. CP0 refit은 빈 int64 draw와 초기 sampler 상태를 대조하며, Ridge는 FP64 core/FP32 normalization/FP64 inverse로 표기한다. 새 결과의 실제 통과 여부는 최종 receipt/VALIDATION에 따른다.

## 비용 측정 전 최종 독립 리뷰 교정

정확도 큐가 정상 종료한 뒤, 실패한 데이터·solver 감사를 무조건 COMPLETE로 승격하던 실행 게이트와 감사 CLI의 종료 상태를 교정했다. A01은 실제 pre-T 감사 통과, A04는 synthetic solver 감사 통과를 각각 요구한다. 등록된 M5 접근 차단은 허용하며, 보고 검증은 필수 감사 실패를 전체 실패로 반영한다. 실패 재현 6개와 정상 대조 3개를 추가했고, 교정 후 관련 테스트 11개가 통과했다. 이번 실제 데이터 감사는 처음부터 PASS_WITH_M5_BLOCK이므로 기존 수치의 무효화 사유가 아니다.

아직 실행하지 않은 Jena CPU 비용의 Ridge FP64 core 표기와 parameter 집계도 교정했다. route별 F+adapter 또는 Ridge coefficient 수와, F·모든 adapter·Ridge가 함께 적재된 공유 프로세스 수를 분리한다. RSS는 계속 공유 프로세스의 sampled peak이며 route activation memory로 해석하지 않는다. parity 실패 route는 latency NA와 실제 반복 수 0을 유지한다.

부모 계획 `a7cb073516999163f9abbf95dd1e92b60f5bf0bf05caf7a0dabfbf9371e210ed`의 계획·receipt·job·profile·score와 변경 전 두 source를 attempts에 보존했다. 새 계획 `1fc765b20c3e0cd2f94f49b3e88f2bf6ff7fe6f27e1829f2c3f5c11bc1fdb196`에서 바뀐 scientific identity는 미실행 E02/E03 두 작업뿐이다. 나머지 104개 identity와 완료한 정확도 결과는 동일하며, 새 계획으로 A03 72개/공개 artifact SHA 144개 검산을 다시 통과했다. 자원 원장의 시작 시각과 누적 사용량은 유지한다.
