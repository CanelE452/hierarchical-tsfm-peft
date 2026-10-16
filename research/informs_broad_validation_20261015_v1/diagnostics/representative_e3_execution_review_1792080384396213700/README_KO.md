# Beijing 대표 3 epoch 연장 실행 검토

[확인] G·B0(K=3)·B2(K=3)는 A_fit 검색을 기존 1 epoch에서 실제 3 epoch까지 연장했다. 각 조건의 추가 검색은 1,245 updates·19,920 origin exposures이며, 누적 검색은 1,868 updates·29,880 exposures이다. A_stop 정규화 MSE는 세 조건 모두 1 epoch를 다시 선택했다. 선택된 길이로 A60에서 새 refit 754 updates·12,063 exposures를 실행했으며 이전 A60 결과를 복사하지 않았다고 실제 provenance에 기록되어 있다.

새 선택 모델의 model_state_sha 및 scaler_sha는 이전 __e1과 같다. 새 bundle SHA는 run/spec/contract/search metadata를 포함해 다르다. 이 둘을 가중치 변화와 혼동하지 않는다. 새 V manifest 3개는 이전 V와 예측·target·mask·origin·channel fingerprints 및 점수가 모두 같다. 이는 저장 메타데이터 비교이며 이 검토에서 weights/NPZ를 열거나 예측·FP64 검산을 재실행하지 않았다.

`SUMMARY.csv`는 원수치, `REVIEW.json`은 실제 carry verifier가 확인한 항목과 한계, `INPUTS_SHA256.json`은 읽기 전 봉인하고 종료 전 재확인한 공개/캐시 상대 경로 입력 증거다. optimizer·RNG·sampler를 포함한 non-contract payload fingerprint 보존은 성공한 실제 migration 기록으로 확인했으며, 이 검토가 독립 tensor 검산을 새로 수행했다는 뜻은 아니다. 최초 원상태에 대한 외부 서명이 없었다는 provenance 한계도 보존했다.

추가 학습을 수행하지 않았다는 결론도, 3 epoch로 성능이 개선됐다는 결론도 성립하지 않는다. 현재는 Beijing 한 seed의 대표 G·quarter B0/B2에 한정되며, 4자료·전체 K·모든 가족의 최소 학습 충족이나 수렴, 독립 TEST 일반화 증거로 확대하지 않는다.
