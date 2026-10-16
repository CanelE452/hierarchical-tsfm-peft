# route_predictions 최종 T 충분통계 pack 경로 수정

[확인] 기존 표준 10열 `route_predictions/<id>/T_final/PUBLIC_ERROR_STATS.csv.gz`는 stats pack의 두 root 허용 목록에서 빠져 있었다. `pack()`의 입력 검사와 `_load_manifest()`의 복원·archive 검사에 `route_predictions`만 추가했다. 원 CSV schema와 split·cohort 값, 압축·SHA·크기·경로·symlink·tar 검사는 유지했다.

[확인] 합성 T_final 통계로 csv_common9와 기존 opaque_gzip fallback 각각 pack → inspect_archive → restore를 수행했고, 원 CSV/gzip byte 및 SHA가 모두 일치했다. 임시 원본 파일을 제거하여 복원 전 clone 상태를 만든 뒤 manifest의 원 경로를 virtual_files에 등록한 publication 검사도 통과했다. E0 경로와 7열 통계를 계속 거부하는 회귀를 추가했다.

- [수정 전 회귀](RED.json) · [수정 후 명시 검사](GREEN.json)
- [원문 대비 변경](SOURCE_DIFF.txt)

`test_stats_pack` 12개와 `test_publication_check` 25개, 총 37개가 통과했다. CUDA_VISIBLE_DEVICES=-1, OMP/MKL/OPENBLAS 각각 2 thread이며 torch/tensorflow/jax import는 없었다. 실제 campaign pack·기존 archive·원자료·배열·모델·GPU·보호 TEST·root pytest·git 변경은 하지 않았다. 이전 전체 324개 검사는 이 새 소스의 검증으로 다시 주장하지 않는다.

E0 7열 gzip은 별도 원형 공개하며 이번 수정은 E0 pack 지원을 추가하지 않는다. 실제 최종 payload 20MiB/250MiB·privacy·링크 전수 검사는 아직 수행하지 않았다. 현재 진행 snapshot이며 실험·게시 완료 선언이 아니다.
