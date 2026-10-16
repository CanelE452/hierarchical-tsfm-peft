# E0 충분통계와 stats pack 호환성 — 소스 읽기 검토

[확인] E0의 `legacy_e0/evaluation/<id>/cpN/PUBLIC_ERROR_STATS.csv.gz`는 현재 stats pack의 지원 형식과 다르다. 최종 공개는 E0 원 gzip을 원형으로 유지하고 표준 campaign 통계만 pack하는 최소 대응을 사용한다. 이 문서는 소스 흐름 검토이며 실제 E0 예측이나 최종 payload 검사를 실행한 결과가 아니다.

## 입력 소스 SHA256

```text
41b7aa9bf55469fd199c9d7f4661c17d0bc245fb779a32e754b7c7628d55e740  legacy_evaluation.py
0d8c4e4cc9881b70907acf7a154133a64cb582227c1df357521a7668e7fde3fd  stats_pack.py
697476f0ba518c3d0d6888cc9744aac9e2e46cb614221a425e949803934cd1d6  publication_check.py
9c796c2378dde5513246aa046d8cf680a2a328e9bd8943cb6b872cb94f5ee101  publication_projection.py

```

[확인] [legacy_evaluation.py](../../legacy_evaluation.py)의 `score_fp64()`와 `evaluate_checkpoint()`는 다음 7열을 그대로 gzip에 저장한다.

```text
origin,channel,horizon_start,horizon_end,sse,sae,count
```

Manifest는 nested `identity`에 legacy_id·checkpoint_update 등을 기록하고, split=`legacy_B_development`, 별도 legacy cohort, sibling 통계의 `public_stats_sha256`를 기록한다. 표준 campaign의 최상위 run_id·public_stats_relative_path를 전제할 수 없다. B는 개발 데이터이며 새 V/T와 합치지 않는다.

[확인] [stats_pack.py](../../stats_pack.py)는 prediction manifest를 수집하지 않는다. 공개 파일 목록에서 `/PUBLIC_ERROR_STATS.csv.gz`로 끝나는 경로를 수집하며, 원본에 다음 10열을 요구한다. csv_common9는 run_id 열을 prefix로 따로 보존하는 기존 저장 모드일 뿐 7열을 변환하는 기능이 아니다.

```text
run_id,split,cohort,origin,channel,horizon_start,horizon_end,sse,sae,count
```

pack 입력과 `_load_manifest()`는 허용 root만 받는다. `legacy_e0`는 허용하지 않는다. 경로를 우회해도 `_csv_schema()`가 7열을 거부한다. `opaque_gzip` 역시 원 CSV schema 검증을 생략하지 않으며, archive의 `inspect_archive()`와 restore의 `_original_csv()`에서도 같은 검증을 거친다. [publication_check.py](../../publication_check.py)의 archive 검사는 이 API를 호출하므로 수동 archive 생성만으로 우회할 수 없다.

표준 10열 T 통계의 `route_predictions` 누락은 별도로 확인하여 stats_pack의 두 root 허용 목록에 추가했고 합성 회귀로 검증했다. E0 7열 지원은 추가하지 않았다. [route 경로 변경 증빙](../stats_route_predictions_1792070186457362300/README_KO.md)에 범위가 기록돼 있다.

## 최종 공개 목록 구성

- `file-list`는 E0 manifest·reference·rescore·원 gzip을 포함한 전체 안전 공개 목록으로 유지한다.
- `--only`에는 허용된 표준 root의 10열 통계 경로만 명시한다. 이 경우 pack은 선택한 원 gzip만 archive로 대체하고 E0 원 gzip은 공개 목록에 남긴다.
- `--only` 없이 전체 파일 목록을 자동 수집하면 E0 경로에서 실패한다. 반대로 file-list 자체에서 E0를 빼면 원형 별도 공개도 누락되므로, 두 목록의 용도를 구분한다.
- final payload에서는 archive와 별도 공개 E0 gzip을 합산한다. pack으로 대체한 표준 원 gzip을 물리 게시량에 다시 더하지 않는다. 복원 예상 통계량은 별도 지표다.

## 원형 보존과 남은 실제 검사

[확인] 기존 E0 gzip에 가짜 run_id/split/cohort를 넣거나 10열로 덮어쓰면 `public_stats_sha256` 및 기존 rescore의 원문 증거가 달라진다. 파일명만 옮겨 허용 root로 가장해도 형식 불일치가 남는다. 원문·seal·개발 B의 구분을 보존한다.

원형 gzip의 publication 경로는 generic gzip UTF-8·privacy·파일 크기 검사를 받으며 표준 10열 검증을 요구하지 않는다. 이것은 모든 실제 E0 산출물이 공개 조건을 통과했다는 뜻이 아니다. 최종 시점에 파일별 20MiB, 전체 물리 payload 250MiB, 링크의 정확한 case·포함 범위, secret·개인정보·로컬 절대경로를 실제 파일로 검사해야 한다. E0의 결과와 사고 증빙을 숨기거나 원본을 삭제하는 작업은 하지 않는다.

이번 검토에서는 소스 텍스트와 SHA만 읽었으며 실제 E0 원자료·모델·weights·배열·GPU·보호 TEST·실제 예측·전체 pack·최종 payload 검사·게시·git 변경을 수행하지 않았다. 현재 캠페인은 진행 중이며 최종 완료 보고가 아니다.
