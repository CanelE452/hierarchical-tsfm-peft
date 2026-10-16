# 공개 오차 충분통계의 저장 형식

`stats_pack.py`는 원 `PUBLIC_ERROR_STATS.csv.gz`를 바꾸지 않고 main 안에 공개할 저장 사본을 만든다. pack을 채택한 clone에서는 **restore 전 원 통계 경로가 비어 있다.** 이 한계를 숨기지 않으며, 공개 문서의 직접 링크는 pack 색인과 복원 안내로 연결해야 한다. 원 JSON·점수·receipt·학습 계약·실패 기록·예측 manifest는 보존한다. 다른 공개 장소나 Cloud/Release를 사용하지 않는다.

## 실제 샘플 근거

2026-10-15 완료된 공개 통계로 CPU에서 측정했다. 원자료·예측 NPZ·가중치를 읽지 않았다.

- Electricity B0 K81: 원 gzip 1,152,819 byte → XZ preset6 999,072 byte, 13.3% 감소, 압축 1.08초.
- PeMS08 Raw Ridge: 원 gzip 1,630,369 byte → XZ preset6 1,351,008 byte, 17.1% 감소, 압축 2.19초.
- Electricity B0/B2/B4 K81와 PeMS08 CURRENT/SEASONAL/RIDGE 6개: 원 gzip 합 8,340,114 byte → 전체 CSV solid tar.xz 5,027,312 byte, 39.7% 감소, 10.22초.
- 같은 6개의 run ID 상수열만 매핑으로 분리하고 같은 9열 원문을 공유하면 4개 고유 CSV member, 4,717,160 byte, 43.4% 감소, 9.56초다. 이 비율을 전체 campaign의 보장된 압축률로 외삽하지 않는다.

개별 preset9의 추가 효과가 거의 없어 도구는 preset6을 사용한다. 이 샘플 6개 모두 원 CSV byte SHA와 원 gzip byte SHA를 실제 재생성으로 확인했다. 측정 상세는 `stats_packs/sample_*/SAMPLE.json`과 `sample_common_*/SAMPLE.json`에 보존한다. 측정용 archive는 최종 검증된 pack과 구분하며 공개 후보에서 자동으로 제외된다.

완성 도구의 같은 6개 검증은 `stats_packs/pack_1792062943212834900/`에 남겼다. 원 8,340,114 byte → archive 4,722,924 byte, 43.37% 감소, 생성과 별도 임시 복원 전체 17.86초, Windows peak working set 232.125 MiB였다. source 원 SHA는 모두 그대로였다. 이 pack을 만든 source는 `source_snapshots/stats_pack_sample_a7d2e6f4687a.py`에 보존했다. 이후 공개 검사에 원 CSV SHA 결박을 추가했고, 같은 실제 pack을 강화된 검사기로 다시 검사해 PASS했다. 전체 캠페인 물량의 pack 실행 기록은 아니다.

## 원 byte를 보존하는 방법

숫자 파싱·반올림·행 정렬·통계 축약을 하지 않는다. CSV header가 10열 충분통계 schema인지 확인한 후 다음 두 저장 모드 중 하나를 사용한다.

- `csv_common9`: 모든 행의 첫 run ID가 같은 경우에만 그 접두어를 manifest에 그대로 보존한다. 나머지 9열은 줄 끝까지 원 byte를 보존한다. SHA가 같은 9열 payload는 하나의 member를 공유한다. 복원 시 각 원 run ID를 다시 붙이고 전체 CSV SHA를 확인한다.
- `opaque_gzip`: 원 gzip을 재생성할 deflate recipe를 찾지 못하거나 run ID가 상수열이 아니면 원 gzip byte 자체를 member로 저장한다. CSV를 새 gzip으로 임의 재작성하지 않는다.

gzip header·trailer는 hex로, deflate level·wbits·memLevel·strategy·flush 순서와 zlib compile/runtime 버전은 manifest에 기록한다. 현재 실제 샘플은 zlib 1.3.2, level9, raw deflate wbits=-15, memLevel8, default strategy, Z_SYNC_FLUSH 후 Z_FINISH에서 원 gzip byte가 정확히 일치했다.

CSV SHA는 환경과 무관한 원문 식별자다. 원 gzip byte 재생성은 zlib 구현에 영향을 받을 수 있으므로 둘을 구분한다. 다른 환경에서 CSV SHA가 같아도 원 gzip SHA가 다르면 restore는 실패하며 다른 byte를 원 경로에 쓰지 않는다. opaque 모드는 원 gzip을 보존하므로 이러한 재생성 의존성이 없다.

## 형식과 제한

새 `stats_packs/pack_*/` 안에 `part_0000.tar.xz`와 `MANIFEST.json`, 공개 파일 목록, 설명, 기존 Markdown 링크 수정 목록을 만든다. 모든 archive는 20 MiB 미만이다. XZ preset6과 canonical USTAR만 사용하며, member는 검증된 통계 CSV 또는 원 opaque 통계 gzip이다. NPZ·모델·원자료·임의 binary를 포함하지 않는다.

member 최대 64 MiB, 한 tar의 해제량 최대 128 MiB, XZ decoder memory limit 256 MiB, manifest 최대 8 MiB, 총 restore 출력 기본 상한 3 GiB다. source별 처리·64 MiB 압축 그룹으로 메모리 사용을 제한하며 LZMA 자체는 단일 스레드다. 대규모 실물 pack은 별도 승인된 실행 순서에서만 수행한다.

복원은 archive SHA/size, 정확한 정규 member 집합·경로·개수·크기·SHA, canonical tar byte, CSV schema와 값 범위, 원 CSV/gzip SHA를 검사한다. symlink·절대경로·상위 경로 탈출·중복 경로·알 수 없는 member·뒤에 숨긴 데이터·연결 XZ stream·과다 해제량을 거부한다. `extractall`을 사용하지 않는다.

`publication_check.py`는 `stats_packs/pack_*/*.tar.xz`만 허용하고 manifest와 해당 검증 API를 사용해 추출 없이 검사한다. 해제된 통계 텍스트에도 기존 secret/개인 경로 검사를 적용한다. 원 gzip·PNG·SVG 검사 동작은 유지했다. `.xz`를 허용되지 않는 확장자나 base64 JSON으로 위장하지 않는다.

## 생성과 복원

전체 생성은 최종 T/후처리가 끝난 뒤 ROOT 실행 단계에서 수행한다. 최신 projection의 명시 파일 목록을 입력하면 다른 source·receipt·점수 파일을 유지하면서 실제 pack된 표준 통계만 새 payload에서 제외한다.

```text
python stats_pack.py pack --campaign . --file-list publication_projections/snapshot_<final>/PAYLOAD_FILES.txt
```

샘플 검사는 `--only predictions/<run>/V_select/PUBLIC_ERROR_STATS.csv.gz`를 반복해 수행할 수 있다. 매번 새 디렉터리를 만들며 기존 결과는 덮어쓰지 않는다. 생성 과정은 별도 임시 디렉터리로 전체 stats를 복원하고 모든 원 gzip/CSV SHA가 맞아야 완성 manifest를 쓴다. 검증 도중 실패한 pack은 완성 게시물로 취급하지 않는다.

clone의 repository 루트에서 실행한다.

```text
python research/informs_broad_validation_20261015_v1/stats_pack.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_<final>/MANIFEST.json --destination .
```

기존 파일이 원 SHA와 같으면 읽기만 한다. 불일치 파일은 덮어쓰지 않고 실패한다. 원본 campaign 파일을 삭제하거나 수정하지 않는다. 복원 중 자원이 끝나더라도 이미 정확히 복원된 파일은 다음 실행에서 읽기 검증 후 재사용한다.

`MARKDOWN_LINK_REPLACEMENTS.json`은 원 통계 경로로 직접 연결된 기존 문서 목록만 보고한다. 기존 Markdown을 자동 수정하지 않는다. 최종 색인 연결·중간 projection 문서 공개 제외·allowlist 통합·250 MiB 확인은 최종 ROOT 게시 검사에서 수행한다. 이 도구는 prepare, stage, commit, push를 실행하지 않는다.
