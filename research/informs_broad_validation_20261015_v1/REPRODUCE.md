# 재현·원수치 복원

**PARTIAL.** 실제 자산·선택·검산 계약과 검증된 무손실 복원 절차다. 자동 학습·T 재사용·commit/push 명령이 아니다. 원격 게시 상태는 이 문서를 포함한 Git 커밋으로 확인한다.

먼저 [데이터 계약](DATA_CONTRACT.json), CONFIG_LOCK, 초기/확장 registry, 모델 pin과 실제 checkpoint/normalization/cohort hash를 확인한다. 원 데이터는 공식 자산 계약을 따라 private cache에 두며 main에 복제하지 않는다. M5 접근·라이선스 차단은 우회하지 않는다. 기존 실제 환경을 사용하고 이 문서가 새 설치·백본 재학습을 요구하지 않는다.

공개 자료를 clone한 후 `$Repo`를 저장소 루트, `$Campaign`을 그 아래 `research/informs_broad_validation_20261015_v1`, `$Python`을 기존 프로젝트 Python으로 둔다. 실제 기계 경로·token을 문서에 적지 않는다. `INFORMS_ASSET_ROOT`, `INFORMS_CACHE_ROOT`는 해당 실행 환경의 자산과 private cache를 가리킨다.

## 바이트 정확 복원

[pack 실행·복원 문서](publication_derivatives/final_md_20261016_v2/docs/doc_0071.md)의 두 실제 명령을 사용한다. K8 복원의 `--public-root`는 campaign 폴더이고 aux95 복원의 `--destination`은 저장소 루트다. 같은 원본은 건너뛰고 다른 기존 파일은 덮지 않는다.

기존 충분통계 pack은 canonical manifest와 alias extension의 원 경로·원 SHA 복원을 각각 수행해야 한다. 새 canonical pack 하나만으로 모든 alias가 생긴다고 가정하지 않는다. T46은 [실제 전수 복원 문서](publication_derivatives/final_md_20261016_v2/docs/doc_0010.md)의 명령을 사용한다. 저장소 루트에서 다음을 실행한다.

```text
python research/informs_broad_validation_20261015_v1/stats_pack.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_1792097901794080200/MANIFEST.json --destination .
```

원46 CSV/gzip 바이트를 별도 임시 복원·반복 복원했고 원gzip은 로컬 보존했다. 기존 pack/alias 및 Markdown backup은 각각 명시된 manifest의 원 경로·SHA로 복원한다. 최종 공개 allowlist/복원 색인의 원격 clone 검증은 게시 단계에 남는다. 행/열/정밀도 축소나 점수 tolerance로 중복을 정의하지 않는다.

현재 aux95의 정확한 원 CSV는 [manifest](publication_derivatives/gzip_aux95_1792094747124151400/GZIP_MANIFEST.json)와 각 gzip에 남아 있다. 소스 API가 원 JSON/CSV 경로를 요구하면 먼저 복원한다. Markdown 렌더용 안전 파생본과 원문 바이트 복원용 gzip을 구분한다. 추가 [aux2995 원형 복원 패키지](publication_derivatives/gzip_extra_aux_20261016_v1/README_KO.md)는 원2995파일31,709,263B·gzip6,722,015B이며 [전수·독립·반복 복원](publication_derivatives/gzip_extra_aux_20261016_v1/BUILD_RESULT.json)이 실제 PASS했다. [manifest](publication_derivatives/gzip_extra_aux_20261016_v1/GZIP_MANIFEST.json)의 원경로·원 SHA·gzip SHA로 결속하고 저장소 루트에서 다음을 실행한다. 추가 [77파일](publication_derivatives/gzip_extra_aux_20261016_v2/BUILD_RESULT.json)도 별도 전수·독립·반복 PASS이며 두 묶음 합3072원파일36,797,468B→물리9,893,189B다. 실제 [원바이트 복원 색인](RESTORE_INDEX_KO.md)에 전체 묶음별 명령이 있고 최종 원격 clone 검증은 별도다. [후속 실제 집계·포장 감사](FINAL_POSTPROCESS_AUDIT.json)는 후속 완료를 원 AUDIT_REPORT의 당시 pending snapshot과 구분한다.

```text
python research/informs_broad_validation_20261015_v1/publication_derivatives/gzip_extra_aux_20261016_v1/restore_originals.py --destination .
python research/informs_broad_validation_20261015_v1/publication_derivatives/gzip_extra_aux_20261016_v2/restore_originals.py --destination .
python research/informs_broad_validation_20261015_v1/publication_derivatives/gzip_extra_aux_20261016_v3/restore_originals.py --destination .
```

## 검증 범위

독립 scorer는 예측·target·mask·origins·channel_ids와 시간 계약·scaler/cohort/source/selected artifact를 대조하고 FP64로 채널 macro 오차를 다시 계산한다. public 충분통계 재집계와 원예측 독립 재채점, checkpoint 실제 재추론은 별도 영수증으로 남긴다. actual replay50의50PASS는 신규41+기존 exact9이며 고유 replay70과 겹친다. 이를 합산하거나 전체 학습 재실행으로 해석하지 않는다.

공식 T46 batch SHA `3e812216…e9e`, selection lock SHA `c3493746…a51f1`, exact46 manifest score·source/scaler/artifact ID를 확인한다. E0 registry SHA `b67252ae…3181`의55 B checkpoint MATCH는 already-seen legacy 개발 구간이며 새 T와 구분한다. [최종 whole 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0583.json.gz)는 표준499(V405+T46+M48), 공개통계451, legacy B55, errors0/global true/입력2557불변을 확인했다. 실제 RESULT 파일 SHA는3707c98024497180fdc4a2f566b0c83d08bcf24e6e14f0961c8666fb8287c956이다. [제공 public entry](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0592.json.gz)도 원 entry를 수정하지 않고451행/451gzip을 재집계했으며 임시 복원본 삭제와 입력 불변을 확인했다. 복원·저장값 검산·실제 replay는 서로 대체하지 않는다.

정보 경계 검사는 원본을 바꾸지 않은 미래 복사본 개입에서 검사한 통계·PCA·past/P/E의 불변성을 확인했다. F 사전학습 중복, legacy 전처리 계보, 모든 미래 의존성의 보편 부재를 증명한 것이 아니다.

CPU 검사는 campaign의 명시된 unittest 범위와 새 보조 helper 단독 검사를 각각 확인한다. bare root pytest를 재현 명령으로 사용하지 않는다. 이미 실행된 [Labour 보조 smoke 사고](publication_derivatives/final_md_20261016_v2/docs/doc_0019.md)는 전체 held-out 수치 역직렬화와 TRAIN2 update 뒤 실패한 범위 위반으로 별도 보존했다. 이 검사를 본 실험 PASS로 세지 않으며 세션 전체 보호값 접근0을 주장하지 않는다. 고정 test source·실제 종료코드·device 환경이 있는 영수증을 따른다.

## 보고서·그림 재생성

실제 저장 예측을 검산하고 registry를 안정화한 후 기존 API로 고유한 새 snapshot을 만든다. 이는 학습·새 모델 추론을 하지 않지만 저장 예측을 읽어 재채점하는 CPU 작업이다.

```powershell
& $Python -X utf8 -u -B (Join-Path $Campaign 'report_figures_cost_namespace_layout_v3.py') `
  --public-root $Campaign --private-root $env:INFORMS_CACHE_ROOT --split V_select
& $Python -X utf8 -u -B (Join-Path $Campaign 'report_figures_cost_namespace_layout_v3.py') `
  --public-root $Campaign --private-root $env:INFORMS_CACHE_ROOT --split T_final
& $Python -X utf8 -u -B (Join-Path $Campaign 'report_campaign.py') `
  --public-root $Campaign --private-root $env:INFORMS_CACHE_ROOT
```

원 report_figures.py의 list diagnostic/cost namespace 오류와 v2·원부분 snapshot은 보존했다. 위 v3는 source39·과학 계산·선택을 바꾸지 않는 layout 교정본(SHA `2bb7e52b0820233d373cb3f3bb990d1a15439ef3b2dadd0895ff2045c0bb22eb`)이며 실제 V405/71stem·T46/52stem을 생성했다. V34/T24 원수치 CSV는 v2와 byte-identical이다. 대표 가독성 검토와 전그림 검토를 구분한다. [학습 보완9stem/18이미지](publication_derivatives/gzip_extra_aux_20261016_v2/files/item_0000.json.gz)는 V5자료·T4자료의 동일 원 CSV4개로 생성했고 Beijing V의 범례 분리를 실제 확인했다. 나머지 전그림 가독성 PASS로 확대하지 않는다. 최종 채택본·원수치 CSV·run ID는 [PUBLIC_FIGURES_KO.md](PUBLIC_FIGURES_KO.md)를 따른다. 같은 조건의 다른 snapshot 재생성은 byte-identical 이미지 재현을 자동 보장하지 않는다.

T 그림은 실제 locked SELECTION_LOCK과 감사 가능한 T manifest가 있을 때만 기존 `--split T_final`을 사용한다. 보고서 재생성을 모델·K 재선택에 사용하지 않는다. 여섯 그림 그룹·전체 수치 CSV·missing/PENDING 표시·자료별 물리시간·PeMS elapsed 축을 유지한다. 대표 trace 원자료 배열은 공개하지 않는다.

학습 재개는 [RESUME.md](RESUME.md)의 실제 상태·source·등록·자원 조건을 따른다. 선택 잠금 이후 새로운 학습은 기존 T에 되먹이지 않는 별도 개발 범위로 분리한다. final 비용은 실제 full pipeline/fresh residency/row variant와 artifact가 같은 조건끼리만 비교하며 cached-parent 학습 비용을 배포 비용으로 바꾸지 않는다.

## K 비용과 원수치 감사의 별도 복원

[fresh12 묶음](diagnostics/publication_multik_fresh_lossless_20261016_v1/bundle_20261016_actual_01/README_KO.md)은 기존 K8 묶음을 외부 exact 참조하고 fresh2160trial을 포함한2234원본을 복원한다. 전체20scope3600trial=기존1440+fresh2160이며 [옛 부분166 묶음](diagnostics/publication_multik_historical_partial166_20261016_v1/bundle_1792101441341745500/README_KO.md)은 별도 역사적 실패로 복원한다. 원166을 최종3600에 더하지 않는다. 둘 다 원 bundle_multik.py의 restore API를 사용한다.

```text
python research/informs_broad_validation_20261015_v1/diagnostics/publication_multik_lossless_1792094047117000000/bundle_multik.py restore --manifest research/informs_broad_validation_20261015_v1/diagnostics/publication_multik_fresh_lossless_20261016_v1/bundle_20261016_actual_01/MANIFEST.json --public-root research/informs_broad_validation_20261015_v1
python research/informs_broad_validation_20261015_v1/diagnostics/publication_multik_lossless_1792094047117000000/bundle_multik.py restore --manifest research/informs_broad_validation_20261015_v1/diagnostics/publication_multik_historical_partial166_20261016_v1/bundle_1792101441341745500/MANIFEST.json --public-root research/informs_broad_validation_20261015_v1
```

독립 [core 비용 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0290.json.gz)는137report행/96receipt/6850rawtrial/274summary, [K20 감사](publication_derivatives/gzip_extra_aux_20261016_v1/files/item_0505.json.gz)는20scope/3600trial을 원 receipt·CSV·ledger·SHA와 대사했다. core의 mean/median/min/max/p90 재계산과 K20의 median·개별 trial/ledger/resource-valid 대사는 새 측정이나 메모리 재측정이 아니다. **K20 p90은 계산하지 않았다.** 옛 PeMS LoRA fresh2의 exact reuse와 old/new balanced group 차이도 분리한다. 원 aggregate의 timed값을 stage_probe나 instrumented transition과 섞지 않는다.

공개 MD5개·CSV·그림 색인·무손실 복원·실제 remote main SHA의 최종 clone 검증은 게시 단계의 남은 항목이다. 지금 문서를 게시 완료 영수증으로 사용하지 않는다.


## 잠금 route의 공개 충분통계 분해 재현

[실제52쌍](publication_derivatives/gzip_extra_aux_20261016_v2/files/item_0012.json.gz)은 기존 잠금의 모델/역할을 유지한 T 관찰 뒤 보고용 분해다. 원 target/NPZ/model 없이 공개 통계로 수행하며 기존14쌍과 독립합산하지 않는다. 원 public statistics와 aux metadata를 바이트 정확 복원한 뒤 기본 metadata-only 명령을 확인한다. 실제 재집계는 아래 `--execute-report`에서만 수행하고 새 snapshot을 만든다.

```powershell
& $Python -X utf8 -B (Join-Path $Campaign 'diagnostics/postfinal_locked_route_pairs_20261016_v1/locked_route_pairs_scope_v2.py') `
  --public-root $Campaign `
  --scores (Join-Path $Campaign 'reports/snapshot_20261015T215213Z_1792101133423713300/SCORES.csv') `
  --input-manifest (Join-Path $Campaign 'reports/snapshot_20261015T215213Z_1792101133423713300/INPUT_MANIFEST.json') `
  --pattern-evaluation (Join-Path $Campaign 'patterns/snapshot_20261015T093749Z_1792057069715991800/evaluation_20261015T221722Z_1792102642656382000') `
  --time-blocks (Join-Path $Campaign 'diagnostics/time_blocks_1792102643194860300') `
  --execute-report
```

실제 보조표 생성63.104초·verified52/failures0와13개 합성 검사를 별도 영수증으로 구분한다. 원 target값 재검증·새 추론·T 선택은 없다. 52쌍 연결 명세를 T 이전 사전등록으로 표현하지 않는다.

## 최종 공개 파일 바이트 검증

최종조립에서 생성하는 PUBLIC_MANIFEST.sha256은 원격게시나 수치·링크 검증과 별도로 명시된 repository-relative 파일의 SHA256과 개별/전체 byte 한도를 검사한다. manifest 자신은 해시 목록에서 제외하되 전체용량에는 포함한다. 기존 publication_check.prepare/verify의 별도 전체 receipt/allowlist와 호환을 주장하지 않는다. manifest 실제 생성 후에만 아래를 실행한다.

```powershell
& $Python -X utf8 -B (Join-Path $Campaign 'diagnostics/publication_selected_only_20261016_v2/verify_public_manifest.py') `
  --repo-root $Repo --manifest (Join-Path $Campaign 'PUBLIC_MANIFEST.sha256')
```

이것은 파일바이트·명시용량 검사이며 숫자재집계·학습완료·상대링크·복원전수·원격Git 검증을 대신하지 않는다.
