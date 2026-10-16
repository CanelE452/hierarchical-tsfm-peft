# 공개 원수치·원바이트 복원

원수치 행·조건·정밀도를 줄이지 않고 공개 파일 한도를 맞추기 위해 충분통계와 부속 자료를 무손실로 묶었다. 원본은 로컬에 보존했다. 각 manifest의 원경로·원 SHA·payload SHA를 확인한다. 원본과 같은 파일은 건너뛰고 다르면 실패하므로 기존 파일을 덮어쓰지 않는다. 모델 가중치·전체 원자료·원 target 배열은 이 공개 묶음에 포함하지 않는다.

복원은 데이터셋을 새로 선택하거나 학습을 재개하는 명령이 아니다. 공개 SSE/SAE/count의 재채점과 JSON/CSV byte 복원을 위한 것이다. 새 clone에서 저장소 루트를 현재 디렉터리로 사용한다. `R`은 `research/informs_broad_validation_20261015_v1`이다.

## 공개 충분통계

[stats pack 코드](stats_pack.py)와 [T46 manifest](stats_packs/pack_1792097901794080200/MANIFEST.json)를 사용한다. `stats_packs`의 기존 V pack도 각각 같은 형식으로 복원한다. 원 gzip byte 재생성은 zlib 환경 차이가 있으면 실패할 수 있으며 원 SHA 기준을 완화하지 않는다.

```powershell
$R = 'research/informs_broad_validation_20261015_v1'
Get-ChildItem -LiteralPath (Join-Path $R 'stats_packs') -Filter MANIFEST.json -Recurse | ForEach-Object {
  python (Join-Path $R 'stats_pack.py') restore --manifest $_.FullName --destination .
  if ($LASTEXITCODE -ne 0) { throw 'stats restore failed' }
}
Get-ChildItem -LiteralPath (Join-Path $R 'stats_packs') -Filter 'ALIAS_EXTENSION_*.json' -Recurse | ForEach-Object {
  python (Join-Path $R 'stats_pack.py') restore --manifest $_.FullName --destination .
  if ($LASTEXITCODE -ne 0) { throw 'stats alias restore failed' }
}
python (Join-Path $R 'publication_projections/lossless_1792092403733272600/restore_internal_statistics.py') --destination .
```

## 부속 JSON·CSV 및 Markdown

- [기존 aux95 manifest](publication_derivatives/gzip_aux95_1792094747124151400/GZIP_MANIFEST.json)
- [새2995 manifest](publication_derivatives/gzip_extra_aux_20261016_v1/GZIP_MANIFEST.json), [실제 전수·독립·반복 복원](publication_derivatives/gzip_extra_aux_20261016_v1/BUILD_RESULT.json)
- [추가77 manifest](publication_derivatives/gzip_extra_aux_20261016_v2/GZIP_MANIFEST.json), [실제 전수·독립·반복 복원](publication_derivatives/gzip_extra_aux_20261016_v2/BUILD_RESULT.json)
- [추가7 manifest](publication_derivatives/gzip_extra_aux_20261016_v3/GZIP_MANIFEST.json), [실제 전수·독립·반복 복원](publication_derivatives/gzip_extra_aux_20261016_v3/BUILD_RESULT.json)
- [최종 Markdown 원본 manifest](publication_derivatives/final_md_20261016_v2/GZIP_MANIFEST.json)

각 폴더의 `restore_originals.py --destination .`를 실행한다. 예를 들면 다음과 같다. 최종 Markdown의 원본과 GitHub용 링크 파생본은 경로가 다르므로 원본 byte를 원할 때만 원본을 복원한다. 렌더링용 파생본은 그대로 유지된다.

```powershell
python (Join-Path $R 'publication_derivatives/gzip_aux95_1792094747124151400/restore_originals.py') --destination .
python (Join-Path $R 'publication_derivatives/gzip_extra_aux_20261016_v1/restore_originals.py') --destination .
python (Join-Path $R 'publication_derivatives/gzip_extra_aux_20261016_v2/restore_originals.py') --destination .
python (Join-Path $R 'publication_derivatives/gzip_extra_aux_20261016_v3/restore_originals.py') --destination .
python (Join-Path $R 'publication_derivatives/final_md_20261016_v2/restore_originals.py') --destination .
```

## 다중 K 원 trial·실행 장부

[기존8·1440 trial](diagnostics/publication_multik_lossless_1792094047117000000/bundle_1792096101396562900/MANIFEST.json), [신규12·2160 trial](diagnostics/publication_multik_fresh_lossless_20261016_v1/bundle_20261016_actual_01/MANIFEST.json), [이전 부분166 보존](diagnostics/publication_multik_historical_partial166_20261016_v1/bundle_1792101441341745500/MANIFEST.json)은 서로 범위가 다르다. 최종20/3600에 이전 부분166을 다시 더하지 않는다. 기존 복원 코드의 `restore --manifest <manifest> --public-root $R`를 사용한다.

```powershell
$Bundle = Join-Path $R 'diagnostics/publication_multik_lossless_1792094047117000000/bundle_multik.py'
python $Bundle restore --manifest (Join-Path $R 'diagnostics/publication_multik_lossless_1792094047117000000/bundle_1792096101396562900/MANIFEST.json') --public-root $R
python $Bundle restore --manifest (Join-Path $R 'diagnostics/publication_multik_fresh_lossless_20261016_v1/bundle_20261016_actual_01/MANIFEST.json') --public-root $R
python $Bundle restore --manifest (Join-Path $R 'diagnostics/publication_multik_historical_partial166_20261016_v1/bundle_1792101441341745500/MANIFEST.json') --public-root $R
```

전수 복원 PASS는 원 target의 새 독립 감사나 성능 개선·학습 완료를 뜻하지 않는다. [정규화 지표와 공개 검산](AUDIT_REPORT.json), [후속 실제 집계·포장 검산](FINAL_POSTPROCESS_AUDIT.json), [상세 재현 조건](REPRODUCE.md)을 함께 읽는다.
