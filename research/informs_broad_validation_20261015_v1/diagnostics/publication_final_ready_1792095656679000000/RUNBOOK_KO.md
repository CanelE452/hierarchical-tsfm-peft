# 최종 safe Markdown·명시 게시 후보 준비

이 준비물은 이전 `phase2_aux_v3.py`의 코드펜스 분리·영향 전이·Markdown mapping 우선 치환과 `phase3_candidate.py`의 기존 publication 검증기 호출 방식을 명시 입력형으로 분리한 것이다. 실제 후보 생성·압축·전수 검사·T 읽기는 아직 하지 않았다. 새로운 원자료·모델·GPU 경로를 만들지 않는다.

private cache 상대경로 `publication_final_ready_1792095656679000000/final_candidate.py`가 실행 helper다. 기존 base 4,600 FILE_LIST와 SHA를 고정한다. 그 외 추가 파일은 ROOT가 승인한 **명시 목록과 각 목록 SHA**만 받는다. 저장소 전체를 탐색해 임의 파일을 게시 대상으로 추가하지 않는다. 소유 출력 폴더에서 직접 생성한 backup·파생 Markdown·mapping·복원 helper만 별도로 추가한다.

## 입력 확정 후 실행

ROOT가 실제 aux95와 K8 bundle을 실행하고 최종 산출물 목록을 확정한 뒤 `SPEC_TEMPLATE.json`의 placeholder를 실제 경로·SHA로 채운 private spec을 만든다. 이 템플릿은 실행 가능한 최종 spec이 아니다. 미래의 T46·E0B·비용·replay·감사·최종 report/그림은 실제 완료된 파일의 정확한 allowlist를 `add_lists`에 추가한다. 기존 40/44 검토나 숫자를 이유로 신규 46/50을 제외하는 로직은 없다.

```powershell
$Helper = Join-Path $Cache 'publication_final_ready_1792095656679000000/final_candidate.py'
$Public = Join-Path $Repo 'research/informs_broad_validation_20261015_v1/publication_derivatives/final_md_RUN_ID'
$Private = Join-Path $Cache 'publication_final_candidate_RUN_ID'
# 실제 proof·명시 목록을 확인하되 생성하지 않음
& $Python -X utf8 -u -B $Helper --repo-root $Repo --spec $FinalSpec `
  --public-output $Public --private-output $Private
# 모든 입력 확정·정밀 비용 측정 종료 후 ROOT가 별도 실행
& $Python -X utf8 -u -B $Helper --repo-root $Repo --spec $FinalSpec `
  --public-output $Public --private-output $Private --execute
```

고유한 새 public `final_md_*` 폴더와 private 후보 폴더만 허용한다. 기존 v3·원본 Markdown·원 후보·registry·source는 수정하지 않는다. 자동 Git stage/commit/push는 없다.

## 입력과 제외의 한계

- base FILE_LIST는 SHA `c05f96e08df37ca12de9d016df788dcfec387fad6bbf673844d92f8d3298ee6d`와 4,600개를 요구한다. base inventory CSV도 SHA로 고정한다.
- `add_lists`는 각각 SHA로 고정한 저장소 상대경로 목록이다. 실제 aux95 payload·원 목록·기존 복원 helper·manifest·결과와 K8 payload·manifest·결과가 빠지면 실패한다.
- aux95는 기존 고정 목록의 각 원 경로·크기·SHA와 manifest 95개가 모두 일치하고 실제 `PASS_ALL95_BYTE_EXACT` 영수증에 연결되어야 한다.
- K8은 8 scope·1,440 trial의 실제 전체 임시 복원 PASS와 manifest SHA가 연결되어야 한다. trial 원문은 복원용 manifest로, 큰 원 JSON은 정확 gzip으로 연결한다. CSV·다른 원 영수증은 그대로 보존한다.
- 실제 후보를 만들 때 모든 aux/K8 payload의 현재 바이트·SHA를 검증된 manifest와 다시 대조한다. 원 측정을 다시 실행하지 않는다.
- `virtual_lists`는 이미 증명된 충분통계 복원 경로만 사용한다. aux 원문을 virtual로 넣어 잘못된 Markdown 링크를 숨길 수 없다.
- 일반적인 drop 인자는 없다. 후보에서 빠지는 것은 실제 무손실 표현으로 대체한 aux/K8 원 경로와 새 파생 Markdown으로 대체한 원 Markdown뿐이다. 조건·숫자·강한 대조·미완료 경로를 삭제하지 않는다.

## Markdown 보존과 전이

기존 v3의 original→render mapping을 사용해 과거 원문을 향한 링크도 현재 선택된 render에 연결한다. 새 aux/K8 대상에 직접 연결된 Markdown과 그 Markdown을 참조하는 선택된 Markdown 전체에 전이 closure를 적용한다. 새 Markdown→Markdown mapping은 `.md.gz` backup보다 우선한다.

원문은 `read_bytes().decode('utf-8')`로 읽어 BOM·CRLF를 보존한다. 기존 closed code-fence 분리와 publication 검증기의 Markdown/reference/HTML regex를 사용한다. 상대 링크의 목적지 span만 새 위치 기준으로 바꾸고 query·fragment를 유지한다. 외부 URL과 코드펜스는 그대로 둔다. 바뀐 목적지를 제외한 출력 바이트를 다시 모아 원문 바이트와 비교한다.

영향받은 모든 원 Markdown의 바이트는 별도 gzip으로 보존한다. 기존 v3 `restore_originals.py`를 바이트 그대로 복사하며 같은 `exact_auxiliary_gzip_v1` manifest를 사용한다. 별도 임시 저장소에 전수 복원·원 바이트 비교·반복 복원을 완료해야 한다. 기존 원 Markdown도 끝에 다시 바이트 비교한다. clone에서는 새 폴더의 `restore_originals.py --destination <저장소루트>`로 원 경로를 복원한다.

## 검증과 성공 조건

기존 `publication_check.inspect`를 새 파일 또는 base inventory SHA에서 달라진 파일에 적용한다. 기존 후보의 동일 SHA 파일은 기존 검사 증거를 재사용한다. 전체 선택된 Markdown/SVG의 상대 링크·대소문자는 기존 `_link`로 다시 검사한다. 기존 PNG/SVG·gzip·통계 archive의 해당 새 파일 검사도 기존 검증기를 사용한다. `virtual` 충분통계는 원문을 근사로 대체하는 권한이 아니다.

현재 알려진 registry HOLD를 예외로 허용하지 않는다. ROOT가 실제 canonical registry를 명시 목록에 포함하면 같은 검증기로 해결 여부를 확인한다. 링크/내용 검사 오류 0, 개별 20MiB 미만, 전체 250MiB 미만, root Python/원 Markdown 불변이어야 성공 FILE_LIST를 쓴다. 실패한 새 출력과 private 검사 결과는 남기지만 성공 후보를 만들지 않는다.

성공 private 폴더에는 `FILE_LIST.txt`, `CANDIDATE_FILES_WITH_SHA.csv`, `EXCLUDED_PRESERVED_FILES.txt`, `RESULT.json`, `NEW_CONTENT_INSPECTION.json`, `WHOLE_LINK_CHECK.json`이 생긴다. 이 결과도 최종 보고서의 과학적 완료나 Git push 완료를 뜻하지 않는다. 실제 최종 파일·T 범위·전체 비용·미완료 항목은 ROOT가 영수증에 맞게 기록한다.
