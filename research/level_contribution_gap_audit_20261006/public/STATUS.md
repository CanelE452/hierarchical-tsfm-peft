> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 [publication_manifest.json](publication_manifest.json)에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.

# LEVEL 기여 감사 — 상태

2026-10-06, 증거 감사 및 보고 완료. 대상은 원래 고정 PCA LEVEL이며 후속 A/B/GROUP 등의 성과는 대조 이력으로만 사용한다.

- [확인] 작업 루트와 origin은 CanelE452/hierarchical-tsfm-peft. 로컬 main과 원격 main은 `2154e7f2b0e9878164e77536b8deaa9f372ac760`로 일치한다.
- [확인] 기존 미게시 Chronos-2 비교와 Peacock matched PCA RAW가 로컬에 있다. 원격 부재를 미실행으로 해석하지 않는다.
- 허용 작업: 기존 기록·소스·선택·예측·검산의 연결과 보고. 이번 신규 fit/optimizer update/model forward/GPU/다운로드는 모두 0이다.
- 완료 순서: 실행 상태 및 이력 확인 → v1~v16과 로컬 보충 증거 감사 → 선행·주장·반론 평가 → 산출물 및 출처 검증.
- 완료 실험 재실행, 새로운 학습형 E/D 적합, 기존 TEST 재선택, 원본 발표물 수정, commit/push는 수행하지 않는다.

## 상태·승인·진행 중 작업

[확인] 초기·종료 조회에서 이 프로젝트의 Python/학습/Chronos 작업 프로세스는 관찰되지 않았다. 다른 프로젝트의 Python 프로세스는 보존했다. Codex 채팅 조회에서 이전 `TSFM PEFT 연구 주제 선정`은 notLoaded였으며 현재 감사 채팅이 active였다. 이전 채팅의 실제 사용자 Chronos 요청과 완료 turn을 읽었다. 원격 실행 서버 목록·원격 프로세스 관리 권한까지 전수 확인했다는 뜻은 아니다. 관련 원장에 미종료 작업이 있다면 현재 OS 프로세스와 대조한 한계를 유지한다.

현재 승인 범위는 첨부 기준의 증거 감사이며 신규 학습은 0회다. 과거 Chronos 요청은 실제 사용자 메시지 `<private-chat-record>`에서 확인했다. 이전 PLAN 파일 존재만을 승인 근거로 삼지 않았다. 이전 Chronos는 실제 완료돼 재개할 부분이 없다. 이전 남은 GPU 7,956.0035초를 신규 연구 예산으로 사용하지 않았다.

로컬 미게시 결과는 v17 Peacock matched PCA RAW와 이번 LEVEL Chronos 비교다. main 하나만 원격에서 조회됐고 SHA는 그대로다. 새 branch/PR/commit/push, 프로세스 종료, reset/clean/stash, 원본 발표물 덮어쓰기를 수행하지 않았다.

## 완료·부분·미확인 구분

- 완료 결과 연결: v1 학습형 E/D, v3 선형 대체, v5~v7 원형/RAW/직접 대조, v8~v10 보존 공간/단계적/ridge의 정확도, v11 U/Gamma/A, v12 Peacock, v13 GROUP, v14 비선형 및 저장 P/Q 교체, v15 좌표/decoder 정확도, v16 시간 압축, 로컬 v17 RAW, 현재 세 자료 Chronos/Small MV·UNI 및 비용.
- 일부만 완료: v10 비용68/78행, v15 GPU비용606/612행(Robin6행 누락). PASS를 전회차 완료로 해석하지 않았다.
- 실행 완료·원인미분리: U/Gamma/GROUP/P/Q 교체·좌표·시간 압축의 변화 원인. step0와 재사용은 추가 개선이 아니다.
- 이번 조회 미확인: 현재 Robin/Peacock/Jena의 동일 계약 학습 E/D-only 직접 대조. 전체 `result.json` 검색의 대상204개 중 E/D mode 미기재156개라는 검색 한계도 보존했다. 초기 Electricity 결과는 대체가 아니다.
- 미입증: 독립 방법론 신규성·고정 PCA 최적성·큰 분산 배분 인과·새 자료/백본 일반성·사전학습 비중복.

## 실제 계산과 검증 수준

이번 신규 fit/계수 적합/optimizer update/PCA 적합/model forward/GPU/다운로드/새 비용 측정/새 성능 재채점/bootstrap/PQ 교체는 모두 0이다. 기존 점수와 자체검산은 읽었고, 새 작업은 파일·출처·계약·구조 확인 및 보고였다.

- early: 열거한 계획→source→fit→selection→prediction→evaluation→checks 파일 실재, 기존 checkpoint/prediction 4개 제한 hash 예시.
- later: prediction/checkpoint642개 참조·297개 고유 경로의 실재·명시 크기, 24개 고유 파일/30검사 연산의 표본 hash. 참조 수는 실험 수가 아니다.
- Chronos: 연결된157개 파일의 집계 hash 검사와13개 봉인 파일 검사에서 누락·불일치0. 개별 actual digest 출력은 미보존하여 null로 남겼다. 신규 예측12개의 키·FP32·형상·원점 순서·유한성도 확인했다. 이전 forward/parity/점수 검산은 재실행하지 않았다.
- 최종: 18개 증거 행의 출처 경로, UTF-8/CSV/JSON 형식, 보고서 링크·표시값 출처, Git 보존과 실제 산출물을 검사했다. `verification.json`의 direct / recorded 구분이 기준이다.

CPU는 읽기·hash·소형 metadata/NPZ 구조 검사에만 사용했다. 전 감사의 CPU process time을 합산 계측하지 않았으므로 임의 수치를 만들지 않았다. 도구별 실제 종료·wall time은 실행 기록에 있으며 최종 검증 subprocess의 시간은 verification에 따로 남긴다. 대형 중간배열·새 `.cache` 폴더는 생성하지 않았다. 저장량은 verification의 실제 파일크기 합이며1GiB 상한보다 작다. 과거 Chronos GPU2,843.9965초/CPU37.8452초/다운로드111,750,017bytes는 **이전 작업 사용량**이다.

## 산출물과 판정

[최종 비판 보고](CRITICAL_REVIEW_KO.md), [증거 색인](evidence_index.csv), [다음 결정](NEXT_ACTIONS.md), [검증 receipt](verification.json). early/later/chronos artifact JSON은 직접 확인한 범위를 보존한다.

`evidence_index.csv`의 열은 이번 신규 스키마다. 주장/모델식/정보 계약/계획/source/fit/selection/prediction/점수/검산/재사용/반례/완료상태/판정/뒤집을 증거를 구분한다. 경로 배열은 JSON 문자열이며 기존 실험의 key 이름을 추정한 것이 아니다. 빈 개별 출처 칸은 색인 미수록을 뜻하며 미실행 확정이 아니다.

[판단] b 보완과 제한된 자료의 잔차 입력 효과는 조건부 유지, 일반 실용 우위는 축소, 독립 방법론 기여 확보와 원형 확장은 보류한다. 신규 E/D 계획은 이 공백을 숨기지 않고, 현재 중요한 결정을 바꾸지 않는 이유로 제안하지 않았다. 새 방향별 진단도 동일 이유로 생략했다.
