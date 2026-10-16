# Beijing 핵심 대조 검산·보고 준비

상태: **PREPARATION_ONLY / 실제 감사·보고 생성 0**. 이 문서는 현재 진행 중인 두 fit의 성공·완료를 뜻하지 않는다. 기존 N4 산출물·원 adapter·PLAN·통계·모델은 수정하지 않았다. ROOT가 실제 fit 두 개와 POST 여덟 단계를 마친 뒤 원 영수증을 검토하고 명시적으로 실행한다.

준비 파일은 `audit_core_contrast.py`, `test_audit_core_contrast.py`, `build_core_contrast.py`, `test_build_core_contrast.py`, 이 문서 다섯 개다. 감사기는 stdlib만 사용하며 모델·Task·원자료·NPZ·체크포인트 텐서·보호 T를 수치 해석하지 않는다. 프로세스를 조회하거나 시작·종료하지 않는다. 기존 N4의 고정된 stdlib helper 소스에서 순수 산술·입력 결속 기능을 재사용한다.

## 답할 질문과 정확한 세 조건

1. 같은 K6에서 independent B0에 B2 앞단을 더할 근거가 있는가?
2. 같은 B2에서 K6과 K11=C의 정확도와 전체 비용 경계는 어떠한가?

```text
role    run_id                                                        source
B0_K6   postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1                새 N9
B2_K6   postT_cpu_beijing_E1_B2_k6_s92601_required_v1                   기존 N4 재사용
B2_K11  postT_cpu_beijing_E1_B2_k11_s92601_required_v1                  새 N8
```

세 조건은 C11/L512/H48/seed92601, A50 끝10519·A_stop/A60 끝12622, V[14726,16830)의43 origins·20539 observed labels, A60 scaler와 target/mask/origin/channel 지문을 공유해야 한다. B0와 B2 모두 frozen Bolt를 사용한다. K6 PCA 초기화와 K11 identity 초기화, 앞단·K·모수 용량·선택 refit·실제 노출량이 함께 다르므로 pure encoder·동등 compute·순수 압축 정보 손실 또는 동일 D의 효과로 부르지 않는다. noBolt 재실험이 아니다.

## 실제 실행에 필요한 결속

아직 존재·완료를 가정하지 않는 두 값은 최종 fit dispatch `RESULT.json` SHA와 최종 POST `RESULT.json` SHA다. ROOT가 실제 파일의 SHA를 명시 인자로 제공해야 한다. 현재 감사 소스 자체 SHA도 명시해야 한다.

감사기는 다음 고정 소스와 원 증거를 결속한다.

- N9 PLAN `55e40b087c2a1104e5e290053f61ab4c6fc4dd5d88ab8f446ddc27c511d3eba6`, runner `a1a58875ccf1715d51dce036618768b9521a623f7adf0b78ad6d43562495125d`.
- N8 PLAN `78b0a614d0b96c342a6996673f0cd735733718e259072f35bde9b983f7d813aa`, runner `e8fbb4a57144c470b8b49ae147a84f45689f34b3ba78d46325407606e7e94e7e`.
- 원 fit dispatch 소스 `3e48df1d2c6ee9cfdca0d2be425486c92e829d7a6682f1e2e743c55d023ee882`, 실제 fit SPEC `3faf463d4c89269a47d20e83929d76b5ae7a23ae6c7766377aa2e92e9d765390`, POSTv2 소스 `b4b7f4cdb00835b333c1b491bf9c514b4ba032a6370ea4d1dbf67a871708f192`, 원 guard 소스 `82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b`.
- 원 multik PLAN `2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec`의 source39+immutable9 및 새 두 PLAN의 고정 의존성.
- 기존 N4 감사 `72c8174301959f0c9de7eeeae34212b54452f7dda95192785898b7a525c04a68`, 보고 CSV `563f1140b673af0e7a187b1a05338d5e5d1b32b22de7eb7c7ff225c1d0ea74cc`, 보고 binding `ce51ecfb61308a8833013fc2be2a2124b51ad2f8b5409904e7964616503c1f6d`, B2 K6 manifest `2462712b8596c76400980a20733705eab68950da572b7c6dc4e79da95e912a73`.
- stdlib 산술 helper `933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04`와 입력·학습·cohort helper `8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd`.

기존 N4 B2 K6는 원 감사의 PASS와 그 입력 SHA에 묶인 fit·stage·manifest·비용 영수증·기존 공개 CSV를 재사용한다. 기존 충분통계·모델·예측을 다시 수치 해석하거나 재실행하지 않는다. 새 두 조건은 원 search0/.25/1/3과 fresh selected A60 refit의 실제 update/exposure·stage receipt, 선택 artifact/manifest, 실제 FP64 rescore와 fresh CPU6 replay를 요구한다.

실제 guard는 개별 worker returncode/stop reason, STARTED의 PID/create_time과 owned tree, tree join/errors/remaining, SPEC/helper SHA, 정규화된 effective 경로·자원·argv/cwd/max_seconds, 관측 group/worker RSS를 확인한다. 양쪽 fit과 전체 POST의 강한 종료·핀 보존 영수증을 요구한다. 감사기가 현재 PID를 조회하거나 관측되지 않은 후손의 부재까지 증명하는 것은 아니다.

## 산술 및 비용 계약

새 두 조건의 공개 충분통계는 각각43×11×3=1419 band 행이다. 총2838행이며 dense point 수나20539 observed label 수와 다르다. 채널별 SSE/count·SAE/count를 `math.fsum`으로 구하고 관측된 채널을 동일 가중 평균한다. global SSE/global count로 바꾸지 않는다. 원 manifest/SCORES/별도 full-array FP64 receipt와 원 atol1e-9/rtol1e-7로 대사하고, 기존 N4와 origin×channel×band 관측 count binding도 일치해야 한다.

새 비용은 두 조건×B1/B8의4개, 각 warm10·ordinary timed20·별도 intrusive stage20의50행으로 총200행이다. CPU2에서 history/mask→전처리→P/E/F/D→FP64 inverse/CPU 반환 전체 route를 실제 측정한 원 영수증만 허용한다. first V origins와 물리 input/mask SHA는 기존 B2 K6의 같은 batch와 같아야 한다. K6·K11에 따른 실제 F streams를 구분하며 B1/B8은 row128 이하라 각 trial physical call1이다. 새4조건 총7650 streams·200 calls,8개 summary의40개 지연시간 수치를 검산한다. B8 시간을8로 나누거나 K/C를 속도비로 대입하지 않는다.

원 비교 정책대로 timed/stage 각각 min/median/`fsum` mean/max/linear p90를 재계산한다. **cost 요약의1 ULP 차이도 NOT_COMPLETE**로 남기고 새 공차를 만들지 않는다. 각 비용은120초 이내 fresh quiet proof, 실제 rescore/replay outer SHA, source/scaler/artifact 및 종료 영수증을 요구한다. 비용 구간끼리 또는 기록된 fit/validation 구간과 겹치면 거부한다. stage 합=whole을 주장하지 않는다.

원 측정에 F만의 resident bytes는 없다. CSV의 `F_resident_bytes`는 NA(빈 필드), scope는 `NOT_SEPARATELY_RECORDED`다. 전체 `resident_weights_cpu_MiB`, 관측 tree peak RSS, F instance1·frozen hash·runtime streams/calls를 각각 보고한다. 모델 파일 바이트·전체 CPU weights·텐서 크기를 F만의 실측 메모리로 대체하지 않는다.

## ROOT 실행 형식

아래는 **실행하지 않은 형식**이다. 경로 변수와 SHA 자리표시는 ROOT가 실제 파일에서 채운다. 원 guard/자원·보호 T 계약은 바뀌지 않으며 감사·builder가 모델 작업이나 다른 큐를 시작하지 않는다.

```text
python audit_core_contrast.py --execute --root-triggered
  --public-root <PUBLIC_ROOT> --cache-root <CACHE_ROOT>
  --dispatch-sha <ACTUAL_TERMINAL_FIT_RESULT_SHA>
  --post-sha <ACTUAL_TERMINAL_POST_RESULT_SHA>
  --audit-sha <CURRENT_AUDITOR_SHA>
  --output <PUBLIC_ROOT>/priority_valid_minima_20261016_v1/INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json

python build_core_contrast.py --execute --root-triggered
  --public-root <PUBLIC_ROOT> --cache-root <CACHE_ROOT>
  --audit-path <ACTUAL_NEW_AUDIT_PATH> --audit-sha <ACTUAL_NEW_AUDIT_SHA>
  --builder-sha <CURRENT_BUILDER_SHA>
```

감사 input이 없거나 partial/불일치이면 새 JSON에 `NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED`를 남긴다. 원 자료를 바꾸거나 성공을 가정하지 않는다. 기존 출력이 있으면 입력을 읽기 전에 거부한다. 부분 완료의 유용한 결과는 ROOT의 별도 수동 partial 보고가 필요하며 세 행 완료 builder는 실행할 수 없다.

builder는 실제 새 audit PASS와 입력 SHA 보존·기존 N4 정확한 CSV/manifest를 다시 결속한 후 아래 세 파일만 CreateNew한다. 공개 binding에는 repo/cache 상대경로만 넣는다.

```text
CORE_CONTRAST_RESULTS_v1.csv
CORE_CONTRAST_KO_v1.md
CORE_CONTRAST_BINDINGS_v1.json
```

## 준비 검증 및 남은 확인

실제 통계/trial 숫자·NPZ·Task·모델·checkpoint 텐서·보호 T·프로세스·Git을 읽거나 실행하지 않고 stdlib 합성16개를 실행했다. 최초 소스 부재 RED, guard의 effective max_seconds 불일치 fixture RED를 확인한 뒤 구현/보완했고 최종16/16 PASS(0.049초, 전체0.188초, exit0)였다. 원48과 두 계획·의존성·고정 helper를 합친60개 metadata/source pin의 전후 SHA도 같았다. 두 CLI의 기본 실행은 input_reads0/files_written0/scientific_or_child_operations0 및 NumPy/Torch import false를 반환했다. 실행은 Python stdin의 작은 unittest suite였으며 root-wide pytest·실제 모델 테스트가 아니다.

검사는 macro 채널 균형, band count 동일성, K11을 K6 counter로 오인하는 경우, 누락/중복/잘못된 stage trial, 원 cost1 ULP 거부, 실제 replay·artifact binding, guard SPEC/STARTED PID-create/tree/resource/cwd/max_seconds, partial POST 거부, 기존 출력 보존, 원정밀도 CSV/NA memory/reuse 표기를 다룬다.

실제 두 fit의 최종 종료·POST8개·신규 sufficient statistics/trial·원 guard receipt와의 end-to-end 연결은 **아직 미검증**이다. 현재 실제 audit/report 산출0이며 미래 RESULT SHA를 추정하지 않았다. 준비본은 ROOT의 전체 source review와 실제 완료 증거 검토 후에만 사용할 수 있다. 한 seed·반복 dev V·제한된3 검색이며 검사 통과가 성능 개선·인과성·충분 수렴·독립 일반화 또는 전체 MASTER 완료를 뜻하지 않는다.
