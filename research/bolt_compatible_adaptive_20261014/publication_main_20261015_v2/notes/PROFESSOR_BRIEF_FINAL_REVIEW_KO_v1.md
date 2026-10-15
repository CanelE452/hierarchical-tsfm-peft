> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인](../README_KO.md)을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.

# 교수 미팅용 방법론 메모 최종 독립 검토

검토일: 2026-10-15. 검토 대상은 [교수 미팅용 메모](../reports/professor_method_brief_v1/REPORT_KO.md)와 직접 열어 본 [도식 v3](../reports/professor_method_brief_v1/figures_v3/method_spec.png)다. 검토 중 발견한 v2 도식·본문의 두 표현은 ROOT가 수정했고, 수정된 문서를 다시 읽었다. 이 검토자는 원 메모·도식·과학 소스·결과를 편집하지 않았다.

**[판단] 현재 문서에서 교수님께 설명하기 전에 막아야 할 수식·핵심 원수치·링크 오류는 추가로 발견하지 못했다.** 구현과 비교 절차를 설명할 수 있다는 판단이며, 정확도 문제 해결·독립 일반화·학술적 신규성의 승인은 아니다.

검토한 메모 snapshot SHA256은 `8235b54edc896015214a5fb29a07a126f08e86cf070b58f807692fea426cbea3`다. 이후 본문이나 도식이 바뀌면 이 판정은 그 변경까지 자동으로 포함하지 않는다.

## 발견한 두 표현과 수정 확인

1. **[확인·해소] 미래 label의 역할.** v2 도식의 “Future labels appear only in FIT training.”은 실제 TUNE label 선정과 DEV label 채점까지 부정하는 문장으로 읽혔다. v3는 “FIT labels fit coefficients; TUNE labels select; exposed DEV labels score”와 “no future labels enter inference”로 나눴다. 배포 입력과 감독·선정·채점의 구분이 실제 절차와 맞는다. 기존 도식은 보존됐다.
2. **[확인·해소] 정규화와 PCA center.** 원 본문의 “scaler는 정규화 전에 이미 적용됐다”는 순서가 모호했다. 현재는 encoder 전에 FIT mean/std로 정규화했다고 쓰고, 그림·수식의 `μ_E`를 scaler 평균과 구분한다. 이 수정은 수치나 방법 변경이 아니다.

## 수식과 학습 범위

**[확인]** [원 correction 모듈](../../projection_followup_v1/horizon_decoder_delta_v1.py)과 [실행 controller](../../projection_followup_v1/run_horizon_decoder_diagnostic_v1.py)를 읽은 범위에서 메모의 식은 일치한다. `E:C×K`, `U:H×K`, lag-major `φ:7K`, `D_old/ΔD:7K×C` 차원이 맞고, 각 lag의 origin 기준은 `z_(t−lag)`다. 미래 시점은 예측값을 쓰며, origin 이전 known prefix를 그대로 쓴다. RAMP의 `h=1`은 0, `h=48`은 1이고, 48개 독립 디코더를 학습한 것이 아니다.

**[확인]** 이번 잔차 보정에서 PCA E·center·원 과거 재구성 D·F는 고정이고 ΔD만 FIT의 실제 관측 미래 residual로 적합한다. 실제 F ungated FIT 특징의 uncentered RMS를 네 arm이 공유한다. C별 관측 `SSE/N_c + α||표준화 slope||²`는 코드의 `count*α` ridge 분모와 맞는다. 무절편·무중심화, α0의 singular cutoff, 양수 α의 모든 singular 방향 사용, FP64 적합/FP32 배포, 별도 correction 덧셈과 ZERO byte-copy를 구분한 설명도 맞는다.

**[확인]** Raw baseline을 `current+(F−current)`로 다시 계산하지 않는 FP32 예외, Raw와 PCA의 서로 다른 추가 계수 수, FLAT과 current-C persistence의 차이, current-C 및 K 과거 buffer가 남아 strict K-only가 아니라는 제한이 드러나 있다. E(Y)와 수치 공간 minimum은 진단이며 이 보정의 배포·fit·선정으로 섞이지 않는다.

## 원수치와 비용 단위

NPZ를 열지 않고 [과학 RESULT](../evidence/projection_followup_v1/runs/horizon_decoder_diagnostic_v1/RESULT.json.gz), [원 비용 RESULT](../../projection_followup_v1/runs/horizon_decoder_cost_v1/RESULT.json), [반복 비용 RESULT](../../projection_followup_v1/runs/horizon_decoder_cost_repeat_v1/RESULT.json)의 JSON metadata와 메모를 대조했다.

- **[확인]** 정확도 표의 6행×3개 MSE 18개가 저장된 all-observed normalized channel-macro MSE의 표시 자릿수와 일치한다. 동일 공간·arm의 F 대 FLAT는 TUNE 12/12, DEV 12/12에서 F가 낮다. PCA RAMP의 자기 baseline 대비 DEV 악화는 4/4이고, 작은 K 보정의 Raw 대비 DEV 승리는 0/8이다. complete-C 원수치를 all-observed 값으로 바꿔 비교한 오류는 발견하지 못했다.
- **[확인]** 비용 표의 6행×원·반복 12개 값은 **B8 batch median ms**다. ms/window가 아니며, 메모의 “8로 나눈다”는 설명이 맞는다. 두 실행 전체 144개 행의 median 및 per-window 값을 저장 integer-ns 표본에서 다시 계산해 144/144 일치했다. Robin K8 B1의 Raw 대비 ratio-of-medians `1.062715258309972`와 `1.0036407766990292`도 메모 표시값과 맞는다. 이 계산은 F나 타이머 재측정이 아니다.
- **[확인]** [native RESULT](../../projection_followup_v1/runs/new_direction_native_v1/RESULT.json)의 Robin K8 두 DEV 값 `0.6290895742200046/0.6073359754376242`, Beijing K4 `0.7366342455840759/0.800842428334142`는 메모의 반올림과 일치한다. [ridge 확장 RESULT](../../projection_followup_v1/runs/ridge_regularization_extension_v1/RESULT.json)의 Beijing 원 C history DEV `0.630001184564156`도 일치한다.
- **[확인]** [cross-moment 감사](../../projection_followup_v1/audits/PREDICTIVE_CROSS_MOMENT_AUDIT_v1.json)의 Robin K8 E(Y) oracle-common-mask DEV `0.18166649606446122`와 같은 mask Raw `0.3404018870282048`가 메모와 일치한다. 미래를 안다는 조건의 진단이며 과거 예측 가능성의 증거로 쓰지 않았다. [cross RESULT](../../projection_followup_v1/runs/predictive_cross_moment_v1/RESULT.json)의 Robin FIT 과거 timestamp 출현 131072/고유1545도 일치한다.
- **[확인]** [오차항 RESULT](../evidence/projection_followup_v1/runs/horizon_correction_error_terms_v1/RESULT.json.gz)에서 Robin K4 STATIC의 TUNE interaction `−0.0293414237132`와 DEV `+0.0372698915775`, K8 STATIC DEV의 interaction `−0.0071809085412`·energy `+0.146077857460`·ΔMSE `+0.138896948919`는 메모의 수치·부호·해석과 맞는다. 최대 closure `9.700573677662305e−15`, 전체72/h3456/구간216/per-C52416도 일치한다. 이는 저장 endpoint 항등식의 보고이며 새로운 인과 실험은 아니다.

원·반복 비용을 각각 남기고 이전 76셀 cohort의 유리한 분모를 차용하지 않은 점, warm2/실측5·겹치는 첫B 창·공유 메모리·실행 간 원인 미확정 변동을 적은 점은 유지할 필요가 있다. 이번 검토는 별도의 속도 신뢰구간이나 독립 latency 재현을 만들지 않았다.

## 주장 범위·선행·과거 보호 incident

**[확인]** 메모는 “재구성 목적만 학습했다”는 잘못된 역사로 되돌아가지 않는다. 예측 목적 E/D·E-only·Q gradient·forecast D가 실행됐다는 사실과, 유한 학습 예산에서 그것이 Raw 격차를 해결하지 못했다는 관측을 구분한다. 모든 예측기 대비 F의 필요성, 동적/인과 그래프 발견, representation 학습 성공, full D_h·비선형 D 전체의 불가능성, 분포 변화의 유일한 원인 등으로 확대하지 않는다. 반복 노출된 DEV에서 적응적으로 설계한 연구라는 제한도 유지한다.

선행의 넓은 계열 설명은 공식 자료와 맞는다. [AdaPTS](https://proceedings.mlr.press/v267/benechehab25a.html)는 multivariate 입력을 latent feature로 바꿔 univariate FM을 차원별 적용하는 adapter를 다룬다. [ORTCL](https://ojs.aaai.org/index.php/AAAI/article/view/39526)은 입력·출력 feature의 orthogonal rotation을 사용하는 streaming continual learning 방법이다. [Ben Taieb·Hyndman](https://proceedings.mlr.press/v32/taieb14.html)은 horizon별 direct strategy로 recursive forecast를 보정한다. 메모는 이들과의 동일 구현·우선권 또는 신규성 완전 조사를 주장하지 않는다. 본 검토도 포괄적인 novelty search가 아니다.

**[확인]** [과거 Electricity 접근 incident 원문](../references/research/level_compressor_fullscreen_20261011/PROTECTED_ACCESS_INCIDENT.json)은 전체 Arrow target을 두 번 materialize한 보호 접근 위반을 기록한다. 기록상 그 값은 모델 학습·추론·채점·방법/하이퍼파라미터 선정에 사용되지 않았고, Robin/Jena/Peacock 계약을 변경하지 않았다. 현재 continuation의 새 접근0과 연구 전체 역사의 접근0을 분리한 메모의 설명은 맞다. 필요하면 상담 중 “별도 Electricity 조사 접근 위반이며 현재 Robin/Beijing 예측 성능 계산에 쓴 자료라는 뜻은 아니다”라고 보충할 수 있으나, 위반 자체를 없던 것으로 설명해서는 안 된다.

## 링크·그림·검토 한계

**[확인]** 수정된 메모의 Markdown href 23개 중 외부 URL3개를 제외한 로컬20개는 모두 실제 경로에 존재했다. v3 PNG를 직접 열어 label·식·화살표를 확인했다. FIT/TUNE/DEV 역할과 μ_E가 명확하고, 상단 anchor·하단 known-past buffer·추가 correction을 구분해 읽을 수 있었다. SVG와 링크 존재도 확인했지만 모든 뷰어의 렌더링이나 GitHub 게시 상태까지 확인한 것은 아니다.

검토자는 SOURCE/JSON/MD·도식만 읽고, stdlib로 JSON 원수치·integer-ns median·로컬 경로를 대조했다. 실제 연구 NPZ·원데이터·checkpoint·모델은 열지 않았다. 새 fit·F forward·VJP·optimizer·비용·과학 감사·보고서 생성·Git 작업은 실행하지 않았다. 기존 PASS 원문을 확인하고 숫자가 일치함을 재검토한 범위이며, 기존 과학 검산과 별개의 세 번째 완전 독립 재현 또는 미노출 holdout 검증이라고 부르지 않는다.
