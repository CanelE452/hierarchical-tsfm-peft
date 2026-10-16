# 기존 full-C B2 두 조건의 prefix 실행 준비

**역사적 준비 문서의 공개 링크 사본.** 아래 actual0은 실행 전 snapshot이다. 현재 Beijing K11 완료와 Jena K21 미실행은 [실제 상태](RESULT_STATUS_KO_v1.md) 및 [현재 결과](../priority_valid_minima_20261016_v1/CORE_CONTRAST_KO_v1.md)를 따른다. 원 로컬 문서는 보존했고 공개용 MASTER 링크만 바꿨다.

**[확인] 현재는 코드·작은 메타데이터·stdlib 합성 검사 준비만 완료했다. 새 실제 Task/모델/체크포인트 해석, probe, fit, 전체 V, FP64 검산, CPU6 및 비용 실행은 모두0이다.** [PLAN_DRAFT.json](PLAN_DRAFT.json)은 `DRAFT_NOT_EXECUTABLE`, `runner_sha256=null`, `ROOT_after_N4_report_and_main_admitted=false`다. ROOT가 N4 보고·main 게시 후 원 목적과 잔여 자원을 다시 판단하고 새 PLAN을 봉인하기 전에는 실행할 수 없다. 자동 큐나 scheduler는 없다.

## 원래 등록된 두 조건

```text
dataset  original CPU run ID                                         C=K  safe prefix  A50/A60 ends  V bounds       V origins/observed
Jena     postT_cpu_jena_E1_B2_k21_s92601_remaining_v1                 21   0:16772      10482/12579   14675:16772    43/43344
Beijing  postT_cpu_beijing_E1_B2_k11_s92601_required_v1                11   0:16830      10519/12622   14726:16830    43/20539
```

Jena는 [core remaining REGISTRY](../postT_V_only_cpu_core_remaining_20261016_v1/REGISTRY.csv), Beijing은 [required minima REGISTRY](../postT_V_only_cpu_required_minima_20261016_v1/REGISTRY.csv)의 기존 CPU ID를 그대로 사용하고, 산출물만 이 새 namespace에 둔다. 원 spec의 stage_label과 provenance까지 대사한다. 기존 namespace의 결과·fit claim·partial은 실행 입장에서 다시 확인해 보존하며 자동 재개하지 않는다. 기존 폐기 probe claim은 fit 완료로 세지 않는다.

[원 MASTER](../plan/MASTER_PLAN_PUBLIC.md)의209–225행과300행은 최소 K1/4,K1/2,C 뒤 전체 grid를 채우도록 구분한다. **K9/16/241/128의 full-grid 미실행8과 K=C 필수8은 다르다.** K=C의 원 GPU `.25` screen은 존재하며 이번 준비는 그중 두 B2의 새 CPU3 search 깊이에 대한 adapter gap이다. 두 조건이 끝나도 K=C 최소8이나 full-grid8 전체가 완료되는 것은 아니다. 최초 Jena4 cap은 앞선 P1/P2 범위이고 이번2의 자동 실행 근거로 쓰지 않는다. 입장은 기존 필수 대조의 별도 후순위 ROOT 판단이다.

## 유지한 학습·표현 조건

L512/H48, seed92601, original B2, lr0.001, AdamW/weight_decay0, clip1, effective16/micro4, CPU2, F row128, checkpoint_inputs=False를 유지한다. 원 A50에서0/.25/1/3을 A_stop으로 선택하고 새 A60 모델을 선택된 길이만 refit한다. Jena search9923 origins/29769 exposures/1863 updates와 Beijing9960/29880/1868은 원 endpoint별 마지막 짧은 batch를 포함한 수다. 선택 refit 길이는0/.25/1/3 중 실제 값으로 기록하며 같다고 가정하지 않는다.

원 `fit_base.create_model`98–113행의 **K=C, k_grid 없음** 초기화는 `E=I`, center0이다. 실제 허가된 fit에서는 A50/A60의 이 초기화 metadata와 작은 basis/center를 검사하고 원 `_stage`를 호출하도록 준비했다. 이후 학습된 E를 identity로 고정하지 않는다. 고정 full-C PCA rotation 및 basis-transpose inverse와 다른 대조이며, 원 B2의 PED forecast MSE 학습과 frozen F를 그대로 사용한다. 새 방법이나 새 K를 추가하지 않는다.

## 과거 정보·원 계약

Jena는 고정 `safe_jena_prefix.load_jena_prefix`, Beijing은 고정 N3 `safe_task`/bounded reader를 사용한다. 전자는 source35135행의 F-order 열별 prefix0:16772, 후자는 source28051행의 C-order prefix0:16830만 수치화한다. 원 permitted20965/21038의 full Task SHA는 **INHERITED_NOT_RECOMPUTED**이고 실제 prefix SHA와 구분한다. 원 split bounds를 보존하며 짧은 Task 길이로 split을 다시 만들지 않는다.

실제 허가된 실행에서는 A50 mean/std/count, A50/A60 scaler와 원 V43 target/mask/origin/channel 네 지문을 다시 대사한다. `data.load_task`, `data.split_ranges(truncated)`, 전체 fit/prediction_cache/route_catalog 진입점은 호출하지 않는다. 전체 opaque asset SHA와 ZIP buffering은 tail byte I/O를 일으킬 수 있으므로 T byte I/O0이라고 부르지 않는다. 새 T 수치·target·점수 연산은 허용하지 않는다.

## 기존 증거와 새 실행의 구분

두 원 upstream은 `primary_jena_E1_B2_k21_s92601`, `primary_beijing_E1_B2_k11_s92601`이다. 원 recipe는 CUDA 학습/epochs[.25]이고 receipt는 제한 screen이다. 원 V manifest에 실제 추론 device는 기록돼 있지 않다. 원 자료·축·scaler·V 입력 지문을 확인하는 reference로만 연결하며, CPU 동등 정확도나 새 CPU3 완료·정확 continuation으로 재명명하지 않는다. 원 weights/optimizer/sampler를 가져오지 않는다.

Jena full-C CPU probe는 기존 증거가 없다. Beijing [기존 probe](../postT_V_only_cpu_required_minima_20261016_v1/diagnostics/probe_root_v1_04/RESULT.json)는 폐기된3 updates/48 exposures이며 fit·전체 V가 아니다. 이 adapter에 새 probe 모드나 자동 probe를 추가하지 않았다. **실제 full-C CPU peak RSS·학습 시간·완주 가능성은 미측정**이다. 필요한 자원 판단은 ROOT 입장과 외부 guard에서 해야 한다.

## 준비된 실행 경로

- `fit`: 원 registered spec 그대로 원 create_model/_stage, 새 selected bundle, 전체 CPU V43 physical 추론을 준비했다. private checkpoint/NPZ와 public receipt/curve/error sufficient stats/SCORES/manifest를 분리한다.
- `rescore`: 같은 safe V archive의 byte/member 지문을 검사하고 기존 독립 FP64 math.fsum channel macro·band 통계와 공개 SCORES를 대사한다. Torch/모델을 import하지 않는다.
- `replay`: 같은 selected artifact를 원 `deployment.Route` 전체 constructor로 새 CPU6(4+2) 추론하고 저장 V subset과 atol1e-5/rtol1e-4로 비교한다. 새 seed나 전체 V 일반화 검증이 아니다.
- `cost`: 해당 artifact/manifest의 실제 rescore/replay RESULT/SHA와 fresh120s quiet proof를 요구한다. 새 원 Route로 physical→preprocess/P/E/F/D→FP64 inverse/CPU output을 매 trial 실행하도록 준비했다. 이 경로는 V forecast NPZ를 읽지 않고 manifest metadata와 기존 검산 증거만 결속한다.

```text
dataset   full V43 streams/calls  CPU6 streams/calls  B1 trial streams/calls  B8 trial streams/calls
Jena      903/11                 126/2                21/1                    168/2
Beijing   473/11                  66/2                11/1                     88/1
```

각 B1/B8 비용은10 warmup·20 ordinary timed·별도20 intrusive stage다. B8은8개 batch 전체이며 stage sum=whole을 주장하지 않는다. F 상주 weights·CPU RSS·GPU peak·학습/로드 비용은 구분해야 한다. 실제 counters와 checkpoint/model/norm/source/scaler를 전후 대사하며 기존 결과·모델·코드를 수정하지 않는다. 원/새 namespace artifact 경로를 연결할 때 전역 monkeypatch나 옛 sealed allowlist 확장은 사용하지 않는다.

## 실제 준비 검증과 남은 검증

[확인] 기존69개 source/작은 metadata/opaque byte SHA를 대사했다. 원 source39+immutable9, 원 PLAN/REG/source와 두 registered spec, upstream GPU-training quarter reference 및 Beijing 폐기 probe, 고정 helper source를 확인했다. **새 raw asset/model bytes 재해시나 원 prefix 수치 해석은 하지 않았다.** prefix loader의 기존 header/실제 검산 증거는 상속한 근거이며 이 준비의 새 수치 검증이 아니다.

stdlib 합성 검사15개가 통과했다. 최초 미구현 상태11 assertion RED와 metadata-only cost validator의1 assertion RED를 확인했다. identity와 PCA rotation 구분, 두 원 spec/새 namespace/foreign state, prefix와 고정 T bounds, endpoint별 활동량, C21/C11 동적 F counters, quiet·exclusive output, ROOT 새 봉인, deadline의 부분 결과 보존을 검사했다. deadline fixture의 이전 분기는 메모리에서만 재구성하며 실제 checkpoint를 사용하지 않는다.

```text
Ran 15 tests in 0.085s — OK
wall_seconds=0.08565230015665293
RSS_bytes=32419840; OS_peak_working_set_bytes=35594240 (<512MiB)
numpy_imported=false; torch_imported=false
actual Task/model/CP/NPZ/fit/forward/probe/cost=0
```

ROOT의 새 `PLAN.json`, 현재 runner SHA, 긍정 post-N4 report/main admission, 명시 ROOT trigger가 모두 필요하다. 준비 담당자는 아래 명령이나 큐를 실행하지 않았다.

```powershell
& $Python -X utf8 -B "$CampaignRoot/priority_fullC_b2_20261016_v1/fullC_b2_runner.py" `
  --execute --root-triggered --public-root $CampaignRoot --cache-root $CacheRoot --asset-root $AssetRoot `
  --plan-sha $NewSealedPlanSHA --run-id $ExactOriginalCPUId --mode fit `
  --deadline 1792222550 --output $NewExclusiveDiagnostic
```

rescore/replay 뒤 cost에는 batch1/8·quiet proof/SHA·실제 rescore/replay proof/SHA를 추가한다. ROOT의 외부 CPU2/worker RSS3GiB/최대2개·합6GiB guard, storage30GiB+reserve256MiB, GPU0 및 원 deadline1792222550을 유지한다. 비용은 실제 학습·모델 worker와 겹치지 않아야 한다. trigger flag 자체는 OS quiet나 process-tree closure의 증거가 아니다.

원 stage의 optimizer-boundary/deadline checkpoint 동작만 상속하며 임의 signal autosave를 보장하지 않는다. 부분 결과는 보존하고 자동 재개/덮어쓰기를 거부한다. full V·실제 FP64·checkpoint replay·비용·수렴/자원·독립 일반화는 모두 아직 미검증이다. 한 seed와 반복 개발 V이며 두 조건의 성공이나 방법론 우위를 이 준비만으로 주장하지 않는다.
