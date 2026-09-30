**v11은 세 자료에서 A와 B를 각각의 직접 대조 및 외부 기준선과 비교하는 개발 실험으로 확정하겠습니다.** 기존 LEVEL 부모 6개와 Robin/Jena 직접 NLinear 4개를 재사용하고, Hog 직접 NLinear만 새로 적합합니다. 기본 실행은 **신경망 40fit + Gamma 12fit**, 기술복구 예비 8회를 포함해 **최대 60attempt**입니다.

이번에는 저장소·기존 기록·파일 존재와 해시·설치 소스·공식 자료만 읽었습니다. **새 학습·예측·채점·benchmark·프로젝트 파일 수정·commit/push는 하지 않았습니다.**

원안에서 바꿔 제안하는 것은 비용 측정의 반복 수입니다. 전체 방법과 소배치 grid를 유지하면서 **모든 구성의 24원점 반복을 20→10회로 통일**합니다. 아래 실행량과 시간 근거를 포함해 이 변경까지 한 번에 승인받겠습니다.

**1. 출발 상태와 이번에 판단할 주장**

[확인] 로컬 `main`과 실시간 원격 `origin/main`은 모두 다음 커밋입니다.

`efc029f8aae619b499071967d6b4764cb55fb802`

작업 저장소는 [hierarchical-tsfm-peft](E:/CODING/proj/hierarchical-tsfm-peft), origin은 `https://github.com/CanelE452/hierarchical-tsfm-peft.git`입니다. 추적 파일의 수정은 없습니다. 기존 `_docs/history` 임시 기록, `chronos-2-finetuned/`, 발표 그림 및 다른 연구 폴더의 untracked 파일은 보존합니다.

프로젝트·상위 경로에서 추가 `AGENTS.md`는 발견되지 않았고, 적용되는 [전역 AGENTS.md](C:/Users/User/.codex/AGENTS.md)를 확인했습니다. 과거 연구별 계약과 위반 기록은 그대로 두고, 이번 승인 범위는 별도 원장으로 관리합니다.

다음 두 폴더는 현재 존재하지 않습니다. 승인 후 생성합니다.

```text
E:/CODING/proj/hierarchical-tsfm-peft/research/
  tsfm_peft_internal_vs_subspace_v11_20260930/

E:/CODING/proj/hierarchical-tsfm-peft/.cache/
  tsfm_peft_internal_vs_subspace_v11_20260930/
```

이번에 검증할 것은 다음 세 가지입니다.

- **A의 내부 적응 가치:** 기존 LEVEL의 Q 예측을 유지하면서 잠재 TSFM 내부 LoRA가 정확도를 회복하는가?
- **B의 공간 학습 가치:** 고정 PCA와 동일한 NLinear 출발점보다 U 학습이 도움이 되는가? 그 효과가 Gamma 혼합만으로도 얻어지는가?
- **실제 선택 가치:** F0·미압축 LoRA·직접 NLinear 및 정상적인 소배치 실행을 고려해 정확도–메모리–시간에서 선택할 이유가 생기는가?

A/B는 부모와 학습 단계가 다른 전체 시스템입니다. A–B 차이를 순수한 내부 적응 또는 공간 선택의 인과효과로 해석하지 않습니다.

**2. 실제 재사용 대상**

아래 경로 표기를 줄이기 위해 캐시 루트를 정의합니다. 모두 실제 확인한 위치입니다.

```text
R  = E:/CODING/proj/hierarchical-tsfm-peft

C4 = R/.cache/tsfm_peft_fresh_group_v4_20260927
C5 = R/.cache/tsfm_peft_accuracy_recovery_v5_20260927
C6 = R/.cache/tsfm_peft_level_confirmation_v6_20260928
C7 = R/.cache/tsfm_peft_practical_controls_v7_20260928

아래 run의 checkpoint 경로 = 해당 C*/runs/<run ID>/best.pt
seed 쌍의 표시 순서 = 92601, 92602
```

[확인] 핵심 부모와 S*의 실제 checkpoint 및 SHA-256입니다.

```text
Hog LEVEL — C5
v5_hog_level_res_92601
d021957c4ec44f0fcb1753cbd8b9df39d25adcda41bdffe6d217520ce3698433
v5_hog_level_res_92602
30267e143fc3bf4842e136fcc801e4de0b8d5350f64c2965eac2b2a5ac28dccc

Robin LEVEL — C6
v6_robin_level_res_92601
fffabef217686fdd7193206ee6011f833b696b6bbcaa3655d6c4e8d4f88ca647
v6_robin_level_res_92602
b65baa1d1dc13d0146e238345b8aba4af38718cc62fbdcb72d78a8f0c568d02c

Jena LEVEL — C7
v7_jena_level_res_92601
2dd129ab4483e1cc359647f9522c35afff74d01f8d45578d5c92943134ec0459
v7_jena_level_res_92602
4057b762a43c83d6d762cbd15fe12857cc481226e2ddcd08da905643c359e5e8

Robin DIRECT_NLINEAR — C7
v7_robin_direct_nlinear_lr0.001_92601
7632df5802548d8a5314747c8d873cc2669cbed2633b4d52a385ea2e1b735604
v7_robin_direct_nlinear_lr0.001_92602
e14a9821ec7e31e8ea7c5f96fdff7fc0dcd1c1cb01b1932cccf3a89e8cffae6f

Jena DIRECT_NLINEAR — C7
v7_jena_direct_nlinear_lr0.001_92601
8c60869dedf90bb72a69857ebce61560f610a134bb1ad3370a043d88202ae79d
v7_jena_direct_nlinear_lr0.001_92602
f521e2b0eeecf4c68422941160ea7dafc180f97044cef9151707a64572caf3aa
```

Hog에서 동일 계약의 직접 원채널 NLinear는 찾지 못했습니다. 따라서 다른 선형 모델로 대신하지 않고 **두 LR×두 seed, 4fit**을 포함합니다.

초기 state와 PCA 연결도 확인했습니다. 아래 해시는 가독성을 위한 **SHA-256 앞 12자리**입니다. 승인 후 manifest에는 전체 값을 기록합니다.

```text
자료    초기 state 실제 경로                         seed별 해시
Hog     C5/initial/hog_level_{seed}.pt                 315dcb5010ef / 2f993421d33f
Robin   C6/initial/seed_{seed}.pt                      82eb23a5894f / 356b45715ef3
Jena    C7/initial/jena/edg_seed_{seed}.pt              f39efbf406af / b95f25fa0601

Robin S C7/initial/robin/direct_seed_{seed}.pt          9138a9b9b6bf / a5a448285888
Jena S  C7/initial/jena/direct_seed_{seed}.pt           ee5ff3d2dad0 / 9eecf447766e

PCA basis/decoder tensor 해시
Hog     4fafd94f089fb425270a8f3da05542c788cc11552133c68d036a9f8d174e0747
Robin   352532dd5a659db657de52c3333a7101c11345f82c499c1ec04d75256390e0f9
Jena    de502c98819a1757c5b92bf513a44f47c609275f66363feef8790009246ac283
```

A에는 이 초기 G가 아니라 **각 best checkpoint의 학습된 G***를 사용합니다. 초기 state는 출처·구조·동결 상태를 확인하는 데 사용합니다.

저장 예측도 재사용 가능합니다.

```text
역할                  실제 예측 위치                                   해시 앞 12자리

Hog LEVEL             C5/evaluations/hog_level_01/<run ID>.npz          791c75c4df83 / c48b9a54e7aa
Hog F0/native LoRA    C4/v4_test_01_predictions/<run ID>.npz            b8035a990065
                                                                       255e51113de8 / aea88e1f5ab4

Robin LEVEL           C6/evaluation01_predictions/<run ID>.npz         82ba61b3e2a8 / 4ac3c80aab00
Robin F0/native LoRA  같은 폴더                                        aeb50b502c9d
                                                                       7857869b3493 / 8f4927b5ed0d
Robin DIRECT          C7/robin_eval01_predictions/<run ID>.npz          629e1674e4ae / e5d64d1f7beb

Jena LEVEL            C7/jena_eval01_predictions/<run ID>.npz           262ef5690e46 / 2848e0f3eb4d
Jena DIRECT           같은 폴더                                        a9b4203d2c6f / 7f5db918a543
Jena F0/native LoRA   같은 폴더                                        d433daed68b5
                                                                       7c1f72fec22e / 0044d21fc097
```

F0/native LoRA run ID는 각각 `v4_hog_f0`, `v4_hog_lora_{seed}`, `v6_robin_f0`, `v6_robin_lora_{seed}`, `v7_jena_f0`, `v7_jena_lora_{seed}`입니다. checkpoint는 각 C4/C6/C7의 대응 run에 있습니다. native LoRA checkpoint 해시 앞 12자리는 Hog `c159862e3da5/868b1dfd5892`, Robin `b824c28f403d/f47a07acce78`, Jena `883ca19277ea/1dd12f6045d3`입니다.

병합 모델은 이 어댑터와 공식 백본에서 **별도 배포 복사본**으로 만듭니다. 과거 timing만 재사용하거나 출처가 다른 merged weight를 끼워 넣지 않습니다.

Robin/Jena S*의 선택은 [selected_robin.json](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_practical_controls_v7_20260928/selected_robin.json), [selected_jena.json](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_practical_controls_v7_20260928/selected_jena.json)에 연결합니다. 기존 두 LR의 seed 평균 VAL은 다음과 같아 모두 `1e-3`이 엄격하게 낮았습니다.

```text
자료      LR 1e-3       LR 1e-4
Robin     0.435457819    0.437953086
Jena      0.488382899    0.506150966
```

기존 v7의 정확 동률 우선순위는 새 v11과 다르지만 실제 동률이 아니므로 재사용 결과에는 영향을 주지 않습니다. 이 차이를 기록하고 재학습하지 않습니다.

**3. 데이터·환경 계약**

[확인] 다음 배열이 로컬에 있으며 기존 기록의 해시와 연결됩니다.

```text
Hog
C4/data/trainval.npz
d721612f985cbd435a4a53573338e8bc0a77cd5c8fb984bbc8d1ac5883e2172f
C4/data/test.npz
5a1ffdb00fdbefeb70318b2e514d46d731ff4957a5d63ae7a33e69b974a9524b

Robin
C6/data/trainval.npz
04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14
C6/data/test.npz
ecb32c192c56140a2b896b16d7936f8e9a983d78bef0bdec1ef6d35e2ddb6aa1

Jena
C7/data/jena_trainval.npz
29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933
C7/data/jena_test.npz
f188ee288c1a3c4722d673a4882a0c271f79ea3f161a725f976869da780161e5
```

배열의 실제 key는 `x, finite, times, columns, mean, std, basis`와 분할·원점 항목입니다. 기존 채널 순서와 `train_origins`, `val_origins`, `test_a_origins`, `test_b_origins`를 그대로 사용합니다. 전체 NPZ 해시에 이 정보가 포함되며, 실행 manifest에서 개별 원점·채널·mask 식별자도 연결합니다.

```text
자료    C/K    간격     TRAIN 원점   VAL 원점   평가 A/B   TRAIN 위상
Robin   17/5   1시간       1,945          30      40/40       24
Hog     32/8   1시간       1,945          30      40/40       24
Jena    21/6   10분       25,648         371     365/365     144
```

모두 L512/H48, 평가 stride는 24관측입니다. 따라서 Jena는 문맥 85시간 20분, 예측 8시간, 평가 원점 간격 4시간입니다.

Robin/Hog의 목표 구간은 기존 반개방 구간을 유지합니다.

```text
TRAIN   [2016-01-23, 2016-04-15)
VAL     [2016-04-17, 2016-05-18)
TEST-A  [2016-05-20, 2016-06-30)
TEST-B  [2016-06-30, 2016-08-10)
문맥 확보 시작: 2016-01-01
```

Jena는 실제 시작 시각을 유지합니다.

```text
TRAIN   [2024-01-01 00:10, 2024-07-01 00:00)
VAL     [2024-07-01,       2024-09-01)
TEST-A  [2024-09-01,       2024-11-01)
TEST-B  [2024-11-01,       2025-01-01)
```

TRAIN 평균·표준편차·PCA와 입력 결측 대체를 재사용합니다. 관측 mask는 유지하고, 관측된 0이나 불리한 채널을 제거하지 않습니다. Jena의 기존 sentinel 처리도 유지합니다. 별도 timezone 변환이나 날짜 재매핑은 하지 않습니다.

[확인] 현재 환경은 Python 3.11.16, PyTorch 2.10.0+cu128, Chronos 2.3.2, PEFT 0.21.0, Transformers 5.17.0, SciPy 1.17.1입니다. GPU는 RTX 4070 12GB이며 E 드라이브 여유 공간은 약 442GiB였습니다. 다른 프로젝트의 Python 프로세스가 있어 실행 직전 점유를 다시 확인합니다.

백본은 다음 snapshot을 재사용합니다.

```text
R/.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/
772f3d25d38aec6d914c8949dab4462e2d46f5d8/

model.safetensors SHA-256
06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd
```

추가 설치·raw/백본 다운로드는 필요하지 않은 것으로 확인했습니다. 실제 복원과 출력 정합성은 승인 후 검사할 항목이며, 파일 존재만으로 이미 통과했다고 선언하지 않습니다.

**4. 가까운 선행과 주장 경계**

이번 설계의 기여는 새 LoRA 수학이나 최초의 잠재 어댑터가 아니라, **어떤 적응 위치와 예측 결합이 정상적인 외부 대안 앞에서 유용한가를 직접 비교하는 것**입니다.

- LoRA의 frozen base·zero-update 초기화·저랭크 내부 적응은 기존 원리입니다. 공식 설명과 설치된 PEFT 소스의 초기화·병합 경로를 확인했습니다. [LoRA 공식 출처](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/), [PEFT 문서](https://huggingface.co/docs/peft/en/package_reference/lora)
- AdaPTS의 잠재 입출력 어댑터와 예측 목적 공동학습이 B와 겹칩니다. B의 구체적 차이는 frozen 원채널 S*를 기준으로 tied orthogonal U의 TSFM 보정분을 더하고, U 학습과 사후 Gamma 적합을 분리하는 것입니다. 공식 논문집 정보와 저자 코드의 해당 경로를 읽었고, 이번 조회에서는 최종 PDF 방법절 전체를 확보하지 못했습니다. 이를 완전한 선행중복 검토로 표시하지 않습니다. [AdaPTS](https://proceedings.mlr.press/v267/benechehab25a.html), [저자 코드](https://github.com/abenechehab/AdaPTS)
- S*는 저자 NLinear의 마지막 관측 제거·복원과 shared affine 정의에 대응합니다. 전체 논문 실험의 재현이나 모든 선형 모델의 상한은 아닙니다. [NLinear 저자 구현](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)
- ForeCA는 분산과 시간적 예측 가능성이 다르다는 근거입니다. 특정 TSFM의 추가 이득을 찾는 현재 B의 효과를 보장하지 않습니다. 공식 논문의 도입부와 관련 방법절을 확인했습니다. [ForeCA](https://proceedings.mlr.press/v28/goerg13.html)
- 반복 VAL 선택과 Gamma 적합의 선택 편향을 명시합니다. 세 자료의 이번 평가는 독립 확증이 아닙니다. [Cawley–Talbot](https://jmlr.org/papers/v11/cawley10a.html)

검색 결과의 부재로 최초성을 주장하지 않습니다. 문헌의 남은 확인은 실행과 병행할 수 있으며 학습을 무기한 막는 조건으로 두지 않습니다.

**5. 고정할 A/B 수식과 학습 변수**

\(U_0\in\mathbb R^{C\times K}\), \(P_0=U_0U_0^\top\)이며, \(R_n(a)\)는 시간축 n번 반복입니다.

\[
r_0=X-XP_0,\qquad
q^*(X)=R_H(\operatorname{last}(r_0))
 +G^*\!\left(r_0-R_L(\operatorname{last}(r_0))\right).
\]

\[
\mathrm{BASE\_LEVEL}(X)=F_0(XU_0)U_0^\top+q^*(X).
\]

여기서 G*는 부모 checkpoint의 공유·무편향 `512→32→48` 예측기입니다.

**A:**

\[
\boxed{A(X)=F_\phi(XU_0)U_0^\top+q^*(X)}
\]

새 LoRA만 학습합니다. 원래 TSFM weight, U0/E/D, G*, 통계와 buffer는 고정합니다. LoRA는 `r8/alpha16/dropout0/q,v/bias none`이며 zero-update 초기 출력은 기존 LEVEL과 대응시킵니다.

직접 대조는 다음 두 가지입니다.

- `A − BASE_LEVEL`: 고정 부모 위 내부 적응의 효과.
- `LORA_FULL_MSE(X)=F_\phi(X)`: 동일 LoRA 초기값·LR 후보·점예측 MSE를 사용하는 미압축 적응.

기존 `LORA_NATIVE`는 native quantile loss로 학습한 별도 실용 기준선으로 유지합니다. TEST가 더 좋은 LoRA만 골라 하나의 이름으로 합치지 않습니다.

**B:**

\[
S^*(X)=R_H(\operatorname{last}(X))
 +T^*(X-R_L(\operatorname{last}(X))),
\]

\[
\boxed{
B(U,\Gamma;X)=S^*(X)+
\left[F_0(XU)-S^*(X)U\right]\Gamma U^\top
}
\]

\(T^*\)는 원채널 공유 `Linear(512,48,bias=True)`이며 24,624개 파라미터입니다.

```text
역할                 고정                         새 학습/적합
B_FIXED_U            S*, F0, U0, Gamma=I           없음
B_LEARNED_U          S*, F0, Gamma=I               U
B_FIXED_GAMMA        S*, F0, U0                    Gamma
B_LEARNED_GAMMA      S*, F0, 선택된 U              Gamma
```

반드시 `S*(X)U`를 사용합니다. `S*(XU)`로 바꾸지 않습니다. Gamma=0은 정확히 S* 복귀이고, Gamma=I는 잠재 F0와 S*의 나머지 공간 예측 결합입니다.

B의 encoder/decoder는 같은 U를 공유합니다. LoRA나 G*를 추가하지 않습니다. B는 U가 변하면서 Q 예측도 달라지므로 기존 LEVEL의 Q 보존이라고 쓰지 않습니다.

**U 파라미터화는 다음으로 고정합니다.**

설치된 PyTorch에서 지원을 확인한 `orthogonal(..., orthogonal_map="householder", use_trivialization=True)`를 사용합니다. U0를 복사한 `[C,K]` 파라미터에 한 번 등록하고, 독립 RNG 문맥에서 생성한 raw parameter와 base buffer 전체를 두 LR에 동일하게 복사합니다. 초기 constrained U의 PCA 방향·부호 일치를 검사하고 forward 중 재정렬하지 않습니다. [공식 API](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.parametrizations.orthogonal.html)

```text
자료    등록 raw 변수   C×C base buffer   직교행렬 자유도
Robin        85              289                70
Jena        126              441               105
Hog         256            1,024               220
```

등록 변수 수와 실제 활성 자유도를 구분합니다. 재시작에는 raw/base를 모두 저장하고, 배포에는 constrained U만 고정 행렬로 내보냅니다. U 변화의 효과에는 부분공간 변화와 같은 공간 안의 회전·좌표 변화가 함께 포함됩니다.

**6. 구현과 검사 범위**

기존 runner의 guard를 해제하지 않고, v11 전용 진입점에서 자료와 checkpoint schema를 명시적으로 연결합니다. v5/v6/v7의 동명 `model.py/runtime.py`를 경로에 따라 우연히 import하지 않습니다.

[확인] 기존 고정 압축 분기는 TSFM forward를 `no_grad`로 감쌉니다. 이를 그대로 재사용하면 A/B 학습이 끊길 수 있으므로 다음을 구현 계약으로 둡니다.

- **A:** q* 계산만 `no_grad`; LoRA가 들어간 TSFM은 autograd 안에서 실행합니다. 부모 전체 forward와 새 TSFM을 중복 실행하지 않습니다.
- **B:** S* forward는 `no_grad`가 가능하지만, `X@U`, `S*@U`, F0 입력 경로와 최종 `@U.T`는 모두 graph 안에 둡니다.
- F0 weight 동결과 `eval()`은 유지하되, `predict`의 `no_grad` 경로 대신 확인된 모델 forward를 사용합니다.
- 기존 native scaling·역변환·0.5 분위수·첫 H48 crop을 보존합니다. 0.5 분위수를 평균 예측이라고 바꾸지 않습니다.
- A/FULL LoRA의 같은 자료·seed 초기 adapter tensor를 두 LR에 공유합니다. 표집 RNG는 모델 초기화와 분리합니다.
- B 저장에는 raw U뿐 아니라 parametrization buffer와 export U가 포함되어야 합니다. 기존 `named_parameters` 필터만으로 저장하지 않습니다.
- 병합은 별도 deploy copy에서 수행하고 반환 모델을 사용합니다. `safe_merge`만으로 출력 정합성 검사를 대신하지 않습니다.

검사 허용오차는 아래로 고정합니다.

```text
초기/병합/export/full-chunk parity  atol=1e-5, rtol=1e-4
같은 경로 checkpoint replay        atol=1e-6, rtol=0
UᵀU-I 최대 절대값                  1e-5
```

초기 함수, Q 불변, 의도한 변수의 gradient·실제 갱신, frozen state, row/quantile/mask 축, 저장·복원·재시작을 검사합니다. LoRA zero-B 초기화에서 일부 A gradient가 첫 step에 0인 것과 영구 단절을 구분합니다. 모든 scalar의 변화나 합성 loss 감소를 강제하지 않습니다.

실제 Bolt의 상수·근상수 입력 backward 유한성은 아직 실행 검증하지 않았습니다. 승인 후 확인하고 문제가 나오면 최소 재현과 영향을 기록합니다. 출력 의미를 바꾸는 수정은 기술 수리로 몰래 포함하지 않습니다.

**7. 적합·선택·Gamma 순서**

새 신경망 적합은 다음 공통 레시피를 사용합니다.

```text
seed                   92601, 92602
LR                     1e-4, 1e-3
optimizer              AdamW, weight_decay=0
clip                   norm 1
과제 batch             4
TRAIN 표집             epoch당 위상균형 512원점
최대 epoch             120
조기 종료              strict VAL nonimprovement 6회
scheduler              ReduceLROnPlateau
                       factor=.5, patience=2, relative threshold=1e-4
checkpoint             step0 포함 전체 VAL MSE 최소
epoch 동률             먼저 나온 상태
LR 선택                두 seed의 best VAL MSE 평균
LR 정확 동률           1e-4
```

같은 자료·seed·epoch의 TRAIN origin schedule을 family/LR 간 공유하고 hash를 남깁니다. 고정 TRAIN probe는 Robin/Hog 96원점, Jena 144원점으로 정해 initial/best/final을 비교합니다.

기본 실행 순서는 다음과 같습니다.

1. 승인 원문·manifest·원장·초기 상태와 schedule을 기록합니다.
2. 유한 합성 검사와 실제 모델의 무학습 정합성을 확인합니다.
3. Hog DIRECT 4fit을 수행하고 TRAIN/VAL로 S*를 선택합니다.
4. seed → LR → 자료 `Robin/Jena/Hog` → family `A/FULL_MSE/B_LEARNED_U` 순서로 36fit을 수행합니다.
5. 세 자료의 모든 LR/checkpoint 선택을 끝냅니다.
6. 선택된 fixed/learned U에 Gamma 총 12건을 적합합니다.
7. 선택 manifest를 고정한 뒤 예정된 세 자료 평가를 모두 수행합니다.
8. 같은 회차 비용 측정·진단·판정·검산·게시를 완료합니다.

한 자료의 TEST 결과를 먼저 보고 다른 자료를 취소하지 않습니다. 최대 epoch에서 개선 중이면 학습충분성 제한을 기록합니다. **자동 240epoch 연장은 포함하지 않습니다.**

Gamma는 관측 VAL 행만 사용합니다.

\[
D_{ij}=\left[F_0(XU)-S^*(X)U\right]_{ohj}U_{cj},
\quad b_i=Y_{ohc}-S^*(X)_{ohc},
\quad w_i=\frac{1}{C\,N_c}.
\]

\[
\min_{0\le\gamma\le1}
\left\|\sqrt w\,(D\gamma-b)\right\|_2^2.
\]

CPU FP64에서 설치된 SciPy의 `lsq_linear`를 사용합니다. `method="bvls"`, `lsq_solver="exact"`, `bounds=(0,1)`, `tol=1e-10`, `max_iter=100`으로 고정합니다. 추가 bias·penalty·horizon별 계수는 없습니다. [SciPy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)

solver 상태·경계·rank·직접 목적식·Gamma0/I 참조를 확인하고 FP32 배포 출력과 대조합니다. 실패 후 다시 solve하면 추가 attempt입니다.

S/LR/U/Gamma가 같은 VAL을 반복 사용한다는 한계를 명시합니다. Gamma 이후 U/LR를 다시 고르지 않고, 보정 전후를 모두 보고합니다. 실제 solver 결과가 **정확히 0인 방향만** 생략할 수 있으며 작은 양수를 임의로 0으로 만들지 않습니다.

**8. 평가와 해석**

주 지표는 관측 미래의 채널 동일 가중 MSE입니다. MAE, signed mean error, seed별·기간별·채널별 값을 함께 저장합니다.

전체 점수는 채널별로 A/B 기간의 오차합과 관측수를 합친 후 채널평균하고, 마지막에 두 seed 손실을 평균합니다. 예측 ensemble이나 세 자료의 원 MSE 평균을 만들지 않습니다.

주 비교는 다음과 같이 고정합니다.

```text
외부 선택 가치     A/B ↔ F0 / FULL_LORA_MSE / LORA_NATIVE / DIRECT
A 내부 효과        A ↔ BASE_LEVEL
B 공간 학습        B_LEARNED_U ↔ B_FIXED_U       (둘 다 Gamma=I)
단순 혼합 효과     각 B의 Gamma 적합 전 ↔ 후
전체 시스템        A ↔ B
역사적 참고        같은 계약의 v8/v9/v10
```

A의 Q 제거 결과는 동일 학습 checkpoint의 main-only fit0 참조로 보고할 수 있습니다. Q 없이 별도로 학습한 압축 LoRA의 최적 결과라고 부르지 않습니다.

기존 계약에 따라 Robin/Hog 7원점, Jena 42원점의 시간 블록을 기간 안에서 재표집합니다. 2,000회, 고정 seed `9262026`을 사용하고 채널·방법·대응 seed를 함께 움직입니다. 이 구간은 선택된 두 seed에 조건부이며 부모·LR·Gamma 선택 불확실성 전체를 포함하지 않습니다.

공간 분석은 모든 비교군이 공유하는 완전관측 target 벡터에서만 수행합니다.

- A: 공통 U0의 P/Q 오류.
- B: 공통 U0 기준과 선택 U 기준을 별도로 제시.
- coverage를 보고하고, 결측 target을 0으로 채워 투영하지 않음.
- U 거리·principal angle은 해석 단서이며 새 난이도 지표나 라우팅 규칙으로 사용하지 않음.

세 자료는 모두 노출된 개발 평가입니다. TEST 후 구조·LR·계수·자료·기간을 다시 선택하지 않습니다.

**9. 비용 측정의 실제 구성과 반복 수**

비용은 결과의 유불리와 관계없이 예정된 선택 모델과 기준선을 같은 세션에서 측정합니다. 비용 입력은 **기존 VAL 원점의 시간순 첫 24개**로 고정합니다.

```text
공통 환경          FP32, TF32 off, CPU 4 threads
배치               B1, B4
예열               구성/shape별 전체 wrapper 호출 10회
본 측정            전체 24원점 × 10passes
순서               균형 있게 바꾼 3블록
drift 확인         자료·블록별 F0 B1/B4 앞뒤 sentinel
시간 안            표준화 CPU 입력 → 전송 → online forward
                   → 전체 원채널 CPU 출력·조합
시간 밖            model load, disk, 채점, hash, 원장 기록
```

모델 load 등도 전체 GPU job 점유 예산에서는 빠지지 않습니다. 동기화·warmup·peak reset·allocator 정리를 동일하게 적용합니다. [PyTorch benchmark 안내](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

자료별 instance는 최대 19개입니다.

```text
압축 12개
  BASE_LEVEL 2 + A 2
  B_FIXED_U 2 + B_LEARNED_U 2
  B_FIXED_GAMMA 2 + B_LEARNED_GAMMA 2

미압축 5개
  F0 1 + FULL_LORA_MSE 2 + LORA_NATIVE 2

DIRECT 2개
```

미압축 chunk grid는 다음과 같습니다.

```text
B1: full C, K
B4: full 4C, 4K, K

Robin B4  68 / 20 / 5
Jena  B4  84 / 24 / 6
Hog   B4 128 / 32 / 8
```

압축 모델에도 B1 full K, B4 full 4K 및 K행 chunk를 제공합니다. Gamma의 정확한 0 때문에 활성 방향이 줄면 실제 활성 K와 대응 실행을 기록합니다. full/chunk 출력 정합성을 먼저 확인하며 시간축이나 미래 H를 나누지 않습니다.

실제 최대 측정량은 다음과 같습니다.

```text
GPU 구성 / 자료
  미압축  5 instances × 5 batch/chunk 구성 = 25
  압축   12 instances × 3 구성             = 36
  DIRECT  2 instances × 2 batch            =  4
  합계                                      65

세 자료 × 3블록 기본 측정 행               585
F0 앞뒤 sentinel                           36
GPU 측정 행 합계                          621
DIRECT CPU 측정 행                         36

GPU 본 측정 원점 처리 수
621 × 10passes × 24원점 = 149,040

예열 포함 최대 TSFM 호출 수
Robin 62,220 + Jena 62,220 + Hog 64,320 = 188,760
```

이 수는 exact-function 중복 제거나 Gamma0 fallback 이전의 상한입니다. 상태·함수·실행이 실제로 같은 경우만 합치며 TEST 성능으로 구성을 제외하지 않습니다.

[추정] 기존 v7/v9/v10 비용 job의 실제 실행률을 적용하면 10passes 안은 약 **45–100분** 규모입니다. 이는 새 실측이 아니며 완료시간 보장이 아닙니다. 20passes는 느린 기존 실행률에서 측정 자체가 약 2시간을 넘길 가능성이 있어, **grid를 보존하는 대신 반복 수를 사전에 줄이는 안**을 선택했습니다.

보고 항목은 B1 지연, B4 처리량, peak allocated/reserved, resident allocation, 가능한 process memory, 전체 배포 parameter/bytes, 실제 학습 peak/time입니다. 등록 학습 변수·활성 gradient·실제 갱신·과거 부모를 포함한 누적 적합량도 구분합니다.

A는 q*와 병합 TSFM 하나, B는 S*·export U·Gamma·필요한 TSFM만 배포합니다. Gamma0의 S fallback은 F0를 실제로 로드하지 않는 별도 경로로 표시합니다. DIRECT CPU 비용은 별도 장치의 실용 비교입니다.

불리한 블록과 drift를 보존합니다. 필요하면 총예산 안에서 제한된 진단 1회, 최대 GPU 10분만 사용합니다. 원인이 미확정이어도 유효한 정확도 결과는 완료하고 비용 우열만 제한합니다.

**10. 예산과 수리 권한**

```text
실자료 적합                               기본 attempt
A              3자료 × 2LR × 2seed             12
B_LEARNED_U    3자료 × 2LR × 2seed             12
FULL_LORA_MSE  3자료 × 2LR × 2seed             12
Hog DIRECT             2LR × 2seed              4
신경망 기본 합계                                40

Gamma          3자료 × 2U종류 × 2seed          12
기본 실자료 합계                                52

기술복구·실패·재시도 공용 예비                    8
전체 상한                                      60
```

예비 8회는 신경망/계수 적합이 공유합니다. 부모/S 복구·기술실패·재시도에만 사용하며 새로운 LR·rank·K·후보 탐색에 전용하지 않습니다. 재개해 추가 update하거나 다시 solve하면 새 attempt입니다.

```text
GPU                     1장, 동시 학습 1개
GPU job 점유            최대 28,800초
CPU 계수 적합·분석      최대 7,200초
CPU 검사                누적 최대 1,800초
합성 optimizer          최대 6세션, 세션 전체 10update 이하
신규 저장               최대 10GiB
raw/백본 다운로드       0
```

GPU 운영 배분은 복원·검사 0.5h, 학습·VAL 4.5h, 평가 0.5h, 비용 2h, 기술여유 0.5h입니다. 부분 배분은 조정할 수 있지만 **총 8시간은 자동 증액하지 않습니다.**

4.5시간/40fit은 평균 약 405초의 배분입니다. 과거 native LoRA fit은 자료에 따라 약 95–448초였지만 새 A/B/FULL의 소요시간은 아직 모릅니다. 최초 정상 fit들의 실제 epoch 처리율로 남은 전체 작업량을 확인하고, 부족하면 자료를 조용히 빼지 않고 범위·예산 변경을 보고합니다.

모든 job은 시작 전에 원장에 예약합니다. 부모 job과 내부 fit 시간을 이중 합산하지 않습니다. 실자료 optimizer smoke는 예정 fit의 첫 step으로 포함하며 무료 검사로 처리하지 않습니다.

다른 프로젝트 프로세스를 종료하지 않습니다. 자원 충돌 대기는 회당 30분·누적 60분까지로 하고, 대기와 GPU 점유를 구분합니다. 새 서버·유료 서비스·전역환경·driver/power 변경은 범위 밖입니다.

승인 범위의 API·경로·직렬화·복원·집계 오류는 원본 실패와 diff를 남기고 수리합니다. TEST 이후 재현 가능한 채점·복원 버그는 영향받은 비교 전체를 함께 정정하되, 모델 선택이나 수식까지 바뀌어야 하면 별도 변경안으로 보고합니다.

**11. 결과별 결정과 완료 산출물**

- **A가 부모를 개선하고 외부 대안 앞에 선택 영역을 만듦:** 해당 정확도·메모리·시간 조건에서 A 유지.
- **A가 부모만 개선하고 외부 대안에 계속 밀림:** 내부 적응 효과는 인정하되 실용 기여는 미확보.
- **B 학습 U가 fixed U보다 유리하고 TSFM의 nonzero 기여가 남음:** B 유지 후보. 부분공간 변화와 내부 회전의 분리는 후속 과제.
- **fixed U 또는 fixed U+Gamma로 충분함:** U 학습을 삭제하거나 보조 결과로 낮춤.
- **Gamma가 대부분 0:** NLinear 복귀로 해석. TSFM 성공으로 표시하지 않음.
- **자료별로 서로 다름:** 적용 조건을 보고하되 TEST 승자를 합친 선택기를 만들지 않음.
- **둘 다 정상 기준선에 지배됨:** 현재 두 설계를 보류/기각. 남은 예산으로 새 모듈을 시작하지 않음.
- **일부 기술 차단:** 유효한 다른 결과는 보존하되 A/B 전체 비교 완료라고 선언하지 않음.

완료는 논문 주제 확보가 아니라 **유지·단순화·보류·기각 중 무엇을 선택할지 근거 있게 결정하는 것**입니다.

승인 후 산출물은 새 폴더에 다음 역할로 만듭니다.

- `REQUEST_PLAN.txt`, 승인된 실제 계획 `PLAN.md`, 사용자 승인 원문 `APPROVAL.txt`.
- `STATUS.md`, 공통 원장, reuse/data/initial/schedule/protocol manifest.
- 실행별 result/curve/checkpoint receipt, 선택 및 Gamma solver 기록.
- 전체·기간·seed·채널 평가 JSON/CSV, 비용 원반복과 요약.
- 필요한 비교 그림, claim/evidence 표, `METHOD_UPDATE.md`, `TOPIC_DECISION.md`.
- 최종 수치·복원·출처 검산과 publication receipt.

원고 전체 재작성이나 논문 제출은 포함하지 않습니다. raw·weight·대용량 prediction은 로컬 캐시에 두고 공개 결과는 hash로 연결합니다.

관련 신규 파일과 결과 색인만 개별 검증·스테이징하여 일반 commit 후 `origin/main`에 push하고 live SHA를 확인합니다. 기존 v1–v10, 실패·봉인·위반 기록과 무관한 dirty 파일은 보존합니다. 재개 시 PLAN/STATUS·원장·실제 PID·원격 상태부터 확인합니다.

**실행 전 최종 검토 요약**

```text
부모·S 재사용
  Hog    C5/runs/v5_hog_level_res_{seed}/best.pt
         LEVEL 2개 재사용, 직접 NLinear 신규 4fit
  Robin  C6/runs/v6_robin_level_res_{seed}/best.pt
         C7/runs/v7_robin_direct_nlinear_lr0.001_{seed}/best.pt
         모두 재사용
  Jena   C7/runs/v7_jena_level_res_{seed}/best.pt
         C7/runs/v7_jena_direct_nlinear_lr0.001_{seed}/best.pt
         모두 재사용
  부모·S 실제 checkpoint와 저장 예측 해시 확인 완료

A
  Fφ(XU0)U0ᵀ + q*(X)
  새 LoRA만 학습, U0/G*/원래 backbone 고정
  직접 대조: BASE_LEVEL, 동일 MSE의 미압축 LoRA

B
  S*(X) + [F0(XU) - S*(X)U] Γ Uᵀ
  S*/F0 고정
  먼저 Γ=I에서 U만 학습 → 선택 후 fixed/learned 각각 Γ 적합
  직접 대조: fixed U, Γ 보정 전후, 직접 NLinear

적합 예산
  A12 + B12 + FULL_MSE12 + Hog DIRECT4 = 신경망40
  Gamma12 + 기술복구8
  기본52 / 최대60attempt
  GPU8h, CPU분석2h, CPU검사0.5h, 신규10GiB

선택
  TRAIN 적합 → 전체 VAL의 step0 포함 best checkpoint
  두 seed 평균으로 LR 선택, 정확 동률 1e-4
  세 자료 신경망 선택 완료 → Gamma12 → 전체 평가
  자동240epoch 연장·TEST 재선택 없음

비용
  자료별 최대19instances
  GPU195구성 × 3블록 + sentinel36 = 621측정 행
  DIRECT CPU36행
  공통 VAL24, 예열10, 10passes, FP32/TF32off/CPU4threads
  미압축 B1 {C,K}, B4 {4C,4K,K}
  압축 B1 {K}, B4 {4K,K}
  과거 timing과 직접 비율 비교하지 않음

완료·중단
  외부 대안 앞의 선택 가치와 내부 변경 효과를 분리해 결정
  fixed/Gamma/NLinear로 충분하면 단순화
  지배되면 보류·기각, 기술무효는 별도 표시
  예정 해석 완료 또는 필수 차단·총상한 도달 시 종료
  관련 결과 검산 → 일반 commit → origin/main push 확인
```

**비용 반복을 일괄 10회로 조정한 내용을 포함해, 이 v11 전체 실행계획을 승인하시겠습니까?**