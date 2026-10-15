# Robin L7에서 RECON D를 고정한 E 적응 대조

사전 계획이며 새 실제 실행은0이다. 직전 RECON-init2조건의 학습은 끝났고 독립 감사·새 공간 진단·보고가 진행 중이다. SOURCE/합성 준비는 병행하되 이 묶음의 검산·보고 이후 새 실제 실행을 시작한다.

[확인] RECON D 초기의 E/D 공동 적응은 FORECAST-LS 초기보다 두 seed에서 TUNE·노출 DEV가 낮았지만, DEV .534758/.554825는 학습 전 RECON .488093과 Raw .340402보다 높았다. TUNE는 .102808에서 .083936/.083137로 줄었다. [판단] 학습량을 더 늘리기 전에 **D를 고정한 E 적응과 E/D 공동 적응을 구분**한다. 초기화가 다른 비교에서 개선됐다는 사실만으로 표현이 좋아졌다고 판단하지 않는다.

## 고정 비교와 새 실행

Robin C17/K4·L7·두 minibatch-order seed92601/92602의 E-only2fit을 새로 수행한다. 각600update, 총1,200update이며 직전 같은 RECON 초기 E/D 공동 적응2fit을 재사용한다. 기존 RECON CP0·FORECAST-LS 초기·Raw도 참고한다. 초기 PCA E/center와 기존 FIT RECON D32·과거 입력·L512/H48·FIT256/TUNE64/기노출 VAL30·origin/scaler/mask·current-C anchor·lag0/1/2/4/8/16/32를 유지한다. D32와 center는 고정하고 E만 학습한다. F 파라미터는 동결하되 F 입력 gradient는 유지한다.

원 train_condition의 requires_grad 파라미터 선택·ID검사·None gradient/norm 처리·CP 복원 흐름을 읽기 전용으로 확인했다. 새 wrapper에서 D.requires_grad_(False)를 명시하고 같은 함수를 재사용한다. 실행된 원 SOURCE를 수정하거나 stopped optimizer state를 재개하지 않는다. E의 actual-C 손실 target은 FIT 미래 Y의 정상 감독이며 기존 D 자체는 FIT 과거 RECON에서 적합한 채 고정된다. E와 D의 감독 이력을 따로 명시한다.

seed별600×4 순서, 고정 FIT 관측수 가중 MSE, AdamW lr.001/WD0/clip1, CP0/100/200/400/600·전체 TUNE 최소/이른 tie를 유지한다. clipping은 실제 학습 파라미터 E에만 적용하므로 공동 적응과 E의 유효 scale이 달라질 수 있다. 이 대조는 학습 절차의 비교이며 단일 원인의 순수 효과나 동일 계산량이라고 주장하지 않는다.

## 검증과 후속 진단

실제 CP0가 직전 RECON-init과 같은 상태·예측인지 확인한다. CP0 검사는 optimizer0이다. fresh 모델에서 D.grad가 None인 채 freeze하고 D·center가 optimizer에서 제외되는 경로를 확인한다. 모든10CP의 D bytes가 초기값과 정확히 같고, 선정 FIT/DEV가 그 선정 상태를 사용하는지 검산한다. per-step D.grad/훈련 가능 flag 검사와 저장 시점의 D bytes 확인을 구분하며, CP만으로 매 update bytes를 독립 재생했다고 주장하지 않는다. 기존 F dual hash·행동 계약·입력 원 바인딩을 유지한다. 두 성공/실패 잠금·registry 뒤 DEV를 연다. 한 seed 실패로 다른 경로를 생략하지 않는다.

선정 상태의 TUNE/DEV4개 진단은 같은 complete-C mask·retained1e-15 정의로 계산한다. 고정 D라 joint A는 같지만 E가 만든 known-past offset은 달라질 수 있음을 구분한다. minimum·actual F·진단 E(Y)·latent persistence와 원 C persistence를 보존한다. 미래 oracle는 원인 진단에만 사용하고 배포·선정 입력으로 넣지 않는다. 새 E-only 성능과 공간이 모두 나쁘면 같은 조건의 무조건 장기 학습으로 확대하지 않는다. Raw 개선을 확인한 경우에만 그 상태의 실제 동일 cohort 비용을 다음 단계로 측정하며, 현재 새 deployment timing은 미계획/미측정이다.

원2fit 재사용·새2fit·모든 실패·사용 자원·미완료 항목을 구분해 한국어 MD·원수치·그림으로 보고한다. 이미 노출된 DEV의 실패를 보고 고른 적응적 대조이며 독립 일반화나 새로운 방법의 보편적 성능 근거가 아니다. K streams+K 과거 buffer+현재 C anchor여서 strict K-only가 아니다. 최신 계속 실행 지시 안의 같은 목적에 대한 작업 판단이며 개별 새 승인을 받았다고 쓰지 않는다. TEST·설치/다운로드·큰 백본 파라미터 학습·기존 작업 종료/중단 학습 재개·자동 commit/push·논문/PPT는 포함하지 않는다.
