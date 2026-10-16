# INFORMS 모델 구현

원자료·실제 foundation·GPU를 사용하지 않고 CPU 합성 검사만 수행했다. 모델 정의는 `networks.py`, 검사는 `test_networks.py`, 공식 원문 pin은 `LITERATURE_MODEL_PINS.json`이다. 외부 패키지를 설치하거나 기존 캠페인 파일을 수정하지 않았다.

## 고정 API

- `ForecastAdapter(front_kind,length,horizon,channels,latent_k,pca_center,basis,encoder_kind='pointwise',k_grid=None)`; `forward(x,foundation_callable,latent_k=None)`는 `[B,L,C]→[B,H,C]`다. Foundation은 `[B,L,K]→[B,H,K]` tensor callable이며 root가 정규화·실제 모델·동결·장치를 담당한다.
- `.E`는 `[C,Kmax]`, `.D`는 `[7,Kmax,C]`, `.front`는 독립 Module이다. `.encode(x,latent_k=None)`와 `.decode(x,z,future,latent_k=None)`를 제공한다. center는 buffer이며 PCA basis의 앞 Kmax 열을 초기값으로 쓴다. D의 lag0만 pinv(E), 나머지는0이다.
- 요청한 K의 E column과 **D[:, :K, :]**를 사용한다. Lag-major feature와 D를 일치시킨 후 flatten한다. 실제로 K개 stream만 callable에 전달하며 Kmax를 호출한 뒤 자르지 않는다. k_grid가 있으면 등록된 K만 허용한다.
- D의 lag는 `(0,1,2,4,8,16,32)`이고 원 C의 `x_last`를 anchor로 더한다. L>32, H>0을 검사한다. L512/H48, L512/H12, L364/H28을 검사했다.
- `DirectG(length,horizon,hidden=32).forward(x)`는 channel-shared `L→32→H` innovation과 current anchor다. `ConditionalHead(length,horizon,hidden=32).forward(x,g,b,control=False)` 및 `.forecastforward(...)`는 `g + head(past-current,g-current,b-g)`다. 두 parent tensor를 detach하며 `control=True`는 마지막 입력을 실제 최근 H개 past-current로 바꾼다. 두 대조의 입력 폭·모든 파라미터는 같다. 출력층0 초기화는 CP0 g parity만 보장한다.
- `DirectMixer(kind,length,horizon,channels,hidden=32)`는 별도 직접예측 변형이다. 앞단의 original L 출력과 미래 H head를 구별한다.

## 시간 처리와 메모리

기존 검증된 TemporalFront/DirectMixer의 작은 원리를 새 파일로 복사했다. B0는 항등, B2는 시간/채널 교대 residual MLP, B4는 평균 pooling `(1,2,4)`·대칭 이동평균 residual/trend·fine→coarse residual·coarse→fine trend와 original-grid fusion이다. B1/B3 정의도 보존하지만 등록 여부는 root registry가 정한다. 논문 전체 재현이 아니다. B2/B4는 주어진 전체 과거 창을 사용하며 prefix causal이라고 부르지 않는다.

`encoder_kind='causal7'`는 `(0,1,2,4,8,16,32)`, `'bidirectional7'`은 `(-16,-8,-4,0,4,8,16)`이다. 지연 대조는 B0 앞단만 허용한다. W0는 E, 나머지 여섯 행렬은 `.lag_weights[6,C,Kmax]`의0 초기값이다. 각 shift는 현재 표준화 창 밖을0으로 채운 후 center를 빼며 추가 L+32 과거를 읽지 않는다. 최초32행을 제외한 causal prefix 검사를 통과했다. 양방향 대조는 미래 정답을 보지 않지만 창 안의 뒤쪽 과거 행을 사용한다.

지연 입력 일곱 개를 `[B,L,7C]`로 한꺼번에 쌓지 않고 합산한다. B2/B4 feature MLP hidden은 C와 무관하게32다. C321/L364/K81 forward 형상을 CPU에서 검사했다. 실제 foundation의 activation·GPU memory는 이 검사 범위가 아니다.

## 공식 AdaPTS 최소 포트

공식 `abenechehab/AdaPTS` commit `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`의 `LinearAutoEncoder`, `MultichannelProjector.transform_torch/inverse_transform_torch`, `ADAPTS.adapter_supervised_fine_tuning` 사용 부분을 읽었다. MIT 고지와 저작권을 코드 docstring에 보존했다.

`AdaPTSLinearPort(length,horizon,channels,latent_k)`는 bias를 포함한 affine C→K encoder와 K→C decoder를 사용한다. `decoder(F(encoder(x)))`이고 current anchor·lag7 D가 없으므로 B0를 이름만 바꾼 경로가 아니다. patch size1, RevIN=False, deterministic 경로다. 원 공식 학습기의 무작위 validation 분할·스케일러·재구성 pretraining·Adam loop는 가져오지 않는다. Root의 새 시간 계약·관측 macro loss·Bolt callable로 적합하므로 AdaPTS 논문 전체 실험/확률 추론 재현이 아니다. 이식부의 직접 affine 수치 대응을 CPU에서 검사했다.

## LoRA API와 수명

`LoRAQvAdapter(base_model,rank=8,alpha=16.0,dropout=0.0)`는 leaf 이름 q/v/q_proj/v_proj인 실제 nn.Linear에 부착한다. Base는 root가 **장치·dtype 설정 후** 넘기고 직접 호출한다. 이 객체는 base를 child module로 등록하지 않아서 `.parameters()`와 `.adapter_parameters()`는 A/B만 반환한다. Base gradient flag는 부착 중 동결하고 close 시 원 flag로 돌린다. 기존 base gradient가 있으면 부착을 거부한다.

`.adapter_state_dict()` / `.load_adapter_state_dict(state)`는 adapter-only tensor를 다룬다. `.merge()`는 delta를 weight에 반영하고 hook을 제거한다. `.unmerge()`는 저장한 원 target weight를 copy하여 FP32 뺄셈 오차 없이 복구한 뒤 hook을 재부착한다. `.close()`는 hook 제거·원 weight/gradient flag 복구·전체 base tensor hash 검증을 하며 반복 호출해도 안전하다. Context manager도 제공한다. Base를 부착 뒤 다른 장치로 옮기지 않는다. Root는 각 경로를 finally/context로 닫아야 한다.

CPU toy q/v에서 adapter-only gradient, base gradient 없음, 병합 전후 수치 parity, merge 시 hook0, unmerge/close byte-exact base 복구와 예외 경로 cleanup을 검사했다. 실제 Bolt q/v 선택과 병합 parity는 root의 실제 모델 단계에서 확인해야 한다.

## 검증

`python -B research/informs_broad_validation_20261015_v1/test_networks.py`를 지정된 기존 venv interpreter로 실행했다. 미구현 assertion의 RED를 먼저 확인한 후 10 tests와 9 shape subtests를 GREEN으로 만들었다. 원 RED와 최초 GREEN 로그는 비공개 `.cache/informs_broad_validation_20261015_v1/network_tests_red.log`, `network_tests_green_01.log`에 있다. 합성 optimizer step은 gradient 연결 검사이며 실제 학습 결과로 세지 않는다. 최종 출력층은 첫 backward부터, 그 앞 layer는 출력층 한 번 갱신 후부터 gradient가 흐르는 구조다.
