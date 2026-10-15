# 후속 실행 전에 고정한 세부 조건

R2는 sources_R2에 동결한 코드로 실행 중이다. 다음 세부화는 R3/R4/R5의 실제 학습·후보 점수를 보기 전에 source에 고정하며, C는 아직 채점하지 않았다. 최초 전체 계획이 모든 하이퍼파라미터를 사전등록한 확증 실험이었다고 쓰지 않는다.

- GPU queue에서 R2 학습이 먼저 시작되었으므로 실제 무학습 R1은 그 다음에 실행한다. R1 설정·좌표·SPRINT port는 R2 중 준비·합성 검증했다. R1 결과로 R2 가족을 취소하지 않는다. 계획 순서와 이 실행 순서를 구분한다.
- R3는 B에서 선정한 각 가족의 작은-K 앞단을 사용하고 G의 seed92601 하나를 공유한다. 모든 경로의 G 초기값도 같게 고정한다. 첫 fold5경로×2자료=10fit, 두 번째 fold의 전체 upstream 재적합 뒤 같은5경로×2자료=10fit, 합계20fit이다. G의 두 seed 확증을 했다고 쓰지 않는다.
- R4는 native shared3-feature temporal head에 alpha0/.01 두 개×2가족×2자료=8fit, 직접 작은 Mixer2종×2자료×1seed=4fit, 기존 rank8 q/v LoRA×2자료=2fit, Raw-C block ridge alpha.01×2자료=2fit, 합계16개의 결정적 적합/학습 구성이다. ZERO는 학습 없는 후보다. native_ridge.py의 범용 ALPHAS 메뉴 전체를 실행했다고 세지 않는다. 원C ridge의33특징 중 current-current 열은0이며 실효 rank는 최대32다.
- LoRA는 저장소의 기존 구성 r8/alpha16/dropout0/q,v와 기존 학습률.0001을 사용한다. 그 외 neural route의 기본 학습률은.001이다. 추가 LoRA 탐색을 하지 않는다. 같은 update 상한·관측 미래 손실·B 선택을 유지하며 lr까지 같은 비교라고 부르지 않는다.
- secondary 50/20/30의 B0/B2/B4와 G는 사전 고정 seed92601로 새로 학습한다. primary B[60%,80%)는 secondary C[70%,100%)와 겹치므로 primary B로 고른 seed를 secondary에 가져오지 않는다. 그 가능성은 source review에서 실제 secondary 실행 전에 발견·수정했다.
- B3는 B4 decomposition=False와 같은 parameter/state 구조·초기값·schedule의 분해 제거 재학습 대조다. 같은 dataset/r/seed와 matched endpoint의 기존 R2 B3를 재사용한다. 가지 제거는 새로 학습하며 용량 변화가 함께 있음을 밝힌다.
- SPRINT의 유효 lookback Raw는 동일하게 채워진512 과거를504/432로 crop한 입력으로 측정한다. crop 전에 관측된 값의 결측 carry까지 같게 맞추며, 원자료를 짧은 창만으로 새로 채우는 다른 대조와 섞지 않는다.
- general 비용에서는 공통 F1개가 resident이고, native/direct/G-only exclusive 비용은 실제 F 로드0으로 잰다. LoRA hook은 정상·예외 경로 모두 해제한다. 별도 프로세스는 한 경로씩 실행하며 학습 프로세스 경쟁 여부를 확인한다. OS 전체 GPU 독점 예약을 했다는 뜻은 아니다.
- 선정 checkpoint/native weights·과학 source·fold 통계의 SHA와 선택→비용→C 채점의 hash/시간 연결을 검사한다. 이 검사는 성능 개선이나 미노출 독립 일반화를 뜻하지 않는다.
