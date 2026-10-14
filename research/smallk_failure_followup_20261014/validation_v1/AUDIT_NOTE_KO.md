# D0 배열 독립 감사 노트

[확인] bridge actual 입력과 D0 saved diagnostics 산출물을 CPU에서 다시 읽어 원 RAW/MAIN/MAIN_NATIVE NPZ, bridge NPZ, 저장 CSV/NPZ의 수치를 독립 재계산했다.
[확인] 새 GPU 작업, checkpoint torch 재로딩, 모델 추론, TEST split 접근은 하지 않았다.

- 최종 상태: `COMPLETE`; resource terminal: `None`
- audit body wall seconds: `14.530835500452667` (FollowupLimit RT receipt와 별도 기록; 보충 CPU로 중복 가산하지 않음)
- 허용 오차: `1e-12` float64 absolute reconstruction/table tolerance

## robin

- [확인] source SHA/ASSET ref/mean/std/target/mask/origin/raw_forecast/raw_last 조건: `PASS`
- [확인] CSV 재계산 최대 차이: `0.0`; rank NPZ 최대 차이: `0.0`
- [확인] reconstruction 최대 오차 saved/recomputed: `2.4147350785597155e-15` / `2.4147350785597155e-15`
- [확인] channel/horizon/time/tail gap 재구성 최대 오차: `5.551115123125783e-17`, `3.3306690738754696e-16`, `2.42861286636753e-15`
- [확인] rank projection basis는 FIT RAW forecast delta와 last만으로 재계산했을 때 저장 basis/예측과 일치했다. Y target은 basis fit에 사용하지 않았다.
- [확인] MAIN_NATIVE E/D 동일성은 root D0의 torch-equal metadata 네 건이 모두 true임을 확인했다. 이 감사에서는 checkpoint tensor를 다시 열지 않았다.

대표 tail gap raw values:

- val MAIN_NATIVE seed 92602 Robin_office_Zelma q=0.99 same_split_raw body: gap=0.16916292096575553, candidate=0.20202496567133818, raw=0.03286204470558264, paired=True
- val MAIN_NATIVE seed 92602 Robin_office_Zelma q=0.99 same_split_own body: gap=0.16783321881056412, candidate=0.20069526351614675, raw=0.03286204470558264, paired=False
- val MAIN_NATIVE seed 92601 Robin_office_Zelma q=0.99 same_split_raw body: gap=0.16704229325824516, candidate=0.1999043379638278, raw=0.03286204470558264, paired=True

대표 signed-bias shift raw values:

- val MAIN_NATIVE seed 92601: bias=-0.10902103669150551, raw_bias=-0.05592930327906009, shift=-0.05309173341244542, mse_gap=0.28847172404710836
- val MAIN_NATIVE seed 92602: bias=-0.10157158609751747, raw_bias=-0.05592930327906009, shift=-0.045642282818457384, mse_gap=0.28120768304197535
- tune MAIN_NATIVE seed 92601: bias=0.01161723600513319, raw_bias=-0.02168083574198903, shift=0.033298071747122224, mse_gap=0.05110802598107046

해석 한계: 이 수치는 저장 배열의 정합성과 오차 분해 검증이다. split은 개발/노출 평가 역할을 보존하며 독립 holdout, causal claim, 새 모델 성능 주장을 만들지 않는다.

## beijing_aotizhongxin

- [확인] source SHA/ASSET ref/mean/std/target/mask/origin/raw_forecast/raw_last 조건: `PASS`
- [확인] CSV 재계산 최대 차이: `0.0`; rank NPZ 최대 차이: `0.0`
- [확인] reconstruction 최대 오차 saved/recomputed: `2.6645352591003757e-15` / `2.6645352591003757e-15`
- [확인] channel/horizon/time/tail gap 재구성 최대 오차: `1.6653345369377348e-16`, `1.2351231148954867e-15`, `2.6645352591003757e-15`
- [확인] rank projection basis는 FIT RAW forecast delta와 last만으로 재계산했을 때 저장 basis/예측과 일치했다. Y target은 basis fit에 사용하지 않았다.
- [확인] MAIN_NATIVE E/D 동일성은 root D0의 torch-equal metadata 네 건이 모두 true임을 확인했다. 이 감사에서는 checkpoint tensor를 다시 열지 않았다.

대표 tail gap raw values:

- tune MAIN_NATIVE seed 92601 WSPM q=0.9 same_split_raw body: gap=0.05116230715152409, candidate=0.08345642868696049, raw=0.0322941215354364, paired=True
- tune MAIN_NATIVE seed 92601 WSPM q=0.9 fit_raw body: gap=0.05088909453179011, candidate=0.075773743703465, raw=0.024884649171674888, paired=True
- tune MAIN_NATIVE seed 92601 WSPM q=0.95 fit_raw body: gap=0.048953383223768646, candidate=0.08782579463333316, raw=0.03887241140956452, paired=True

대표 signed-bias shift raw values:

- eval MAIN_NATIVE seed 92602: bias=0.03298612285895383, raw_bias=-0.06163694777950617, shift=0.09462307063846001, mse_gap=0.18251481861162033
- eval MAIN seed 92602: bias=0.009114486914093391, raw_bias=-0.06163694777950617, shift=0.07075143469359957, mse_gap=0.07299442507192533
- eval MAIN_NATIVE seed 92601: bias=-0.006288847485388394, raw_bias=-0.06163694777950617, shift=0.055348100294117776, mse_gap=0.16869093708966332

해석 한계: 이 수치는 저장 배열의 정합성과 오차 분해 검증이다. split은 개발/노출 평가 역할을 보존하며 독립 holdout, causal claim, 새 모델 성능 주장을 만들지 않는다.


## Pre-data execution attempts

[확인] actual D0 audit 전에 세 번의 pre-data 실행 실패가 있었다. 첫 번째는 Windows 명령줄 길이 제한으로 Python body가 시작되지 않았고, 두 번째는 기존 alidation_v1 폴더 감지로 FollowupLimit 생성 전에 중단됐으며, 세 번째는 
esearch import 경로 오류로 FollowupLimit 생성 전에 중단됐다. 세 경우 모두 과학 배열/CSV/JSON 읽기 전 실패이며 데이터 검증 실패로 해석하지 않는다.
