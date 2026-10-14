# Joint E/D 목적 비교의 독립 CPU 검산

상태: PASS_WITHOUT_F_OR_PER_UPDATE_FORWARD_REPLAY; 검사 11216개, 실패 0개.

저장60checkpoint·FIT/TUNE 예측과36selected 출력을 재채점하고, 원 CP/동일 order·FIT teacher mask·실제 Y TUNE 선택·u600 E/D 변화·selected hash를 검산했다.
600curve 기록은 coverage·유한성·목적 손실 조합을 확인했다. 매 update 출력/중간 CP가 없어 그 값 전체를 독립 재-forward했다고 쓰지 않는다. F도 새로 실행하지 않았다.
RAW 목적도 Y로 학습·선택된 초기 CP와 실제 Y TUNE 선택을 사용한다. 무감독/label-free 또는 독립 holdout 연구가 아니다.

FIT 관측/비관측 finite 여부·비관측 NaN 수는 실제 저장 teacher contract로 검산했다. masked-NaN backward 보정은 코드 견고성 확보이며, 실제 NaN이 없는 조건의 예측 부진이나 local cap 종료 원인으로 주장하지 않는다. GPU process wall에는 CPU 자원 검사·입출력도 포함되며 GPU kernel 학습 시간과 같지 않다.

```text
dataset              seed    objective  select     MSE          MAE
robin                92601   Y          100        0.56967172   0.44634479
robin                92601   RAW        200        0.56170226   0.44436612
robin                92601   MIXED      200        0.56585281   0.44453446
robin                92602   Y          0          0.57194670   0.44700767
robin                92602   RAW        100        0.56775779   0.44557836
robin                92602   MIXED      100        0.57009957   0.44608376
beijing_aotizhongxin 92601   Y          0          0.66134456   0.46647248
beijing_aotizhongxin 92601   RAW        0          0.66134456   0.46647248
beijing_aotizhongxin 92601   MIXED      400        0.66513163   0.46802450
beijing_aotizhongxin 92602   Y          600        0.67562897   0.47615686
beijing_aotizhongxin 92602   RAW        200        0.65374830   0.45917311
beijing_aotizhongxin 92602   MIXED      200        0.65198690   0.46425717
```

select0은 원 MAIN 유지다. 실제600학습 실행과 새 선택 모델의 성능 개선은 다른 주장으로 구분한다. 세 목적을 VAL/EVAL 결과로 사후 선별하지 않는다.

[원수치·검사·출처](JOINT_ARRAY_AUDIT_v1.json) · [감사 코드](audit_joint_actual_v1.py)
