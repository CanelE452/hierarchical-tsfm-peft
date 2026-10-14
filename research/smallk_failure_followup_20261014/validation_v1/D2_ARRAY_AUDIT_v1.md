# D2 실제 디코더 재적합의 독립 CPU 검산

상태: PASS_WITHOUT_INDEPENDENT_F_FORWARD_REPLAY; 검사 28365개, 실패 0개.

원 MAIN CP의 파일·codec state hash와 SOURCE_DECODER의 실제 hidden·출력 가중치를 대조했다. absolute latent·현재 last·정규화 통계·target/mask/origin의 원 bridge·MAIN·RAW 일치를 검사했다.
동결 GELU feature와 원/factored D, FIT-only RMS·동일 observed mask·절편 없는 ridge의 모든 일곱 alpha·TUNE 실제 Y MSE 선택·선택 head의 전 split 예측을 CPU에서 다시 계산했다.
원 F를 새로 실행하지 않았다. 저장 actual F 출력과 producer 전후 hash 기록을 확인한 검산이며 독립 holdout이 아니다. 다른 단위를 가진 원단위 채널을 합쳐 우수성을 주장하지 않는다.

```text
dataset              seed    method     alpha       MSE           MAE
robin                92601   RAW        -           0.340401887   0.313192123
robin                92601   MAIN       -           0.567146384   0.447670118
robin                92601   Y          0.001       0.585387019   0.459910034
robin                92601   RAW        0.001       0.573922012   0.454686353
robin                92601   MIXED      0.001       0.575599635   0.454746002
robin                92602   RAW        -           0.340401887   0.313192123
robin                92602   MAIN       -           0.571946704   0.447007678
robin                92602   Y          0.1         0.582314981   0.453049318
robin                92602   RAW        0.1         0.575007940   0.451088401
robin                92602   MIXED      0.1         0.577205772   0.451241074
beijing_aotizhongxin 92601   RAW        -           0.589575414   0.407916509
beijing_aotizhongxin 92601   MAIN       -           0.661344557   0.466472476
beijing_aotizhongxin 92601   Y          0.1         0.671084539   0.466606879
beijing_aotizhongxin 92601   RAW        0.1         0.655374817   0.458122655
beijing_aotizhongxin 92601   MIXED      0.1         0.658916614   0.460462163
beijing_aotizhongxin 92602   RAW        -           0.589575414   0.407916509
beijing_aotizhongxin 92602   MAIN       -           0.662569839   0.473547987
beijing_aotizhongxin 92602   Y          0.01        0.665577143   0.475252932
beijing_aotizhongxin 92602   RAW        0.1         0.665150302   0.465967910
beijing_aotizhongxin 92602   MIXED      0.01        0.658964198   0.467534097
```

최저 VAL/EVAL 방법을 사후 선택하지 않았다. Y/RAW/MIXED 각각의 사전 TUNE 선택 결과를 모두 보존하며 원 head로 fallback하여 개선이라고 쓰지 않는다.

[원수치·검사·출처](D2_ARRAY_AUDIT_v1.json) · [감사 코드](audit_D2_actual_v1.py)
