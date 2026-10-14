# D1 저장 실제 배열의 독립 CPU 검산

상태: PASS_WITHOUT_INDEPENDENT_GPU_REPLAY; 검사 13180개, 실패 0개.

원 F를 GPU에서 다시 실행하지 않았다. 실제 producer의 F 전후 두 state hash·동결 기록과 원 source snapshot SHA를 확인하고, 저장된 absolute latent future에서 FP32 decode를 재계산했다.
FIT RAW forecast−정확한 과거 last만으로 SVD basis subspace/mean을 독립 재계산했다. source snapshot AST에서 basis 적합의 입력에 Y/mask가 들어가지 않는 호출을 확인했다.
모든 실제 output/INPUT/bridge/RAW NPZ의 SHA·array dtype/shape/byte SHA 및 observed 채널 macro 정규화·원단위 MSE/MAE를 다시 계산했다.
Robin VAL과 이미 노출된 Beijing EVAL을 사용한 개발 증거다. 검산 통과는 예측 개선이나 독립 일반화의 검증이 아니다.

```text
dataset              candidate                    K    MSE        MAE         MSEΔ%       MAEΔ%
robin                identity_C                   17   0.3404019  0.3131921   0.000       0.000
robin                through_origin_quarter       4    0.5681031  0.4493145   66.892      43.463
robin                through_origin_half          8    0.5362725  0.4263521   57.541      36.131
robin                through_origin_full_C_rotated17   0.3152730  0.3382431   -7.382      7.999
robin                centered_quarter             4    0.5684507  0.4504965   66.994      43.840
robin                centered_half                8    0.5370614  0.4270845   57.773      36.365
robin                centered_full_C_rotated      17   0.3157236  0.3389082   -7.250      8.211
beijing_aotizhongxin identity_C                   11   0.5895754  0.4079165   -0.000      -0.000
beijing_aotizhongxin through_origin_quarter       2    0.6746169  0.4778962   14.424      17.155
beijing_aotizhongxin through_origin_half          5    0.6080259  0.4445350   3.129       8.977
beijing_aotizhongxin through_origin_full_C_rotated11   0.5814305  0.4246372   -1.381      4.099
beijing_aotizhongxin centered_quarter             2    0.6763188  0.4795440   14.713      17.559
beijing_aotizhongxin centered_half                5    0.6088185  0.4452606   3.264       9.155
beijing_aotizhongxin centered_full_C_rotated      11   0.5811405  0.4249211   -1.431      4.169
```

full-C rotation은 정보량을 줄이지 않아도 F의 입력 좌표계를 바꾼다. 실제 FP32 history 재구성과 독립 FP64 basis 재구성 오차는 JSON에 별도로 보존했다. MSE 감소와 MAE 증가가 함께 나타나면 개선 지표를 선택해 숨기지 않는다.

[원수치·검사·출처](D1_ARRAY_AUDIT_v1.json) · [감사 코드](audit_D1_actual_v1.py)
