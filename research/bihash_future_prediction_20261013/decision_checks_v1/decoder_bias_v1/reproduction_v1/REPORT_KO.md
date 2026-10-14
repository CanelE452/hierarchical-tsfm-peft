# 취소되는 디코더 bias 수치 진단

상태: COMPLETE
실제 Jena NO_ALIGN의 CP0/600 디코더와 저장된 FIT 예측·정답을 사용했다. K 차원 미래/마지막 latent는 명시적으로 만든 synthetic 값이며 Chronos-Bolt의 실제 출력이 아니다.
원 식 `x_last + D(future) - D(last)`에서 두 최종 affine bias의 정확한 미분은 0이다. 미래 B×H 행의 bias 합산과 broadcast 경로의 H→B 합산 순서 차이를 실제 저장 FIT residual gradient와 함께 기록했다.
진단용 bias-free 식은 선형 분기의 `W(future-last)`와 비선형 hidden 차이의 최종 선형 변환을 사용한다. 이 식을 실제 학습에 적용하거나 체크포인트를 수정하지 않았다.
- `codec.codec.decode.bias`: 변경 21개, 최대 절대 차이 0.008123568259
- `codec.residual.2.bias`: 변경 21개, 최대 절대 차이 0.008123563603

torch.float32: original bias gradient 최대 3.725290298e-09, fresh AdamW 한 step 최대 변화 0.0002714181319, bias-free synthetic 출력 최대 차이 4.768371582e-07.

torch.float64: original bias gradient 최대 3.903127821e-18, fresh AdamW 한 step 최대 변화 3.903127821e-13, bias-free synthetic 출력 최대 차이 8.881784197e-16.

이 현상의 재현은 표현학습 실패 원인의 확정이나 bias 제거의 예측 성능 개선 증거가 아니다. 실제 과거 GPU 학습의 정확한 gradient tensor는 저장되지 않았으므로 이번 CPU 재현과 분리한다. AdamW restored 조건은 기존 momentum 자체로 갱신될 수도 있다.
