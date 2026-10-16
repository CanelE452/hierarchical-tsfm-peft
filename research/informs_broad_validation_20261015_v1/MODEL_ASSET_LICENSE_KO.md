# 공식 Bolt mini·tiny 자산 확보

[확인] 두 공식 모델을 익명·무료로 실제 다운로드하고 revision, 파일 길이, Git blob SHA 및 LFS SHA256를 대조했다. 상태는 **자산 준비 완료, 추론 미검증**이다. 모델을 로드하거나 GPU/F를 실행하지 않았다. 설치·로그인·약관 대리 동의도 수행하지 않았다.

## 실제 pin

```text
model   revision                                  tensor elements  weight bytes
mini    251268337516a88e253628c43e1d26ec577b376b     21,236,096       84,956,096
tiny    a0e552de83495b5c28c14c71c374f3e33280b340      8,652,672       34,622,352
```

원본 출처는 Amazon의 [mini 모델](https://huggingface.co/amazon/chronos-bolt-mini/tree/251268337516a88e253628c43e1d26ec577b376b)과 [tiny 모델](https://huggingface.co/amazon/chronos-bolt-tiny/tree/a0e552de83495b5c28c14c71c374f3e33280b340)이다. `HfApi(token=False).model_info(..., files_metadata=True, token=False)`의 실제 응답에서 두 저장소 모두 `private=false`, `gated=false`, 모델 카드 license `apache-2.0`을 확인했다.

각 모델에서 `config.json`, `README.md`, `model.safetensors`만 받았다. 각 파일의 byte와 해시는 [MODEL_ASSETS.json](MODEL_ASSETS.json)에 있다. 모델 private cache 상대 경로는 `models/amazon/chronos-bolt-mini/<revision>` 및 `models/amazon/chronos-bolt-tiny/<revision>`이다. 공개 manifest에는 사용자 이름이나 절대 private 경로를 넣지 않았다. 가중치를 public main으로 복제하지 않는다.

```text
mini model.safetensors SHA256
1a1a4297f132b808c5c7da24e3cce549d519c01ff5c2661fcad404baba018f24

tiny model.safetensors SHA256
75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32
```

## config와 실제 검증 범위

[확인] 두 config의 `chronos_config.context_length=2048`, `prediction_length=64`, input patch size/stride는 16이고 quantile은 0.1부터 0.9까지 9개다. 따라서 계획의 L512, H12/H28/H48 및 중앙값 0.5 요청을 config상 지원한다. 최상위 `n_positions=512`와 시계열 context 길이 2048은 다른 config 필드이므로 혼동하지 않는다. 아키텍처는 `ChronosBoltModelForForecasting`, pipeline은 `ChronosBoltPipeline`, 저장 dtype은 F32다. 두 모델 모두 encoder/decoder 4층이며 mini의 `d_model=384`, tiny는 256이다.

파일 전체를 다시 읽어 SHA256와 Git blob SHA1를 계산했다. safetensors는 header와 `safe_open(..., framework="numpy")` metadata를 대조해 각 101개 tensor의 이름·shape·offset 범위를 확인했다. payload tensor를 메모리로 로드하지 않았다. tensor element 합계는 저장 파일 metadata에서 산출한 값이며, 별도의 학습 가능한 모델 객체의 parameter count를 측정한 것은 아니다.

이어 별도 `--verify-only` 실행에서 **모델별 3파일과 공식 LICENSE 1파일, 총 7파일**의 길이·두 해시 및 safetensors metadata를 재검증했다. 네트워크 요청·모델 로드·GPU 호출은 모두 0이었다. 이 검증은 실제 추론 parity나 정확도/비용 측정 완료를 의미하지 않는다. Raw mini/tiny 실행은 이후 GPU lease 안에서 진행해야 한다.

## 라이선스 근거

두 revision의 실제 모델 카드에 Apache-2.0 선언이 있다. 모델 저장소 자체에는 별도 LICENSE 파일이 없었다. 공식 companion source인 `amazon-science/chronos-forecasting`의 API에서 source commit `1224fafa91e1ecd6674f5204066b159e40b5e4a1`을 pin하고 [해당 commit의 LICENSE 원문](https://github.com/amazon-science/chronos-forecasting/blob/1224fafa91e1ecd6674f5204066b159e40b5e4a1/LICENSE)을 읽어 Apache License 2.0을 확인했다. 이 원문은 모델 revision에 원래 들어 있던 파일인 것처럼 표시하지 않는다.

```text
official source LICENSE bytes:       10,142
Git blob SHA1:                       67db8588217f266eb561f75fae738656325deac9
SHA256:                              09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b
```

Apache-2.0의 재배포 조건은 license 사본 제공, 변경 사실 표시, 관련 저작권·특허·상표·귀속 고지 유지 및 해당되는 NOTICE 고지다. 이번 공개 대상은 취득 코드와 pin/검증 정보이며, 모델 파일은 private cache에 둔다. 원 모델 학습 자료의 모든 권리나 이번 benchmark의 사전학습 노출 여부까지 이 license 확인으로 검증했다고 주장하지 않는다. 공식 모델 카드의 속도·정확도 설명도 본 캠페인의 실측 결과로 인용하지 않는다.

## 자원 계상과 재검증

다운로드 전에 `resources.record_download`로 각 pinned byte 수를 예약했다. 기존 데이터·원문 metadata 전송은 immutable baseline **54,477,296 byte**, 초기 HTTP header/API 조사 보수분은 별도 **2,097,152 byte**이다. 이번 두 모델의 필수 파일과 공식 LICENSE 합계는 **119,608,148 byte**이며 누적 예약은 **176,182,596 byte**, 승인된 5GiB보다 작다. 추가 API metadata는 초기 보수분 안에서 처리했다. 추후 다른 다운로드는 별도 사전 예약이 필요하다.

취득 종료 시 새 private cache 전체 크기는 **228,788,408 byte**로 30GiB 이내였다. 이 값은 그 시점의 snapshot이며 병행 실행이 계속 생성하는 파일의 최종 총량은 아니다. 공개 산출물의 별도 byte도 최종 campaign storage 합계에 포함해야 한다.

[acquire_models.py](acquire_models.py)의 재검증 명령은 다음과 같다. `PRIVATE_CACHE`는 실제 로컬 private cache 경로로 전달한다.

```text
python -X utf8 research/informs_broad_validation_20261015_v1/acquire_models.py --cache PRIVATE_CACHE --verify-only
```

기존 manifest를 덮는 신규 취득은 거부한다. 새로운 취득 manifest가 필요하면 다른 `--output`을 지정한다. 기존 유효 파일은 해시를 대조하여 재사용하며, 실패한 `.part` 파일은 보존한다. 잘못된 길이나 해시, gate/authentication, 라이선스 불일치가 발생하면 해당 자산을 ready로 표기하지 않는다.
