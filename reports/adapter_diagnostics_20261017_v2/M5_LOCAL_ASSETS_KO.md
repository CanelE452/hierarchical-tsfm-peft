# M5 로컬 자산 발견 및 헤더 검증

[확인] 사용자가 승인한 recovery 묶음의 원본 CSV 4개와 ZIP 1개가 실제 존재하며, SHA256·바이트 수·CSV 첫 헤더의 필드 순서를 검증했다. 현재 상태는 `LOCAL_ASSETS_SCHEMA_VERIFIED`다. [공개 증거](M5_LOCAL_ASSET_DISCOVERY.json)에 기본 파일명·해시·전체 header_fields·day_column_range 및 검증 시각이 있다. 실제 경로는 SHA로 결속된 private inventory에만 저장했다.

```text
파일                          bytes       schema
sales_train_validation.csv    120007726   6 IDs + d_1..d_1913
sales_train_evaluation.csv    121736518   6 IDs + d_1..d_1941
calendar.csv                     103469   지정 14개 필드 순서
sell_prices.csv               203395785   store_id,item_id,wm_yr_wk,sell_price
m5-forecasting-accuracy.zip     48009163   opaque SHA/bytes만 확인
```

[확인] CSV 숫자 행을 파싱하지 않았다(`new_numeric_values_parsed=0`). 파일 전체 바이트는 SHA 계산에만 읽었고 CSV 파싱은 각 파일의 첫 헤더 한 줄로 제한했다. ZIP 내부를 열거나 해제하지 않았다. 새 M5 평가·split 등록·fit·모델/F/optimizer 실행·다운로드·원본 수정은 모두 0이다. 다른 저장소의 복사본이나 파생 예측 파일은 선택하지 않았다.

과거 2026-10-16의 HTTP401과 로컬 후보 미발견은 당시 등록된 inventory 범위의 관측이다. PC 전체에 M5가 없다는 증거로 해석할 수 없다. 원래 감사·HTTP401 기록은 보존하며, 이번 발견 증거로 현재 로컬 접근 상태를 보완한다.

파일의 출처 계보와 license는 독립 검증하지 않아 `UNKNOWN`, 이 캠페인의 연구 split은 `NOT_REGISTERED`, 모델 평가는 `NOT_RUN`이다. 로컬 파일 존재·헤더 일치만으로 모델 평가의 유효성이나 데이터 사용 허가가 검증된 것은 아니다. 새 평가나 학습은 실행하지 않는다.

공개 검증 API는 `code/m5_local_assets.py`의 `verify_local_m5_assets(ctx, discovery=None)`다. `ctx`에는 `report`와 `cache` 루트를 제공한다. 기본 동작은 공개 discovery를 읽은 뒤 `cache://m5_local_assets/INVENTORY.json`의 SHA를 먼저 확인하고, 그 typed inventory에 명시된 승인 CSV 4개 및 ZIP 1개만 다시 읽어 바이트·SHA·첫 헤더/schema와 공개 증거를 대조하는 것이다. 성공하면 같은 상태와 검사 개수를 반환한다. inventory/source 변조·누락·헤더 오류는 거부한다.

```python
verify_local_m5_assets({"report": REPORT_ROOT, "cache": PRIVATE_CACHE})
```

[확인] 정상·변조·누락의 CPU fixture 3개를 RED에서 GREEN으로 검증했고, 불완전한 CSV 본문과 opaque archive fixture도 숫자 행/압축 내부를 파싱하지 않고 처리했다. 승인 실물에 같은 API를 실행하여 PASS를 확인했다. 원래 캠페인의 시작 시각·예산을 갱신하지 않았다.
