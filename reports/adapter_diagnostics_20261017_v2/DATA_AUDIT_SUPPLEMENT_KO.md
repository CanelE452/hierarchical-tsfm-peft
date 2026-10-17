# 원 A01 감사의 출처·단위 보완

원 DATA_AUDIT_KO.md와 DATA_AUDIT.json은 검증 당시의 receipt-bound 원본을 보존한다. 아래는 같은 자료의 추가 공개 메타데이터다.

## 자료 출처·단위와 노출 상태 보완

[DATA_SOURCE_DETAILS.json](DATA_SOURCE_DETAILS.json)에 실제 로컬 source SHA·허용 범위·URL·단위·노출 근거를 보존한다. 채널 이름에서 물리 단위를 추정하지 않는다.

beijing: source SHA=78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277; original URL=UNKNOWN: absent from the pinned local source metadata; no external inference; units=UNKNOWN: absent from the pinned local source metadata; no external inference. historically_unseen=False; new_to_source_campaign=False; LOCKED; new numeric T values parsed=0. UNKNOWN per-dataset count; the historical global 46 conditions are not apportioned to every dataset.

jena: source SHA=29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933; original URL=UNKNOWN: absent from the pinned local source metadata; no external inference; units=UNKNOWN: absent from the pinned local source metadata; no external inference. historically_unseen=False; new_to_source_campaign=False; LOCKED; new numeric T values parsed=0. UNKNOWN per-dataset count; the historical global 46 conditions are not apportioned to every dataset.

robin: source SHA=04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14; original URL=UNKNOWN: absent from the pinned local source metadata; no external inference; units=UNKNOWN: absent from the pinned local source metadata; no external inference. historically_unseen=False; new_to_source_campaign=False; LOCKED; new numeric T values parsed=0. UNKNOWN per-dataset count; the historical global 46 conditions are not apportioned to every dataset.

pems08: source SHA=e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075; original URL=https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08/PEMS08.npz; units=Traffic flow as author data; exact physical counting-unit specification absent. historically_unseen=unknown; new_to_source_campaign=True; LOCKED; new numeric T values parsed=0. UNKNOWN per-dataset count; the historical global 46 conditions are not apportioned to every dataset.

electricity: source SHA=bc33039133b9f1ca2e73d6b58d97306dd063f865fd12fbb0920ab312f047ad28; original URL=https://zenodo.org/api/records/3889830/files/electricity_hourly_dataset.ts/content; units=Source calls hourly aggregated electricity consumption; raw UCI kW ancestry; exact aggregation and unit conversion not rederived. historically_unseen=unknown; new_to_source_campaign=True; LOCKED; new numeric T values parsed=0. UNKNOWN per-dataset count; the historical global 46 conditions are not apportioned to every dataset.

M5: 로컬 파일의 헤더 schema·SHA를 확인했다. [M5_LOCAL_ASSET_DISCOVERY.json](M5_LOCAL_ASSET_DISCOVERY.json)과 [M5_LOCAL_ASSETS_KO.md](M5_LOCAL_ASSETS_KO.md)에 근거와 범위를 보존한다. 숫자 행 파싱=0, fit/evaluation=NOT_RUN, C/L/H·split=NOT_REGISTERED다. 성능·자료 품질·단위·첫 사용·사전학습 미노출을 확인한 결과가 아니다. 원 감사의 2026-10-16 HTTP 401·등록 inventory 부재 기록은 과거 접근 기록으로 보존하며, 현재 전체 PC의 파일 부재나 라이선스 거부를 뜻하지 않는다.

pems32: source SHA=e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075; original URL=https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08/PEMS08.npz; units=Traffic flow as author data; exact physical counting-unit specification absent. historically_unseen=unknown; new_to_source_campaign=True; LOCKED; new numeric T values parsed=0. C32 post-T development subset; no numeric T evaluation in this campaign; historical parent exposure does not imply C32 T scoring.

new_to_campaign은 원 DATA_CONTRACT의 과거 캠페인 플래그다. 이번 v2의 신규 독립 자료 또는 사전학습 미노출을 뜻하지 않는다. 정규화 MSE/MAE를 물리 단위 오차로 부르지 않는다.
