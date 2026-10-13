# Source/data contract 검증

- 생성 시각: 2026-10-13 18:21:12
- 상태: `PASS`
- 검증 범위: TRAIN/VAL source NPZ, full/repair/resolved/clean terminal manifests, resolved prediction NPZ materialization, source/snapshot hashes. No TEST data, no training.

## 핵심 결과

- jena: source SHA match `True`, FIT mean max abs 0, FIT std max abs 0, origins {'fit': 256, 'tune': 64, 'val': 371}
- robin: source SHA match `True`, FIT mean max abs 0, FIT std max abs 0, origins {'fit': 256, 'tune': 64, 'val': 30}
- prediction NPZ 검증: 236개 / expected 236
- target observed max abs: 0
- target full max abs: 0
- mismatch counts: sha=0, mean/std=0, origin=0, mask=0, target_observed=0
- TEST data 의심 접근 수: 0

이 PASS는 위 source/contract/materialization 항목에 한정된다. 분석 조건 전체에서 미래 정보 누출이 전혀 없다는 일반 명제까지 확장하지 않는다.

## warnings

- 없음

## errors

- 없음
