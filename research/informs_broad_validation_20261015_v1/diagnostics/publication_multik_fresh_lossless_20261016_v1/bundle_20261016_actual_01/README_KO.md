# Fresh K 비용 무손실 공개 포장

새 측정이 아니라 원 JSON/CSV/영수증의 바이트 동일 공개 파생본입니다. 완료·부분·미실행 상태는 원 BATCH와 full20 큐 그대로 보존합니다.

- 현재 포장: 12 fresh scope, 2160 trial. 계획상 최대12/2160과 구별합니다.
- [복원 원본 목록과 모든 SHA](MANIFEST.json)
- [별도 임시 복원·반복 검산](BUILD_RESULT.json)
- [기존8 재사용 및 원 terminal 바인딩](EXTERNAL_REUSE_BINDINGS.json)
- [기존 K8 공개 포장](../../publication_multik_lossless_1792094047117000000/bundle_1792096101396562900/MANIFEST.json)

복원은 기존 bundle_multik.py의 restore --manifest <이 폴더/MANIFEST.json> --public-root <복원 R>를 사용합니다. 기존과 다른 파일은 덮어쓰지 않습니다. 원 trial 경로는 MANIFEST에서 복원하며 Markdown의 원 trial 직접 링크를 만들지 않습니다.
