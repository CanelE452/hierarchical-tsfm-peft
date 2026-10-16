# INFORMS 실제 진행 상태 (중간 기록)

최종 완료 보고서가 아니다. START_HERE/MASTER/YAML 전체를 읽고 새 실행 중이다. 원 main c7a06e52와 기존 결과를 보존한다. 신규 결과는 아직 게시하지 않았다.

실제 worktree: 현재 이 파일이 있는 Git worktree. private cache는 INFORMS_CACHE_ROOT, 원 자산은 INFORMS_ASSET_ROOT 환경변수로 지정한다. 실제 경로는 비공개 LOCAL_EXECUTION.json에만 있다.

공식321전력(보호2014-10-07 이후 numeric parse제외), PeMS170 flow, 기존안전FIT3종 준비완료. M5공식접근401이라3패널 차단. 새 A50/Astop10에서 선택한epoch로 A60 upstream을 새로 적합한다. Mcal/V/T와 기존3분할의 결과는 재사용하지 않는다.

56개 초기 합성검사 후 독립감사에서lag-Cidentity누락을실제RED로확인했다. lag-C 실행 전 한조건을고쳤고구source/패치/provenance를보존했다. 최초Beijing G/B0K3/B2K3는영향없으며각search156+refit189update를실제완료했다. 모두0.25epoch LIMITED_SCREEN이며1/3epoch및충분수렴은완료가아니다. 이후Beijing핵심queue실행상태는EXPERIMENT_REGISTRY와privateACTIVE_GPU_LEASE/PID로확인한다.

GPU실제probe는C321/B2effective16,micro4,Frow128에서약2.12초/update,peak6.7GiB였다. 30초/micro1 결과도보존한다. 12h는모든grid3epoch+refit의완료보장이아니다. 단일GPU OSlock/processwall누적/크래시보수계상/1h감사보류분을controller에통합했다. 30GiB cache/5GiB free download 제한도새guard로추적한다.

담당코드: data_contract/prepare_data(완료),networks(완료),training/gpu_runtime/rootcampaign(구동),fit_base(구동),evaluation/audit_predictions(완료),combination(구현완료·실제미실행),prediction_cache(구현완료·실제미실행),legacy_continuation(장부12+대조28상태검증완료·실제연장0),deployment/measure_costs(구현중),tiny/mini공식자산취득(진행중).

남은필수: E1가족/data/K최소실학습, Mcaldense실제부모추론+E2실학습, E3양시차actualfit, E4공유모델actualfit, E0실연장(원RNG부재=exactresume아님), V선정·모든선택잠금후T추론, 별도FP64재채점/공개통계재집계/체크포인트재추론, 전체비용,그림6묶음/한국어보고,안전allowlist main push/remotehash. 계획·합성검사를이실행의대체증거로세지않는다.
