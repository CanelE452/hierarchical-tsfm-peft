# 대표 대조군의 실제 ID와 학습 계약

[확인] 이 목록은 기존 V metadata의 기술용 인용이다. T 선택·자동 모델 선택이 아니다. NA는 실제 해당 V가 없다는 뜻이다. A60_end와M_cal_end는 각 자료의 원본 row index exclusive endpoint이며, 원시 시간 간격·H/C가 자료별로 달라 평균하지 않는다. 전체 hash와 source score 위치는 CSV에 있다.

```text
RawSmall | primary_beijing_E1_RAW_BOLT_SMALL_k0_s92601
  V_MSE=0.7968786536971474; MAE=0.5618622043428347; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawSmall | primary_jena_E1_RAW_BOLT_SMALL_k0_s92601
  V_MSE=0.8491941389472357; MAE=0.4813883402485757; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawSmall | primary_electricity_E1_RAW_BOLT_SMALL_k0_s92601
  V_MSE=0.12662080012059046; MAE=0.2027005720721219; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawSmall | primary_pems08_E1_RAW_BOLT_SMALL_k0_s92601
  V_MSE=0.0953549334407595; MAE=0.19633382925361176; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawMini | primary_beijing_E1_RAW_BOLT_MINI_k0_s92601
  V_MSE=0.7892370076405851; MAE=0.5628558178301597; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawMini | primary_jena_E1_RAW_BOLT_MINI_k0_s92601
  V_MSE=0.9347997602692638; MAE=0.501505945656834; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawMini | primary_electricity_E1_RAW_BOLT_MINI_k0_s92601
  V_MSE=0.1336863746484637; MAE=0.20972763399340724; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawMini | primary_pems08_E1_RAW_BOLT_MINI_k0_s92601
  V_MSE=0.09704617438226289; MAE=0.19862252152703616; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawTiny | primary_beijing_E1_RAW_BOLT_TINY_k0_s92601
  V_MSE=0.8239191101251634; MAE=0.5727370594360998; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawTiny | primary_jena_E1_RAW_BOLT_TINY_k0_s92601
  V_MSE=1.3227551535537496; MAE=0.5855640659694938; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawTiny | primary_electricity_E1_RAW_BOLT_TINY_k0_s92601
  V_MSE=0.1407919520047067; MAE=0.21841953816014792; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
RawTiny | primary_pems08_E1_RAW_BOLT_TINY_k0_s92601
  V_MSE=0.09406309486711996; MAE=0.19373333188244285; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=fixed/no checkpoint
  fit=Official frozen foundation; no optimizer; A60 normalization only; device=no optimizer; training_status=NO_TRAINING; source_files_current=not applicable/metadata absent
Ridge_A60 | primary_beijing_E1_RAW_RIDGE_k0_s92601
  V_MSE=0.8315416367816169; MAE=0.6073577987792294; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=All legal A50 origins closed form; A_stop alpha choice; fresh all-origin A60 refit; device=cpu; training_status=CLOSED_FORM_ALL_LEGAL_A_ORIGINS; source_files_current=not applicable/metadata absent
Ridge_A60 | primary_jena_E1_RAW_RIDGE_k0_s92601
  V_MSE=0.5566463357048567; MAE=0.40552180175406594; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=A_stop
  fit=All legal A50 origins closed form; A_stop alpha choice; fresh all-origin A60 refit; device=cpu; training_status=CLOSED_FORM_ALL_LEGAL_A_ORIGINS; source_files_current=not applicable/metadata absent
Ridge_A60 | primary_electricity_E1_RAW_RIDGE_k0_s92601
  V_MSE=0.3193732185113921; MAE=0.40482331800492677; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=A_stop
  fit=All legal A50 origins closed form; A_stop alpha choice; fresh all-origin A60 refit; device=cpu; training_status=CLOSED_FORM_ALL_LEGAL_A_ORIGINS; source_files_current=not applicable/metadata absent
Ridge_A60 | primary_pems08_E1_RAW_RIDGE_k0_s92601
  V_MSE=0.22899594314673508; MAE=0.2325320321186678; endpoints=optimizer0/closed form; selected_epoch=NA; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=A_stop
  fit=All legal A50 origins closed form; A_stop alpha choice; fresh all-origin A60 refit; device=cpu; training_status=CLOSED_FORM_ALL_LEGAL_A_ORIGINS; source_files_current=not applicable/metadata absent
G_initial025 | primary_beijing_E1_DIRECT_G_k0_s92601
  V_MSE=0.8091814805925012; MAE=0.5993137650617376; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=False
G_initial025 | primary_jena_E1_DIRECT_G_k0_s92601
  V_MSE=0.6545615255198267; MAE=0.44190991873619756; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_initial025 | primary_electricity_E1_DIRECT_G_k0_s92601
  V_MSE=0.2275109583497733; MAE=0.31698772686915977; endpoints=0.25; selected_epoch=0.25; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_initial025 | primary_pems08_E1_DIRECT_G_k0_s92601
  V_MSE=0.11612389204300003; MAE=0.20879919431676036; endpoints=0.25; selected_epoch=0.25; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_GPU_to1 | primary_beijing_E1_DIRECT_G_k0_s92601__e1
  V_MSE=0.8142803948589289; MAE=0.6066850920769683; endpoints=0.25|1.0; selected_epoch=1.0; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_GPU_to1 | primary_jena_E1_DIRECT_G_k0_s92601__e1
  V_MSE=0.6037864771004047; MAE=0.4387272958898847; endpoints=0.25|1.0; selected_epoch=1.0; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_GPU_to1 | primary_electricity_E1_DIRECT_G_k0_s92601__e1
  V_MSE=0.18622973045202748; MAE=0.27841584333492236; endpoints=0.25|1.0; selected_epoch=1.0; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_GPU_to1 | primary_pems08_E1_DIRECT_G_k0_s92601__e1
  V_MSE=0.10605044045284828; MAE=0.19622655831868363; endpoints=0.25|1.0; selected_epoch=1.0; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_GPU_to3 | primary_beijing_E1_DIRECT_G_k0_s92601__e1__e3
  V_MSE=0.8142803948589289; MAE=0.6066850920769683; endpoints=0.25|1.0|3.0; selected_epoch=1.0; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED; source_files_current=True
G_CPU_fresh3 | primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3
  V_MSE=0.632217057436123; MAE=0.44152128982113786; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cpu; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
G_CPU_fresh3 | primary_electricity_E1_DIRECT_G_k0_s92601__cpu_fresh_e3
  V_MSE=0.17339512102560362; MAE=0.26282206795224494; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cpu; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
G_CPU_fresh3 | primary_pems08_E1_DIRECT_G_k0_s92601__cpu_fresh_e3
  V_MSE=0.09938846978422584; MAE=0.18905427923411572; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cpu; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
G_AplusM025 | primary_beijing_E2_G_REFIT_APLUSM_k0_s92601_G
  V_MSE=0.8036789861977347; MAE=0.599552047729934; endpoints=0.25; selected_epoch=0.25; actual_fit_end=14726; A60_end=12622; M_end=14726; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_AplusM025 | primary_jena_E2_G_REFIT_APLUSM_k0_s92601_G
  V_MSE=0.6238911519107087; MAE=0.4418792754537655; endpoints=0.25; selected_epoch=0.25; actual_fit_end=14675; A60_end=12579; M_end=14675; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_AplusM025 | primary_electricity_E2_G_REFIT_APLUSM_k0_s92601_G
  V_MSE=0.22570318568493; MAE=0.3162348541973617; endpoints=0.25; selected_epoch=0.25; actual_fit_end=16968; A60_end=14544; M_end=16968; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_AplusM025 | primary_pems08_E2_G_REFIT_APLUSM_k0_s92601_G
  V_MSE=0.10896194708345815; MAE=0.2058163318581044; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12499; A60_end=10713; M_end=12499; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
G_AplusM3 | primary_beijing_E2_G_REFIT_APLUSM_k0_s92601_G__e3
  V_MSE=0.7987709673709295; MAE=0.5919192512693504; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=14726; A60_end=12622; M_end=14726; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
G_AplusM3 | primary_jena_E2_G_REFIT_APLUSM_k0_s92601_G__e3
  V_MSE=0.597991468992935; MAE=0.4317919848047176; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=14675; A60_end=12579; M_end=14675; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
G_AplusM3 | primary_electricity_E2_G_REFIT_APLUSM_k0_s92601_G__e3
  V_MSE=0.17258087646733639; MAE=0.26122791070786455; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=16968; A60_end=14544; M_end=16968; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
G_AplusM3 | primary_pems08_E2_G_REFIT_APLUSM_k0_s92601_G__e3
  V_MSE=0.09296125438490886; MAE=0.18791465824021997; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=12499; A60_end=10713; M_end=12499; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
LoRA025 | primary_beijing_E1_LORA_R8_k0_s92601
  V_MSE=0.735095062494602; MAE=0.5581049329262888; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA025 | primary_jena_E1_LORA_R8_k0_s92601
  V_MSE=0.6262660149807455; MAE=0.41678830712550113; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA025 | primary_electricity_E1_LORA_R8_k0_s92601
  V_MSE=0.12662080012059046; MAE=0.2027005720721219; endpoints=0.25; selected_epoch=0.0; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA025 | primary_pems08_E1_LORA_R8_k0_s92601
  V_MSE=0.08472487374298254; MAE=0.18174260538928147; endpoints=0.25; selected_epoch=0.25; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA_GPU3 | primary_beijing_E1_LORA_R8_k0_s92601__e3
  V_MSE=0.735095062494602; MAE=0.5581049329262888; endpoints=0.25|1.0|3.0; selected_epoch=0.25; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED; source_files_current=True
LoRA_AplusM025 | primary_beijing_E2_LORA_REFIT_APLUSM_k0_s92601_LORA
  V_MSE=0.7316976797218803; MAE=0.5580353397663483; endpoints=0.25; selected_epoch=0.25; actual_fit_end=14726; A60_end=12622; M_end=14726; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA_AplusM025 | primary_jena_E2_LORA_REFIT_APLUSM_k0_s92601_LORA
  V_MSE=0.5991261211039864; MAE=0.40136281572084137; endpoints=0.25; selected_epoch=0.25; actual_fit_end=14675; A60_end=12579; M_end=14675; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA_AplusM025 | primary_electricity_E2_LORA_REFIT_APLUSM_k0_s92601_LORA
  V_MSE=0.1242335161525936; MAE=0.20273291708728988; endpoints=0.25; selected_epoch=0.25; actual_fit_end=16968; A60_end=14544; M_end=16968; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
LoRA_AplusM025 | primary_pems08_E2_LORA_REFIT_APLUSM_k0_s92601_LORA
  V_MSE=0.0832601324075793; MAE=0.17992605162807035; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12499; A60_end=10713; M_end=12499; selection=V_select
  fit=A+M direct supervised fit; A60 normalization; V checkpoint selection; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
AdaPTS_port025 | primary_beijing_E1_ADAPTS_BOLT_PORT_k3_s92601
  V_MSE=1.108223614169821; MAE=0.7803135717095645; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
AdaPTS_port3 | primary_beijing_E1_ADAPTS_BOLT_PORT_k3_s92601__e3
  V_MSE=0.8338202577336109; MAE=0.6428439901877439; endpoints=0.25|1.0|3.0; selected_epoch=3.0; actual_fit_end=12622; A60_end=12622; M_end=14726; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=UNDERTRAINED_OR_BUDGET_LIMIT; source_files_current=True
AdaPTS_port025 | primary_jena_E1_ADAPTS_BOLT_PORT_k6_s92601
  V_MSE=1.1843258230081575; MAE=0.69820782792687; endpoints=0.25; selected_epoch=0.25; actual_fit_end=12579; A60_end=12579; M_end=14675; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
AdaPTS_port025 | primary_electricity_E1_ADAPTS_BOLT_PORT_k81_s92601
  V_MSE=0.33823160481413944; MAE=0.388694478044822; endpoints=0.25; selected_epoch=0.25; actual_fit_end=14544; A60_end=14544; M_end=16968; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
AdaPTS_port025 | primary_pems08_E1_ADAPTS_BOLT_PORT_k43_s92601
  V_MSE=0.5271178103560189; MAE=0.3769197045836124; endpoints=0.25; selected_epoch=0.25; actual_fit_end=10713; A60_end=10713; M_end=12499; selection=A_stop
  fit=A50 search / A_stop checkpoint selection / fresh A60 refit; device=cuda; training_status=LIMITED_SCREEN; source_files_current=True
```

`optimizer_updates`은 base model에서 search_updates, H/control에서는 해당 fit의 updates, ridge에서는 actual_optimizer_updates를 옮긴 필드다. fresh A60 refit은 `fresh_refit_updates`에 따로 있어 앞 필드를 전체 학습 update라고 읽지 않는다. `attempted_epoch_endpoints`는 실제 receipt의 요청/시도 endpoint이며 선택값은 `selected_epoch`다. 미수렴 판정은 원 receipt를 유지한다.
