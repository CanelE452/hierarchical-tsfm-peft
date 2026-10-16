# Exact existing cost_campaign CLI commands for ROOT review; executing this file starts GPU cost jobs.
# Run only after K20/previous cost owner terminal and exact plan/source/assets reviewed.
# No automatic selection/T/registry/Git writes. Each dataset retains the full protocol.
$taskAssetRoot = $env:INFORMS_ASSET_ROOT
$taskPublicRoot = $env:INFORMS_PUBLIC_ROOT
if (-not $taskAssetRoot -or -not $taskPublicRoot -or -not $env:INFORMS_CACHE_ROOT) { throw 'Set the three INFORMS roots first.' }
$taskPython = Join-Path $taskAssetRoot '.venv/Scripts/python.exe'
$taskCLI = Join-Path $taskPublicRoot 'cost_campaign.py'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

# CORE_SAME_G_beijing : 18 conditions, new same-G five-route balanced scope
& $taskPython -X utf8 -u -B $taskCLI '--dataset' 'beijing' '--reserve-final-seconds' '3600' '--run-id' 'primary_beijing_E1_RAW_BOLT_SMALL_k0_s92601' '--run-id' 'primary_beijing_E1_DIRECT_G_k0_s92601__e1__e3' '--run-id' 'primary_beijing_E2_CONDITIONAL_k3_s92601_B0__xp_e3_4336644dd851' '--run-id' 'primary_beijing_E2_CONDITIONAL_k3_s92601_B2__xp_e3_4336644dd851' '--run-id' 'primary_beijing_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_6a1e0351808b'

# CORE_SAME_G_jena : 18 conditions, new same-G five-route balanced scope
& $taskPython -X utf8 -u -B $taskCLI '--dataset' 'jena' '--reserve-final-seconds' '3600' '--run-id' 'primary_jena_E1_RAW_BOLT_SMALL_k0_s92601' '--run-id' 'primary_jena_E1_DIRECT_G_k0_s92601__e1' '--run-id' 'primary_jena_E2_CONDITIONAL_k6_s92601_B0__xp_e1_1aaf055cbe2c' '--run-id' 'primary_jena_E2_CONDITIONAL_k6_s92601_B2__xp_e1_1aaf055cbe2c' '--run-id' 'primary_jena_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_7943f4e9d13b'

# CORE_SAME_G_electricity : 18 conditions, new same-G five-route balanced scope
& $taskPython -X utf8 -u -B $taskCLI '--dataset' 'electricity' '--reserve-final-seconds' '3600' '--run-id' 'primary_electricity_E1_RAW_BOLT_SMALL_k0_s92601' '--run-id' 'primary_electricity_E1_DIRECT_G_k0_s92601__e1' '--run-id' 'primary_electricity_E2_CONDITIONAL_k81_s92601_B0__xp_e1_9549829e7a68' '--run-id' 'primary_electricity_E2_CONDITIONAL_k81_s92601_B2__xp_e1_9549829e7a68' '--run-id' 'primary_electricity_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_2a8965c95f86'

# CORE_SAME_G_pems08 : 18 conditions, new same-G five-route balanced scope
& $taskPython -X utf8 -u -B $taskCLI '--dataset' 'pems08' '--reserve-final-seconds' '3600' '--run-id' 'primary_pems08_E1_RAW_BOLT_SMALL_k0_s92601' '--run-id' 'primary_pems08_E1_DIRECT_G_k0_s92601__e1' '--run-id' 'primary_pems08_E2_CONDITIONAL_k43_s92601_B0__xp_e1_93c81fd937ea' '--run-id' 'primary_pems08_E2_CONDITIONAL_k43_s92601_B2__xp_e1_93c81fd937ea' '--run-id' 'primary_pems08_E2_CONDITIONAL_k0_s92601_RAW__xp_deep_g_raw_control_6d69e6bb44c5'

