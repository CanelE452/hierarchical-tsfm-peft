# Fresh serial fits only. Root must launch after current sampling/audits finish.
# Producer checkpoint events expose actual saved updates; this queue never resumes.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runnerPath = Join-Path $PSScriptRoot 'run_forecast_sequence_v1.py'
$lockPath = Join-Path $PSScriptRoot 'METHOD_LOCK_FORECAST_SEQUENCE_v1.md'
$runtimePath = Join-Path $PSScriptRoot 'continuation_runtime_v2.py'
$coreRuntimePath = Join-Path $PSScriptRoot 'continuation_runtime_v1.py'
$authorizationPath = Join-Path $PSScriptRoot 'continuation_authorization_v1.json'
$manifestPath = Join-Path $PSScriptRoot 'continuation_run_manifest_v1.json'
$resourceFolder = Join-Path $PSScriptRoot 'continuation_resources_v1'
$activeLeasePath = Join-Path $resourceFolder 'ACTIVE.json'
$bridgePath = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/input_bridge/old_conditions_actual_v1/RESULT.json'
$queueFolder = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/forecast_sequence_v1'
$sourceSha = 'bace190480f46f11cd51ced5048d74e06d94559b2b0c8756d63e9ddf35ec76b7'
$lockSha = '0aac5bc927eb7bbce2329e3648ccf1fabc3860abcef4d9b9b03aceca2ac3a850'
$runtimeSha = '48f585a1f205be638f1ba01fd5cf3ab99abf27a2dd4538fb2d37d4e5a7fbefb9'
$coreRuntimeSha = '1ef7b67a55002cbb696900b6d26cdf99c6c33bf6ca5e1dc44b99d38394a83058'
$lockedSources = @{
    'forecast_sequence_decoder.py' = 'f61bbeb1bdb104c420bbf46cd41d3b8b74c1d9931dc326739191b8303108853d'
    'test_run_forecast_sequence_v1.py' = '62e61d46029dc42434d20fde1e7b1cc2f0f6ab5d537012cae610f48511e19f8f'
    'test_forecast_sequence_decoder.py' = '28b42e5baa392be3aaf97837701d5407295b91ec56d27dcb69c5d0a280fc3c5a'
    'fast_storage_scan_v1.py' = 'cb0f2644a6b6cdc5bd2389027c54a6146218ec79e6250553ec72eb4c43f57395'
}
$queueSourceSha = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceTestProofPath = Join-Path $PSScriptRoot 'validation_v1/FORECAST_SEQUENCE_SOURCE_FIXTURES_v1.json'
$sourceTestProofSha = '8ff5f32191002a9e587a5f61258418855180c3837600756bc563a73894d3fedf'
$preparedValidation = [pscustomobject]@{
    status = 'ACTUAL_CPU_DRIVER_FIXTURES_PASS'
    runner_CPU_fixtures = 12
    new_finalization_CPU_fixtures = 2
    new_finalization_status = 'ACTUAL_PASS'
    actual_proof = [pscustomobject]@{path=$sourceTestProofPath;sha256=$sourceTestProofSha}
    queue_PS_parser_error_count = 0
    decoder_CPU_fixtures = 12
    runtime_CPU_fixtures = 7
    source_review = 'Independent original-API and scientific-contract review reported no blocking issue'
    scope = 'CPU fixtures and source review; no real-data performance or independent F-forward claim'
}
$conditions = [System.Collections.Generic.List[object]]::new()
foreach ($datasetName in @('robin', 'beijing_aotizhongxin')) {
    foreach ($seedValue in @(92601, 92602)) {
        foreach ($headChoice in @(@{name='STATIC_RF1';suffix='static'}, @{name='TEMPORAL_RF3';suffix='temporal'})) {
            $conditions.Add([pscustomobject]@{
                dataset=$datasetName;seed=$seedValue;head=$headChoice.name
                tag=('sequence_' + $datasetName + '_' + $headChoice.suffix + '_s' + $seedValue + '_v1')
            })
        }
    }
}

# Check fresh ownership before creating any queue artifact. Never remove a lease.
if ((Get-FileHash -LiteralPath $sourceTestProofPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceTestProofSha) {
    throw 'Actual CPU fixture proof SHA mismatch'
}
$sourceTestProof = Get-Content -LiteralPath $sourceTestProofPath -Raw -Encoding utf8 | ConvertFrom-Json
if (($sourceTestProof.status -ne 'COMPLETE') -or ($sourceTestProof.tests_run -ne 12) -or
    ($sourceTestProof.failures -ne 0) -or ($sourceTestProof.errors -ne 0) -or ($sourceTestProof.PS_queue_parser_error_count -ne 0)) {
    throw 'Actual CPU fixture proof is not a completed 12-fixture pass'
}
if (-not (Test-Path -LiteralPath $authorizationPath) -or -not (Test-Path -LiteralPath $manifestPath)) {
    throw 'Explicit approved authorization and fixed run manifest are required'
}
if (Test-Path -LiteralPath $activeLeasePath) { throw 'Actual scientific job lease is active; root must review its terminal state first' }
$authorization = Get-Content -LiteralPath $authorizationPath -Raw -Encoding utf8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$authorizationSha = (Get-FileHash -LiteralPath $authorizationPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestSha = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if (($authorization.status -ne 'APPROVED') -or ($authorization.origin -ne 'human_user_response') -or
    -not $authorization.response_identifier -or -not $authorization.user_response -or
    ($manifest.authorization_sha256 -ne $authorizationSha) -or ($manifest.runtime_source_sha256 -ne $coreRuntimeSha)) {
    throw 'Human approval or fixed manifest binding is invalid'
}
$bridgeSha = (Get-FileHash -LiteralPath $bridgePath -Algorithm SHA256).Hash.ToLowerInvariant()
foreach ($condition in $conditions) {
    if (Test-Path -LiteralPath (Join-Path $queueFolder $condition.tag)) { throw ('Fresh condition folder already exists: ' + $condition.tag) }
}
if (-not (Test-Path -LiteralPath $queueFolder)) { New-Item -ItemType Directory -Path $queueFolder | Out-Null }
$logPath = Join-Path $queueFolder 'FORECAST_SEQUENCE_QUEUE_v1.jsonl'
$terminalPath = Join-Path $queueFolder 'FORECAST_SEQUENCE_QUEUE_v1.json'
if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $terminalPath)) { throw 'Queue artifacts already exist' }
$logStream = [System.IO.File]::Open($logPath, [System.IO.FileMode]::CreateNew)
$logStream.Dispose()
$rows = [System.Collections.Generic.List[object]]::new()
$queueStatus = 'RUNNING'
$queueError = $null
$stoppedBeforeTag = $null
$startedAt = [DateTimeOffset]::UtcNow.ToString('o')
try {
    foreach ($condition in $conditions) {
        if (((Get-FileHash -LiteralPath $runnerPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceSha) -or
            ((Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $lockSha) -or
            ((Get-FileHash -LiteralPath $runtimePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $runtimeSha) -or
            ((Get-FileHash -LiteralPath $coreRuntimePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $coreRuntimeSha) -or
            ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $queueSourceSha)) {
            throw 'Locked runner/protocol/runtime/queue source changed'
        }
        foreach ($fileName in $lockedSources.Keys) {
            if ((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $fileName) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $lockedSources[$fileName]) {
                throw ('Locked helper or prepared fixture changed: ' + $fileName)
            }
        }
        if (((Get-FileHash -LiteralPath $authorizationPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $authorizationSha) -or
            ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestSha) -or
            ((Get-FileHash -LiteralPath $bridgePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $bridgeSha)) {
            throw 'Approval, fixed manifest or original input bridge changed'
        }
        if (Test-Path -LiteralPath $activeLeasePath) { throw 'Active lease remains; no further independent fit may start' }
        $gpuCharged = 0.0
        $cpuCharged = 0.0
        if (Test-Path -LiteralPath $resourceFolder) {
            foreach ($receiptFile in Get-ChildItem -LiteralPath $resourceFolder -Filter '*.json' -File) {
                $receiptValue = Get-Content -LiteralPath $receiptFile.FullName -Raw -Encoding utf8 | ConvertFrom-Json
                if ($receiptValue.status -notin @('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED')) {
                    throw ('Nonterminal actual resource receipt: ' + $receiptFile.FullName)
                }
                if ($receiptValue.gpu) { $gpuCharged += [double]$receiptValue.wall_seconds }
                else { $cpuCharged += [double]$receiptValue.wall_seconds }
            }
        }
        $nowUnix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        if (($nowUnix -ge [double]$manifest.expires_unix) -or
            ($gpuCharged + 150.0 -gt [double]$authorization.limits.gpu_process_wall_seconds) -or
            ($cpuCharged -ge [double]$authorization.limits.cpu_process_wall_seconds)) {
            $queueStatus = 'BUDGET_HELD_PRESERVED'
            $queueError = 'Campaign expired, CPU cap spent, or less than one declared 150-second fit remains in GPU cap'
            $stoppedBeforeTag = $condition.tag
            Write-Output ($([pscustomobject]@{event='SEQUENCE_QUEUE_GLOBAL_HOLD';tag=$condition.tag;reason=$queueError;gpu_charged=$gpuCharged;cpu_charged=$cpuCharged}) | ConvertTo-Json -Compress)
            break
        }
        $runFolder = Join-Path $queueFolder $condition.tag
        if (Test-Path -LiteralPath $runFolder) { throw ('Fresh run folder already exists: ' + $condition.tag) }
        Write-Output ($([pscustomobject]@{event='SEQUENCE_QUEUE_START';dataset=$condition.dataset;seed=$condition.seed;head=$condition.head;tag=$condition.tag;local_fit_cap_seconds=150}) | ConvertTo-Json -Compress)
        & $pythonPath $runnerPath --execute --dataset $condition.dataset --seed $condition.seed --head $condition.head --input-bridge $bridgePath --tag $condition.tag
        $processExit = $LASTEXITCODE
        $resultPath = Join-Path $runFolder 'RESULT.json'
        if (-not (Test-Path -LiteralPath $resultPath)) { throw ('Missing actual terminal RESULT; resource state requires review: ' + $condition.tag) }
        $resultValue = Get-Content -LiteralPath $resultPath -Raw -Encoding utf8 | ConvertFrom-Json
        if (($resultValue.dataset -ne $condition.dataset) -or ([int]$resultValue.seed -ne $condition.seed) -or
            ($resultValue.head -ne $condition.head) -or ($resultValue.tag -ne $condition.tag)) { throw 'Actual producer identity mismatch' }
        $resourceRecord = $resultValue.shared_resource_receipt
        if (-not $resourceRecord.path -or -not $resourceRecord.sha256) { throw 'Actual terminal resource receipt unavailable; review before continuing' }
        $resourcePath = if ([System.IO.Path]::IsPathRooted($resourceRecord.path)) { $resourceRecord.path } else { Join-Path $projectRoot $resourceRecord.path }
        if ((Get-FileHash -LiteralPath $resourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $resourceRecord.sha256) { throw 'Actual terminal resource receipt SHA mismatch' }
        $actualResource = Get-Content -LiteralPath $resourcePath -Raw -Encoding utf8 | ConvertFrom-Json
        if (Test-Path -LiteralPath $activeLeasePath) { throw 'Actual process ended with active lease preserved; review before continuing' }
        $failureMessages = @($resultValue.failure.message, $resultValue.resource_terminal_error.message,
                             $resourceRecord.terminal_error.message, $resourceRecord.check_failure.message) | Where-Object { $_ }
        $localOnlyHold = ($resultValue.status -eq 'BUDGET_HELD_PRESERVED') -and
            ($failureMessages -contains 'FORECAST_SEQUENCE_FIT_PROCESS_WALL_150S') -and
            (@($failureMessages | Where-Object { $_ -ne 'FORECAST_SEQUENCE_FIT_PROCESS_WALL_150S' }).Count -eq 0)
        $row = [pscustomobject]@{
            tag=$condition.tag;dataset=$condition.dataset;seed=$condition.seed;head=$condition.head
            exit_code=$processExit;status=$resultValue.status;selected_new_update=$resultValue.selected_new_update
            actual_optimizer_updates=@($resultValue.curve).Count
            actual_saved_checkpoint_updates=@($resultValue.checkpoints | ForEach-Object { $_.new_update })
            learning_executed=$resultValue.learning_executed
            producer_result=[pscustomobject]@{path=$resultPath;sha256=(Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash.ToLowerInvariant()}
            shared_resource_receipt=$resourceRecord;actual_process_wall_seconds=$actualResource.wall_seconds
            local_fit_cap_only_hold=$localOnlyHold;failure=$resultValue.failure
        }
        $rows.Add($row)
        $line = $row | ConvertTo-Json -Compress -Depth 8
        Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
        Write-Output $line
        if (($resultValue.status -eq 'BUDGET_HELD_PRESERVED') -and -not $localOnlyHold) {
            $queueStatus = 'BUDGET_HELD_PRESERVED'
            $queueError = 'Global campaign/device/GPU/CPU/storage resource hold; pending independent fits preserved'
            break
        }
        if (($resultValue.status -notin @('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED')) -or
            ($resourceRecord.terminal_status -notin @('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED'))) {
            throw 'Actual condition or resource receipt is not terminal'
        }
        # A path failure or only the local 150s fit hold does not invalidate other conditions.
        # The next iteration rechecks all global resources and the exclusive lease.
    }
    if ($queueStatus -eq 'RUNNING') {
        $failedRows = @($rows | Where-Object { ($_.status -ne 'COMPLETE') -or ($_.exit_code -ne 0) })
        $queueStatus = if (($rows.Count -eq 8) -and ($failedRows.Count -eq 0)) { 'COMPLETE' } else { 'PARTIAL_WITH_FAILURES' }
    }
} catch {
    if ($queueStatus -ne 'BUDGET_HELD_PRESERVED') { $queueStatus = 'FAILED_PRESERVED' }
    $queueError = $_.Exception.Message
    Write-Output ($([pscustomobject]@{event='SEQUENCE_QUEUE_FAILED_PRESERVED';message=$queueError}) | ConvertTo-Json -Compress)
} finally {
    $terminal = [pscustomobject]@{
        schema='smallk_forecast_sequence_queue_v1';status=$queueStatus
        expected_conditions=8;actual_terminal_conditions=$rows.Count
        started_utc=$startedAt;finished_utc=[DateTimeOffset]::UtcNow.ToString('o')
        source_sha256=$sourceSha;lock_sha256=$lockSha;runtime_sha256=$runtimeSha;core_runtime_sha256=$coreRuntimeSha;queue_sha256=$queueSourceSha
        authorization=[pscustomobject]@{path=$authorizationPath;sha256=$authorizationSha}
        run_manifest=[pscustomobject]@{path=$manifestPath;sha256=$manifestSha}
        input_bridge=[pscustomobject]@{path=$bridgePath;sha256=$bridgeSha}
        preparation_validation=$preparedValidation
        expected_tags=@($conditions | ForEach-Object { $_.tag })
        partial_checkpoint_resumed=$false;prior_files_overwritten=$false;protected_TEST_accessed=$false
        failure=$queueError;stopped_before_tag=$stoppedBeforeTag;conditions=$rows
    }
    $terminalStream = [System.IO.File]::Open($terminalPath, [System.IO.FileMode]::CreateNew)
    $terminalWriter = [System.IO.StreamWriter]::new($terminalStream, [System.Text.UTF8Encoding]::new($false))
    $terminalWriter.WriteLine(($terminal | ConvertTo-Json -Depth 12))
    $terminalWriter.Dispose()
}
if ($queueStatus -ne 'COMPLETE') { exit 1 }
