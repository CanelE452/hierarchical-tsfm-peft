# Fresh matched-readout serial queue for the three remaining original conditions.
# SOURCE-only until ROOT executes after the first matched_robin_s92601_v1 producer is terminal.
$ErrorActionPreference = 'Stop'

$env:CUDA_VISIBLE_DEVICES = ''
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '2' }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = '2' }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = '2' }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = '2' }

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runnerPath = Join-Path $PSScriptRoot 'run_matched_readout_v1.py'
$runtimePath = Join-Path $PSScriptRoot 'matched_readout_runtime_v1.py'
$lockPath = Join-Path $PSScriptRoot 'METHOD_LOCK_MATCHED_READOUT_v1.md'
$authorizationPath = Join-Path $PSScriptRoot 'matched_readout_authorization_v1.json'
$manifestPath = Join-Path $PSScriptRoot 'matched_readout_run_manifest_v1.json'
$resourceFolder = Join-Path $PSScriptRoot 'matched_readout_resources_v1'
$activeLeasePath = Join-Path $resourceFolder 'ACTIVE.json'
$bridgePath = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/input_bridge/old_conditions_actual_v1/RESULT.json'
$queueFolder = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/matched_readout_v1'
$stageFolder = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/stage_separation_v1/native'
$stageReportManifest = Join-Path $PSScriptRoot 'report_stage_separation_v2/MANIFEST.json'
$queueResult = Join-Path $queueFolder 'MATCHED_READOUT_QUEUE_v1.json'
$queueLog = Join-Path $queueFolder 'MATCHED_READOUT_QUEUE_v1.jsonl'

$expectedRunnerSha = 'f4cff8e6bfe563ff22387e77b9fd5e1ad8e3614b0e89e90678c364ea82077e4a'
$expectedRuntimeSha = '1729067cf8bc9c929eab56fb358dc9187627afa502e9fec5a39d770252234664'
$expectedLockSha = 'ff4da28461baa6d641f6d0460dc9154b35fffa774286c242a1776dcc20f82cf0'
$reserveSeconds = 5.0
$utf8 = New-Object System.Text.UTF8Encoding($false)

$conditions = @(
    [pscustomobject]@{dataset='robin'; short='robin'; seed=92602; tag='matched_robin_s92602_v1'},
    [pscustomobject]@{dataset='beijing_aotizhongxin'; short='beijing'; seed=92601; tag='matched_beijing_s92601_v1'},
    [pscustomobject]@{dataset='beijing_aotizhongxin'; short='beijing'; seed=92602; tag='matched_beijing_s92602_v1'}
)

function Get-ShaLower([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-Json([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Write-NewJson([string]$Path, [object]$Value) {
    $handle = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
    try {
        $bytes = $utf8.GetBytes(($Value | ConvertTo-Json -Depth 80) + [Environment]::NewLine)
        $handle.Write($bytes, 0, $bytes.Length)
    } finally {
        $handle.Dispose()
    }
}

function Append-JsonLine([string]$Path, [object]$Value) {
    [System.IO.File]::AppendAllText($Path, (($Value | ConvertTo-Json -Depth 50 -Compress) + [Environment]::NewLine), $utf8)
}

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) { return $PathValue }
    return (Join-Path $projectRoot $PathValue)
}

function Assert-LockedSources([string]$QueueSha) {
    if ((Get-ShaLower $runnerPath) -ne $expectedRunnerSha) { throw 'Frozen matched runner source changed before dispatch' }
    if ((Get-ShaLower $runtimePath) -ne $expectedRuntimeSha) { throw 'Frozen matched runtime source changed before dispatch' }
    if ((Get-ShaLower $lockPath) -ne $expectedLockSha) { throw 'Frozen matched method lock changed before dispatch' }
    if ((Get-ShaLower $PSCommandPath) -ne $QueueSha) { throw 'Matched queue source changed during dispatch' }
}

function Get-ResourceState([string]$AuthorizationSha, [string]$ManifestSha) {
    $cpuCharged = 0.0
    $gpuCharged = 0.0
    $receipts = [System.Collections.Generic.List[object]]::new()
    if (Test-Path -LiteralPath $resourceFolder) {
        foreach ($receiptFile in Get-ChildItem -LiteralPath $resourceFolder -Filter '*.json' -File) {
            if ($receiptFile.Name -eq 'ACTIVE.json') { continue }
            $receipt = Read-Json $receiptFile.FullName
            if ($receipt.status -notin @('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED')) {
                throw ('Nonterminal matched-readout resource receipt: ' + $receiptFile.FullName)
            }
            if (($receipt.authorization.sha256 -ne $AuthorizationSha) -or ($receipt.run_manifest.sha256 -ne $ManifestSha)) {
                throw ('Matched-readout receipt belongs to a different authorization or manifest: ' + $receiptFile.FullName)
            }
            if ($receipt.gpu) { $gpuCharged += [double]$receipt.wall_seconds } else { $cpuCharged += [double]$receipt.wall_seconds }
            $receipts.Add([pscustomobject]@{path=$receiptFile.FullName;sha256=(Get-ShaLower $receiptFile.FullName);status=$receipt.status;gpu=$receipt.gpu;wall_seconds=[double]$receipt.wall_seconds;tag=$receipt.tag})
        }
    }
    $active = $null
    if (Test-Path -LiteralPath $activeLeasePath) {
        $active = Read-Json $activeLeasePath
    }
    return [pscustomobject]@{cpu_charged=$cpuCharged;gpu_charged=$gpuCharged;receipt_count=$receipts.Count;receipts=$receipts;active_lease=$active}
}

function Get-DescendantProcessIds([int]$RootPid) {
    $all = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    $frontier = [System.Collections.Generic.List[int]]::new()
    $frontier.Add($RootPid)
    $descendants = [System.Collections.Generic.List[int]]::new()
    while ($frontier.Count -gt 0) {
        $parent = $frontier[0]
        $frontier.RemoveAt(0)
        foreach ($candidate in $all | Where-Object { [int]$_.ParentProcessId -eq $parent }) {
            $childId = [int]$candidate.ProcessId
            if (-not $descendants.Contains($childId)) {
                $descendants.Add($childId)
                $frontier.Add($childId)
            }
        }
    }
    return @($descendants)
}

function Stop-NewProcessTree([int]$RootPid) {
    $targets = @($RootPid) + @(Get-DescendantProcessIds -RootPid $RootPid)
    $uniqueTargets = @($targets | Sort-Object -Unique | Sort-Object -Descending)
    foreach ($targetId in $uniqueTargets) {
        try { Stop-Process -Id $targetId -Force -ErrorAction SilentlyContinue } catch { }
    }
    return $uniqueTargets
}

function Get-StageNativeResult([object]$Condition) {
    return Join-Path $stageFolder (('native_' + $Condition.short + '_s' + $Condition.seed + '_v1/RESULT.json'))
}

function Start-MatchedCondition([object]$Condition, [object]$Authorization, [object]$Manifest, [string]$AuthorizationSha, [string]$ManifestSha) {
    $resourceBefore = Get-ResourceState -AuthorizationSha $AuthorizationSha -ManifestSha $ManifestSha
    if ($null -ne $resourceBefore.active_lease) { throw 'Matched-readout active lease present before dispatch; review terminal state first' }
    $nowUnix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $cpuRemainingBefore = [double]$Authorization.limits.cpu_process_wall_seconds - [double]$resourceBefore.cpu_charged
    $wallRemainingBefore = [double]$Manifest.expires_unix - [double]$nowUnix
    if (($cpuRemainingBefore -le $reserveSeconds) -or ($wallRemainingBefore -le $reserveSeconds)) {
        return [pscustomobject]@{status='BUDGET_HELD_PRESERVED';not_started=$true;reason='Less than queue reserve remains before start';resource_before=$resourceBefore;cpu_remaining=$cpuRemainingBefore;wall_remaining=$wallRemainingBefore}
    }

    $stageNativeResult = Get-StageNativeResult -Condition $Condition
    if (-not (Test-Path -LiteralPath $stageNativeResult)) { throw ('Stage native result missing: ' + $stageNativeResult) }
    $runFolder = Join-Path $queueFolder $Condition.tag
    if (Test-Path -LiteralPath $runFolder) { throw ('Fresh matched producer directory already exists: ' + $Condition.tag) }
    $stdoutPath = Join-Path $queueFolder ($Condition.tag + '.stdout.log')
    $stderrPath = Join-Path $queueFolder ($Condition.tag + '.stderr.log')
    if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath)) { throw ('Fresh process logs already exist for tag: ' + $Condition.tag) }

    $argumentList = @(
        $runnerPath, '--execute',
        '--dataset', $Condition.dataset,
        '--seed', [string]$Condition.seed,
        '--input-bridge', $bridgePath,
        '--stage-native-result', $stageNativeResult,
        '--stage-report-manifest', $stageReportManifest,
        '--tag', $Condition.tag
    )
    $process = Start-Process -FilePath $pythonPath -ArgumentList $argumentList -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
    $startedUtc = [DateTimeOffset]::UtcNow
    $startedUnix = $startedUtc.ToUnixTimeSeconds()
    $processBirth = $null
    try { $processBirth = ([DateTimeOffset]$process.StartTime.ToUniversalTime()).ToUnixTimeSeconds() } catch { $processBirth = $startedUnix }
    $killed = $false
    $killReason = $null
    $killedPids = @()
    $monitorSamples = [System.Collections.Generic.List[object]]::new()

    while (-not $process.HasExited) {
        Start-Sleep -Milliseconds 500
        $elapsed = ([DateTimeOffset]::UtcNow - $startedUtc).TotalSeconds
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $projectedCpu = [double]$resourceBefore.cpu_charged + [double]$elapsed
        $wallRemaining = [double]$Manifest.expires_unix - [double]$now
        if (($projectedCpu -ge ([double]$Authorization.limits.cpu_process_wall_seconds - $reserveSeconds)) -or ($wallRemaining -le $reserveSeconds)) {
            $killReason = 'Outer supervisor reserve reached before cooperative runner terminal receipt'
            $killedPids = Stop-NewProcessTree -RootPid $process.Id
            $killed = $true
            break
        }
        if (($monitorSamples.Count -eq 0) -or ($elapsed -ge (($monitorSamples[-1].elapsed_seconds) + 10.0))) {
            $monitorSamples.Add([pscustomobject]@{elapsed_seconds=$elapsed;projected_cpu_charged=$projectedCpu;wall_remaining_seconds=$wallRemaining})
        }
        $process.Refresh()
    }
    try { $process.WaitForExit(5000) | Out-Null } catch { }
    $exitCode = $null
    try { $exitCode = [int]$process.ExitCode } catch { }
    $resultPath = Join-Path $runFolder 'RESULT.json'
    $resourceAfter = Get-ResourceState -AuthorizationSha $AuthorizationSha -ManifestSha $ManifestSha
    $row = [pscustomobject]@{
        dataset=$Condition.dataset;seed=$Condition.seed;tag=$Condition.tag
        started_utc=$startedUtc.ToString('o');root_pid=$process.Id;process_birth_unix=$processBirth
        exit_code=$exitCode;killed_by_queue_supervisor=$killed;kill_reason=$killReason;killed_pids=$killedPids
        stdout_log=$stdoutPath;stderr_log=$stderrPath;stage_native_result=$stageNativeResult
        resource_before=[pscustomobject]@{cpu_charged=$resourceBefore.cpu_charged;gpu_charged=$resourceBefore.gpu_charged;receipt_count=$resourceBefore.receipt_count}
        resource_after=[pscustomobject]@{cpu_charged=$resourceAfter.cpu_charged;gpu_charged=$resourceAfter.gpu_charged;receipt_count=$resourceAfter.receipt_count;active_lease_present=($null -ne $resourceAfter.active_lease)}
        monitor_samples=$monitorSamples
    }
    if (Test-Path -LiteralPath $resultPath) {
        $resultValue = Read-Json $resultPath
        $row | Add-Member -NotePropertyName status -NotePropertyValue $resultValue.status
        $row | Add-Member -NotePropertyName body_wall_seconds -NotePropertyValue $resultValue.body_wall_seconds
        $row | Add-Member -NotePropertyName families -NotePropertyValue $resultValue.families
        $row | Add-Member -NotePropertyName producer_result -NotePropertyValue ([pscustomobject]@{path=$resultPath;sha256=(Get-ShaLower $resultPath)})
        $receiptRecord = $resultValue.resource_receipt
        $row | Add-Member -NotePropertyName resource_receipt -NotePropertyValue $receiptRecord
        if ($receiptRecord.path -and $receiptRecord.sha256) {
            $resourcePath = Resolve-RepoPath $receiptRecord.path
            if ((Get-ShaLower $resourcePath) -ne $receiptRecord.sha256) { throw ('Matched resource receipt SHA mismatch for tag: ' + $Condition.tag) }
        }
        if (($resultValue.dataset -ne $Condition.dataset) -or ([int]$resultValue.seed -ne [int]$Condition.seed)) { throw ('Matched producer identity mismatch: ' + $Condition.tag) }
    } else {
        $status = if ($killed) { 'BUDGET_HELD_PRESERVED' } else { 'FAILED_NO_RESULT' }
        $row | Add-Member -NotePropertyName status -NotePropertyValue $status
        $row | Add-Member -NotePropertyName error -NotePropertyValue 'No producer RESULT.json exists; no terminal scientific status is invented.'
        if ($null -ne $resourceAfter.active_lease) { $row | Add-Member -NotePropertyName active_lease_requires_review -NotePropertyValue $true }
    }
    return $row
}

$queueSha = (Get-ShaLower $PSCommandPath)
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv python is required' }
Assert-LockedSources -QueueSha $queueSha
if (-not (Test-Path -LiteralPath $authorizationPath) -or -not (Test-Path -LiteralPath $manifestPath)) { throw 'Fresh matched authorization and run manifest are required' }
if (-not (Test-Path -LiteralPath $bridgePath)) { throw 'Actual input bridge RESULT is required' }
if (-not (Test-Path -LiteralPath $stageReportManifest)) { throw 'Canonical stage report manifest is required' }
if (-not (Test-Path -LiteralPath $queueFolder)) { New-Item -ItemType Directory -Path $queueFolder | Out-Null }
if ((Test-Path -LiteralPath $queueResult) -or (Test-Path -LiteralPath $queueLog)) { throw 'Fresh matched queue artifacts already exist; do not overwrite/restart' }
foreach ($condition in $conditions) {
    if (Test-Path -LiteralPath (Join-Path $queueFolder $condition.tag)) { throw ('Fresh matched tag already exists: ' + $condition.tag) }
}
$logStream = [System.IO.File]::Open($queueLog, [System.IO.FileMode]::CreateNew)
$logStream.Dispose()

$authorization = Read-Json $authorizationPath
$manifest = Read-Json $manifestPath
$authorizationSha = Get-ShaLower $authorizationPath
$manifestSha = Get-ShaLower $manifestPath
$bridgeSha = Get-ShaLower $bridgePath
if (($authorization.status -ne 'APPROVED') -or ($authorization.origin -ne 'human_user_response') -or
    ($authorization.phase -ne 'matched_readout_information_v1') -or ($authorization.gpu_allowed -ne $false) -or
    ($authorization.cuda_allowed -ne $false) -or ($authorization.F_access_allowed -ne $false) -or
    ($manifest.schema -ne 'smallk_matched_readout_run_manifest_v1') -or ($manifest.phase -ne 'matched_readout_information_v1') -or
    ($manifest.authorization_sha256 -ne $authorizationSha) -or ($manifest.runtime_source_sha256 -ne $expectedRuntimeSha) -or
    ($manifest.gpu_allowed -ne $false) -or ($manifest.cuda_allowed -ne $false) -or ($manifest.F_access_allowed -ne $false)) {
    throw 'Fresh matched authorization or fixed run manifest binding is invalid'
}
$startedAt = [DateTimeOffset]::UtcNow.ToString('o')
$rows = [System.Collections.Generic.List[object]]::new()
$queueStatus = 'RUNNING'
$queueError = $null
$stoppedBeforeTag = $null

try {
    foreach ($condition in $conditions) {
        Assert-LockedSources -QueueSha $queueSha
        if ((Get-ShaLower $authorizationPath) -ne $authorizationSha) { throw 'Matched authorization changed during queue' }
        if ((Get-ShaLower $manifestPath) -ne $manifestSha) { throw 'Matched run manifest changed during queue' }
        if ((Get-ShaLower $bridgePath) -ne $bridgeSha) { throw 'Input bridge changed during queue' }
        $preState = Get-ResourceState -AuthorizationSha $authorizationSha -ManifestSha $manifestSha
        if ($null -ne $preState.active_lease) {
            $queueStatus = 'BUDGET_HELD_PRESERVED'
            $queueError = 'Active matched-readout lease present before start; no new job dispatched.'
            $stoppedBeforeTag = $condition.tag
            break
        }
        $preUnix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        if ((([double]$authorization.limits.cpu_process_wall_seconds - [double]$preState.cpu_charged) -le $reserveSeconds) -or
            (([double]$manifest.expires_unix - [double]$preUnix) -le $reserveSeconds)) {
            $queueStatus = 'BUDGET_HELD_PRESERVED'
            $queueError = 'Matched-readout CPU or wall reserve exhausted before next job.'
            $stoppedBeforeTag = $condition.tag
            break
        }
        Write-Output ($([pscustomobject]@{event='MATCHED_QUEUE_START';dataset=$condition.dataset;seed=$condition.seed;tag=$condition.tag;reserve_seconds=$reserveSeconds}) | ConvertTo-Json -Compress)
        $row = Start-MatchedCondition -Condition $condition -Authorization $authorization -Manifest $manifest -AuthorizationSha $authorizationSha -ManifestSha $manifestSha
        $rows.Add($row)
        Append-JsonLine -Path $queueLog -Value $row
        Write-Output ($([pscustomobject]@{event='MATCHED_QUEUE_CONDITION_TERMINAL';dataset=$row.dataset;seed=$row.seed;tag=$row.tag;status=$row.status;exit_code=$row.exit_code}) | ConvertTo-Json -Compress)
        if ($row.status -eq 'BUDGET_HELD_PRESERVED') {
            $queueStatus = 'BUDGET_HELD_PRESERVED'
            $queueError = 'Matched budget held; remaining conditions not dispatched.'
            break
        }
        if ($row.resource_after.active_lease_present) {
            $queueStatus = 'BUDGET_HELD_PRESERVED'
            $queueError = 'Active matched-readout lease remained after condition; review before continuing.'
            break
        }
    }
    if ($queueStatus -eq 'RUNNING') {
        $queueStatus = if ($rows.Count -eq $conditions.Count -and @($rows | Where-Object { $_.status -ne 'COMPLETE' }).Count -eq 0) { 'COMPLETE' } else { 'FAILED_PRESERVED' }
    }
} catch {
    if ($queueStatus -ne 'BUDGET_HELD_PRESERVED') { $queueStatus = 'FAILED_PRESERVED' }
    $queueError = $_.Exception.Message
    Write-Output ($([pscustomobject]@{event='MATCHED_QUEUE_FAILED_PRESERVED';message=$queueError}) | ConvertTo-Json -Compress)
} finally {
    $finalState = $null
    try { $finalState = Get-ResourceState -AuthorizationSha $authorizationSha -ManifestSha $manifestSha } catch { $finalState = [pscustomobject]@{error=$_.Exception.Message} }
    $terminal = [pscustomobject]@{
        schema='smallk_matched_readout_queue_v1';status=$queueStatus
        expected_remaining_conditions=$conditions.Count;actual_terminal_conditions=$rows.Count
        started_utc=$startedAt;finished_utc=[DateTimeOffset]::UtcNow.ToString('o')
        runner_sha256=$expectedRunnerSha;runtime_sha256=$expectedRuntimeSha;method_lock_sha256=$expectedLockSha;queue_sha256=$queueSha
        authorization=[pscustomobject]@{path=$authorizationPath;sha256=$authorizationSha}
        run_manifest=[pscustomobject]@{path=$manifestPath;sha256=$manifestSha}
        input_bridge=[pscustomobject]@{path=$bridgePath;sha256=$bridgeSha}
        queue_log=[pscustomobject]@{path=$queueLog}
        expected_tags=@($conditions | ForEach-Object { $_.tag })
        previously_started_tag='matched_robin_s92601_v1'
        outer_supervision=[pscustomobject]@{reserve_seconds=$reserveSeconds;kill_scope='only Start-Process root pid and descendants from this queue';never_kill_existing_or_old_jobs=$true}
        partial_checkpoint_resumed=$false;prior_files_overwritten=$false;protected_TEST_accessed=$false;GPU_execution=$false
        stopped_before_tag=$stoppedBeforeTag;failure=$queueError;final_resource_state=$finalState;conditions=$rows
    }
    Write-NewJson -Path $queueResult -Value $terminal
    Write-Output "MATCHED_QUEUE_TERMINAL status=$queueStatus completed=$($rows.Count)"
}

if ($queueStatus -eq 'COMPLETE') { exit 0 }
exit 1
