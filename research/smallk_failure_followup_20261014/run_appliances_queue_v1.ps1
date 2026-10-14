$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runnerPath = Join-Path $PSScriptRoot 'run_appliances_losses_v1.py'
$lockPath = Join-Path $PSScriptRoot 'METHOD_LOCK_APPLIANCES_LOSS_v1.md'
$bridgePath = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/appliances_input_bridge/phase_v2/RESULT.json'
$queueFolder = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/appliances_losses'
$sourceSha = '0e7b969dddf0bd30fc828e88494b8d2e5bdf8ff6fc7f0e5981cfa921d84ddeb3'
$lockSha = '11b275500197bcc77a0c10b6b4d3277ba45760e6c6fd8ffe97c9cadeb632d06c'
if (-not (Test-Path -LiteralPath $queueFolder)) { New-Item -ItemType Directory -Path $queueFolder | Out-Null }
$logPath = Join-Path $queueFolder 'APPLIANCES_QUEUE_v1.jsonl'
$terminalPath = Join-Path $queueFolder 'APPLIANCES_QUEUE_v1.json'
if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $terminalPath)) { throw 'Queue artifacts already exist' }
$logStream = [System.IO.File]::Open($logPath, [System.IO.FileMode]::CreateNew)
$logStream.Dispose()
$rows = [System.Collections.Generic.List[object]]::new()
$queueStatus = 'COMPLETE'
$startedAt = [DateTimeOffset]::UtcNow.ToString('o')
try {
    foreach ($seedValue in @(92601, 92602)) {
        foreach ($objectiveName in @('MSE', 'MAE', 'HUBER_DELTA1')) {
            if (((Get-FileHash -LiteralPath $runnerPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceSha) -or
                ((Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $lockSha)) { throw 'Locked runner or protocol source changed' }
            $runTag = 'appliances_' + $objectiveName.ToLowerInvariant() + '_s' + $seedValue + '_v1'
            $runFolder = Join-Path $queueFolder $runTag
            if (Test-Path -LiteralPath $runFolder) { throw ('Run folder already exists: ' + $runFolder) }
            & $pythonPath $runnerPath --execute --dataset appliances --seed $seedValue --objective $objectiveName --input-bridge $bridgePath --tag $runTag
            $processExit = $LASTEXITCODE
            $resultPath = Join-Path $runFolder 'RESULT.json'
            if (-not (Test-Path -LiteralPath $resultPath)) { throw ('Missing actual terminal result: ' + $runTag) }
            $resultValue = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
            $row = [pscustomobject]@{tag=$runTag;dataset='appliances';seed=$seedValue;objective=$objectiveName;exit_code=$processExit;status=$resultValue.status;selected_new_update=$resultValue.selected_new_update;producer_result=[pscustomobject]@{path=$resultPath;sha256=(Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash.ToLowerInvariant()}}
            $rows.Add($row)
            $line = $row | ConvertTo-Json -Compress -Depth 4
            Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
            Write-Output $line
            if ($resultValue.status -eq 'BUDGET_HELD_PRESERVED') { $queueStatus = 'BUDGET_HELD_PRESERVED'; throw 'Resource guard held this run; pending queue stopped and existing artifacts preserved' }
            if (($processExit -ne 0) -or ($resultValue.status -ne 'COMPLETE')) { $queueStatus = 'PARTIAL_WITH_FAILURES' }
        }
    }
} catch {
    if ($queueStatus -ne 'BUDGET_HELD_PRESERVED') { $queueStatus = 'FAILED_PRESERVED' }
    $queueError = $_.Exception.Message
    throw
} finally {
    $terminal = [pscustomobject]@{status=$queueStatus;schema='smallk_appliances_loss_queue_v1';expected_conditions=6;actual_terminal_conditions=$rows.Count;started_utc=$startedAt;finished_utc=[DateTimeOffset]::UtcNow.ToString('o');source_sha256=$sourceSha;lock_sha256=$lockSha;failure=$queueError;conditions=$rows}
    $terminalStream = [System.IO.File]::Open($terminalPath, [System.IO.FileMode]::CreateNew)
    $terminalWriter = [System.IO.StreamWriter]::new($terminalStream, [System.Text.UTF8Encoding]::new($false))
    $terminalWriter.WriteLine(($terminal | ConvertTo-Json -Depth 6))
    $terminalWriter.Dispose()
}
