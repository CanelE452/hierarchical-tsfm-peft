$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runnerPath = Join-Path $PSScriptRoot 'run_joint_objectives_v2.py'
$lockPath = Join-Path $PSScriptRoot 'METHOD_LOCK_JOINT_v2.md'
$bridgePath = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/input_bridge/old_conditions_actual_v1/RESULT.json'
$queueFolder = Join-Path $projectRoot '.cache/smallk_failure_followup_20261014/joint_objectives'
$sourceSha = '352cb68c499a84ce9e5e32b95f4adfb2af96c89b226da28eec44630a3b26d21f'
$lockSha = '43902321ae0a2caec166b050171f6fa164165dba727cf4d8bfabafd356e4e101'
if (-not (Test-Path -LiteralPath $queueFolder)) { New-Item -ItemType Directory -Path $queueFolder | Out-Null }
$logPath = Join-Path $queueFolder 'JOINT_QUEUE_v2.jsonl'
$terminalPath = Join-Path $queueFolder 'JOINT_QUEUE_v2.json'
if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $terminalPath)) { throw 'Queue artifacts already exist' }
$logStream = [System.IO.File]::Open($logPath, [System.IO.FileMode]::CreateNew)
$logStream.Dispose()
$rows = [System.Collections.Generic.List[object]]::new()
$queueStatus = 'COMPLETE'
$startedAt = [DateTimeOffset]::UtcNow.ToString('o')
try {
    foreach ($datasetName in @('robin', 'beijing_aotizhongxin')) {
        foreach ($seedValue in @(92601, 92602)) {
            foreach ($objectiveName in @('Y', 'RAW', 'MIXED')) {
                if (((Get-FileHash -LiteralPath $runnerPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceSha) -or
                    ((Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $lockSha)) { throw 'Locked runner or protocol source changed' }
                $runTag = 'joint_' + $datasetName + '_' + $objectiveName.ToLowerInvariant() + '_s' + $seedValue + '_v2'
                $runFolder = Join-Path $queueFolder $runTag
                if (Test-Path -LiteralPath $runFolder) { throw ('Run folder already exists: ' + $runFolder) }
                & $pythonPath $runnerPath --execute --dataset $datasetName --seed $seedValue --objective $objectiveName --input-bridge $bridgePath --tag $runTag
                $processExit = $LASTEXITCODE
                $resultPath = Join-Path $runFolder 'RESULT.json'
                if (-not (Test-Path -LiteralPath $resultPath)) { throw ('Missing actual terminal result: ' + $runTag) }
                $resultValue = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
                $row = [pscustomobject]@{tag=$runTag;dataset=$datasetName;seed=$seedValue;objective=$objectiveName;exit_code=$processExit;status=$resultValue.status;selected_new_update=$resultValue.selected_new_update;result_path=$resultPath;result_sha256=(Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash.ToLowerInvariant()}
                $rows.Add($row)
                $line = $row | ConvertTo-Json -Compress
                Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
                Write-Output $line
                if ($resultValue.status -eq 'BUDGET_HELD_PRESERVED') { $queueStatus = 'BUDGET_HELD_PRESERVED'; throw 'Shared or local resource guard held this run; queue stopped with existing artifacts preserved' }
                if (($processExit -ne 0) -or ($resultValue.status -ne 'COMPLETE')) { $queueStatus = 'PARTIAL_WITH_FAILURES' }
            }
        }
    }
} catch {
    if ($queueStatus -ne 'BUDGET_HELD_PRESERVED') { $queueStatus = 'FAILED_PRESERVED' }
    $queueError = $_.Exception.Message
    throw
} finally {
    $terminal = [pscustomobject]@{status=$queueStatus;expected_conditions=12;actual_terminal_conditions=$rows.Count;started_utc=$startedAt;finished_utc=[DateTimeOffset]::UtcNow.ToString('o');source_sha256=$sourceSha;lock_sha256=$lockSha;failure=$queueError;conditions=$rows}
    $payload = $terminal | ConvertTo-Json -Depth 5
    $terminalStream = [System.IO.File]::Open($terminalPath, [System.IO.FileMode]::CreateNew)
    $terminalWriter = [System.IO.StreamWriter]::new($terminalStream, [System.Text.UTF8Encoding]::new($false))
    $terminalWriter.WriteLine($payload)
    $terminalWriter.Dispose()
}
