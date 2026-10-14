$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $workspace '.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot 'run_stage_native_ridge_v1.py'
$expectedRunnerSha = '38a64a4fd17616488a632cde4c212a6b699bafc9eaa7aad69b47d65e27ce3a38'
if ((Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash.ToLower() -ne $expectedRunnerSha) {
    throw 'Frozen native scientific source changed before dispatch'
}
$cache = Join-Path $workspace '.cache/smallk_failure_followup_20261014'
$bridge = Join-Path $cache 'input_bridge/old_conditions_actual_v1/RESULT.json'
$folder = Join-Path $cache 'stage_separation_v1/native'
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$queueResult = Join-Path $folder 'NATIVE_QUEUE_v1.json'
$queueLog = Join-Path $folder 'NATIVE_QUEUE_v1.jsonl'
if ((Test-Path -LiteralPath $queueResult) -or (Test-Path -LiteralPath $queueLog)) {
    throw 'Fresh native queue artifact already exists; do not overwrite/restart'
}
$conditions = @(
    @{dataset='robin'; short='robin'; seed=92601},
    @{dataset='robin'; short='robin'; seed=92602},
    @{dataset='beijing_aotizhongxin'; short='beijing'; seed=92601},
    @{dataset='beijing_aotizhongxin'; short='beijing'; seed=92602}
)
foreach ($condition in $conditions) {
    $tag = "native_$($condition.short)_s$($condition.seed)_v1"
    if (Test-Path -LiteralPath (Join-Path $folder $tag)) { throw "Fresh producer directory already exists: $tag" }
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$started = [DateTimeOffset]::UtcNow.ToString('o')
$rows = @()
$held = $false
foreach ($condition in $conditions) {
    $tag = "native_$($condition.short)_s$($condition.seed)_v1"
    $d2 = Join-Path $cache "condition_probes/d2_$($condition.short)_s$($condition.seed)_v1/RESULT.json"
    $oracle = Join-Path $cache "stage_separation_v1/oracle/oracle_$($condition.short)_s$($condition.seed)_v1/RESULT.json"
    Write-Output "NATIVE_START dataset=$($condition.dataset) seed=$($condition.seed) tag=$tag"
    & $python $runner --execute --dataset $condition.dataset --seed $condition.seed --input-bridge $bridge --d2-result $d2 --oracle-result $oracle --tag $tag
    $exitCode = $LASTEXITCODE
    $resultPath = Join-Path $folder "$tag/RESULT.json"
    $row = @{dataset=$condition.dataset;seed=$condition.seed;tag=$tag;exit_code=$exitCode;producer_path=$resultPath}
    if (Test-Path -LiteralPath $resultPath) {
        $producer = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row.status = $producer.status
        $row.producer_sha256 = (Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash.ToLower()
        $row.total_runner_wall_seconds = $producer.total_runner_wall_seconds
        $row.resource_receipt = $producer.resource_receipt
        $row.family_statuses = @{}
        foreach ($family in @('O1','N1','N2','N3')) { $row.family_statuses[$family] = $producer.families.$family.status }
        if ($producer.status -eq 'BUDGET_HELD_PRESERVED') { $held = $true }
    } else {
        $row.status = 'FAILED_NO_RESULT'
        $row.error = 'Inspect retained process output; no completed producer is invented.'
        if ($exitCode -ne 0) {
            $lease = Join-Path $PSScriptRoot 'stage_separation_resources_v1/ACTIVE.json'
            if (Test-Path -LiteralPath $lease) { $held = $true; $row.active_lease_requires_review = $true }
        }
    }
    $rows += $row
    $text = $row | ConvertTo-Json -Depth 30 -Compress
    [System.IO.File]::AppendAllText($queueLog, $text + [Environment]::NewLine, $utf8)
    Write-Output "NATIVE_TERMINAL $text"
    if ($held) { Write-Output 'NATIVE_QUEUE_BUDGET_OR_LEASE_HELD: independent remaining conditions not dispatched'; break }
}
$complete = ($rows.Count -eq 4 -and @($rows | Where-Object { $_.status -ne 'COMPLETE' -or $_.exit_code -ne 0 }).Count -eq 0)
$status = if ($complete) { 'COMPLETE' } elseif ($held) { 'BUDGET_HELD_PRESERVED' } else { 'FAILED_PRESERVED' }
$summary = @{status=$status;started_utc=$started;finished_utc=[DateTimeOffset]::UtcNow.ToString('o');conditions=$rows;expected_conditions=4;runner_sha256=$expectedRunnerSha;new_gpu_jobs=0;protected_TEST_accessed=$false;old_campaign_restart=$false}
$handle = [System.IO.File]::Open($queueResult, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try { $bytes = $utf8.GetBytes(($summary | ConvertTo-Json -Depth 40) + [Environment]::NewLine); $handle.Write($bytes,0,$bytes.Length) } finally { $handle.Dispose() }
Write-Output "NATIVE_QUEUE_TERMINAL status=$status completed=$($rows.Count)"
if ($complete) { exit 0 } else { exit 1 }
