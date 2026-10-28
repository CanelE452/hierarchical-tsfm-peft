param([switch]$Final)
$ErrorActionPreference = 'Stop'
$runtimeRoot = 'C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies'
$nodeExe = Join-Path $runtimeRoot 'node/bin/node.exe'
$pythonExe = Join-Path $runtimeRoot 'python/python.exe'
$pdfRenderExe = Join-Path $runtimeRoot 'native/poppler/Library/bin/pdftoppm.exe'
$previousNodePath = $env:NODE_PATH
$previousPythonUtf8 = $env:PYTHONUTF8
$env:NODE_PATH = Join-Path $runtimeRoot 'node/node_modules'
$env:PYTHONUTF8 = '1'
$buildArgs = @((Join-Path $PSScriptRoot 'build_deck.cjs'))
$checkArgs = @((Join-Path $PSScriptRoot 'validate_deck.py'))
if ($Final) { $buildArgs += '--final'; $checkArgs += '--final' }
try {
    & $nodeExe @buildArgs
    if ($LASTEXITCODE -ne 0) { throw 'Deck creation did not pass the data gate.' }
    & $pythonExe @checkArgs --decorate
    if ($LASTEXITCODE -ne 0) { throw 'Native chart decoration failed.' }
    $stem = if ($Final) { 'Temporal_Channel_Input_Adaptation_revised' } else { 'Temporal_Channel_Adaptation_Jena_DRAFT' }
    $pptxPath = Join-Path $PSScriptRoot ($stem+'.pptx')
    $pdfPath = Join-Path $PSScriptRoot ($stem+'.pdf')
    & (Join-Path $PSScriptRoot 'export_pdf.ps1') -PptxPath $pptxPath -PdfPath $pdfPath
    $powerPointWasRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
    $checkApp = New-Object -ComObject PowerPoint.Application
    $checkDeck = $null
    try {
        $checkDeck = $checkApp.Presentations.Open($pptxPath, -1, 0, 0)
        if ($checkDeck.Slides.Count -ne 19) { throw 'Expected 19 slides in normal PowerPoint open.' }
        $openCheck = @{status='PASS';normal_open=$true;slide_count=$checkDeck.Slides.Count;final=[bool]$Final}
        $openCheck | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot ('OFFICE_OPEN_'+$(if($Final){'FINAL'}else{'DRAFT'})+'.json')) -Encoding utf8
    } finally {
        if ($null -ne $checkDeck) { $checkDeck.Close(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($checkDeck) }
        if (!$powerPointWasRunning) { $checkApp.Quit() }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($checkApp)
    }
    $renderName = $(if($Final){'final_'}else{'draft_'})+(Get-Date -Format 'yyyyMMdd_HHmmss')
    $renderDir = Join-Path (Join-Path $PSScriptRoot 'renders') $renderName
    [void](New-Item -ItemType Directory -Path $renderDir)
    & $pdfRenderExe -png -scale-to 1600 $pdfPath (Join-Path $renderDir 'slide')
    if ($LASTEXITCODE -ne 0) { throw 'PDF slide rendering failed.' }
    & $pythonExe @checkArgs --renders $renderDir
    if ($LASTEXITCODE -ne 0) { throw 'Presentation validation failed.' }
    Write-Output ('Rendered slides: '+$renderDir)
} finally {
    $env:NODE_PATH = $previousNodePath
    $env:PYTHONUTF8 = $previousPythonUtf8
}
