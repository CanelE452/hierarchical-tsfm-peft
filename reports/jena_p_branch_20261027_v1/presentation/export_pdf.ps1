param(
    [Parameter(Mandatory=$true)][string]$PptxPath,
    [Parameter(Mandatory=$true)][string]$PdfPath
)
$ErrorActionPreference = 'Stop'
$presentationRoot = [System.IO.Path]::GetFullPath($PSScriptRoot) + [System.IO.Path]::DirectorySeparatorChar
$inputFile = [System.IO.Path]::GetFullPath($PptxPath)
$outputFile = [System.IO.Path]::GetFullPath($PdfPath)
$normalizedFile = [System.IO.Path]::ChangeExtension($inputFile, '.office.pptx')
if (!$inputFile.StartsWith($presentationRoot, [System.StringComparison]::OrdinalIgnoreCase) -or !$outputFile.StartsWith($presentationRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Presentation export is restricted to this campaign presentation folder.'
}
$powerPointWasRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$powerPointApp = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $powerPointApp.Presentations.Open2007($inputFile, 0, 0, 0, -1)
    # Normalize the generated OOXML through Office, preserving editable shapes and notes.
    $deck.SaveAs($normalizedFile, 24)
    $deck.SaveAs($outputFile, 32)
    Write-Output $outputFile
} finally {
    if ($null -ne $deck) { $deck.Close(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($deck) }
    if (!$powerPointWasRunning) { $powerPointApp.Quit() }
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPointApp)
}
Copy-Item -LiteralPath $normalizedFile -Destination $inputFile -Force
Remove-Item -LiteralPath $normalizedFile
